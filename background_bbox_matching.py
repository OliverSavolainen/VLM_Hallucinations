import json
import numpy as np
import os
import h5py

def match_bbox_with_background(extracted_objects_path='intermediate_outputs/objects_with_bounding_boxes.jsonl',
                               segmentations_path='data/segmentation_masks/segmentation_masks_ade20k.h5',
                               label_file_path='data/pope_labels.jsonl',
                               output_path="pipeline_outputs/object_background_matching.jsonl",
                               hallucination_threshold=0.75,model_bbox_scale=1000.0, model_name="cogvlm"):

    segmentations = {}
    # Read segmentation masks
    with h5py.File(segmentations_path, "r") as hf:
        for image_name in hf.keys():
            segmentations[image_name] = hf.get(image_name)[:]

    # Read objects and bounding boxes
    with open(extracted_objects_path, 'r') as file:
        extracted_objects = [json.loads(line) for line in file]


    labels = load_labels(label_file_path)

    valid_objects = []
    for obj in extracted_objects:
        image_name = obj['question_id']
        bbox = obj["bounding_box"]

        label = find_label_for_prompt(obj["prompt"], image_name, labels)
        if label is None:
            continue

        obj['label'] = label

        # No answers, for no answer we do the same as POPE
        if obj["is_no_answer"]:
            # Incorrect label
            if label == "yes":
                obj["is_pope_hallucination"] = True
                obj["is_our_hallucination"] = True
                obj['is_misclassification'] = False
            # Correct label
            else:
                obj["is_pope_hallucination"] = False
                obj["is_our_hallucination"] = False
                obj['is_misclassification'] = False

        # Yes answer, for yes answers POPE only considers label we consider segmentations
        else:
            # POPE hallucinatios
            # Correct label
            if label == "yes":
                obj["is_pope_hallucination"] = False

            # Incorrect label
            else:
                obj["is_pope_hallucination"] = True


            # if yes with empty bbox we do same as pope
            if bbox == "":
                obj['is_our_hallucination'] = obj["is_pope_hallucination"]
                obj['is_misclassification'] = False
                valid_objects.append(obj)
                continue

            # compare bbox

            # Scale bbox coords to segmentation mask size
            segmentation_mask = segmentations[image_name]
            x0, y0, x1, y1 = scale_bbox(bbox, segmentation_mask.shape,model_bbox_scale)
            # Get the segmentation mask corresponding to the bbox
            masked_bbox = segmentation_mask[x0:x1, y0:y1]
            # Count number of background pixels (zeros)
            num_zeros = np.sum(masked_bbox == 0)
            background_ratio = num_zeros / masked_bbox.size if num_zeros != 0 else 0.0

            obj['background_ratio'] = background_ratio
            if background_ratio > hallucination_threshold:
                obj['is_our_hallucination'] = True
                obj['is_misclassification'] = False
            else:
                obj['is_our_hallucination'] = False
                obj['is_misclassification'] = True

        valid_objects.append(obj)

    segmentation_mask_type = segmentations_path.split('/')[-1].replace('.h5', '')
    output_path = output_path.replace('.jsonl', "_" + segmentation_mask_type + "_" + str(hallucination_threshold) + "_" + model_name + ".jsonl")

    with open(output_path, 'w') as file:
        _ = [file.write(json.dumps(obj) + "\n") for obj in valid_objects]

def scale_bbox(bbox, new_scale,model_bbox_scale):
    corners = bbox.split(",")
    float_corners = [float(corner) / model_bbox_scale for corner in corners]
    scaled_x0, scaled_x1 = (int(float_corners[0] * new_scale[0]),
                            int(float_corners[2] * new_scale[0]))
    scaled_y0, scaled_y1 = (int(float_corners[1] * new_scale[1]),
                            int(float_corners[3] * new_scale[1]))
    return scaled_x0, scaled_y0, scaled_x1, scaled_y1

def load_labels(label_file_path):
    labels = {}
    with open(label_file_path, 'r') as file:
        for line in file:
            label = json.loads(line)
            image_name = label['image']
            prompt = label['text']
            labels[(image_name, prompt)] = label['label']
    return labels

def find_label_for_prompt(prompt, image_name, labels):
    for (img, q_prompt), label in labels.items():
        if img == image_name and q_prompt in prompt:
            return label
    return None