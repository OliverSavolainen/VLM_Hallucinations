import argparse
import re
import json
from collections import defaultdict

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer, util

import utils

def run_pipeline(args):
    # Extract objects and bboxes from answer file
    processed_objects = []
    with open(args.input_file, 'r') as file:
        for line in file:
            try:
                json_line = json.loads(line)
                file_name = json_line.get("file", "")

                text = json_line.get("text", "")

                prompt = json_line.get("prompt", "")
                object_name = utils.extract_object_name(prompt)

                # If no answer no need for bboxes
                isNoAnswer = utils.extract_no_answers(text)
                if isNoAnswer:
                    processed_objects.append(utils.get_object_output_dict(file_name, prompt, object_name, "", True, False))

                # Yes answers
                else:
                    # Extract bboxes
                    bounding_boxes = utils.extract_objects_with_bounding_boxes(text, args.bbox_regex)
                    # no bbox
                    if len(bounding_boxes) == 0:
                        processed_objects.append(
                            utils.get_object_output_dict(file_name, prompt, object_name, "", False, True))

                    # Only 1 bbox
                    elif len(bounding_boxes) == 1:
                        for obj, bbox in bounding_boxes:
                            processed_objects.append(
                                utils.get_object_output_dict(file_name, prompt, object_name, bbox, False, False))

                    # multiple bboxes select question object
                    else:
                        q_obj_found = False
                        for obj, bbox in bounding_boxes:
                            if q_obj_found:
                                break
                            if obj.lowercase() == object_name.lowercase():
                                processed_objects.append(
                                    utils.get_object_output_dict(file_name, prompt, object_name, bbox, False, False))
                                q_obj_found = True

                        if not q_obj_found:
                            # Not question object bbox
                            processed_objects.append(
                                utils.get_object_output_dict(file_name, prompt, object_name, "", False, True))

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    # Save bboxes to intermediate outputs
    with open(args.bbox_output_file, 'w') as output_file:
        for obj in processed_objects:
            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')

    print(f"Objects with bounding boxes saved to '{args.bbox_output_file}'")
    # Initialize the sentence transformer model
    model = SentenceTransformer(args.sentence_transformer)

    # Load POPE labels
    labels = utils.load_labels(args.pope_labels_file)

    # Read ground truth labels in coco format
    with open(args.bbox_labels_file, 'r') as file:
        annotations = json.load(file)
    # Extract categories from labels
    categories = annotations["categories"]
    object_classes = {category["id"]: category["name"] for category in categories}

    # Get information about images from coco labels
    images = annotations["images"]
    image_sizes = {}
    for image_info in images:
        image_sizes[image_info["id"]] = (image_info["width"], image_info["height"])

    if args.use_segmentation_mask:

        segmentations = {}
        # Read segmentation masks
        with h5py.File(args.segmentation_masks, "r") as hf:
            for image_name in hf.keys():
                segmentations[image_name] = hf.get(image_name)[:]
    else:
        # Get annotations
        annotations = annotations['annotations']
        bbox_labels = defaultdict(list)
        for annotation in annotations:
            # Image id as dict key
            id = annotation["image_id"]
            # Add bbox list and object category id as tuple
            bbox_labels[id].append((annotation["bbox"], annotation["category_id"]))

    valid_objects = []
    for obj in processed_objects:
        image_name = obj['question_id']
        bbox = obj["bounding_box"]
        # Extract id from name
        image_id = int(image_name[13:-4])
        # Get image size
        img_size = image_sizes[image_id]

        label = utils.find_label_for_prompt(obj["prompt"], image_name, labels)
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
            # POPE hallucinations
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



            # Scale bbox coords to image size
            x0, y0, x1, y1 = utils.scale_bbox(bbox, img_size, args.model_bbox_scale)
            scaled_bbox = [x0, y0, x1, y1]
            obj["bounding_box"] = scaled_bbox

            # Use background pixel ratio with segmentation masks
            if args.use_segmentation_mask:
                # Compare bbox with background pixels
                segmentation_mask = segmentations[image_name]

                # Get the segmentation mask corresponding to the bbox
                masked_bbox = segmentation_mask[x0:x1, y0:y1]
                # Count number of background pixels (zeros)
                num_zeros = np.sum(masked_bbox == 0)
                background_ratio = num_zeros / masked_bbox.size if num_zeros != 0 else 0.0

                obj['background_ratio'] = background_ratio
                # Background pixels above threshold, is hallucination
                if background_ratio > args.hallucination_threshold:
                    obj['is_our_hallucination'] = True
                    obj['is_misclassification'] = False
                # Background pixels below threshold is misclassification
                else:
                    obj['is_our_hallucination'] = False
                    obj['is_misclassification'] = True
            #Use bbox matching
            else:
                # Get ground truth bounding boxes
                gt_bboxes = bbox_labels[image_id]
                # Find ground truth bbox with largest iou
                bbox_match_box, bbox_match_object, bbox_match_iou = utils.compare_bboxes(scaled_bbox, gt_bboxes)
                # Add matched bbox info
                obj["bbox_match_box"] = bbox_match_box
                obj["bbox_match_object"] = utils.get_object_name(bbox_match_object, object_classes)
                obj["bbox_match_iou"] = bbox_match_iou

                # Hallucination if iou less than threshold
                if bbox_match_iou < args.hallucination_threshold:
                    obj["is_our_hallucination"] = True
                    obj['is_misclassification'] = False
                else:
                    # soft match object name
                    object_embedding = model.encode(object_name, convert_to_tensor=True)
                    matched_class = obj['bbox_match_object']
                    gt_object_embedding = model.encode(matched_class, convert_to_tensor=True)
                    cosine_score = util.cos_sim(object_embedding, gt_object_embedding)
                    obj["cosine_similarity"] = cosine_score.tolist()

                    # above 0.9 match -> correct classification
                    if cosine_score > 0.9:
                        obj["is_our_hallucination"] = False
                        obj["is_misclassification"] = False
                    # below 0.9 but above threshold -> misclassification
                    elif cosine_score > args.misclassification_threshold:
                        obj["is_our_hallucination"] = False
                        obj['is_misclassification'] = True
                    # below threshold -> hallucination
                    else:
                        obj["is_our_hallucination"] = True
                        obj['is_misclassification'] = False

        valid_objects.append(obj)

    segmentation_mask_type = args.segmentation_masks.split('/')[-1].replace('.h5', '')[19:]
    output_path = args.output_file.replace('.jsonl', "_" + args.model_name + "_" + segmentation_mask_type + "_" +
                                           str(args.hallucination_threshold) + ".jsonl")

    with open(output_path, 'w') as file:
        _ = [file.write(json.dumps(obj) + "\n") for obj in valid_objects]

    return valid_objects


if __name__ == '__main__':


    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Grounded POPE pipeline")

    # Grounded model name
    parser.add_argument("--model_name", type=str, default="cog_vlm", help="Name of the grounded model.")

    # Pipeline input, outputs from the grounded model
    parser.add_argument("--input_file", type=str, default="model_outputs/prompts_cogvlm_outputs.jsonl",
                        help="Path to the JSONL file containing model answers to POPE questions.")

    # Output file of the pipeline
    parser.add_argument("--output_file", type=str, default="pipeline_outputs/G_POPE.jsonl",
                        help="Path to results JSONL file.")

    # Intermediate output containing bboxes and objects extracted from answer file
    parser.add_argument("--bbox_output_file", type=str, default='intermediate_outputs/pope_objects_with_bboxes.jsonl',
                        help="Path to JSONL file for extracted objects and their bounding boxes.")

    # POPE Labels
    parser.add_argument("--pope_labels_file", type=str, default="coco/coco_pope_adversarial.json",
                        help="POPE labels file, value for this determines if answers from POPE are considered when evaluating.")
    # COCO Annotations
    parser.add_argument("--bbox_labels_file", type=str, default="data/bbox_pope_images/labels.json",
                        help="Path to the JSONL file containing grountruth bounding boxes.")

    # Segmentation masks to be used for background foreground separation
    parser.add_argument("--segmentation_masks", type=str,
                        default="data/segmentation_masks/segmentation_masks_ade20k.h5",
                        help="Path to the h5 file containing segmentation masks.")

    # Ratio of background pixels to be considered hallucination
    parser.add_argument("--hallucination_threshold", type=float, default=0.9,
                        help="Ratio of background pixels in bbox to classify as hallucination. Float between 0.0 and 1.0.")
    # Misclassification threshold
    parser.add_argument("--misclassification_threshold", type=float, default=0.3,
                        help="Threshold for cosine similarity between generated and best match bounding box object. "
                             "Cosine similarity less than the threshold is considered misclassification. Float between 0.0 and 1.0.")

    # Bbox coordinate scale of the grounded model
    parser.add_argument("--model_bbox_scale", type=float, default=1000.0,
                        help="Coordinate range of the grounded model bounding boxes in (max_width, max_height) format as tuple of floats. "
                             "1000.0 for CogVLM, 1.0 for Shikra.")

    # Bounding box regex
    parser.add_argument("--bbox_regex", type=str, default="\[\[(\d{3},\d{3},\d{3},\d{3})\]\]",
                        help="Regex of bounding box depending on the grounded model bbox format. Cogvlm = \[\[(\d{3},\d{3},\d{3},\d{3})\]\] "
                             ", shikra = \[(\d+\.\d+(?:,\d+\.\d+){3}(?:;\d+\.\d+(?:,\d+\.\d+){3})*)\]  ")

    # Boolean to use segmentation masks or bbox matching
    parser.add_argument("--use_segmentation_mask", type=bool,
                        default=False,
                        help="Flag to use segmentation masks or bbox matching.")

    # Pretrained sentence transformer
    parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",
                        help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")

    args = parser.parse_args()

    run_pipeline(args)




















