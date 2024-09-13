import argparse
import re
import json
import h5py
import numpy as np
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

    segmentations = {}
    # Read segmentation masks
    with h5py.File(args.segmentation_masks, "r") as hf:
        for image_name in hf.keys():
            segmentations[image_name] = hf.get(image_name)[:]

    # Load POPE labels
    labels = utils.load_labels(args.pope_labels_file)

    valid_objects = []
    for obj in processed_objects:
        image_name = obj['question_id']
        bbox = obj["bounding_box"]

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

            # Compare bbox with background pixels
            # Scale bbox coords to segmentation mask size
            segmentation_mask = segmentations[image_name]
            x0, y0, x1, y1 = utils.scale_bbox(bbox, segmentation_mask.shape, args.model_bbox_scale)
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

    # Segmentation masks to be used for background foreground separation
    parser.add_argument("--segmentation_masks", type=str,
                        default="data/segmentation_masks/segmentation_masks_ade20k.h5",
                        help="Path to the h5 file containing segmentation masks.")

    # Ratio of background pixels to be considered hallucination
    parser.add_argument("--hallucination_threshold", type=float, default=0.75,
                        help="Ratio of background pixels in bbox to classify as hallucination. Float between 0.0 and 1.0.")

    # Bbox coordinate scale of the grounded model
    parser.add_argument("--model_bbox_scale", type=float, default=1000.0,
                        help="Coordinate range of the grounded model bounding boxes in (max_width, max_height) format as tuple of floats. "
                             "1000.0 for CogVLM, 1.0 for Shikra.")

    # Bounding box regex
    parser.add_argument("--bbox_regex", type=str, default="\[\[(\d{3},\d{3},\d{3},\d{3})\]\]",
                        help="Regex of bounding box depending on the grounded model bbox format. Cogvlm = \[\[(\d{3},\d{3},\d{3},\d{3})\]\] "
                             ", shikra = \[(\d+\.\d+(?:,\d+\.\d+){3}(?:;\d+\.\d+(?:,\d+\.\d+){3})*)\]  ")

    args = parser.parse_args()

    run_pipeline(args)




















