import jsonlines
import argparse
import re
import json
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from torchvision.ops import box_iou
import utils

def run_pipeline(args):

    # Read the JSONL file and process each line
    extracted_objects = []

    with open(args.input_file, 'r') as file:
        for line in file:
            try:
                json_line = json.loads(line)
                file_name = json_line.get("file", "")

                text = json_line.get("text", "")
                objects = utils.extract_objects_with_bounding_boxes(text, args.bbox_regex)
                for obj, bbox in objects:
                    extracted_objects.append({
                        "question_id": file_name,
                        "object_name": obj.capitalize(),
                        "bounding_box": bbox
                    })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    # Save the processed objects to a new JSONL file
    with open(args.extracted_objects_file, 'w') as output_file:
        for obj in extracted_objects:
            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')

    print(f"Objects with bounding boxes saved to '{args.extracted_objects_file}'")

    # Read ground truth labels in coco format
    with open(args.labels_file, 'r') as file:
        labels = json.load(file)

    # Extract categories from labels
    categories = labels["categories"]
    object_classes = {category["id"]: category["name"] for category in categories}

    # Get information about images from coco labels
    images = labels["images"]

    # Extract image sizes for each image
    image_sizes = {}
    for image_info in images:
        image_sizes[image_info["id"]] = (image_info["width"], image_info["height"])

    # Get annotations
    annotations = labels['annotations']

    # Extract bounding box labels from annotations
    bbox_labels = defaultdict(list)
    for label in annotations:
        # Image id as dict key
        id = label["image_id"]
        # Add bbox list and object category id as tuple
        bbox_labels[id].append((label["bbox"], label["category_id"]))

    for object in extracted_objects:
        # Get image name
        image_name = object['question_id']
        # Extract id from name
        image_id = int(image_name[13:-4])
        # Get image size
        img_size = image_sizes[image_id]
        # Get extracted bbox and scale it img_size
        scaled_bbox = utils.scale_bbox(object["bounding_box"], img_size, args.model_bbox_scale)
        object["bounding_box"] = scaled_bbox
        # Get ground truth bounding boxes
        gt_bboxes = bbox_labels[image_id]
        # Find ground truth bbox with largest iou
        bbox_match_box, bbox_match_object, bbox_match_iou = utils.compare_bboxes(scaled_bbox, gt_bboxes)
        # Add matched bbox info
        object["bbox_match_box"] = bbox_match_box
        object["bbox_match_object"] = utils.get_object_name(bbox_match_object, object_classes)
        object["bbox_match_iou"] = bbox_match_iou

        # Hallucination if iou less than threshold
        if bbox_match_iou < args.hallucination_threshold:
            object["is_hallucination"] = True
        else:
            object["is_hallucination"] = False

    # Save results in jsonl
    with open(args.iou_matched_objects_file, 'w') as file:
        _ = [file.write(json.dumps(object) + "\n") for object in extracted_objects]

    # Initialize the sentence transformer model
    model = SentenceTransformer(args.sentence_transformer)

    output_file_path = args.output_file.replace('.jsonl', "_hth_" + str(args.hallucination_threshold) + "_mth_" + str(
        args.misclassification_threshold) + "_" + args.model_name + ".jsonl")

    for obj in extracted_objects:
        if obj["is_hallucination"]:
            obj["is_misclassification"] = False
            continue

        object_name = obj['object_name']
        object_embedding = model.encode(object_name, convert_to_tensor=True)
        matched_class = obj['bbox_match_object']
        gt_object_embedding = model.encode(matched_class, convert_to_tensor=True)
        cosine_score = util.cos_sim(object_embedding, gt_object_embedding)
        obj["cosine_similarity"] = cosine_score.tolist()

        # above 0.9 match -> correct classification
        if cosine_score > 0.9:
            obj["is_misclassification"] = False
        # below 0.9 but above threshold -> misclassification
        elif cosine_score > args.misclassification_threshold:
            obj['is_misclassification'] = True
        # below threshold -> hallucination
        else:
            obj["is_hallucination"] = True
            obj['is_misclassification'] = False

        # Print and write the output with the matched class
        print(
            f"Image ID: {obj['question_id']}, Object: {object_name}, Matched COCO Class: {matched_class}, Score: {cosine_score}")

    with open(output_file_path, 'w') as file:
        _ = [file.write(json.dumps(object) + "\n") for object in extracted_objects]
































if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Grounded Captioning VLM Hallucination Pipeline")
    # Input answer file
    parser.add_argument("--input_file", type=str, default="captioning_outputs_cogvlm.jsonl",
                        help="Path to the JSONL file containing generated captions by the VLM.")
    # Annotations
    parser.add_argument("--labels_file", type=str, default="data/bbox_pope_images/labels.json",
                        help="Path to the JSONL file containing grountruth bounding boxes.")
    # Intermediate outputs
    # Extracted objects
    parser.add_argument("--extracted_objects_file", type=str,
                        default='intermediate_outputs/objects_with_bounding_boxes.jsonl',
                        help="Path to JSONL file for extracted objects and their bounding boxes.")
    # IoU matched object
    parser.add_argument("--iou_matched_objects_file", type=str, default='intermediate_outputs/bbox_iou_matched.jsonl',
                        help="Path to JSONL file for generated objects matched with gt bounding boxes.")
    # Final output file
    parser.add_argument("--output_file", type=str, default="pipeline_outputs/bbox_hallucinations.jsonl",
                        help="Path to results JSONL file.")
    # Model name
    parser.add_argument("--model_name", type=str, default="cog_vlm", help="Name of the grounded model.")
    # Model bbox scale
    parser.add_argument("--model_bbox_scale", type=float, default=1000.0,
                        help="Coordinate range of the grounded model bounding boxes in (max_width, max_height) format as tuple of floats. "
                             "1000.0 for CogVLM, 1.0 for Shikra.")
    # Bounding box regex
    parser.add_argument("--bbox_regex", type=str, default="\[\[(\d{3},\d{3},\d{3},\d{3})\]\]",
                        help="Regex of bounding box depending on the grounded model bbox format. Cogvlm = \[\[(\d{3},\d{3},\d{3},\d{3})\]\] "
                             ", shikra = \[(\d+\.\d+(?:,\d+\.\d+){3}(?:;\d+\.\d+(?:,\d+\.\d+){3})*)\]  ")
    # Thresholds
    # IoU Hallucination threshold
    parser.add_argument("--hallucination_threshold", type=float, default=0.5,
                        help="Threshold for minimum bounding box iou to not classify as hallucination. Float between 0.0 and 1.0.")
    # Misclassification threshold
    parser.add_argument("--misclassification_threshold", type=float, default=0.3,
                        help="Threshold for cosine similarity between generated and best match bounding box object. "
                             "Cosine similarity less than the threshold is considered misclassification. Float between 0.0 and 1.0.")
    # Pretrained sentence transformer
    parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",
                        help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")
    args = parser.parse_args()

    run_pipeline(args)




