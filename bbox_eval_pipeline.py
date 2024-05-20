import object_extraction
import object_matching
import bbox_IoU_matching
import json
import argparse
from sentence_transformers import SentenceTransformer, util

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Evaluation Pipeline")
parser.add_argument("--input_file", type=str, default="bbox_pope_images/outputs.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
parser.add_argument("--labels", type=str, default="bbox_pope_images/labels.json",  help="Path to the JSONL file containing grountruth bounding boxes.")


parser.add_argument("--output_file", type=str, default="bbox_hallucinations.jsonl",help="Path to results JSONL file.")

parser.add_argument("--matched_objects_output_file", type=str, default="matched.jsonl",help="Path to output the results JSONL file.")
parser.add_argument("--object_classes_file",type=str,default="coco_classes.jsonl",help="Path to JSONL file containing list of object classes to match the VLM captions against.")
parser.add_argument("--bbox_output_file",type=str,default='objects_with_bounding_boxes.jsonl',help="Path to JSONL file for extracted objects and their bounding boxes.")
parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")
args = parser.parse_args()


# Extract object and bounding boxes
object_extraction.extract_objects(args.input_file, args.bbox_output_file)

# Initialize the sentence transformer model
model = SentenceTransformer(args.sentence_transformer)

# Read object classes
with open(args.object_classes_file, "r") as file:
    object_classes = json.load(file)

# Match extracted object with target objects using a pretrained Sentence Transformer
object_matching.find_coco_matches(args.bbox_output_file, args.matched_objects_output_file,object_classes, model=model)

bbox_IoU_matching.match_bbox_iou(args.labels, args.bbox_output_file, args.output_file)
