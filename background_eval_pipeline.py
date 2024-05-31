import json
import argparse
import pope_object_extraction
import background_bbox_matching
from sentence_transformers import SentenceTransformer, util

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Evaluation with foreground background matching pipeline")
##TODO change it to POPE answer files
parser.add_argument("--input_file", type=str, default="model_outputs/prompts_cogvlm_outputs.jsonl", help="Path to the JSONL file containing answers to POPE.")
parser.add_argument("--segmentation_masks", type=str, default="data/segmentation_masks/segmentation_masks_ade20k.h5",  help="Path to the h5 file containing segmentation masks.")

# Model bbox scale
parser.add_argument("--model_bbox_scale", type=float, default=1000.0,  help="Coordinate range of the grounded model bounding boxes in (max_width, max_height) format as tuple of floats. "
                                                                                     "1000.0 for CogVLM, 1.0 for Shikra.")
# Model name
parser.add_argument("--model_name", type=str, default="cog_vlm",  help="Name of the grounded model.")

# Bounding box regex
parser.add_argument("--bbox_regex", type=str, default="\[\[(\d{3},\d{3},\d{3},\d{3})\]\]",  help="Regex of bounding box depending on the grounded model bbox format. Cogvlm = \[\[(\d{3},\d{3},\d{3},\d{3})\]\] "
                                                                                                 ", shikra = \[(\d+\.\d+(?:,\d+\.\d+){3}(?:;\d+\.\d+(?:,\d+\.\d+){3})*)\]  ")

parser.add_argument("--output_file", type=str, default="pipeline_outputs/background_hallucinations.jsonl",help="Path to results JSONL file.")

parser.add_argument("--bbox_output_file",type=str,default='intermediate_outputs/pope_objects_with_bboxes.jsonl',help="Path to JSONL file for extracted objects and their bounding boxes.")
parser.add_argument("--hallucination_threshold", type=float, default=0.75,help="Ratio of background pixels in bbox to classify as hallucination. Float between 0.0 and 1.0.")

parser.add_argument("--pope_questions_file", type=str,default="coco/coco_pope_adversarial.json", help="POPE answer file, value for this determines if answers from POPE are considered when evaluating.")
parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")
args = parser.parse_args()

# Extract object and bounding boxes
pope_object_extraction.extract_objects(args.input_file, args.bbox_output_file,args.bbox_regex)

# Check if bbox corresponds to background or foreground
background_bbox_matching.match_bbox_with_background(extracted_objects_path=args.bbox_output_file,
                            segmentations_path=args.segmentation_masks,output_path=args.output_file,
                            hallucination_threshold=args.hallucination_threshold,
                              label_file_path=args.pope_questions_file, model_bbox_scale=args.model_bbox_scale, model_name=args.model_name)

