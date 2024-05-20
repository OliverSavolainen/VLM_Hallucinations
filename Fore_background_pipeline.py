import json
import argparse
from sentence_transformers import SentenceTransformer, util

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Evaluation with foreground background matching pipeline")
##TODO change it to POPE answer files
parser.add_argument("--input_file", type=str, default="bbox_pope_images/outputs.jsonl", help="Path to the JSONL file containing answers to POPE.")
parser.add_argument("--segmentation_labels", type=str, default="segmentation_masks_voc2012.h5",  help="Path to the h5 file containing segmentation masks.")


parser.add_argument("--output_file", type=str, default="foreground_background_hallucinations.jsonl",help="Path to results JSONL file.")

parser.add_argument("--bbox_output_file",type=str,default='objects_with_bboxes.jsonl',help="Path to JSONL file for extracted objects and their bounding boxes.")


parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")
args = parser.parse_args()
