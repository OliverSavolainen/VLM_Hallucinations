import object_extraction
import object_class_matching
import bbox_iou
import argparse
from sentence_transformers import SentenceTransformer


# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Evaluation Pipeline")
# Input answer file
parser.add_argument("--input_file", type=str, default="captioning_outputs_cogvlm.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Ground truth labels
parser.add_argument("--labels", type=str, default="data/bbox_pope_images/labels.json",  help="Path to the JSONL file containing grountruth bounding boxes.")
# Final output file
parser.add_argument("--output_file", type=str, default="pipeline_outputs/bbox_hallucinations.jsonl",help="Path to results JSONL file.")
# Intermediate outputs
parser.add_argument("--extracted_objects_file",type=str,default='intermediate_outputs/objects_with_bounding_boxes.jsonl',help="Path to JSONL file for extracted objects and their bounding boxes.")
parser.add_argument("--iou_matched_objects_file",type=str,default='intermediate_outputs/bbox_iou_matched.jsonl',help="Path to JSONL file for generated objects matched with gt bounding boxes.")
# Thresholds
parser.add_argument("--hallucination_threshold", type=float, default=0.5,help="Threshold for minimum bounding box iou to not classify as hallucination. Float between 0.0 and 1.0.")
parser.add_argument("--misclassification_threshold", type=float, default=0.5,help="Threshold for cosine similarity between generated and best match bounding box object. "
                                                                                  "Cosine similarity less than the threshold is considered misclassification. Float between 0.0 and 1.0.")
# Pretrained sentence transformer
parser.add_argument("--sentence_transformer", type=str, default="all-MiniLM-L6-v2",help="Pretrained sentence transformer model to be used for matching extracted objects to target object classes.")
args = parser.parse_args()


parser.add_argument("--matched_objects_output_file", type=str, default="intermediate_outputs/matched.jsonl",help="Path to output the results JSONL file.")

# Extract object and bounding boxes
object_extraction.extract_objects(input_file_path=args.input_file,
                                  output_file_path=args.extracted_objects_file)

# Match bboxes
bbox_iou.match_bboxes(labels_path=args.labels,
                      extracted_objects_path=args.extracted_objects_file,
                      output_path=args.iou_matched_objects_file,
                      hallucination_threshold=args.hallucination_threshold)

# Initialize the sentence transformer model
model = SentenceTransformer(args.sentence_transformer)

output_file_path = args.output_file.replace('.jsonl',"_hth_" + str(args.hallucination_threshold) + "_mth_" + str(args.misclassification_threshold) + ".jsonl")

# Compare generated object names with best matched bbox object name to identify misclassifications
object_class_matching.match_object_classes(jsonl_file=args.iou_matched_objects_file,
                                           output_file=output_file_path,
                                           model=model,
                                           threshold=args.misclassification_threshold)


# Match extracted object with target objects using a pretrained Sentence Transformer
#object_matching.find_coco_matches(args.bbox_output_file, args.matched_objects_output_file,object_classes, model=model)

#bbox_IoU_matching.match_bbox_iou(args.labels, args.bbox_output_file, args.output_file)
