import argparse
from grounded_pope_pipeline import run_pipeline
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def evaluate_threshold(pipeline_output_file, gt_file):

    pipeline_output = [json.loads(q) for q in open(pipeline_output_file, 'r')]
    ground_truths = [json.loads(q) for q in open(gt_file, 'r')]

    y_pred = []

    for pipe_out in pipeline_output:
        if pipe_out['is_our_hallucination']:
            y_pred.append("Hallucination")
        elif pipe_out['is_misclassification']:
            y_pred.append("Misclassification")
        else:
            y_pred.append("Correct")

    y_true = [ground_truth["ground_truth"] for ground_truth in ground_truths]


    labels = ["Correct", "Hallucination", "Misclassification"]
    labels_true = ["Correct_Truth", "Hallucination_Truth", "Misclassification_Truth"]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels_true, columns=labels)
    print(conf_matrix_df)

    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    # Append confusion matrix to CSV file
    with open('confusion_matrix.csv', 'a') as f:
        conf_matrix_df.to_csv(f, header=f.tell() == 0)  # Write header only if file is empty

    # Append classification report to text file
    with open('classification_report.txt', 'a') as f:
        f.write('\n' + '-' * 40 + '\n')  # Add a separator for readability
        f.write(report)






















if __name__ == '__main__':

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for threshold in thresholds:
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
        parser.add_argument("--bbox_output_file", type=str,
                            default='intermediate_outputs/pope_objects_with_bboxes.jsonl',
                            help="Path to JSONL file for extracted objects and their bounding boxes.")
        # POPE Labels
        parser.add_argument("--pope_labels_file", type=str, default="coco/coco_pope_adversarial.json",
                            help="POPE labels file, value for this determines if answers from POPE are considered when evaluating.")
        # Segmentation masks to be used for background foreground separation
        parser.add_argument("--segmentation_masks", type=str,
                            default="data/segmentation_masks/segmentation_masks_ade20k.h5",
                            help="Path to the h5 file containing segmentation masks.")
        # Ratio of background pixels to be considered hallucination
        parser.add_argument("--hallucination_threshold", type=float, default=threshold,
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


        pipeline_output_file = f"pipeline_outputs/G_POPE_cog_vlm_ade20k_{threshold}.jsonl"

        evaluate_threshold(pipeline_output_file=pipeline_output_file, gt_file="labeled_grounded_pope_answers.jsonl")