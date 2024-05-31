import json
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Complete Evaluation")
# Input answer file
parser.add_argument("--input_file", type=str, default="model_outputs/prompts_cogvlm_outputs.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Input the pipeline output
parser.add_argument("--pipeline_output_file", type=str, default="pipeline_outputs/background_hallucinations_ade20k_0.75.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Final output file
parser.add_argument("--output_file", type=str, default="results/background_results.jsonl", help="Path to results JSONL file.")
args = parser.parse_args()

ans_file = args.input_file
pipeline_output_file = args.pipeline_output_file
output_file = args.output_file

answers = [json.loads(q) for q in open(ans_file, 'r')]

pipeline_out =[json.loads(q) for q in open(pipeline_output_file, 'r')]

# Count the number of hallucinations and total objects
hallucination_no_bb_count = sum(1 for pipeline_out in pipeline_out if pipeline_out['is_hallucination'] and pipeline_out['bounding_box'] == "")
hallucination_with_bb_and_correct_label_count = sum(1 for pipeline_out in pipeline_out if pipeline_out['is_hallucination'] and pipeline_out['bounding_box'] != "" and pipeline_out['label'] == True)
hallucination_with_bb_and_incorrect_label_count = sum(1 for pipeline_out in pipeline_out if pipeline_out['is_hallucination'] and pipeline_out['bounding_box'] != "" and pipeline_out['label'] == False)
miscl_count = sum(1 for pipeline_out in pipeline_out if not pipeline_out['is_hallucination'] and pipeline_out['bounding_box'] != "" and pipeline_out['label'] == False)

total_objects = len(pipeline_out)

# Calculate accuracy
hallucination_no_bb_rate = hallucination_no_bb_count / total_objects if total_objects > 0 else 0
hallucination_with_bb_and_correct_label_rate = hallucination_with_bb_and_correct_label_count / total_objects if total_objects > 0 else 0
hallucination_with_bb_and_incorrect_label_rate = hallucination_with_bb_and_incorrect_label_count / total_objects if total_objects > 0 else 0
miscl_rate = miscl_count / total_objects if total_objects > 0 else 0
accuracy = (total_objects - hallucination_no_bb_count - hallucination_with_bb_and_correct_label_count - hallucination_with_bb_and_incorrect_label_count) / total_objects if total_objects > 0 else 0

print('Total objects: {}'.format(total_objects))
print('Hallucinations with no bounding boxes: {}'.format(hallucination_no_bb_count))
print('Hallucinations with bounding boxes and "yes" labels: {}'.format(hallucination_with_bb_and_correct_label_count))
print('Hallucinations with bounding boxes and "no" labels: {}'.format(hallucination_with_bb_and_incorrect_label_count))
print('Hallucination rate with no bounding boxes: {}'.format(hallucination_no_bb_rate))
print('Hallucination rate with bounding boxes and "yes" labels: {}'.format(hallucination_with_bb_and_correct_label_rate))
print('Hallucination rate with bounding boxes and "no" labels: {}'.format(hallucination_with_bb_and_incorrect_label_rate))
print('Misclassification rate: {}'.format(miscl_rate))
print('Accuracy: {}'.format(accuracy))

results = {
    'Total objects': total_objects,
    'Accuracy': accuracy
}

with open(output_file, 'w') as f:
    f.write(json.dumps(results) + "\n")
