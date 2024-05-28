import json
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Complete Evaluation")
# Input answer file
parser.add_argument("--input_file", type=str, default="model_outputs/prompts_cogvlm_outputs.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Input the pipeline output
parser.add_argument("--pipeline_output_file", type=str, default="pipeline_outputs/background_hallucinations_ade20k_0.25.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Final output file
parser.add_argument("--output_file", type=str, default="results/background_results.jsonl", help="Path to results JSONL file.")
args = parser.parse_args()

ans_file = args.input_file
pipeline_output_file = args.pipeline_output_file
output_file = args.output_file

answers = [json.loads(q) for q in open(ans_file, 'r')]

pipeline_out =[json.loads(q) for q in open(pipeline_output_file, 'r')]

# Count the number of hallucinations and total objects
hallucination_count = sum(1 for pipeline_out in pipeline_out if pipeline_out['is_hallucination'])
misclassification_count = sum(1 for pipeline_out in pipeline_out if 'is_misclassification' in pipeline_out and pipeline_out['is_misclassification'])
total_objects = len(pipeline_out)

# Calculate accuracy
hallucination_rate = hallucination_count / total_objects if total_objects > 0 else 0
misclassification_rate = misclassification_count / total_objects if total_objects > 0 else 0
accuracy = (total_objects - hallucination_count - misclassification_count) / total_objects if total_objects > 0 else 0

print('Total objects: {}'.format(total_objects))
print('Hallucinations: {}'.format(hallucination_count))
print('Hallucination Rate: {}'.format(hallucination_rate))
print('Misclassifications: {}'.format(misclassification_count))
print('Misclassification Rate: {}'.format(misclassification_rate))
print('Accuracy: {}'.format(accuracy))

results = {
    'Total objects': total_objects,
    'Hallucinations': hallucination_count,
    'Hallucination Rate': hallucination_rate,
    'Accuracy': accuracy
}

with open(output_file, 'w') as f:
    f.write(json.dumps(results) + "\n")
