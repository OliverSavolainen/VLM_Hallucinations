import json
import argparse

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Grounded VLM Hallucination Complete Evaluation")
# Input answer file
parser.add_argument("--input_file", type=str, default="pipeline_outputs/background_hallucinations_ade20k_0.75.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
# Final output file
parser.add_argument("--output_file", type=str, default="results/background_results.jsonl", help="Path to results JSONL file.")
args = parser.parse_args()

ans_file = args.input_file
output_file = args.output_file

answers = [json.loads(q) for q in open(ans_file, 'r')]

# Count the number of hallucinations and total objects
hallucination_count = sum(1 for answer in answers if answer['is_hallucination'])
total_objects = len(answers)

# Calculate accuracy
hallucination_rate = hallucination_count / total_objects if total_objects > 0 else 0
accuracy = (total_objects - hallucination_count) / total_objects if total_objects > 0 else 0

print('Total objects: {}'.format(total_objects))
print('Hallucinations: {}'.format(hallucination_count))
print('Hallucination Rate: {}'.format(hallucination_rate))
print('Accuracy: {}'.format(accuracy))

results = {
    'Total objects': total_objects,
    'Hallucinations': hallucination_count,
    'Hallucination Rate': hallucination_rate,
    'Accuracy': accuracy
}

with open(output_file, 'w') as f:
    f.write(json.dumps(results) + "\n")
