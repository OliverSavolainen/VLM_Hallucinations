import json
import argparse

parser = argparse.ArgumentParser(description="Grounded VLM Hallucination and Misclassification Evaluation")
parser.add_argument("--input_file", type=str, default="pipeline_outputs/bbox_hallucinations_hth_0.5_mth_0.5.jsonl", help="Path to the JSONL file containing generated captions by the VLM.")
parser.add_argument("--output_file", type=str, default="results/bbox_results.jsonl", help="Path to results JSONL file.")
args = parser.parse_args()

ans_file = args.input_file
output_file = args.output_file

answers = [json.loads(q) for q in open(ans_file, 'r')]

# Count the number of hallucinations, misclassifications, and total objects
hallucination_count = sum(1 for answer in answers if answer['is_hallucination'] and not answer['is_misclassification'])
misclassification_count = sum(1 for answer in answers if answer['is_misclassification'])
#both_count = sum(1 for answer in answers if 'is_hallucination' in answer and answer['is_misclassification'])
total_objects = len(answers)

# Calculate rates for hallucinations and misclassifications
hallucination_rate = hallucination_count / total_objects if total_objects > 0 else 0
misclassification_rate = misclassification_count  / total_objects if total_objects > 0 else 0
accuracy_hallucination = (total_objects - hallucination_count) / total_objects if total_objects > 0 else 0
accuracy_misclassification = (total_objects - misclassification_count) / total_objects if total_objects > 0 else 0

print('Total objects: {}'.format(total_objects))
print('Hallucinations: {}'.format(hallucination_count))
print('Misclassifications: {}'.format(misclassification_count))
print('Hallucination Rate: {}'.format(hallucination_rate))
print('Misclassification Rate: {}'.format(misclassification_rate))
print('Accuracy (Hallucination): {}'.format(accuracy_hallucination))
print('Accuracy (Misclassification): {}'.format(accuracy_misclassification))

results = {
    'Total objects': total_objects,
    'Hallucinations': hallucination_count,
    'Misclassifications': misclassification_count,
    'Hallucination Rate': hallucination_rate,
    'Misclassification Rate': misclassification_rate,
    'Accuracy (Hallucination)': accuracy_hallucination,
    'Accuracy (Misclassification)': accuracy_misclassification,
}

with open(output_file, 'w') as f:
    f.write(json.dumps(results) + "\n")