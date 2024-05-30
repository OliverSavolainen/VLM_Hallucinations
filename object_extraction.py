import re
import json

def extract_objects_with_bounding_boxes(text):
    pattern = r'(\b\w+\b)\s*\[\[(\d{3},\d{3},\d{3},\d{3})\]\]'
    matches = re.findall(pattern, text)
    return matches

def extract_objects(input_file_path, output_file_path):
    input_jsonl_file_path = input_file_path
    output_jsonl_file_path = output_file_path

    # Read the JSONL file and process each line
    processed_objects = []

    with open(input_jsonl_file_path, 'r') as file:
        for line in file:
            try:
                json_line = json.loads(line)
                file_name = json_line.get("file", "")

                text = json_line.get("text", "")
                
                objects = extract_objects_with_bounding_boxes(text)
                for obj, bbox in objects:
                    processed_objects.append({
                        "question_id": file_name,
                        "object_name": obj.capitalize(),
                        "bounding_box": bbox
                    })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    # Save the processed objects to a new JSONL file
    with open(output_jsonl_file_path, 'w') as output_file:
        for obj in processed_objects:
            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')

    print(f"Objects with bounding boxes saved to '{output_jsonl_file_path}'")
