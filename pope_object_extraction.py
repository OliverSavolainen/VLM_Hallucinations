import re
import json

def extract_objects_with_bounding_boxes(text):
    # Find all bounding boxes in the format [[123,456,789,012]]
    pattern = r'\[\[(\d{3},\d{3},\d{3},\d{3})\]\]'
    matches = re.findall(pattern, text)
    return matches

def extract_object_name(prompt):
    # Capture the object name from the prompt
    pattern = r'Is there a (\w+) in the image\?'
    match = re.search(pattern, prompt)
    if match:
        return match.group(1)
    return None

def extract_objects(input_file_path, output_file_path):
    # Only if both object name and bounding boxes are found, the object along with the bounding box is appended to processed_objects.
    input_jsonl_file_path = input_file_path
    output_jsonl_file_path = output_file_path

    processed_objects = []

    with open(input_jsonl_file_path, 'r') as file:
        for line in file:
            try:
                json_line = json.loads(line)
                file_name = json_line.get("file", "")

                text = json_line.get("text", "")
                    
                prompt = json_line.get("prompt", "")
                object_name = extract_object_name(prompt)

                if object_name:
                    bounding_boxes = extract_objects_with_bounding_boxes(text)
                    if bounding_boxes:
                        for bbox in bounding_boxes:
                            processed_objects.append({
                                "question_id": file_name,
                                "prompt": prompt,
                                "object_name": object_name.capitalize(),
                                "bounding_box": bbox
                            })
                    else:
                        processed_objects.append({
                        "question_id": file_name,
                        "prompt": prompt,
                        "object_name": object_name.capitalize(),
                        "bounding_box": ""
                        })
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    with open(output_jsonl_file_path, 'w') as output_file:
        for obj in processed_objects:
            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')

    print(f"Objects with bounding boxes saved to '{output_jsonl_file_path}'")