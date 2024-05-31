import re
import json


# cogvlm regex = \[\[(\d{3},\d{3},\d{3},\d{3})\]\]
# shikra regex = \s*\[(\d+\.\d+,\d+\.\d+,\d+\.\d+,\d+\.\d+)\]
def extract_objects_with_bounding_boxes(text,bbox_regex):
    pattern = r'(\b\w+\b)\s*' + bbox_regex
    matches = re.findall(pattern, text)

    results = []
    for obj, bbox_str in matches:
        # Split the bounding box string by semicolons for multiple instances
        bboxes = bbox_str.split(';')
        # Convert each bounding box to a list of floats and pair with the object
        for bbox in bboxes:

            results.append((obj, bbox))


    return results

def extract_objects(input_file_path, output_file_path, bbox_regex):
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
                objects = extract_objects_with_bounding_boxes(text,bbox_regex)
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
