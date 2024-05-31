import re
import json

def extract_no_answers(text):
    pattern = r"\b(no|not)\b"
    matches = re.findall(pattern, text,re.IGNORECASE)
    if len(matches) > 0:
        return True
    else:
        return False



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
def extract_object_name(prompt):
    # Capture the object name from the prompt
    pattern = r'Is there a ([\w\s]+) in the image\?'
    match = re.search(pattern, prompt)
    if match:
        return match.group(1)
    return None

def extract_objects(input_file_path, output_file_path,bbox_regex):
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


                # If no answer no need for bboxes
                isNoAnswer = extract_no_answers(text)
                if isNoAnswer:

                   processed_objects.append({
                            "question_id": file_name,
                            "prompt": prompt,
                            "object_name": object_name.capitalize() if object_name else "",
                            "bounding_box": "",
                            "is_no_answer": isNoAnswer,
                            "is_yes_no_bbox": False
                        })

                # Yes answers
                else:
                    # Extract bboxes
                    bounding_boxes = extract_objects_with_bounding_boxes(text,bbox_regex)

                    # no bbox
                    if len(bounding_boxes) == 0:

                        processed_objects.append({
                            "question_id": file_name,
                            "prompt": prompt,
                            "object_name": object_name.capitalize() if object_name else "",
                            "bounding_box": "",
                            "is_no_answer": isNoAnswer,
                            "is_yes_no_bbox": True
                        })
                    # Only 1 bbox
                    elif len(bounding_boxes) == 1:
                        for obj, bbox in bounding_boxes:
                            processed_objects.append({
                            "question_id": file_name,
                            "prompt": prompt,
                            "object_name": object_name.capitalize() if object_name else "",
                            "bounding_box": bbox,
                            "is_no_answer": isNoAnswer,
                            "is_yes_no_bbox": False
                            })

                    # multiple bboxes select question object
                    else:
                        q_obj_found = False
                        for obj, bbox in bounding_boxes:
                            if q_obj_found:
                                break
                            if obj.lowercase() == object_name.lowercas():
                                processed_objects.append({
                                    "question_id": file_name,
                                    "prompt": prompt,
                                    "object_name": object_name.capitalize() if object_name else "",
                                    "bounding_box": bbox,
                                    "is_no_answer": isNoAnswer,
                                    "is_yes_no_bbox": False
                                })
                                q_obj_found = True
                        if not q_obj_found:
                            # Not question object bbox
                            processed_objects.append({
                                "question_id": file_name,
                                "prompt": prompt,
                                "object_name": object_name.capitalize() if object_name else "",
                                "bounding_box": "",
                                "is_no_answer": isNoAnswer,
                                "is_yes_no_bbox": True
                                })



            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    with open(output_jsonl_file_path, 'w') as output_file:
        for obj in processed_objects:

            json_line = json.dumps(obj)
            output_file.write(json_line + '\n')

    print(f"Objects with bounding boxes saved to '{output_jsonl_file_path}'")


