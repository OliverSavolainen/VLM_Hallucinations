import jsonlines
import argparse
import re
import json
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from torchvision.ops import box_iou

def extract_no_answers(text):
    pattern = r"\b(no|not)\b"
    matches = re.findall(pattern, text,re.IGNORECASE)
    if len(matches) > 0:
        return True
    else:
        return False

def extract_object_name(prompt):
    pattern = r'Is there a ([\w\s]+) in the image\?'
    match = re.search(pattern, prompt)
    if match:
        return match.group(1)
    return None
def extract_objects_with_bounding_boxes(text,bbox_regex):
    pattern = r'(\b\w+\b)\s*' + bbox_regex

    matches = re.findall(pattern, text)

    results = []

    # No match try with only bbox without object name
    if len(matches) == 0:
        matches = re.findall(bbox_regex, text)
        for bbox_str in matches:
            # Split the bounding box string by semicolons for multiple instances
            bboxes = bbox_str.split(';')
            # Convert each bounding box to a list of floats and pair with the object
            for bbox in bboxes:
                results.append(("no_object_name", bbox))

    else:
        for obj, bbox_str in matches:
            # Split the bounding box string by semicolons for multiple instances
            bboxes = bbox_str.split(';')
            # Convert each bounding box to a list of floats and pair with the object
            for bbox in bboxes:
                results.append((obj, bbox))
    return results

def get_object_output_dict(question_id,prompt,object_name,bbox,is_no_answer,is_yes_no_bbox):

    return { "question_id": question_id,
        "prompt": prompt,
        "object_name": object_name.capitalize() if object_name else "",
        "bounding_box": bbox,
        "is_no_answer": is_no_answer,
        "is_yes_no_bbox": is_yes_no_bbox}
def get_object_name(category_id, object_classes):
    # Convert category id to object name

    return object_classes.get(category_id, "unknown")

def convert_bbox_format_coco2torch(bbox):
    # Convert each bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
    # COCO format to torch format for iou
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0] + bbox[2]  # x_min + width
    y_max = bbox[1] + bbox[3]  # y_min + height
    converted_bbox = [x_min, y_min, x_max, y_max]
    return converted_bbox


def scale_bbox(bbox, new_scale, model_bbox_scale):
    corners = bbox.split(",")
    float_corners = [float(corner) / model_bbox_scale for corner in corners]
    scaled_x0, scaled_x1 = (int(float_corners[0] * new_scale[0]),
                                int(float_corners[2] * new_scale[0]))
    scaled_y0, scaled_y1 = (int(float_corners[1] * new_scale[1]),
                                int(float_corners[3] * new_scale[1]))
    return scaled_x0, scaled_y0, scaled_x1, scaled_y1


def compare_bboxes(bbox: str, gt_bboxes: list):
    # Compares the generated bounding box against ground truth bounding boxes in the image
    best_iou = 0
    best_box = None
    best_box_object = 0

    for gt_bbox, category_id in gt_bboxes:
        # Convert format to match
        converted_gt_bbox = convert_bbox_format_coco2torch(gt_bbox)
        # Calculate iou
        iou = calculate_iou_torch(bbox, converted_gt_bbox)

        # Set initial bbox as the best box
        if best_box == None:
            best_box = converted_gt_bbox
            best_iou = iou
            best_box_object = category_id

        # Update best box when a better bounding box is found
        if iou > best_iou:
            best_iou = iou
            best_box = converted_gt_bbox
            best_box_object = category_id

    return best_box, best_box_object, best_iou

def calculate_iou_torch(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes using PyTorch.

    Parameters
    ----------
    box1 : list or tuple
        The (x1, y1, x2, y2) coordinates of the first bounding box.
    box2 : list or tuple
        The (x1, y1, x2, y2) coordinates of the second bounding box.

    Returns
    -------
    float
        IoU of box1 and box2.
    """
    box1 = torch.tensor(box1).unsqueeze(0)  # Convert to tensor and add batch dimension
    box2 = torch.tensor(box2).unsqueeze(0)  # Convert to tensor and add batch dimension

    iou = box_iou(box1, box2)

    return iou.item()

def find_coco_matches(jsonl_file, output_file, coco_classes,model):
    coco_embeddings = model.encode(coco_classes, convert_to_tensor=True)

    with jsonlines.open(jsonl_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            object_name = obj['object_name']
            object_embedding = model.encode(object_name, convert_to_tensor=True)
            cosine_scores = util.cos_sim(object_embedding, coco_embeddings)

            # Find the highest scoring COCO class
            max_score_index = cosine_scores.argmax()
            matched_class = coco_classes[max_score_index]

            # Print and write the output with the matched class
            print(f"Image ID: {obj['question_id']}, Object: {object_name}, Matched COCO Class: {matched_class}, Score: {cosine_scores[0, max_score_index].item()}")
            obj['class'] = matched_class
            writer.write(obj)

def load_labels(label_file_path):
    labels = {}
    with open(label_file_path, 'r') as file:
        for line in file:
            label = json.loads(line)
            image_name = label['image']
            prompt = label['text']
            labels[(image_name, prompt)] = label['label']
    return labels

def find_label_for_prompt(prompt, image_name, labels):
    for (img, q_prompt), label in labels.items():
        if img == image_name and q_prompt in prompt:
            return label
    return None


