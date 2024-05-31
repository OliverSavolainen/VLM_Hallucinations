import json
from collections import defaultdict
import torch
from torchvision.ops import box_iou
def match_bboxes(labels_path='data/bbox_pope_images/labels.json',
                 extracted_objects_path ='intermediate_outputs/objects_with_bounding_boxes.jsonl',
                 output_path = 'intermediate_outputs/bbox_iou_matched.jsonl', hallucination_threshold=0.5,
                 model_bbox_scale=1000.0):


    # Read ground truth labels in coco format
    with open(labels_path, 'r') as file:
        labels = json.load(file)
    
    # Extract categories from labels
    categories = labels["categories"]
    object_classes = {category["id"]: category["name"] for category in categories}

    # Read objects and bounding boxes generated by grounded VLM
    with open(extracted_objects_path, 'r') as file:
        extracted_objects = [json.loads(line) for line in file]

    # Get information about images from coco labels
    images = labels["images"]

    # Extract image sizes for each image
    image_sizes = {}
    for image_info in images:
        image_sizes[image_info["id"]] = (image_info["width"], image_info["height"])

    # Get annotations
    annotations = labels['annotations']

    # Extract bounding box labels from annotations
    bbox_labels = defaultdict(list)
    for label in annotations:
        # Image id as dict key
        id = label["image_id"]
        # Add bbox list and object category id as tuple
        bbox_labels[id].append((label["bbox"], label["category_id"]))

    for object in extracted_objects:
        # Get image name
        image_name = object['question_id']
        # Extract id from name
        image_id = int(image_name[13:-4])
        # Get image size
        img_size = image_sizes[image_id]
        # Get extracted bbox and scale it img_size
        scaled_bbox = scale_bbox(object["bounding_box"], img_size, model_bbox_scale)
        object["bounding_box"] = scaled_bbox
        # Object class
        object_class = object["object_name"]
        # Get ground truth bounding boxes
        gt_bboxes = bbox_labels[image_id]
        # Find ground truth bbox with largest iou
        bbox_match_box, bbox_match_object, bbox_match_iou = compare_bboxes(scaled_bbox, gt_bboxes)
        # Add matched bbox info
        object["bbox_match_box"] = bbox_match_box
        object["bbox_match_object"] = get_object_name(bbox_match_object, object_classes)
        object["bbox_match_iou"] = bbox_match_iou

        # Hallucination if iou less than threshold
        if bbox_match_iou < hallucination_threshold:
            object["is_hallucination"] = True
        else:
            object["is_hallucination"] = False

    # Save results in jsonl
    with open(output_path, 'w') as file:
        _ = [file.write(json.dumps(object) + "\n") for object in extracted_objects]


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

def scale_bbox(bbox, new_scale,model_bbox_scale):
    # Convert from 1000x1000 cogvlm to image size given in new_scale
    # Also converts generated bbox from string to list of floats
    corners = bbox.split(",")
    float_corners = [ float(corner)/model_bbox_scale for corner in corners]
    scaled_x0, scaled_x1 = (float_corners[0] * new_scale[0],
                            float_corners[2] * new_scale[0])
    scaled_y0, scaled_y1 = (float_corners[1] * new_scale[1],
                            float_corners[3] * new_scale[1])
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