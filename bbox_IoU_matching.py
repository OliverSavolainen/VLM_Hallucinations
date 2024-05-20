import json
import pandas as pd


def process_bounding_box_strings(bboxes):
    # Assuming bboxes is a list of string bounding boxes like ["392,130,765,834", ...]
    return list(map(int, bboxes.split(',')))

def scale_bboxes(row):
    scaled_bboxes = []
    for bbox in row['gt_bboxes']:
        x_min = int(bbox[0] / 1000 * row['width'])
        y_min = int(bbox[1] / 1000 * row['height'])
        x_max = int(bbox[2] / 1000 * row['width'])
        y_max = int(bbox[3] / 1000 * row['height'])
        scaled_bboxes.append([x_min, y_min, x_max, y_max])
    return scaled_bboxes


def scale_bbox(row):
    bbox = row['bounding_box']

    x_min = int(bbox[0] / 1000 * row['width'])
    y_min = int(bbox[1] / 1000 * row['height'])
    x_max = int(bbox[2] / 1000 * row['width'])
    y_max = int(bbox[3] / 1000 * row['height'])

    return [x_min, y_min, x_max, y_max]

def convert_bbox_format(bboxes):
    # Convert each bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
    converted_bboxes = []
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[0] + bbox[2]  # x_min + width
        y_max = bbox[1] + bbox[3]  # y_min + height
        converted_bboxes.append([x_min, y_min, x_max, y_max])
    return converted_bboxes

def match_bbox_iou(labels_path='bbox_pope_images/labels.json',bboxes_path ='objects_with_bounding_boxes.jsonl', output_path="bbox_hallucinations.jsonl"):

    LABELS_PATH = labels_path
    BBOXES_PATH = bboxes_path

    # Open the labels file for reading
    with open(LABELS_PATH, 'r') as file:
        labels = json.load(file)

    # Open the outputs file for reading
    with open(BBOXES_PATH, 'r') as file:
        outputs = file.readlines()
        outputs = [json.loads(line) for line in outputs]  # Convert JSON strings into dictionaries

    # Create a dataframe from the images data
    img_data = labels['images']
    img_df = pd.DataFrame(img_data)

    # Create a dataframe from the annotations data
    bbox_data = labels['annotations']
    bbox_df = pd.DataFrame(bbox_data)

    # Create a dataframe from the outputs data
    outputs_df = pd.DataFrame(outputs)

    # Rename the 'id' column in img_df to 'image_id' to match bbox_df for merging
    img_df = img_df.rename(columns={'id': 'image_id'})

    # Merge bbox_df with the selected columns from img_df
    bbox_df = pd.merge(bbox_df, img_df[['image_id', 'height', 'width', 'file_name']], on='image_id', how='left')

    bbox_df.head()
    # Applying the function to each bounding box entry
    outputs_df['bounding_box'] = outputs_df['bounding_box'].apply(lambda x: process_bounding_box_strings(x))


    # Group outputs_df by 'question_id' and aggregate 'bounding_box' into lists
    grouped_outputs = outputs_df.groupby('question_id')['bounding_box'].agg(list).reset_index()

    # Rename the 'question_id' column to 'file_name' to match the merged_df for merging
    grouped_outputs_renamed = grouped_outputs.rename(columns={'question_id': 'file_name'})

    # Merge merged_df with the grouped_outputs DataFrame
    final_df = pd.merge(bbox_df, grouped_outputs_renamed, on='file_name', how='left')

    # Create 'pope_outputs' column. Fill NaN values with empty lists if no bounding boxes are associated
    final_df['pope_bboxes'] = final_df['bounding_box'].apply(lambda x: x if isinstance(x, list) else [])

    # Drop the 'bounding_box' column if not needed, as it duplicates the 'pope_outputs'
    final_df.drop(columns=['bounding_box'], inplace=True)

    final_df = final_df[final_df['pope_bboxes'].apply(lambda x: len(x) > 0)]
    # final_df['pope_bboxes'] = final_df['pope_bboxes'].apply(lambda x: [x[i:i + 4] for i in range(0, len(x), 4)])
    # final_df['pope_bboxes'] = final_df['pope_bboxes'].apply(process_bounding_boxes)

    # Group bbox_df by 'file_name' and aggregate 'bbox' into lists
    grouped_bboxes = bbox_df.groupby('file_name')['bbox'].agg(list).reset_index()

    # Merge outputs_df with the grouped_bboxes DataFrame
    outputs_df = pd.merge(outputs_df, grouped_bboxes, left_on='question_id', right_on='file_name', how='left')

    # Create 'gt_bboxes' column. Fill NaN values with empty lists if no bounding boxes are associated
    outputs_df['gt_bboxes'] = outputs_df['bbox'].apply(lambda x: x if isinstance(x, list) else [])

    # Optionally, drop the 'bbox' column if not needed, as it duplicates the 'gt_bboxes'
    outputs_df.drop(columns=['bbox'], inplace=True)

    outputs_df = pd.merge(outputs_df, img_df[[ 'height', 'width', 'file_name']], on='file_name', how='left')
    # outputs_df = outputs_df.drop(columns=['file_name_x', 'file_name_y'])

    # Apply the function to each row of the DataFrame
    # final_df['scaled_pope_bboxes'] = final_df.apply(scale_bboxes, axis=1)
    # final_df.head()

    outputs_df['bounding_box'] = outputs_df.apply(scale_bbox, axis=1)

    # Apply the conversion function to the 'gt_bboxes' column
    outputs_df['gt_bboxes'] = outputs_df['gt_bboxes'].apply(convert_bbox_format)

    # Assuming the DataFrame 'df' has columns 'bbox' and 'scaled_bboxes'
    outputs_df[['IoU', 'corresponding_pope_bbox']] = outputs_df.apply(find_best_iou_and_bbox, axis=1)

    # Calculate the accuracy for the 'outputs_df' DataFrame
    iou_accuracy = calculate_accuracy(outputs_df, 'IoU')

    # Print the computed accuracy
    print(f"The accuracy of bounding boxes with IoU > 0.5 is {iou_accuracy:.2f}%.")

    # Select the necessary columns and compute the 'hallucinates' column
    return_df = outputs_df[['question_id', 'object_name', 'bounding_box', 'IoU']].copy()
    return_df['hallucinates'] = return_df['IoU'] <= 0.5

    # Save DataFrame to a JSON file
    return_df.to_json(output_path, orient='records', lines=True)




def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area by using the formula: union(A,B) = A + B - Inter(A,B)
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou


def find_best_iou_and_bbox(row):
    ground_truth_bbox = row['bounding_box']
    best_iou = 0
    best_bbox = None

    for pope_bbox in row['gt_bboxes']:
        iou = calculate_iou(ground_truth_bbox, pope_bbox)
        # iou = get_iou(ground_truth_bbox, pope_bbox)
        if iou > best_iou:
            best_iou = iou
            best_bbox = pope_bbox

    return pd.Series([best_iou, best_bbox], index=['IoU', 'corresponding_pope_bbox'])




def calculate_accuracy(df, column_name, iou_threshold=0.5):
    # Count the number of IoU values greater than the threshold
    count_above_threshold = (df[column_name] > iou_threshold).sum()

    # Calculate the total number of IoU entries
    total_count = len(df[column_name])

    # Calculate accuracy as the percentage of IoUs above the threshold
    accuracy = (count_above_threshold / total_count) * 100 if total_count > 0 else 0
    return accuracy



