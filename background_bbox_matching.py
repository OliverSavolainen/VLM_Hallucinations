import json
import numpy as np
import os
import h5py


def match_bbox_with_background(extracted_objects__path='intermediate_outputs/objects_with_bounding_boxes.jsonl',
                               segmentations_path='data/segmentation_masks/segmentation_masks_voc2012.h5',
                               output_path="pipeline_outputs/object_background_matching.jsonl",
                               hallucination_threshold=0.75):

    segmentations = {}
    # Read segmentation masks
    with h5py.File(segmentations_path, "r") as hf:
        for image_name in hf.keys():
            segmentations[image_name] = hf.get(image_name)[:]
    # Read objects and bounding boxes
    with open(extracted_objects__path, 'r') as file:
        extracted_objects = [json.loads(line) for line in file]

    for object in extracted_objects:
        # Get image name
        image_name = object['question_id']
        # Get extracted bbox
        bbox = object["bounding_box"]
        # Get segmentation mask
        segmentation_mask = segmentations[image_name]
        # Scale bbox coords to coco image size (same as segmentation mask size) from 1000x1000
        x0, y0, x1, y1 = scale_bbox(bbox, segmentation_mask.shape)
        # Get the segmentation mask corresponding to the bbox
        masked_bbox = segmentation_mask[x0:x1, y0:y1]
        # Count number of background pixels (zeros)
        num_zeros = np.sum(masked_bbox == 0)

        if num_zeros != 0:
            background_ratio = num_zeros / masked_bbox.size
        else:
            background_ratio = 0.0

        object['background_ratio'] = background_ratio
        # Threshold to consider hallucination
        if background_ratio > hallucination_threshold:
            object['is_hallucination'] = True
        else:
            object['is_hallucination'] = False

    # Gets the type of segmentation mask (ade20k, voc2012 or potentially coco)
    segmentation_mask_type = segmentations_path[43:-3]
    output_path = output_path.replace('.jsonl',("_" + segmentation_mask_type + "_" + str(hallucination_threshold) + ".jsonl"))

    with open(output_path, 'w') as file:
        _ = [file.write(json.dumps(object) + "\n") for object in extracted_objects]

def scale_bbox(bbox, new_scale):
    # Convert from 1000x1000 cogvlm to 640x480 coco
    corners = bbox.split(",")
    float_corners = [ float(corner)/1000 for corner in corners]
    scaled_x0, scaled_x1 = (int(float_corners[0] * new_scale[0]),
                            int(float_corners[2] * new_scale[0]))
    scaled_y0, scaled_y1 = (int(float_corners[1] * new_scale[1]),
                            int(float_corners[3] * new_scale[0]))
    return scaled_x0, scaled_y0, scaled_x1, scaled_y1