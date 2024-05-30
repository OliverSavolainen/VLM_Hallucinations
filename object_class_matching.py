import json
import argparse
from sentence_transformers import SentenceTransformer, util
import jsonlines
"""
# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Match objects in images to COCO classes using semantic similarity and output results.")
parser.add_argument("--jsonl_file", type=str, required=True, help="Path to the JSONL file containing image object data.")
parser.add_argument("--output_file", type=str, default="matched.jsonl",help="Path to output the modified JSONL data.")
args = parser.parse_args()

coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "TV", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
"""
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

def match_object_classes(jsonl_file, output_file, model, threshold):

    with jsonlines.open(jsonl_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            if obj["is_hallucination"]:
                obj["is_misclassification"] = False
                writer.write(obj)
                continue

            object_name = obj['object_name']
            object_embedding = model.encode(object_name, convert_to_tensor=True)
            matched_class = obj['bbox_match_object']
            gt_object_embedding = model.encode(matched_class, convert_to_tensor=True)
            cosine_score = util.cos_sim(object_embedding, gt_object_embedding)
            obj["cosine_similarity"] = cosine_score.tolist()

            # above 0.9 match -> correct classification
            if cosine_score > 0.9:
                obj["is_misclassification"] = False
            # below 0.9 but above threshold -> misclassification
            elif cosine_score > threshold:
                obj['is_misclassification'] = True
            # below threshold -> hallucination
            else:
                obj["is_hallucination"] = True
                obj['is_misclassification'] = False

            # Print and write the output with the matched class
            print(f"Image ID: {obj['question_id']}, Object: {object_name}, Matched COCO Class: {matched_class}, Score: {cosine_score}")
            writer.write(obj)

# Process the JSONL file to find matches and write results
#find_coco_matches(args.jsonl_file, args.output_file, coco_classes)