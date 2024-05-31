import json

questions_path = "./coco/captioning_questions_shikra.jsonl"
answers_path = "./model_outputs/shikra_GROUNDED_CAPTIONING.jsonl"
adapted_answers_path = "./model_outputs/shikra_GROUNDED_CAPTIONING_adapted.jsonl"

questions = []
with open(questions_path, 'r') as f:
    for line in f:
        questions.append(json.loads(line))

answers = []
with open(answers_path,"r") as f:
    for line in f:
        answers.append(json.loads(line))

for i, question in enumerate(questions):
    answers[i]["text"] = answers[i]["pred"]
    answers[i]["file"] = question["image"]

with open(adapted_answers_path, 'w') as f:
    for answer in answers:
        json_str = json.dumps(answer)
        f.write(json_str + "\n")