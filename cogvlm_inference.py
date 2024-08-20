import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import os
import json
import shortuuid
import requests
from io import BytesIO
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images for captioning")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Optional path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-grounding-generalist-hf", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help="Tokenizer identifier or path")
    parser.add_argument("--quant", type=int, default=8, help="Quantization bits")
    parser.add_argument("--query", type=str, default="Describe the image accurately and in detail.", help="Default query for captioning")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision floating point (16-bit)")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision floating point (16-bit)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_number", type=int, default=1, help="Batch number to process")
    parser.add_argument("--total_batches", type=int, default=4, help="Total number of batches")
    parser.add_argument("--template", type=str, help="Template for generating new prompts, includes <expr> as a placeholder.")
    parser.add_argument("--remove_second_line", action="store_true", help="Remove everything after a newline in the prompt")
    parser.add_argument("--category", type=str, default="adversarial", help="Category of prompts to filter and use from the prompts file.")
    return parser.parse_args()

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=(args.quant == 8),
        load_in_4bit=(args.quant == 4)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        trust_remote_code=True,
    ).eval()
    return model, device

def load_prompts(prompts_file, remove_second_line, category):
    prompt_dict = {}
    with open(prompts_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                image_name = data.get('image')
                prompt_text = data.get('text')
                prompt_category = data.get('category', 'wrong')
                if prompt_category != category:
                    continue
                
                if image_name is None or prompt_text is None:
                    print(f"Skipping malformed line: {line}")
                    continue

                if remove_second_line:
                    prompt_text = prompt_text.split('\n', 1)[0]

                if image_name not in prompt_dict:
                    prompt_dict[image_name] = []

                if prompt_text not in prompt_dict[image_name]:
                    prompt_dict[image_name].append(prompt_text)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

    return prompt_dict

def load_images(args):
    if args.folder_path:
        image_files = {filename: os.path.join(args.folder_path, filename)
                       for filename in os.listdir(args.folder_path)
                       if filename.endswith((".jpg", ".png"))}
    elif args.url_file:
        with open(args.url_file, 'r') as file:
            data = json.load(file)
            image_files = {item['file_name']: item['coco_url']
                           for item in data['images']}
    else:
        raise ValueError("No valid input source provided. Please specify a folder path or a JSON file.")
    
    # Apply batching
    items_per_batch = math.ceil(len(image_files) / args.total_batches)
    start_index = (args.batch_number - 1) * items_per_batch
    end_index = start_index + items_per_batch
    batched_files = dict(list(image_files.items())[start_index:end_index])
    
    total_images = len(batched_files)
    print(f"Total images to process in this batch: {total_images}")
    
    images = {}
    for i, (name, path) in enumerate(batched_files.items(), start=1):
        print(f"Processing image {i} of {total_images}: {name}")
        if args.url_file:
            response = requests.get(path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        images[name] = image
    return images

def extract_object(prompt):
    """Extracts the object from the given prompt using regex."""
    match = re.search(r"Is there a (.*?) in the image\?", prompt)
    if match:
        return match.group(1)
    return None

def create_new_prompt(template, obj):
    """Replaces <expr> in the template with the object extracted."""
    return template.replace("<expr>", obj)

def main():
    args = parse_args()
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16
    model, device = load_model(args)
    images = load_images(args)
    tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)

    if args.prompts_file:
        prompts_dict = load_prompts(args.prompts_file, args.remove_second_line, args.category)

    file_mode = 'a' if os.path.exists("model_outputs/prompts_cogvlm_outputs.jsonl") else 'w'

    with open("model_outputs/prompts_cogvlm_outputs.jsonl", file_mode) as ans_file:
        for filename, image in images.items():
            queries = prompts_dict.get(filename, [args.query]) if args.prompts_file else [args.query]
            
            for query in queries:
                if args.template:
                    object_name = extract_object(query)
                    if object_name:
                        query = create_new_prompt(args.template, object_name)
                    else:
                        print("Problem with converting the template")
                input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=[], images=[image]
                )
                inputs = {
                    "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
                    "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
                    "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
                    "images": [[input_by_model["images"][0].to(device).to(torch_type)]]
                    if image is not None
                    else None,
                }
                gen_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                }

                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs["input_ids"].shape[1] :]
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split("</s>")[0]
                    print(f"Image: {filename}")
                    print(query)
                    print("\nCog:", response)

                ans_id = shortuuid.uuid()
                result = {
                    "file": filename,
                    "prompt": query,
                    "text": response,
                    "answer_id": ans_id
                }
                ans_file.write(json.dumps(result) + '\n')

    print("Results written to JSON file successfully.")

if __name__ == "__main__":
    main()