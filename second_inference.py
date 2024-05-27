import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import math
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="Qwen/Qwen-VL", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="Qwen/Qwen-VL", help="Tokenizer identifier or path")
    parser.add_argument("--query", type=str, default="Generate the caption in English with grounding:", help="Default query for captioning")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_number", type=int, default=1, help="Batch number to process")
    parser.add_argument("--total_batches", type=int, default=4, help="Total number of batches")
    return parser.parse_args()

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=(args.quant == 4),  # Default to 4-bit quantization
        load_in_4bit=(args.quant == 4)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        trust_remote_code=True,
    ).eval()  # .eval() without .to(device)
    return model, device

def load_image_urls(url_file):
    with open(url_file, 'r') as file:
        data = json.load(file)
        image_files = {item['file_name']: item['coco_url'] for item in data['images']}
    return image_files

def load_prompts(prompts_file):
    prompt_dict = {}
    with open(prompts_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_name = data['image']
            prompt_text = data['text']
            if image_name not in prompt_dict:
                prompt_dict[image_name] = []
            if prompt_text not in prompt_dict[image_name]:
                prompt_dict[image_name].append(prompt_text)
    return prompt_dict

def main():
    args = parse_args()
    model, device = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, trust_remote_code=True)

    image_files = load_image_urls(args.url_file)
    prompts_dict = load_prompts(args.prompts_file) if args.prompts_file else {}

    total_images = len(image_files)
    items_per_batch = math.ceil(total_images / args.total_batches)
    start_index = (args.batch_number - 1) * items_per_batch
    end_index = start_index + items_per_batch
    batched_files = dict(list(image_files.items())[start_index:end_index])

    outputs = []
    for filename, path in tqdm(batched_files.items()):
        prompts = prompts_dict.get(filename, [args.query])
        for prompt_text in prompts:
            query = [{'image': path}, {'text': prompt_text}]
            inputs = tokenizer.from_list_format(query)
            inputs = tokenizer(inputs, return_tensors='pt').to(device)
            pred = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            print(f"Image: {filename}")
            print(f"Prompt: {prompt_text}")
            print(f"Response: {response}")
            outputs.append({
                "file": filename,
                "prompt": prompt_text,
                "text": response,
            })

    with open(f"outputs_batch_{args.batch_number}.jsonl", 'w') as ans_file:
        for output in outputs:
            ans_file.write(json.dumps(output) + '\n')

if __name__ == "__main__":
    main()