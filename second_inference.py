import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torchvision import transforms
import os
import json
import requests
from io import BytesIO
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images for captioning")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Optional path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="Qwen/Qwen-VL", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="Qwen/Qwen-VL", help="Tokenizer identifier or path")
    parser.add_argument("--quant", type=int, default=8, help="Quantization bits")
    parser.add_argument("--query", type=str, default="Describe the image accurately and in detail.", help="Default query for captioning")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_number", type=int, default=1, help="Batch number to process")
    parser.add_argument("--total_batches", type=int, default=4, help="Total number of batches")
    parser.add_argument("--template", type=str, help="Template for generating new prompts, includes <expr> as a placeholder.")
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

def preprocess_image(image_path):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    preprocess_transform = transforms.Compose([
        lambda img: img.convert("RGB"),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path)
    return preprocess_transform(image).unsqueeze(0)

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
    
    items_per_batch = math.ceil(len(image_files) / args.total_batches)
    start_index = (args.batch_number - 1) * items_per_batch
    end_index = start_index + items_per_batch
    batched_files = dict(list(image_files.items())[start_index:end_index])
    
    return batched_files

def main():
    args = parse_args()
    model, device = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer)
    image_files = load_images(args)
    
    for filename, path in image_files.items():
        if args.url_file:
            response = requests.get(path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        
        image_tensor = preprocess_image(path).to(device)
        
        inputs = tokenizer(args.query, return_tensors="pt").to(device)
        inputs['pixel_values'] = image_tensor

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Image: {filename}")
            print(response)

if __name__ == "__main__":
    main()