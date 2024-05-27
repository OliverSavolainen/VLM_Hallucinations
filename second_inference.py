import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import requests
from io import BytesIO
import math
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images for captioning")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Optional path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="Qwen/Qwen-VL", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="Qwen/Qwen-VL", help="Tokenizer identifier or path")
    parser.add_argument("--quant", type=int, default=8, help="Quantization bits")
    parser.add_argument("--query", type=str, default="Describe the image accurately and in detail.", help="Default query for captioning")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision floating point (16-bit)")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision floating point (16-bit)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--batch_number", type=int, default=1, help="Batch number to process")
    parser.add_argument("--total_batches", type=int, default=4, help="Total number of batches")
    parser.add_argument("--template", type=str, help="Template for generating new prompts, includes <expr> as a placeholder.")
    return parser.parse_args()

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
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
    ).to(device).eval()
    return model, device

def load_image_files(folder_path=None, url_file=None):
    image_files = {}
    if folder_path:
        image_files = {filename: os.path.join(folder_path, filename)
                       for filename in os.listdir(folder_path)
                       if filename.endswith((".jpg", ".png"))}
    elif url_file:
        with open(url_file, 'r') as file:
            data = json.load(file)
            image_files = {item['file_name']: item['coco_url']
                           for item in data['images']}
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

def preprocess_image(image):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    preprocess_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image_tensor = preprocess_transform(image)
    return image_tensor.unsqueeze(0).float()

def main():
    args = parse_args()
    model, device = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, trust_remote_code=True)

    image_files = load_image_files(folder_path=args.folder_path, url_file=args.url_file)
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
            # Download and preprocess image
            if path.startswith('http'):
                response = requests.get(path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
            image_tensor = preprocess_image(image).to(device)
            
            # Create input with the image tensor and text prompt
            query = [{'image': image_tensor}, {'text': prompt_text}]
            inputs = tokenizer.from_list_format(query)
            inputs = tokenizer(inputs, return_tensors='pt').to(device)
            
            # Generate caption
            pred = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            
            # Print and collect results
            print(f"Image: {filename}")
            print(f"Prompt: {prompt_text}")
            print(f"Response: {response}")
            outputs.append({
                "file": filename,
                "prompt": prompt_text,
                "text": response,
            })

    # Save outputs to a JSONL file
    with open(f"outputs_batch_{args.batch_number}.jsonl", 'w') as ans_file:
        for output in outputs:
            ans_file.write(json.dumps(output) + '\n')

if __name__ == "__main__":
    main()