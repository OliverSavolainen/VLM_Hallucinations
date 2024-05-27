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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ImageTextDataset(Dataset):
    def __init__(self, image_files, tokenizer, prompt_template, prompts_dict=None, query=None):
        self.image_files = image_files
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.prompts_dict = prompts_dict
        self.query = query
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for filename, path in self.image_files.items():
            if self.prompts_dict and filename in self.prompts_dict:
                for prompt_text in self.prompts_dict[filename]:
                    data.append((filename, path, prompt_text))
            else:
                data.append((filename, path, self.query))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, path, prompt_text = self.data[idx]
        if path.startswith('http'):
            response = requests.get(path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path).convert("RGB")
        #image_tensor = self.preprocess_image(image)
        input_text = self.prompt_template.format("", prompt_text)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.squeeze()
        attention_mask = self.tokenizer(input_text, return_tensors="pt").attention_mask.squeeze()
        return input_ids, attention_mask, image, filename, prompt_text

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images for captioning")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="Qwen/Qwen-VL", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="Qwen/Qwen-VL", help="Tokenizer identifier or path")
    parser.add_argument("--quant", type=int, default=4, help="Quantization bits")
    parser.add_argument("--query", type=str, default="Describe the image accurately and in detail.", help="Default query for captioning")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision floating point (16-bit)")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision floating point (16-bit)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_number", type=int, default=1, help="Batch number to process")
    parser.add_argument("--total_batches", type=int, default=4, help="Total number of batches")
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

def main():
    args = parse_args()
    model, device = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_template = '<img>{}</img><ref>{}</ref><box>'
    image_files = load_image_files(folder_path=args.folder_path, url_file=args.url_file)

    prompts_dict = None
    if args.prompts_file:
        prompts_dict = load_prompts(args.prompts_file)

    total_images = len(image_files)
    items_per_batch = math.ceil(total_images / args.total_batches)
    start_index = (args.batch_number - 1) * items_per_batch
    end_index = start_index + items_per_batch
    batched_files = dict(list(image_files.items())[start_index:end_index])

    dataset = ImageTextDataset(batched_files, tokenizer, prompt_template, prompts_dict, query=args.query)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    outputs = []
    for batch in tqdm(dataloader):
        for item in batch:
            input_ids, attention_mask, image_tensor, filename, prompt_text = item
            input_ids, attention_mask, image_tensor = input_ids.to(device), attention_mask.to(device), image_tensor
            print(f"Prompt: {prompt_text}")
            print(f"Image: {filename}")
            inputs = {
                'input_ids': input_ids.unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0)
            }

            pred = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                length_penalty=1,
                num_return_sequences=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            answers = [
                tokenizer.decode(output[inputs['input_ids'].size(1):], skip_special_tokens=True)
                for output in pred
            ]

            for answer in answers:
                print(f"Image: {filename}")
                print(f"Prompt: {prompt_text}")
                print(f"Answer: {answer}")
                outputs.append({
                    "file": filename,
                    "prompt": prompt_text,
                    "text": answer,
                })

    with open(f"outputs_batch_{args.batch_number}.jsonl", 'w') as ans_file:
        for output in outputs:
            ans_file.write(json.dumps(output) + '\n')

if __name__ == "__main__":
    main()