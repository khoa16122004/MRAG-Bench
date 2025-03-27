import os
import json
import math
import io
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset

def bench_data_loader(args, image_placeholder="<image>", special_token=None):
    """ 
    Data loader for benchmarking models
    Args:
        args: arguments
        image_placeholder: placeholder string for image
    Returns:
        generator: a generator that yields data (queries, image paths, ...) for each sample
    """
    # Data
    mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")
    
    for item in tqdm(mrag_bench):
        
        qs_id = item['id'] 
        qs = item['question']
        ans = item['answer']
        gt_choice = item['answer_choice']
        scenario = item['scenario']
        choices_A = item['A']
        choices_B = item['B']
        choices_C = item['C']
        choices_D = item['D']
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images]
        
        image = item['image'].convert("RGB") # input image


        
        ### our evaluation instuction for all the models 
        if args.use_rag:
            image_files = [image] + gt_images
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}{image_placeholder}\n"

        else:
            image_files = [image]
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}\n"
            
        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        final_qs = prompt + qs
        
        yield {
            "id": qs_id, 
            "question": final_qs, 
            "image_files": image_files, 
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "aspect": item['aspect']
        }


class CustomImageDataset(Dataset):
    def __init__(self, extract_dir):
        self.extract_dir = extract_dir
        self.ids = os.listdir(extract_dir)
        
        

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        id_ = self.ids[idx]
        id_dir = os.path.join(self.extract_dir, id_)
        
        imgs = []
        
        for file_name in os.listdir(id_dir):
            if file_name.startswith("gt_"):
                image = Image.open(os.path.join(id_dir, file_name)).convert("RGB")
                imgs.append(image)
            elif file_name.strartswith("question_img"):
                question_img = Image.open(os.path.join(id_dir, file_name)).convert("RGB")   
            else:
                with open(os.path.join(id_dir, file_name), 'r') as f:
                     lines = f.readlines()
                     lines = [line.strip() for line in lines]
                     question = lines[0].split(':')[1].strip()
                     choice_A = lines[2].split(':')[1].strip()
                     choice_B = lines[3].split(':')[1].strip()
                     choice_C = lines[4].split(':')[1].strip()
                     choice_D = lines[5].split(':')[1].strip()
                     gt_choice = lines[7].split(':')[1].strip()     
                     
            return question_img, question, (choice_A, choice_B, choice_C, choice_D, )     