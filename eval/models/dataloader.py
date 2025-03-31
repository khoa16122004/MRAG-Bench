import os
import json
import math
import io
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
import datasets

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
        samples_gt = {
            'A': choices_A,
            'B': choices_B,
            'C': choices_C,
            'D': choices_D,
            'gt_choice': gt_choice
        }
        gt_images = [ib.convert("RGB").resize((350, 500)) if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images]
        
        image = item['image'].convert("RGB").resize((350, 500)) # input image
        
        # prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly. {image_placeholder}"
        prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer reason with the option's letter for your answer from the given choices directly. {image_placeholder}"
        image_files = [image]
        for i in range(args.num_retrieval):
            image_files.append(gt_images[i])
            prompt += image_placeholder

        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}\n"
        final_qs = prompt + "\n" + qs
        
        yield {
            "sample_gt": samples_gt, 
            "id": qs_id, 
            "question": final_qs, 
            "image_files": image_files, 
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "aspect": item['aspect']
        }


def multi_QA_loader(image_placeholder):
    dataset = datasets.load_dataset("TIGER-Lab/Mantis-Instruct", "multi_vqa", revision="script")
    
    for item in tqdm(dataset['train']):
        id = item['id']
        image_data = item['images']
        img_path = [image['path'] for image in image_data]
        img_files = [Image.open(path).convert("RGB") for path in img_path]
        conversations = item['conversation']
        number_of_image = len(img_path)

        for i, conversation in enumerate(conversations):
            if i % 2 == 0: # user
                question = conversation['content']
                # remove the "<image>" placeholder from question
                question = question.replace("<image>", "") + image_placeholder * number_of_image
            else: # assistant
                answer = conversation['content']
                yield {
                    "id": id,
                    "question": question,
                    "image_files": img_files,
                    "answer": answer,
                    "num_image": number_of_image
                }
                break
                