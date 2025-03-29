import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import imageio

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from PIL import Image
import math
from torchvision import transforms

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(22520691)


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataloader import bench_data_loader 

def init_model(args):
    special_token = None
    if "llava" in args.model_name:
        from llava_ import LLava
        image_token = "<image>"
        model = LLava(args.pretrained, args.model_name)

    elif "openflamingo" in args.model_name:
        from openflamingo_ import OpenFlamingo
        image_token = "<image>"
        special_token = "<|endofchunk|>"

        model = OpenFlamingo(args.pretrained)
    
    elif "mantis" in args.model_name:
        from mantis_ import Mantis
        image_token = "<image>"
        model = Mantis(args.pretrained)
        
    elif "deepseek" in args.model_name:
        from deepseek_ import DeepSeek
        image_token = "<image_placeholder>"
        model = DeepSeek(args.pretrained)
    
    return model, image_token, special_token
    


def benchmark(args, img_tensors, qs, sample_gt, pertubation_list, model):
    # if args.bench == mulitchoice
    to_pil = transforms.ToPILImage()
    adv_img_tensors = torch.clip(img_tensors + pertubation_list, 0, 1).cuda()
    adv_pil_images = [to_pil(img) for img in adv_img_tensors.cpu()]
    
    output = model.inference(qs, adv_pil_images)[0]
    if output in ['A', 'B', 'C', 'D']:
        # choice = sample_answer['output']    
        acc = 1 if output != sample_gt['gt_choice'] else 0
        return acc, adv_pil_images
    else:
        raise Exception("Output not in ['A', 'B', 'C', 'D']")
    
    # some how get the embedding from another model  like clip
    # choice_embedidng = 
    
    
    
    # elif args.bench == simple

def save_gif(images, filename, duration=0.2):
    imageio.mimsave(filename, images, duration=duration)
    
def ES_1_1(args, model, image_files, qs, sample_gt, epsilon=0.05, c_increase=1.2, c_decrease=0.8, sigma=1.1):
    totensor = transforms.ToTensor()
    img_tensors = torch.stack([totensor(img) for img in image_files]).cuda()
    
    pertubation_list = torch.randn_like(img_tensors).cuda()
    pertubation_list = torch.clamp(pertubation_list, -epsilon, epsilon)
    
    best_fitness, adv_img_files = benchmark(args, img_tensors, qs, sample_gt, pertubation_list, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]
    success = False
    num_evaluation = 1

    adv_history_0 = [adv_img_files[0]]
    adv_history_1 = [adv_img_files[1]]

    for i in tqdm(range(1, args.max_query)):
        if best_fitness > 0:
            success = True
            break
        
        alpha = torch.randn_like(img_tensors).cuda() * sigma
        alpha = torch.clamp(alpha, -epsilon, epsilon)
        new_pertubation_list = pertubation_list + alpha

        new_fitness, adv_img_files = benchmark(args, img_tensors, qs, sample_gt, new_pertubation_list, model)

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_img_files_adv = adv_img_files
            pertubation_list = new_pertubation_list
            sigma *= c_increase
        else:
            sigma *= c_decrease

        history.append(best_fitness)
        num_evaluation += 1

        adv_history_0.append(adv_img_files[0])
        adv_history_1.append(adv_img_files[1])

    save_gif(adv_history_0, f"{args.id}_adv_0.gif")
    save_gif(adv_history_1, f"{args.id}_adv_1.gif")

    return num_evaluation, pertubation_list, best_img_files_adv, success
    
def main(args):
    model, image_token, special_token = init_model(args)
    acc = 0
    run = 0
    for item in bench_data_loader(args, image_placeholder=image_token, special_token=special_token):
        qs = item['question']
        img_files = item['image_files']
        gt_ans = item['gt_choice']
        sample_gt = item['sample_gt']
        print("Question: ", qs)
        print(f"Answer {gt_ans}: {sample_gt[gt_ans]}")
                
        text_outputs = model.inference(qs, img_files)[0]
        print("original Output: ", text_outputs)

        if text_outputs[0] == gt_ans:
            run += 1
            print("Correct, ready to attack")
            num_evaluation, pertubation_list, img_files_adv, success = ES_1_1(args, model, img_files, qs, sample_gt)
            print("Num evaluation for attacking: ", num_evaluation)
        else:
            print("Wrong, skip")
            continue
        
        for i, img in enumerate(img_files_adv):
            if i == 0:
                img.save(f"question.png")
            else:
                img.save(f"img_adv_{i}.png")
        
        text_outputs = model.inference(qs, img_files_adv)[0]
        print("adv Output: ", text_outputs)     
        if success == True:
            acc += 1
        
    print(f"Accuracy run={run} max_query={args.max_query} num_retreival={args.num_retrieval}: {acc/run}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieval", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--max_query", type=int, default=1000)
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 