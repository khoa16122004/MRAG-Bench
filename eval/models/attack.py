import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid


from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from PIL import Image
import math
from torchvision import transforms

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


def benchmark(args, img_tensors, qs, gt_ans, pertubation_list):
    # if args.bench == mulitchoice
    
    # elif args.bench == simple
    pass


def ES_1_1(args, image_files, qs, gt_ans, epsilon=0.03, c_increase=1.2, c_decrease=0.8, sigma=1.1, max_query=1000):
    totensor = transforms.ToTensor()
    img_tensors = torch.stack([totensor(img) for img in image_files])
    print("Image tensors: ", img_tensors.shape)
    
    pertubation_list = torch.randn_like(img_tensors).cuda()
    pertubation_list = torch.clamp(pertubation_list, -epsilon, epsilon)
    
    best_fitness = benchmark(args, img_tensors, qs, gt_ans, pertubation_list)
    history = [best_fitness]
    
    num_evaluation = 1
    while num_evaluation < max_query and attack_success(best_fitness) == False:
        alpha = torch.randn_like(img_tensors).cuda()
        alpha = torch.clamp(alpha, -epsilon, epsilon)

        new_pertubation_list = pertubation_list + alpha * sigma 
        new_pertubation_list = torch.clamp(new_pertubation_list, -epsilon, epsilon)

        new_fitness = benchmark(args, img_tensors, qs, gt_ans, new_pertubation_list)

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            pertubation_list = new_pertubation_list
            sigma *= c_increase
        else:
            sigma *= c_decrease
            

        history.append(best_fitness)
        
        num_evaluation += 1

    return evaluation, pertubation_list
    
def main(args):
    model, image_token, special_token = init_model(args)
    
    for item in bench_data_loader(args, image_placeholder=image_token, special_token=special_token):
        qs = item['question']
        img_files = item['image_files']
        gt_ans = item['gt_choice']
        
        
        
        
        print(gt_ans)
        
        text_outputs = model.inference(qs, img_files)
        print("Output: ", text_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieval", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 