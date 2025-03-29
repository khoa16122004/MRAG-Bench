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
    


def benchmark(args, img_tensors, qs, sample_answer, pertubation_list, model):
    # if args.bench == mulitchoice
    to_pil = transforms.ToPILImage()
    adv_img_tensors = img_tensors + pertubation_list
    adv_pil_images = [to_pil(img) for img in adv_img_tensors.cpu()]
    
    output = model.inference(qs, adv_pil_images)[0]
    if output in ['A', 'B', 'C', 'D']:
        # choice = sample_answer['output']    
        acc = 1 if output == sample_answer['gt_choice'] else 0
        return acc, adv_pil_images
    else:
        raise Exception("Output not in ['A', 'B', 'C', 'D']")
    
    # some how get the embedding from another model  like clip
    # choice_embedidng = 
    
    
    
    # elif args.bench == simple


def ES_1_1(args, model, image_files, qs, gt_ans, epsilon=0.03, c_increase=1.2, c_decrease=0.8, sigma=1.1, max_query=1000):
    totensor = transforms.ToTensor()
    img_tensors = torch.stack([totensor(img) for img in image_files])
    print("Image tensors: ", img_tensors.shape)
    
    pertubation_list = torch.randn_like(img_tensors).cuda()
    pertubation_list = torch.clamp(pertubation_list, -epsilon, epsilon)
    
    best_fitness, adv_img_files = benchmark(args, img_tensors, qs, gt_ans, pertubation_list, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]

    num_evaluation = 1
    while num_evaluation < max_query and best_fitness > 0:
        alpha = torch.randn_like(img_tensors).cuda()
        alpha = torch.clamp(alpha, -epsilon, epsilon)

        new_pertubation_list = pertubation_list + alpha * sigma 
        new_pertubation_list = torch.clamp(new_pertubation_list, -epsilon, epsilon)

        new_fitness, adv_img_files = benchmark(args, img_tensors, qs, gt_ans, new_pertubation_list, model)

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_img_files_adv = adv_img_files
            pertubation_list = new_pertubation_list
            sigma *= c_increase
        else:
            sigma *= c_decrease
            

        history.append(best_fitness)
        
        num_evaluation += 1

    return num_evaluation, pertubation_list, best_img_files_adv
    
def main(args):
    model, image_token, special_token = init_model(args)
    
    for item in bench_data_loader(args, image_placeholder=image_token, special_token=special_token):
        qs = item['question']
        img_files = item['image_files']
        gt_ans = item['gt_choice']
        print("Question: ", qs)
        print(f"Answer {gt_ans}: {item['sample_gt'][gt_ans]}")
                
        text_outputs = model.inference(qs, img_files)[0]
        print("original Output: ", text_outputs)

        if text_outputs[0] == gt_ans:
            print("Correct, ready to attack")
            num_evaluation, pertubation_list, img_files_adv = ES_1_1(args, model, img_files, qs, gt_ans)
            print("Num evaluation for attacking: ", num_evaluation)
        else:
            print("Wrong, skip")
            continue
        
        text_outputs = model.inference(qs, img_files_adv)[0]
        print("adv Output: ", text_outputs)        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieval", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 