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

def main(args):
    model, image_token, special_token = init_model(args)
    
    for item in bench_data_loader(args, image_placeholder=image_token, special_token=special_token):
        qs = item['question']
        img_files = item['image_files']
        gt_ans = item['gt_choice']
        
        qss = [qs, qs, qs]
        batch_img_files = [img_files, img_files, img_files]
        
        text_outputs = model.inference(qss, batch_img_files)
        print(text_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieval", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 