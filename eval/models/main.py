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
from llava_ import LLava

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataloader import bench_data_loader 

def main(args):
    
    if "llava" in args.pretrained:
        image_token = "<image>"
        model = LLava(args.pretrained, args.model_name)

    for item in bench_data_loader(args, image_placeholder=image_token):
        
        qs = item['question']
        img_files = item['image_files']
        
        output = model.inference(qs, img_files)
        print("Output: ", output)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    ############# added for mrag benchmark ####################
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--use_rag", type=lambda x: x.lower() == 'true', default=False, help="Use RAG")
    parser.add_argument("--use_retrieved_examples", type=lambda x: x.lower() == 'true', default=False, help="Use retrieved examples")
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 