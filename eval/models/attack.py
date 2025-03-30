import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import imageio
from bert_score import score
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from PIL import Image
import math
from torchvision import transforms
from dataloader import multi_QA_loader
import warnings
warnings.filterwarnings("ignore")
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

def MultiChoice_benchmark(args, img_tensors, qs, sample_gt, pertubation_list, model):
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
    
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

sim_model = SentenceTransformer('all-MiniLM-L6-v2')
def FreeText_benchmark(args, image_tensors, input_ids, image_sizes, 
                       gt_answer, pertubation_list, model):
    
    adv_img_tensors = image_tensors + pertubation_list
    adv_pil_images = model.decode_image_tensors(adv_img_tensors)
    output = model.inference(input_ids, adv_img_tensors, image_sizes)[0]
    
    print("diff: ", (adv_img_tensors - image_tensors).mean())
    
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(gt_answer, convert_to_tensor=True)

    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    return 1 - similarity, adv_pil_images, output
    
def save_gif(images, filename, duration=0.2):
    imageio.mimsave(filename, images, duration=duration)
    
def ES_1_1(args, benchmark, id, model, 
           image_tensors, image_sizes, input_ids, gt_answer, 
           epsilon=0.05, sigma=1.5):

    pertubation_list = torch.randn_like(image_tensors).cuda()
    pertubation_list = torch.clamp(pertubation_list, -epsilon, epsilon)

    best_fitness, adv_img_files, output = benchmark(args, image_tensors, input_ids, image_sizes, 
                                                    gt_answer, pertubation_list, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]
    success = False
    num_evaluation = 1

    for i in tqdm(range(1, args.max_query)):
        alpha = torch.randn_like(image_tensors).cuda()
        new_pertubation_list =  torch.clamp(pertubation_list + alpha, -epsilon, epsilon)

        new_fitness, adv_img_files, output = benchmark(args, image_tensors, input_ids, image_sizes, 
                                                       gt_answer, new_pertubation_list, model)
        print("current output: ", output)

        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_img_files_adv = adv_img_files
            pertubation_list = new_pertubation_list
        #     sigma *= c_increase
        # else:
        #     sigma *= c_decrease

        print("best fitness: ", best_fitness, "sigma: ", sigma)
        history.append(best_fitness)
        num_evaluation += 1
        if best_fitness > 0:
            success = True
            break
    return num_evaluation, pertubation_list, best_img_files_adv, success
    
def main(args):
    model, image_token, special_token = init_model(args)
    acc = 0
    run = 100
    total_evaluation = 0
    for i, item in enumerate(multi_QA_loader(image_placeholder=image_token)):
        if i == run:
            break
        
        id = item['id']
        qs = item['question']
        img_files = item['image_files'] # list of pil_image
        gt_answer = item['answer']
        
        for j, img in enumerate(img_files):
            img.save(f"clean_{j}.png")
        input_ids, image_tensors, image_sizes = model.repair_input(qs, img_files)
        print("Image tensors shape: ", image_tensors.shape)
        print("Image size: ", image_sizes)
        
        original_output = model.inference(input_ids, image_tensors, image_sizes)[0]
        print("Question: ", qs)
        print("Original output: ", original_output)
        print("Ground truth answer: ", gt_answer)

        num_evaluation, pertubation_list, best_img_files_adv, success =ES_1_1(args, FreeText_benchmark, id, model, 
                                                                              image_tensors, image_sizes, input_ids, gt_answer, 
                                                                              epsilon=args.epsilon, sigma=1.5)
        print("success: ", success)
        if success == True:
            for j, img in enumerate(best_img_files_adv):
                img.save(f"adv_{i}_{j}.png")
                acc += 1
                total_evaluation += num_evaluation
        
        # break
    print(f"Accuracy max_query={args.max_query}:{acc/run} total_evaluation: {total_evaluation}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retrieval", type=int, default=1)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--max_query", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 