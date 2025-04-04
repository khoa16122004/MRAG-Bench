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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

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


    
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

sim_model = SentenceTransformer('all-MiniLM-L6-v2')
def FreeText_benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                       gt_answer, pertubation_list, model):
    
    adv_img_tensors = image_tensors.detach().clone().cuda()
    adv_img_tensors[index_attack] = image_tensors[index_attack] + pertubation_list
    adv_pil_images = model.decode_image_tensors(adv_img_tensors) # torch tensor
    adv_img_tensors_real = model.repair_input(None, adv_pil_images)[1]
    output = model.inference(input_ids, adv_img_tensors_real, image_sizes)[0]    
    
    # cosine similarity
    emb1 = sim_model.encode(output, convert_to_tensor=True)
    emb2 = sim_model.encode(gt_answer, convert_to_tensor=True)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    s1 = 0.1 - similarity
    
    # BLEU score
    bleu = sentence_bleu([gt_answer.split()], output.split())
    s2 = 0.1 - bleu
    
    # number of words
    # num_words = len(output.split())
    # s3 = 0.1 * (10 - num_words)
    
    # # ROUGE
    # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # rouge_scores = scorer.score(gt_answer, output)
    # rouge1 = rouge_scores['rouge1'].fmeasure
    # rouge2 = rouge_scores['rouge2'].fmeasure
    # rougeL = rouge_scores['rougeL'].fmeasure
    # s4 = 0.3 - ((rouge1 + rouge2 + rougeL)/3)
    
    # # METEOR score
    # meteor = meteor_score([gt_answer], output)
    # s5 = 0.3 - meteor
    
    # weighted sum
    # final_score = s1 + s2 + s3 + s4 + s5
    final_score = s1 + s2
    return final_score, adv_pil_images, output, adv_img_tensors
        
    
def ES_1_lambda(args, benchmark, index_attack, model, lambda_,
                image_tensors, image_sizes, input_ids, gt_answer, 
                epsilon=0.05, sigma=1.5, c_increase=1.1, c_decrease=0.9):
    
    # image_tensors: batch_size x 3 x 224 x 224
    best_pertubations = torch.randn_like(image_tensors[index_attack]).cuda()
    best_pertubations = torch.clamp(best_pertubations, -epsilon, epsilon)

    best_fitness, adv_img_files, output, best_adv_img_tensors = benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                                                                          gt_answer, best_pertubations, model)
    best_img_files_adv = adv_img_files
    history = [best_fitness]
    success = False
    num_evaluation = 1
    
    for i in tqdm(range(args.max_query)):
        alpha = torch.randn(lambda_, *image_tensors[index_attack].shape).to(torch.float16).cuda()
        pertubations_list = alpha + best_pertubations * sigma
        pertubations_list = torch.clamp(pertubations_list, -epsilon, epsilon)
        
        current_fitnesses = []
        current_adv_files = []
        current_output = []
        current_adv_img_tensors = []
        for pertubations in pertubations_list:
            fitness, adv_img_files, output, adv_img_tensor = benchmark(args, image_tensors, index_attack, input_ids, image_sizes, 
                                                                       gt_answer, pertubations, model)    
            current_fitnesses.append(fitness)
            current_adv_files.append(adv_img_files)
            current_output.append(output)
            current_adv_img_tensors.append(adv_img_tensor)
        
        num_evaluation += lambda_
            
        current_fitnesses = torch.tensor(current_fitnesses)
        best_id_current_fitness = torch.argmax(current_fitnesses) 
        
        if current_fitnesses[best_id_current_fitness] > best_fitness:
            print("Tốt hơn")
            best_fitness = current_fitnesses[best_id_current_fitness]
            best_pertubations = pertubations_list[best_id_current_fitness]
            best_img_files_adv = current_adv_files[best_id_current_fitness]
            output = current_output[best_id_current_fitness]
            best_adv_img_tensors = current_adv_img_tensors[best_id_current_fitness]
            sigma *= c_increase
        else:
            sigma *= c_decrease
            
        history.append(best_fitness)
        
        print(f"Iteration {i}, best fitness: {best_fitness}, output: {output}\n")

        if best_fitness >= 0:
            success = True
            break
        
    return num_evaluation, history, best_img_files_adv, success, output, best_adv_img_tensors

def main(args):
    
    # repair dir
    experiment_dir = f"{args.prefix_path}_ES_lambda={args.lambda_}_epsilon={args.epsilon}_maxiter={args.max_query}_pretrained={args.pretrained}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # initiallize model
    model, image_token, special_token = init_model(args)
    acc = 0
    
    for i, item in enumerate(multi_QA_loader(image_placeholder=image_token)):
        # repair dir
        sample_dir = os.path.join(experiment_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)
        if i != args.index_sample:
            continue
        if i == args.run:
            break
        
        qs = item['question'] 
        img_files = item['image_files'] # list of pil_image
        gt_answer = item['answer']
        num_image = item['num_image']
        
        input_ids, image_tensors, image_sizes = model.repair_input(qs, img_files)
        
        # repair dir
        clean_dir = os.path.join(sample_dir, "clean_img")
        os.makedirs(clean_dir, exist_ok=True)
        for j, img_clean_files in enumerate(img_files):
            img_clean_files.save(os.path.join(clean_dir, f"{j}.png"))
        
        # inference
        original_output = model.inference(input_ids, image_tensors, image_sizes)[0]
        print("Question: ", qs)
        print("Original output: ", original_output)
        print("Ground truth answer: ", gt_answer)

        for index_attack in range(num_image):
            # repair dir
            index_dir = os.path.join(sample_dir, str(index_attack))
            os.makedirs(index_dir, exist_ok=True)
            
            
            num_evaluation, history, best_img_files_adv, success, output, best_adv_img_tensors =ES_1_lambda(args, FreeText_benchmark, index_attack, model, args.lambda_,
                                                                                                            image_tensors, image_sizes, input_ids, original_output, 
                                                                                                            epsilon=args.epsilon)
            
            # log
            attacked_img_files = best_img_files_adv[index_attack]
            attacked_img_files.save(os.path.join(index_dir, "adv.png"))
            torch.save(best_adv_img_tensors, os.path.join(index_dir, "all_adv.pt"))
            
            with open(os.path.join(index_dir, "history.txt"), "w") as f:          
                for i, fitness in enumerate(history):
                    f.write(f"iteration {i}: {fitness}\n")
            
            with open(os.path.join(index_dir, "output.txt"), "w") as f:
                f.write(f"Question: {qs}\n\n")
                f.write(f"Ground truth answer: {gt_answer}\n\n")
                f.write(f"Original output: {original_output}\n\n")
                f.write(f"Attacked output: {output}\n\n")
                f.write(f"Fitness:  {history[-1]}",)
                f.write(f"Num evaluation: {num_evaluation}\n\n")
                        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_sample", type=int, default=0)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--max_query", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--run", type=int, default=10)
    parser.add_argument("--lambda_", type=int, default=50)
    parser.add_argument("--prefix_path", type=str, default="")
    args = parser.parse_args()

    main(args)

    # python3 eval/models/llava_one_vision.py --answers-file "llava_one_vision_gt_rag_results.jsonl" --use_rag True --use_retrieved_examples False 