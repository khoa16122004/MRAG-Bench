from dataloader import load_dataset
from tqdm import tqdm
from PIL import Image
import io
import os

def extract_data(output_dir):
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
        
        img_dir = os.path.join(output_dir, qs_id)
        os.makedirs(img_dir, exist_ok=True)
        
        with open(os.path.join(img_dir, f"question.txt"), "w") as f:
            f.write(f"Question: {qs}\n\n")
            f.write(f"A: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}\n\n")
            f.write(f"Anwers: {gt_choice}")
            
        
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images] # retrieval image
        for i, img in enumerate(gt_images):
            img.save(os.path.join(img_dir, f"gt_{i}.png"))
        
        
        image = item['image'].convert('RGB') # input-question image, 
        image.save(os.path.join(img_dir, "question_img.png"))
        break
    

extract_data("Extract_image")