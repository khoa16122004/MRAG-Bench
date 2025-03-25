from dataloader import load_dataset
from tqdm import tqdm
from PIL import Image
import io

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
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images] # retrieval image
        
        image = item['image'].convert('RGB') # input-question image, 
        print(image)
        break