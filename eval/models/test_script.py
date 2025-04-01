from llava_ import LLava
import os
from PIL import Image
import torch

model = LLava("llava-onevision-qwen2-7b-ov", "llava-onevision-qwen2-7b-ov") 
img_path_adv = "/data/elo/khoatn/MRAG-Bench/eval/models/test_stt_ES_lambda=50_epsilon=0.1_maxiter=2_pretrained=llava-onevision-qwen2-7b-ov/18/0/adv.png"
clean_img_path = r"/data/elo/khoatn/MRAG-Bench/eval/models/test_stt_ES_lambda=50_epsilon=0.1_maxiter=2_pretrained=llava-onevision-qwen2-7b-ov/18/clean_img"
img_paths = [os.path.join(clean_img_path, path_) for path_ in os.listdir(clean_img_path)]
img_files_clean = [Image.open(img_path).convert("RGB") for img_path in img_paths]
# tmp_adv = img_files_clean[1]
# img_files_clean[0] = tmp_adv
img_files_adv = Image.open(img_path_adv).convert("RGB")
# img_files_clean[1] = img_files_adv
img_files_clean[0] = img_files_adv


for i, img in enumerate(img_files_clean):
    img.save(f"test_{i}.png")
question = "Which images depict activities that are typically seasonal, and what seasons are they most associated with?<image><image><image><image>"
input_ids, image_tensors, image_sizes = model.repair_input(question, img_files_clean)
print("Decoded input: ", model.inference(input_ids, image_tensors, image_sizes))

image_tensors = torch.load("/data/elo/khoatn/MRAG-Bench/eval/models/test_stt_ES_lambda=50_epsilon=0.1_maxiter=2_pretrained=llava-onevision-qwen2-7b-ov/18/0/all_adv.pt")
print("Directed input: ", model.inference(input_ids, image_tensors, image_sizes))


