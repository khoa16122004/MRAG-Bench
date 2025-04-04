from llava_ import LLava
import torch
from time import time
import copy
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
import torch.multiprocessing as mp
from PIL import Image
import os

def inference_worker(rank, model, input_ids, image_tensor, image_size, results, lock, device):
    # torch.cuda.set_device(device)

    outputs = model.inference(input_ids, image_tensor, image_size)
    with lock:
        results.append(outputs)

def run_parallel_inference(model, image_tensors, image_sizes, input_ids, device):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results = manager.list()
    lock = manager.Lock()

    process_images = []
    for rank, image_tensor in enumerate(image_tensors):
        p = mp.Process(target=inference_worker, args=(rank, model, input_ids, image_tensor, image_sizes, results, lock, device))
        p.start()
        process_images.append(p)
    
    for p in process_images:
        p.join()

    return list(results)


if __name__ == '__main__':

    model = LLava("llava-onevision-qwen2-7b-ov", "llava-onevision-qwen2-7b-ov")
    device = "cuda:" + os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print("Device: ", device)
    question = "Which city is the capital of France?<image><image><image>"
    num_batch = 50
    image_files = [Image.open("clean_0.png").convert("RGB") for _ in range(3)]
    input_ids, image_tensors, image_sizes = model.repair_input(question, image_files)
    image_tensors = torch.stack([image_tensors for _ in range(num_batch)])
    print("Image shape:", image_tensors.shape)

    # normal infererence
    start = time()
    result = []
    print(len(result))
    for image_tensor in image_tensors:
        outputs = model.inference(input_ids, image_tensor, image_sizes)
        result.append(outputs)
    print(len(result))
    print("Time inference each sample: ", time() - start)

    input("Wait")

    # parallel infererence
    # start = time()
    # result =run_parallel_inference(model, image_tensors, image_sizes, input_ids, device)
    # print("Time inference multiple samples: ", time() - start)
    streams = [torch.cuda.Stream() for _ in range(num_batch)]
    results = [None] * num_batch
    start = time()
    for i, (image_tensor, stream) in enumerate(zip(image_tensors, streams)):
        with torch.cuda.stream(stream):
            results[i] = model.inference(input_ids, image_tensor, image_sizes)
    print("Time inference multiple samples: ", time() - start)
    torch.cuda.synchronize()
    print(len(result))
    input("Wait")




