from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
from mantis.models.mllava import chat_mllava
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-clip-llama3")

from huggingface_hub import hf_hub_download
import torch

class Mantis:
    def __init__(self, pretrained):
        # Mantis-8B-clip-llama3
        self.processor = MLlavaProcessor.from_pretrained(f"TIGER-Lab/{pretrained}")
        self.model = LlavaForConditionalGeneration.from_pretrained(f"TIGER-Lab/{pretrained}", device_map=f"cuda:{torch.cuda.current_device()}", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        self.generation_kwargs = {"max_new_tokens": 1024, "num_beams": 1, "do_sample": False}
    def inference(self, qs, img_files):
        response, history = chat_mllava(qs, img_files, self.model, self.processor, **self.generation_kwargs)
        
        return response

        
        
        