from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import torch

class LLava:
    def __init__(self, pretrained, model_name):
        
        # llava-next-interleave-7b
        # llava-onevision-qwen2-7b-ov
        self.pretrained = f"lmms-lab/{pretrained}"
        self.model_name = model_name
        self.device = "cuda"
        self.device_map = "auto"
        self.llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        self.llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, model_name, device_map=self.device_map, **self.llava_model_args)
        self.model.eval()
        
    def encode_text_batch(self, qs): 
        inputs_ids = []
        for question in qs:
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_id = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
            inputs_ids.append(input_id)
        
        return torch.stack(inputs_ids)  
    
    def encode_image_batch(self, img_files):
        image_tensors = []
        image_sizes = []
        for batch_img in img_files:
            batch_img_tensor = process_images(batch_img, self.image_processor, self.model.config)
            batch_img_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in batch_img_tensor]

            batch_img_sizes = [image.size for image in batch_img]
            
            image_tensors.append(batch_img_tensor)
            image_sizes.append(batch_img_sizes)


        return image_tensors, image_sizes
        
    def inference(self, qs, img_files): 
        
        input_ids = self.encode_text_batch(qs)
        image_tensors, image_sizes = self.encode_image_batch(img_files)
                    
        with torch.inference_mode():
            cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs