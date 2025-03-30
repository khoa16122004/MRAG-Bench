from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torchvision.transforms as transforms
import copy
import torch
import warnings
import os
class LLava:
    def __init__(self, pretrained, model_name):
        
        # llava-next-interleave-7b
        # llava-onevision-qwen2-7b-ov
        self.pretrained = f"lmms-lab/{pretrained}"
        self.model_name = model_name
        # self.device = "cuda"
        self.device = os.evirion.get("CUDA_VISIBLE_DEVICES", "")
        self.device_map = "auto"
        self.llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        self.llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, model_name, device_map=self.device_map, **self.llava_model_args)
        self.model.eval()
        
    
    def decode_image_tensors(self, image_tensors, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)):
        unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(image_mean, image_std)],
            std=[1 / s for s in image_std]
        )

        to_pil = transforms.ToPILImage()
        pil_images = [to_pil(unnormalize(img.cpu())) for img in image_tensors]

        return pil_images
    
    def repair_input(self, qs, img_files):
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        image_tensors = process_images(img_files, self.image_processor, self.model.config)
        image_tensors = torch.stack([_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors])
        image_sizes = [image.size for image in img_files]
        return input_ids, image_tensors, image_sizes
    
    def inference(self, input_ids, image_tensors, image_sizes):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # input_ids, image_tensors, image_sizes = self.repair_input(qs, img_files)
                        
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
            outputs = text_outputs
            return outputs
        