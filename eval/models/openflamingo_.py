from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch

class OpenFlamingo:
    def __init__(self, pretrained):
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 50277
        
        # OpenFlamingo-9B-vitl-mpt7b
        checkpoint_path = hf_hub_download(f"openflamingo/{pretrained}", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"), strict=False)

        self.model = model.cuda()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
    def inference(self, qs, img_files):
        vision_x = [self.image_processor(image).unsqueeze(0) for image in img_files]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).cuda()
        print(f"PAD TOKEN: {self.tokenizer.pad_token}, PAD TOKEN ID: {self.tokenizer.pad_token_id}")
        lang_x = self.tokenizer([qs], return_tensors="pt")
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].cuda(),
            attention_mask=lang_x["attention_mask"].cuda(),
            max_new_tokens=20,
            num_beams=3,
        )
        output = self.tokenizer.decode(generated_text[0]).strip()
        return output

        
        
        