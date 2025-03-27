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
       
        # OpenFlamingo-9B-vitl-mpt7b
        checkpoint_path = hf_hub_download(f"openflamingo/{pretrained}", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"), strict=False)
       
        self.model = model.cuda()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer.pad_token_id = 50277
        self.tokenizer.padding_side = "left"
    def inference(self, qs, img_files):
        vision_x = [self.image_processor(image).unsqueeze(0) for image in img_files]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).cuda()
        lang_x = self.tokenizer([qs], return_tensors="pt")
        
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].cuda(),
            attention_mask=lang_x["attention_mask"].cuda(),
            max_new_tokens=100,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=5, 
            temperature=0.8,  
            top_p=0.9,  
        )

        output = self.tokenizer.decode(generated_text[0].tolist(), skip_special_tokens=True)
        return output

        
        
        