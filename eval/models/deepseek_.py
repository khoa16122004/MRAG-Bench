from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import torch
from transformers import AutoModelForCausalLM

class DeepSeek:
    def __init__(self, pretrained):
        # deepseek-vl-7b-chat
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(f"deepseek-ai/{pretrained}")
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{pretrained}", trust_remote_code=True)
        self.vl_gpt.to(torch.bfloat16).cuda().eval()
        
        
    def inference(self, qs, img_files):
        conversation = [
            {
                "role": "User",
                "content": qs,
                "images": img_files
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=img_files,
            force_batchify=True
        ).to(self.vl_gpt.device)
        
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
        
        
        