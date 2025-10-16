# FILE: src/models/baselines/direct_vlm_model.py
# --- No changes to the code ---

import torch
import torch.nn as nn
from transformers import ViTModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

class DirectVLM(nn.Module):
    """
    A direct Vision-Language Model (VLM) baseline.
    """
    def __init__(self, vision_model_name, text_model_name):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.text_generator = T5ForConditionalGeneration.from_pretrained(text_model_name)
        
        vision_embedding_dim = self.vision_encoder.config.hidden_size
        text_embedding_dim = self.text_generator.config.d_model
        self.vit_to_t5 = nn.Linear(vision_embedding_dim, text_embedding_dim)

    def forward(self, pixel_values, labels=None, **kwargs):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        cls_token = vision_outputs.last_hidden_state[:, 0:1, :]
        projected_vision_features = self.vit_to_t5(cls_token)
        encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=projected_vision_features)
        
        outputs = self.text_generator(
            labels=labels,
            encoder_outputs=encoder_outputs_for_t5
        )
        return outputs

    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, max_length=256, num_beams=4):
        self.eval()
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        cls_token = vision_outputs.last_hidden_state[:, 0:1, :]
        projected_vision_features = self.vit_to_t5(cls_token)
        encoder_outputs_for_t5 = BaseModelOutput(last_hidden_state=projected_vision_features)
        
        generated_ids = self.text_generator.generate(
            encoder_outputs=encoder_outputs_for_t5,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        generated_json = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_json