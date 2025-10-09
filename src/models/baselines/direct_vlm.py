# src/models/baselines/direct_vlm.py

import torch.nn as nn
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import os

class DirectVLM(nn.Module):
    """
    A standard Vision-Language Model that directly maps images to text.
    This version includes a robust method for loading a fine-tuned model.
    """
    def __init__(self, model_instance):
        """
        Private constructor. Use the class methods `from_config` or `from_trained`
        to create an instance.
        """
        super().__init__()
        self.model = model_instance

    @classmethod
    def from_config(cls, config):
        """
        Creates a new model from a config for training.
        This will show the 'uninitialized weights' warning, which is EXPECTED for training.
        """
        model_instance = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model']['vision_model_name'], 
            config['model']['text_model_name']
        )
        
        # Configure special tokens for the decoder
        model_instance.config.decoder_start_token_id = model_instance.decoder.config.bos_token_id
        model_instance.config.pad_token_id = model_instance.decoder.config.eos_token_id  
        
        return cls(model_instance)

    @classmethod
    def from_trained(cls, model_path):
        """
        Loads a fully fine-tuned model from a directory for inference.
        This should NOT show the 'uninitialized weights' warning.
        """
        model_instance = VisionEncoderDecoderModel.from_pretrained(model_path)
        return cls(model_instance)

    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def predict(self, pixel_values, tokenizer):
        """
        The generation method for inference.
        """
        generated_ids = self.model.generate(
            pixel_values,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            max_length=128, 
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)