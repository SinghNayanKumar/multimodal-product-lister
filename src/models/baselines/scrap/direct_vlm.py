# In src/models/baselines/direct_vlm.py

import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class DirectVLM(nn.Module):
    """
    A standard Vision-Language Model that directly maps images to text.
    This version includes robust methods for loading for both training and inference.
    """
    def __init__(self, model_instance: VisionEncoderDecoderModel):
        """
        Private constructor. It takes a pre-configured VisionEncoderDecoderModel instance.
        Use the class methods `from_config` or `from_trained` to create an object.
        """
        super().__init__()
        self.model = model_instance

    @classmethod
    def from_config(cls, config: dict):
        """
        Creates a new model from a config for training.
        This will show 'uninitialized weights' warnings, which is EXPECTED for training.
        """
        model_instance = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model']['vision_model_name'], 
            config['model']['text_model_name']
        )
        
        # Apply the crucial token configuration
        cls._configure_tokens(model_instance)
        
        return cls(model_instance)

    @classmethod
    def from_trained(cls, model_path: str):
        """
        Loads a fully fine-tuned model from a directory for inference.
        This is the corrected version for your inference script.
        """
        # 1. Load the underlying Hugging Face model from the saved path
        model_instance = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # 2. CRUCIAL: Apply our special token configuration to the loaded model
        cls._configure_tokens(model_instance)
        
        # 3. Return the fully configured model instance
        return cls(model_instance)

    @staticmethod
    def _configure_tokens(model_instance: VisionEncoderDecoderModel):
        """A helper method to apply token settings consistently."""
        decoder_config = model_instance.decoder.config
        
        # Set the token to start decoding with (BOS = Begin of Sentence)
        model_instance.config.decoder_start_token_id = decoder_config.bos_token_id
        
        # Set the token to be used for padding
        model_instance.config.pad_token_id = decoder_config.eos_token_id
        
        # Explicitly set the End-of-Sentence token ID in the main config
        # to resolve ambiguity for the .generate() method.
        model_instance.config.eos_token_id = decoder_config.eos_token_id

    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def predict(self, pixel_values, tokenizer):
        """
        The generation method for inference. Now it will work correctly.
        """
        generated_ids = self.model.generate(
            pixel_values,
            max_length=128, 
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)