# src/models/baselines/direct_vlm_fixed.py

import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class DirectVLM(nn.Module):
    """
    A standard Vision-Language Model that directly maps images to text.
    This version is a simple wrapper around the Hugging Face model, designed to be
    used with the Seq2SeqTrainer.
    """
    def __init__(self, config):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model']['vision_model_name'], 
            config['model']['text_model_name']
        )

    # --- FIX 1: EXPOSE THE MAIN CONFIG ---
    # The Trainer sometimes needs to access the main model config.
    @property
    def config(self):
        return self.model.config

    # --- FIX 2: EXPOSE THE GENERATION CONFIG ---
    # The Trainer needs this to configure the text generation process.
    @property
    def generation_config(self):
        return self.model.generation_config

    def forward(self, pixel_values, labels=None, **kwargs):
        """
        The forward pass for training and loss calculation.
        """
        return self.model(pixel_values=pixel_values, labels=labels, **kwargs)

    # --- FIX 3: EXPOSE THE GENERATE METHOD ---
    # The Trainer needs this for evaluation and prediction.
    def generate(self, *args, **kwargs):
        """
        Passes the generate call to the underlying model.
        """
        return self.model.generate(*args, **kwargs)