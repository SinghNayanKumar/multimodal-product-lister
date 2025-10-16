import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class DirectVLM(nn.Module):
    """
    A standard Vision-Language Model that directly maps images to text without
    any intermediate structured prediction. This serves as the 'Baseline-DirectVLM'.
    We use Hugging Face's VisionEncoderDecoderModel, which is a powerful and
    standard way to implement this baseline.
    """
    def __init__(self, config):
        super().__init__()
        # ANNOTATION: This handy function loads a pre-trained vision encoder (like ViT) and
        # a pre-trained language decoder (like GPT-2) and stitches them together with a
        # cross-attention mechanism, creating a complete image captioning model.
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model']['vision_model_name'], 
            config['model']['text_model_name']
        )
        
        # ANNOTATION: We must configure the special tokens for the decoder to ensure it knows
        # how to start generating text and when to use padding.
        # Configure the special tokens for the decoder
        self.model.config.decoder_start_token_id = self.model.decoder.config.bos_token_id
        self.model.config.pad_token_id = self.model.decoder.config.eos_token_id  


    def forward(self, pixel_values, labels=None, **kwargs):
        """
        The forward pass is very simple. The underlying Hugging Face model
        is designed to automatically calculate the language modeling loss
        when the 'labels' (target token IDs) are provided.
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs

    def predict(self, pixel_values, tokenizer):
        """
        The generation method for inference.
        """
        generated_ids = self.model.generate(
            pixel_values,  # <--- Pass the image tensor positionally
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            max_length=128,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
