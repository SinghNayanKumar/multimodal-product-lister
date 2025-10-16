import torch
import torch.nn as nn
from transformers import GitForCausalLM

class GitBaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_name = config['model']['model_name']
        self.git_model = GitForCausalLM.from_pretrained(model_name)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        """
        The forward pass for training.
        --- FIX: Added 'attention_mask' to arguments and model call ---
        """
        outputs = self.git_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask, # <-- ADDED THIS
            labels=labels
        )
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }

    @torch.no_grad()
    def predict(self, pixel_values, processor, max_length=128, num_beams=4):
        """
        The prediction/inference method.
        Note: For generation, the model creates its own attention mask internally.
        """
        self.eval()
        generated_ids = self.git_model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text