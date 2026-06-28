# taken from the Bad-Teacher repo

import torch
import torch.nn as nn

from transformers import ViTModel


class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:,0])

        return logits