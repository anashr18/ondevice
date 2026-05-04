import torch
import torch.nn as nn
from transformers import AutoModel


class InternViTEncoder(nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/InternViT-300M-448px"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [N_tiles, 3, 448, 448]  bfloat16
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        feats = outputs.last_hidden_state  # [N_tiles, 1024 or 1025, 1024]
        if feats.shape[1] == 1025:
            feats = feats[:, 1:, :]
        return feats  # [N_tiles, 1024, 1024]
