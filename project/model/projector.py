import torch.nn as nn


class LanguageProjection(nn.Module):
    def __init__(self, in_features: int = 768, out_features: int = 896):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
