import torch
import torch.nn as nn


class EmotionRegressor(nn.Module):
    def __init__(
        self,
        wavlm_hidden_size: int = 768,
        num_emotions: int = 6,
    ):
        super().__init__()
        pooled_size = wavlm_hidden_size * 3

        self.layers = nn.Sequential(
            nn.Linear(pooled_size, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 6),
        )

    def forward(self, pooled_features):
        return self.layers(pooled_features)
