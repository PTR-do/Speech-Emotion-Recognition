import torch
import torch.nn as nn


class MultiHeadRegressor(nn.Module):
    def __init__(self, hidden_size: int, num_emotions: int = 6):
        super().__init__()
        self.pooled_size = hidden_size
        self.num_emotions = num_emotions
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Linear(256, 1),
                )
                for _ in range(num_emotions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared(x)
        outputs = [head(shared_features) for head in self.heads]
        emotion_values = torch.cat(outputs, dim=1)
        return emotion_values
