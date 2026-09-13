import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            # input valid_hidden_states = [N_valid, 768]
            # hidden_size = 768
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states = [B, T, 768], attention_mask = [B, T]
        batch_size = hidden_states.size(0)
        pooled = []
        for batch_index in range(batch_size):
            # Select only valid temporal frames.
            valid_hidden_states = hidden_states[
                batch_index,
                attention_mask[batch_index].bool(),
            ]
            # valid_hidden_states = [T_valid, 768]
            scores = self.attention(valid_hidden_states)
            # Normalize attention weights only over valid frames.
            weights = torch.softmax(
                scores,
                dim=0,
            )
            # Weighted sum over valid frames.
            pooled_output = torch.sum(
                weights * valid_hidden_states,
                dim=0,
            )
            # pooled_output = [768]
            pooled.append(pooled_output)
        # pooled = [B, 768]
        return torch.stack(pooled, dim=0)
