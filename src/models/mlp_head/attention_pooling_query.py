import torch
import torch.nn as nn


class LearnableQueryPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        # Learnable query shared across all samples in the batch.
        self.query = nn.Parameter(
            torch.empty(
                1,
                1,
                hidden_size,
            )
        )
        # Initialization of the learnable query.
        nn.init.normal_(
            self.query,
            mean=0.0,
            std=0.02,
        )
        # Linear projection for the WavLM hidden states used as keys.
        self.key_projection = nn.Linear(
            hidden_size,
            hidden_size,
        )
        # Linear projection for the WavLM hidden states used as values.
        self.value_projection = nn.Linear(
            hidden_size,
            hidden_size,
        )
        # Scaling factor used by scaled dot-product attention.
        self.scale = hidden_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        query = self.query
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        attention_scores = torch.matmul(
            query,
            keys.transpose(-2, -1),
        )
        attention_scores = attention_scores * self.scale

        # Apply the temporal attention mask only when provided.
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.bool().unsqueeze(1),
                torch.finfo(attention_scores.dtype).min,
            )

        # Normalize attention scores over the temporal dimension.
        # With a mask, weights are normalized only over valid frames.
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )
        # Compute the weighted sum of the projected value vectors.
        pooled_output = torch.matmul(
            attention_weights,
            values,
        )
        # Shape: [B, D]
        pooled_output = pooled_output.squeeze(1)

        return pooled_output
