import torch
from torch import nn
from src.models.mlp_head.pooling import (
    mean_std_max_pooling,
    mean_std_max_pooling_without_mask,
)


class SERbranch(nn.Module):
    """
    End-to-end WavLM + pooling + emotion regressor model.
    """

    def __init__(
        self,
        wavlm: nn.Module,
        regressor: nn.Module,
    ):
        super().__init__()
        self.wavlm = wavlm
        self.regressor = regressor

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training:
            outputs = self.wavlm(
                waveform,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            # Convert input attention mask to feature attention mask
            feature_attention_mask = self.wavlm._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            pooled_features = mean_std_max_pooling(
                hidden_states,
                feature_attention_mask,
            )
        else:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            device = next(self.wavlm.parameters()).device
            waveform = waveform.to(device)
            outputs = self.wavlm(waveform)
            hidden_states = outputs.last_hidden_state
            pooled_features = mean_std_max_pooling_without_mask(
                hidden_states,
            )
        # Emotion regression
        predictions = self.regressor(pooled_features)
        return predictions
