import torch
from torch import nn


class SERbranch(nn.Module):
    """
    End-to-end WavLM + learnable query attention pooling + emotion regressor model.
    """

    def __init__(
        self,
        wavlm: nn.Module,
        pooling: nn.Module,
        regressor: nn.Module,
    ):
        super().__init__()
        self.wavlm = wavlm
        self.pooling = pooling
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
            # Temporal WavLM hidden states
            hidden_states = outputs.last_hidden_state
            # Convert the audio-level attention mask
            # into a feature-level temporal attention mask.
            feature_attention_mask = self.wavlm._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            # Learnable Query Attention Pooling
            pooled_features = self.pooling(
                hidden_states,
                feature_attention_mask,
            )
        else:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            device = next(self.wavlm.parameters()).device
            waveform = waveform.to(device)
            # WavLM inference
            outputs = self.wavlm(waveform)
            # Temporal WavLM hidden states
            hidden_states = outputs.last_hidden_state
            # Learnable Query Attention Pooling
            pooled_features = self.pooling(
                hidden_states,
            )
        predictions = self.regressor(
            pooled_features,
        )
        return predictions
