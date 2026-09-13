"""
Apply masked temporal Mean/Std/Max pooling to hidden states.
"""

import torch


def mean_std_max_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    hidden_states : torch.Tensor
        WavLM hidden states with shape [batch, time, hidden_size].
    attention_mask : torch.Tensor
        Attention mask with shape [batch, time].
        1 indicates a valid timestep, 0 indicates padding.
    """

    # [batch, time] -> [batch, time, 1]
    mask = attention_mask.unsqueeze(-1).bool()
    # Convert mask to the same dtype as hidden states
    mask_float = mask.to(hidden_states.dtype)
    # Number of valid timesteps for each sample
    valid_count = mask_float.sum(dim=1).clamp(min=1.0)
    # Mean
    mean = (hidden_states * mask_float).sum(dim=1) / valid_count
    # Standard deviation
    centered = hidden_states - mean.unsqueeze(1)
    variance = (centered.pow(2) * mask_float).sum(dim=1) / valid_count
    std = torch.sqrt(variance + 1e-8)
    # Maximum
    masked_hidden = hidden_states.masked_fill(
        ~mask,
        torch.finfo(hidden_states.dtype).min,
    )
    max_values = masked_hidden.max(dim=1).values
    # Concatenate along the feature dimension
    pooled_features = torch.cat(
        [mean, std, max_values],
        dim=-1,
    )
    return pooled_features


def mean_std_max_pooling_without_mask(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # Mean
    mean = hidden_states.mean(dim=1)
    # Standard deviation
    centered = hidden_states - mean.unsqueeze(1)
    variance = centered.pow(2).mean(dim=1)
    std = torch.sqrt(variance + 1e-8)
    # Maximum
    max_values = hidden_states.max(dim=1).values
    # Concatenate along the feature dimension
    pooled_features = torch.cat(
        [mean, std, max_values],
        dim=-1,
    )
    return pooled_features
