"""
Extract pooled WavLM features from the entire dataset.
"""

import torch
from pathlib import Path
from tqdm import tqdm
from src.models.mlp_head.pooling import mean_std_max_pooling


def extract_dataset_features(
    model,
    device,
    dataloader,
    output_path,
):
    """
    Parameters
    model : torch.nn.Module
        WavLM feature extractor.
    device : torch.device
        Device used for feature extraction.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing waveforms, attention masks, and targets.
    output_path : str or pathlib.Path
        Destination .pt file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model = model.to(device)
    model.eval()
    features = []
    targets = []
    total_samples = len(dataloader.dataset)
    processed_samples = 0
    with torch.inference_mode():
        progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Feature extraction",
            unit="batch",
        )
        for batch in progress_bar:
            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"]
            outputs = model(
                waveform,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state
            feature_attention_mask = model._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            pooled_features = mean_std_max_pooling(
                hidden_states,
                feature_attention_mask,
            )
            features.append(pooled_features.cpu())
            targets.append(target.cpu())
            processed_samples += waveform.shape[0]
            progress_bar.set_postfix(
                samples=f"{processed_samples}/{total_samples}",
            )
    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    dataset = {
        "features": features,
        "targets": targets,
    }
    torch.save(
        dataset,
        output_path,
    )
    return dataset
