"""
PyTorch Dataset backed by an HDF5 waveform dataset and collate function.
"""

from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class WaveformDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = Path(h5_path)
        self._h5_file = None
        with h5py.File(self.h5_path, "r") as h5_file:
            self.num_segments = int(h5_file.attrs["num_segments"])
            self.num_samples = int(h5_file.attrs["num_samples"])
            self.sample_rate = int(h5_file.attrs["sample_rate"])
            self.targets_shape = h5_file["targets"].shape

    def _open_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(
                self.h5_path,
                "r",
            )

    def __len__(self):
        return self.num_segments

    def __getitem__(self, index):
        self._open_file()
        start = int(self._h5_file["offsets"][index])
        end = int(self._h5_file["offsets"][index + 1])
        waveform = np.asarray(
            self._h5_file["waveforms"][start:end],
            dtype=np.float32,
        )
        target = np.asarray(
            self._h5_file["targets"][index],
            dtype=np.float32,
        )
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        target = torch.from_numpy(target)
        return {
            "waveform": waveform,
            "target": target,
        }

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


def collate_audio_segments(batch):
    # Remove the channel dimension from each waveform.
    waveforms = [sample["waveform"].squeeze(0) for sample in batch]
    # Store the original length of every waveform before padding.
    lengths = torch.tensor(
        [waveform.shape[-1] for waveform in waveforms],
        dtype=torch.long,
    )
    # Pad all waveforms to the longest waveform in the batch.
    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=0.0,
    )
    max_length = padded_waveforms.shape[1]
    # Create a boolean mask identifying valid audio samples.
    attention_mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    # Stack emotion targets into a single batch tensor.
    targets = torch.stack([sample["target"] for sample in batch])
    return {
        "waveform": padded_waveforms,
        "attention_mask": attention_mask,
        "target": targets,
    }
