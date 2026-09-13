"""
Load a complete audio file or a segment and return a mono waveform resampled to the target sample rate.
Audio preprocessing pipeline:
    audio file
        ↓
    audio loading
        ↓
    mono conversion
        ↓
    resampling to 16 kHz
        ↓
    one-dimensional waveform
"""

from functools import lru_cache
from pathlib import Path
from typing import Union

import soundfile as sf
import torch
import torchaudio


# Return cached metadata for an audio file.
# Metadata is useful when the same audio file is accessed multiple times.
@lru_cache(maxsize=1024)
def _get_audio_metadata(audio_path: str):
    info = sf.info(audio_path)
    return (
        info.samplerate,
        info.frames,
    )


# Return a cached torchaudio resampler.
@lru_cache(maxsize=16)
def _get_resampler(
    original_sample_rate: int,
    target_sample_rate: int,
):
    return torchaudio.transforms.Resample(
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )


def extract_waveform(
    audio_path: Union[str, Path],
    target_sample_rate: int = 16000,
) -> torch.Tensor:

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    if target_sample_rate <= 0:
        raise ValueError(
            f"'target_sample_rate' must be > 0, " f"got {target_sample_rate}"
        )
    # Obtain audio metadata.
    original_sample_rate, total_num_frames = _get_audio_metadata(str(audio_path))
    if total_num_frames <= 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    # Load the complete audio file.
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise ValueError(f"The processed audio is empty: {audio_path}")
    # Convert multi-channel audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )
    # Remove the channel dimension.
    waveform = waveform.squeeze(0)
    # Resample to the target sampling rate.
    if sample_rate != target_sample_rate:
        resampler = _get_resampler(
            sample_rate,
            target_sample_rate,
        )
        waveform = resampler(waveform)
    return waveform


def prepare_segment(
    audio_path: Union[str, Path],
    start: float,
    end: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if start < 0:
        raise ValueError(f"'start' must be >= 0, got {start}")
    if end <= start:
        raise ValueError(
            f"'end' must be greater than 'start'. " f"Got start={start}, end={end}"
        )
    # Obtain audio metadata
    original_sample_rate, total_num_frames = _get_audio_metadata(audio_path)
    print("Audio:", audio_path)
    print("original_sample_rate:", original_sample_rate)
    print("total_num_frames:", total_num_frames)
    print("duration:", total_num_frames / original_sample_rate)
    # Convert timestamps to sample indices
    start_sample = int(start * original_sample_rate)
    end_sample = int(end * original_sample_rate)
    print("start_sample:", start_sample)
    print("end_sample:", end_sample)
    start_sample = max(0, start_sample)
    end_sample = min(end_sample, total_num_frames)
    if end_sample <= start_sample:
        raise ValueError(
            "Requested segment is outside "
            "the audio boundaries: "
            f"start={start}, end={end}"
        )
    num_samples = end_sample - start_sample
    # Load only the requested audio segment
    waveform, sample_rate = torchaudio.load(
        audio_path,
        frame_offset=start_sample,
        num_frames=num_samples,
    )
    # Convert multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )
    # Remove the channel dimension.
    waveform = waveform.squeeze(0)
    if waveform.numel() == 0:
        raise ValueError("The processed audio segment is empty.")
    # Resample
    if sample_rate != target_sample_rate:
        resampler = _get_resampler(
            sample_rate,
            target_sample_rate,
        )
        waveform = resampler(waveform)
    return waveform
