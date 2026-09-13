"""
SER temporal windowing wrapper.
Role in the pipeline:
    speech segment
        ↓
    SERWindowWrapper
        ↓
    temporal windows
        ↓
    SER model
        ↓
    emotion scores for each window

Input:
    mono 16 kHz waveform [T]

Output:
    list[dict]
    Each dict contains:
        "start": window start time relative to the speech segment
        "end": window end time relative to the speech segment
        "emotions": SER model output for the window
"""

import torch
import torch.nn as nn


class SERWindowWrapper(nn.Module):
    def __init__(
        self,
        model,
        window_size: float = 3.0,
        hop_size: float = 0.5,
        sample_rate: int = 16000,
    ):
        """
        Parameters
        model:
            SER model applied independently to each temporal window.
        window_size:
            Window duration in seconds.
        hop_size:
            Distance between the starts of consecutive windows, in seconds.
        sample_rate:
            Sampling rate of the input waveform.
        """

        super().__init__()
        if not callable(model):
            raise TypeError("'model' must be callable, " f"got {type(model).__name__}")
        if window_size <= 0:
            raise ValueError(f"'window_size' must be > 0, got {window_size}")
        if hop_size <= 0:
            raise ValueError(f"'hop_size' must be > 0, got {hop_size}")
        if hop_size > window_size:
            raise ValueError(
                "'hop_size' cannot be greater than 'window_size'. "
                f"Got hop_size={hop_size}, window_size={window_size}"
            )
        if sample_rate <= 0:
            raise ValueError(f"'sample_rate' must be > 0, got {sample_rate}")
        self.model = model
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window_samples = round(window_size * sample_rate)
        self.hop_samples = round(hop_size * sample_rate)

    # Apply the SER model to consecutive temporal windows.
    def __call__(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:

        waveform = self._validate_waveform(waveform)
        if waveform.numel() == 0:
            return []
        windows = self._create_windows(waveform)
        results = []
        for window in windows:
            window_waveform = window["waveform"]
            emotions = self.model(window_waveform)
            results.append(
                {
                    "start": window["start"],
                    "end": window["end"],
                    "emotions": emotions,
                }
            )
        return results

    # Create overlapping temporal windows.
    def _create_windows(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:

        num_samples = waveform.shape[0]
        # If the segment is shorter than one complete window,
        # process the entire segment as a single window.
        if num_samples <= self.window_samples:
            return [
                {
                    "waveform": waveform,
                    "start": 0.0,
                    "end": num_samples / self.sample_rate,
                }
            ]
        windows = []
        start_sample = 0
        while start_sample < num_samples:
            end_sample = start_sample + self.window_samples
            # If a complete window fits inside the remaining audio,
            # process a standard window of window_size seconds.
            if end_sample <= num_samples:
                windows.append(
                    {
                        "waveform": waveform[start_sample:end_sample],
                        "start": start_sample / self.sample_rate,
                        "end": end_sample / self.sample_rate,
                    }
                )
            else:
                # The final window is shorter than window_size and
                # extends until the end of the complete speech segment.
                windows.append(
                    {
                        "waveform": waveform[start_sample:num_samples],
                        "start": start_sample / self.sample_rate,
                        "end": num_samples / self.sample_rate,
                    }
                )
                break
            start_sample += self.hop_samples
        return windows

    # Validate and normalize waveform shape.
    def _validate_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:

        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                "'waveform' must be a torch.Tensor, " f"got {type(waveform).__name__}"
            )
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise ValueError(
                    "Expected a mono waveform with shape "
                    "[T] or [1, T]. "
                    f"Got shape {tuple(waveform.shape)}."
                )
            waveform = waveform.squeeze(0)
        elif waveform.ndim != 1:
            raise ValueError(
                "Expected waveform shape [T] or [1, T]. "
                f"Got shape {tuple(waveform.shape)}."
            )
        if not waveform.is_floating_point():
            waveform = waveform.float()
        return waveform
