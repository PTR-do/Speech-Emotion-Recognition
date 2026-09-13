"""
Speech enhancement + VAD + speech segment extraction.

Pipeline:
    waveform
        ↓
    MP-SENet-DNS enhancement
        ↓
    Silero VAD
        ↓
    speech waveform segments

Input:
    mono 16 kHz waveform
Output:
    list[dict]
    each dict contains a mono 16 kHz speech waveform
    and its start/end timestamps relative to the original waveform
"""

from pathlib import Path
import torch
from MPSENet import MPSENet
from silero_vad import get_speech_timestamps, load_silero_vad


class SpeechSegmentExtractor:
    def __init__(
        self,
        denoiser_checkpoint: str | Path,
        device: str | None = None,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        denoiser_segment_size: int = 64_000,
    ):
        """
        Parameters
        denoiser_checkpoint:
            Local path to the previously downloaded MP-SENet-DNS
            Hugging Face checkpoint.
        device:
            Device used by MP-SENet.
        vad_threshold:
            Silero VAD speech probability threshold.
        min_speech_duration_ms:
            Minimum duration of a detected speech region.
        min_silence_duration_ms:
            Minimum silence duration required to split speech regions.
        speech_pad_ms:
            Padding added around detected speech regions.
        denoiser_segment_size:
            Number of samples used by MP-SENet for each internal
            processing chunk. 64,000 samples = 4 seconds at 16 kHz.
        """

        self.SAMPLE_RATE = 16000
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.denoiser_segment_size = denoiser_segment_size
        checkpoint_path = Path(denoiser_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "MP-SENet checkpoint not found: " f"{checkpoint_path}"
            )
        self.denoiser = MPSENet.from_pretrained(str(checkpoint_path)).to(self.device)
        self.denoiser.eval()
        self.vad_model = load_silero_vad()

    # Enhance the waveform and extract its speech regions.
    @torch.inference_mode()
    def __call__(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:

        waveform = self._validate_waveform(waveform)
        if waveform.numel() == 0:
            return []
        enhanced_waveform = self._denoise(waveform)
        speech_segments = self._extract_speech_segments(enhanced_waveform)
        return speech_segments

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
        if waveform.numel() == 0:
            return waveform
        if not waveform.is_floating_point():
            waveform = waveform.float()
        return waveform

    # Enhance the complete waveform using MP-SENet.
    @torch.inference_mode()
    def _denoise(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:

        waveform_cpu = waveform.detach().cpu()
        enhanced_waveform, sample_rate, _ = self.denoiser(
            waveform_cpu.numpy(),
            segment_size=self.denoiser_segment_size,
        )
        if sample_rate != self.SAMPLE_RATE:
            raise RuntimeError(
                "Unexpected MP-SENet output sample rate: "
                f"{sample_rate}. Expected {self.SAMPLE_RATE} Hz."
            )
        enhanced_waveform = torch.from_numpy(enhanced_waveform).float()
        return enhanced_waveform

    # Silero VAD and extract speech waveforms.
    @torch.inference_mode()
    def _extract_speech_segments(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:

        speech_timestamps = get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=self.SAMPLE_RATE,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        segments = []
        for timestamp in speech_timestamps:
            start_sample = max(
                0,
                int(timestamp["start"]),
            )
            end_sample = min(
                waveform.shape[0],
                int(timestamp["end"]),
            )
            if end_sample <= start_sample:
                continue
            segment = waveform[start_sample:end_sample].clone()
            start = start_sample / self.SAMPLE_RATE
            end = end_sample / self.SAMPLE_RATE
            segments.append(
                {
                    "waveform": segment,
                    "start": start,
                    "end": end,
                }
            )
        return segments
