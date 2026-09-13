"""
End-to-end Speech Emotion Recognition model.

    audio file
        ↓
    extract_waveform()
        ↓
    SpeechSegmentExtractor
        ├── MP-SENet-DNS enhancement
        └── Silero VAD
        ↓
    SpeechSegmentDispatcher
        ↓
    CompositeModel
        ├── SERWindowWrapper
        │     └── SERBranch
        │           ├── WavLM
        │           └── MLP regressor head
        │
        └── WhisperX
              ├── transcription
              └── forced alignment
                    └── word-level timestamps

"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from src.inference.composite_model import CompositeModel
from src.inference.dispatcher import SpeechSegmentDispatcher
from src.inference.preprocessing_pipeline import SpeechSegmentExtractor
from src.inference.audio_processing import (
    extract_waveform,
    prepare_segment,
)


class SerModel:
    def __init__(
        self,
        device: str | None = None,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
        denoiser_segment_size: int = 64_000,
    ):
        self.sample_rate = 16000
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.emotion_names = [
            "happiness",
            "sadness",
            "anger",
            "fear",
            "disgust",
            "surprise",
        ]
        self.extractor = SpeechSegmentExtractor(
            denoiser_checkpoint="src/models/mpsenet/pretrained",
            device=self.device,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            denoiser_segment_size=denoiser_segment_size,
        )
        self.composite_model = CompositeModel()
        self.composite_model.eval()
        self.dispatcher = SpeechSegmentDispatcher(
            model=self.composite_model,
        )

    # Predict from a complete audio file.
    def predict_audio(self, audio_path: str | Path) -> list[dict]:
        waveform = extract_waveform(
            audio_path,
            target_sample_rate=self.sample_rate,
        )
        with torch.inference_mode():
            speech_segments = self.extractor(waveform)
            results = self.dispatcher(speech_segments)
        self.show_results(results)
        return results

    # Predict from a specific segment of an audio file.
    def predict_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
    ) -> list[dict]:
        waveform = prepare_segment(
            audio_path,
            start=start,
            end=end,
            target_sample_rate=self.sample_rate,
        )
        results = self.__predict_waveform(waveform)
        for result in results:
            result["start"] += start
            result["end"] += start
            for window in result["result"]["ser"]:
                window["start"] += start
                window["end"] += start
            for word in result["result"]["whisper"]["words"]:
                word["start"] += start
                word["end"] += start
        self.show_results(results)
        return results

    # Predict from an already extracted waveform.
    def predict_waveform(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:
        with torch.inference_mode():
            speech_segments = self.extractor(waveform)
            results = self.dispatcher(speech_segments)
        self.show_results(results)
        return results

    # Class method
    def __predict_waveform(
        self,
        waveform: torch.Tensor,
    ) -> list[dict]:
        with torch.inference_mode():
            speech_segments = self.extractor(waveform)
            results = self.dispatcher(speech_segments)
        return results

    # Display SER and Whisper results in a single figure.
    def show_results(
        self,
        results: list[dict],
    ) -> None:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ser_axis = axes[0]
        whisper_axis = axes[1]
        self._plot_emotions(
            ser_axis,
            results,
        )
        self._plot_words(
            whisper_axis,
            results,
        )
        whisper_axis.set_xlabel("Time (s)")
        fig.tight_layout()
        plt.show()

    # Plot the six emotion trajectories.
    def _plot_emotions(
        self,
        axis,
        results: list[dict],
    ) -> None:
        for result in results:
            segment_start = result["start"]
            segment_end = result["end"]
            ser_windows = result["result"]["ser"]
            if not ser_windows:
                continue
            for emotion_index, emotion_name in enumerate(self.emotion_names):
                centers = []
                values = []
                for window in ser_windows:
                    start = window["start"]
                    end = window["end"]
                    center = (start + end) / 2.0
                    value = self._get_emotion_value(
                        window["emotions"],
                        emotion_index,
                    )
                    centers.append(center)
                    values.append(value)
                if not centers:
                    continue
                x = [segment_start]
                y = [values[0]]
                x.extend(centers)
                y.extend(values)
                x.append(segment_end)
                y.append(values[-1])
                if len(x) >= 3:
                    interpolator = PchipInterpolator(
                        x,
                        y,
                    )
                    # plot only on the VAD segment
                    smooth_x = torch.linspace(
                        x[0],
                        x[-1],
                        300,
                    ).numpy()
                    smooth_y = interpolator(smooth_x)
                    axis.plot(
                        smooth_x,
                        smooth_y,
                        label=emotion_name,
                    )
                else:
                    axis.plot(
                        x,
                        y,
                        label=emotion_name,
                    )

        axis.set_ylabel("Emotion value")
        axis.set_title(
            "Emotion trajectories",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

    # Extract one emotion value from a prediction.
    def _get_emotion_value(
        self,
        emotions,
        emotion_index: int,
    ) -> float:
        if isinstance(emotions, torch.Tensor):
            values = emotions.detach().cpu().flatten()
            if emotion_index >= values.numel():
                raise IndexError(
                    f"Emotion index {emotion_index} is out of range. "
                    f"Expected at least {emotion_index + 1} values, "
                    f"got {values.numel()}."
                )
            return float(values[emotion_index].clamp(0.0, 1.0).item())

        if isinstance(emotions, (list, tuple)):
            if emotion_index >= len(emotions):
                raise IndexError(
                    f"Emotion index {emotion_index} is out of range. "
                    f"Expected at least {emotion_index + 1} values, "
                    f"got {len(emotions)}."
                )
            value = emotions[emotion_index]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            return float(max(0.0, min(1.0, float(value))))

        if isinstance(emotions, dict):
            emotion_name = self.emotion_names[emotion_index]
            if emotion_name not in emotions:
                raise KeyError(f"Emotion '{emotion_name}' not found in prediction.")
            value = emotions[emotion_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            return float(max(0.0, min(1.0, float(value))))

        raise TypeError(
            "Unsupported emotion output format: " f"{type(emotions).__name__}"
        )

    # Plot Whisper words on a two-line timeline.
    def _plot_words(
        self,
        axis,
        results: list[dict],
    ) -> None:
        words = []

        for result in results:
            whisper_result = result["result"]["whisper"]

            for word in whisper_result.get("words", []):
                words.append(word)

        if not words:
            axis.set_title(
                "Word timeline",
                fontsize=13,
                fontweight="semibold",
                pad=10,
            )
            axis.set_yticks([0, 1])
            axis.set_yticklabels(["", ""])
            return

        axis.set_ylim(-0.5, 1.5)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["", ""])

        for index, word in enumerate(words):
            start = word["start"]
            end = word["end"]
            center = (start + end) / 2
            word_text = word["word"].strip()
            row = index % 2
            # Word orizontal interval
            axis.plot(
                [start, end],
                [row, row],
                linewidth=1.2,
            )
            # Start boundary
            axis.plot(
                [start, start],
                [row - 0.12, row + 0.12],
                linewidth=1.0,
            )
            # End boundary
            axis.plot(
                [end, end],
                [row - 0.12, row + 0.12],
                linewidth=1.0,
            )
            # Center point
            axis.scatter(
                center,
                row,
                s=30,
                zorder=3,
            )
            # Word
            axis.text(
                center,
                row + 0.16,
                word_text,
                ha="center",
                va="bottom",
            )
            # Center timestamp
            axis.text(
                center,
                row - 0.16,
                f"{center:.2f}s",
                ha="center",
                va="top",
                fontsize=8,
            )
        axis.set_title(
            "Word timeline",
            fontsize=13,
            fontweight="semibold",
            pad=10,
        )
