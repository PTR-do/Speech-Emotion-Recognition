"""
Speech segment dispatcher.
The dispatcher is responsible for controlling the flow of speech
segments into the downstream model.
"""

from collections.abc import Callable, Iterable
import torch


class SpeechSegmentDispatcher:
    def __init__(
        self,
        model: Callable[[torch.Tensor], object],
    ):
        """
        Parameters
        model:
            Downstream model that processes one speech waveform at a time.
            The model is responsible for:
                - temporal windowing
                - emotion recognition
                - transcription
                - any other processing performed on the segment
        """
        if not callable(model):
            raise TypeError("'model' must be callable, " f"got {type(model).__name__}")
        self.model = model

    # Process each speech segment independently.
    def __call__(
        self,
        segments: Iterable[dict],
    ) -> list[dict]:
        results = []
        for segment in segments:
            self._validate_segment(segment)
            waveform = segment["waveform"]
            start = segment["start"]
            end = segment["end"]
            result = self.model(waveform)
            result = self._adjust_whisper_timestamps(
                result,
                start,
            )
            results.append(
                {
                    "start": start,
                    "end": end,
                    "result": result,
                }
            )
        return results

    # Align timestamps to the original segment time.
    def _adjust_whisper_timestamps(
        self,
        result: dict,
        segment_start: float,
    ) -> dict:
        whisper_result = result.get("whisper")
        if not isinstance(whisper_result, dict):
            return result
        words = whisper_result.get("words")
        if not isinstance(words, list):
            return result
        for word in words:
            if not isinstance(word, dict):
                continue
            if "start" in word:
                word["start"] += segment_start
            if "end" in word:
                word["end"] += segment_start
        return result

    # Validate the structure of a speech segment.
    def _validate_segment(
        self,
        segment: dict,
    ) -> None:
        if not isinstance(segment, dict):
            raise TypeError(
                "Each speech segment must be a dict, " f"got {type(segment).__name__}"
            )
        required_keys = {
            "waveform",
            "start",
            "end",
        }
        missing_keys = required_keys.difference(segment.keys())
        if missing_keys:
            raise ValueError(
                "Speech segment is missing required keys: " f"{sorted(missing_keys)}"
            )
        if not isinstance(segment["waveform"], torch.Tensor):
            raise TypeError(
                "'waveform' must be a torch.Tensor, "
                f"got {type(segment['waveform']).__name__}"
            )
        if not isinstance(segment["start"], (int, float)):
            raise TypeError(
                "'start' must be a number, " f"got {type(segment['start']).__name__}"
            )
        if not isinstance(segment["end"], (int, float)):
            raise TypeError(
                "'end' must be a number, " f"got {type(segment['end']).__name__}"
            )
        if segment["start"] < 0:
            raise ValueError(f"'start' must be >= 0, got {segment['start']}")
        if segment["end"] <= segment["start"]:
            raise ValueError(
                "'end' must be greater than 'start'. "
                f"Got start={segment['start']}, end={segment['end']}"
            )
