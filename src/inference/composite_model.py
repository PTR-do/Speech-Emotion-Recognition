"""
Composite speech model. Architecture:

    speech waveform
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

Input:
    mono 16 kHz waveform [T]
Output:
    dict containing the results of both branches.
"""

import torch
import torch.nn as nn
import whisperx
from transformers import WavLMModel

from src.models.SER_branch.SER_branch import SERbranch
from src.models.mlp_head.emotion_regressor import EmotionRegressor
from src.inference.SER_window_wrapper import SERWindowWrapper


class CompositeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        regressor = EmotionRegressor()
        regressor.load_state_dict(
            torch.load(
                "src/models/mlp_head/checkpoints/head_train.pth",
                map_location=device,
            )
        )
        wavlm = WavLMModel.from_pretrained("src/models/wavlm/pretrained")
        ser_branch = SERbranch(
            wavlm=wavlm,
            regressor=regressor,
        ).to(device)
        self.ser_branch = SERWindowWrapper(
            model=ser_branch,
            window_size=3.0,
            hop_size=0.5,
            sample_rate=self.sample_rate,
        )
        self.whisper_model = whisperx.load_model(
            "src/models/whisperx/pretrained/large-v3-turbo/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
            str(self.device),
            compute_type=("float16" if self.device.type == "cuda" else "float32"),
        )
        self.alignment_model = None
        self.alignment_metadata = None
        self.alignment_language = None

    # Run both branches on one speech segment.
    def forward(
        self,
        waveform: torch.Tensor,
    ):
        ser_result = self.ser_branch(waveform)
        whisper_result = self._transcribe(waveform)
        return {
            "ser": ser_result,
            "whisper": whisper_result,
        }

    # Run WhisperX transcription and forced alignment.
    def _transcribe(
        self,
        waveform: torch.Tensor,
    ) -> dict:
        waveform = waveform.detach().cpu().float()
        result = self.whisper_model.transcribe(
            waveform.numpy(),
            batch_size=1,
        )
        language = result["language"]
        if self.alignment_model is None or self.alignment_language != language:
            (
                self.alignment_model,
                self.alignment_metadata,
            ) = whisperx.load_align_model(
                language_code=language,
                device=str(self.device),
            )
            self.alignment_language = language
        aligned_result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.alignment_metadata,
            waveform.numpy(),
            str(self.device),
            return_char_alignments=False,
        )
        words = [
            {
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
            }
            for word in aligned_result.get("word_segments", [])
        ]
        text = " ".join(
            segment["text"].strip()
            for segment in result.get("segments", [])
            if segment.get("text")
        )
        return {
            "text": text,
            "words": words,
        }


"""
results are in the form:
{
    "ser": [
        {
            "start": 0.0,
            "end": 3.0,
            "emotions": [...]
        },
        {
            "start": 0.5,
            "end": 3.5,
            "emotions": [...]
        },
        ...
    ],
    "whisper": {
        "text": "...",
        "words": [
            {
                "word": "...",
                "start": 0.42,
                "end": 0.87
            },
            ...
        ]
    }
}
"""
