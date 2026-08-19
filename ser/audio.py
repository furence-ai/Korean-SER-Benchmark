"""오디오 I/O 공통 헬퍼 (백본 무관)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load_audio_16k_mono(path: str) -> np.ndarray:
    """wav 읽기 → mono → 16kHz float32 in [-1, 1] 변환."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav.astype(np.float32)
