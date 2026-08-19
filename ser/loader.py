"""데이터셋 (백본 무관): head-only(캐싱 임베딩) + full-FT(raw audio) + collate + 유틸."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import SAMPLE_RATE, load_audio_16k_mono


# ---------- Head-only: 캐싱된 임베딩 ----------

class EmbeddingDataset(Dataset):
    """Utterance-level pre-pooled .npz 로더. (N, D) feats."""

    def __init__(self, npz_path: str | Path):
        data = np.load(Path(npz_path), allow_pickle=False)
        self.feats = torch.from_numpy(data["feats"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        self.speakers = data["speakers"]
        self.paths = data["paths"]
        assert len(self.feats) == len(self.labels) == len(self.speakers)

    @property
    def embedding_dim(self) -> int:
        return int(self.feats.shape[1])

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feats[idx], self.labels[idx]


class FrameEmbeddingDataset(Dataset):
    """Frame-level .npz 로더. feats (sum_T, D) + sizes/offsets로 utterance마다 (T_i, D) 슬라이스."""

    def __init__(self, npz_path: str | Path):
        data = np.load(Path(npz_path), allow_pickle=False)
        self.feats = data["feats"]                            # (sum_T, D) numpy
        self.sizes = data["sizes"].astype(np.int64)           # (N,)
        self.offsets = data["offsets"].astype(np.int64)       # (N,)
        self.labels = torch.from_numpy(data["labels"]).long() # (N,)
        self.speakers = data["speakers"]
        self.paths = data["paths"]
        assert len(self.sizes) == len(self.offsets) == len(self.labels)

    @property
    def embedding_dim(self) -> int:
        return int(self.feats.shape[1])

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.offsets[idx])
        e = s + int(self.sizes[idx])
        frames = torch.from_numpy(self.feats[s:e].copy()).float()   # (T_i, D)
        return frames, self.labels[idx]


def collate_pad_frames(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 길이 frame 시퀀스 → batch 최장 길이 padding. (feats, padding_mask, labels)."""
    frames_list, labels = zip(*batch)
    sizes = [f.size(0) for f in frames_list]
    max_t = max(sizes)
    bsz = len(frames_list)
    d = frames_list[0].size(-1)

    padded = torch.zeros(bsz, max_t, d, dtype=torch.float32)
    padding_mask = torch.ones(bsz, max_t, dtype=torch.bool)
    for i, (f, s) in enumerate(zip(frames_list, sizes)):
        padded[i, :s] = f
        padding_mask[i, :s] = False
    return padded, padding_mask, torch.stack(list(labels))


def is_frame_level_npz(npz_path: str | Path) -> bool:
    """npz에 'sizes' 키 있으면 frame-level."""
    data = np.load(Path(npz_path), allow_pickle=False)
    return "sizes" in data.files


# ---------- Full FT: manifest에서 raw audio 로드 ----------

class RawAudioDataset(Dataset):
    """manifest jsonl에서 wav 로드. max_sec로 truncate해서 GPU 메모리 제한.

    label_map: 원본 라벨 ID → dense 라벨 remap dict (예: {0:0,1:1,2:2,6:3}).
        None이면 매니페스트 라벨 그대로. map에 없는 라벨 utterance는 드롭.
    transform: crop 후 적용할 파형 증강 콜백 (np.ndarray -> np.ndarray). 기본 None = 증강 OFF.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        max_sec: float = 10.0,
        random_crop: bool = False,
        label_map: dict[int, int] | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        with open(manifest_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        if label_map is not None:
            items = [it for it in items if int(it["label"]) in label_map]
        self.items = items
        self.label_map = label_map
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.random_crop = random_crop
        self.transform = transform

    def _label(self, it: dict) -> int:
        lbl = int(it["label"])
        return self.label_map[lbl] if self.label_map is not None else lbl

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        it = self.items[idx]
        wav = load_audio_16k_mono(it["audio"])
        if len(wav) > self.max_samples:
            if self.random_crop:
                start = np.random.randint(0, len(wav) - self.max_samples + 1)
            else:
                start = 0
            wav = wav[start : start + self.max_samples]
        if self.transform is not None:   # 증강 (기본 OFF)
            wav = self.transform(wav)
        return torch.from_numpy(np.ascontiguousarray(wav)), self._label(it)

    @property
    def labels_tensor(self) -> torch.Tensor:
        return torch.tensor([self._label(it) for it in self.items], dtype=torch.long)

    @property
    def speakers(self) -> np.ndarray:
        """샘플별 화자 ID (items 순서). person_id_from_speaker 로 사람 단위 그룹화에 사용."""
        return np.array([it.get("speaker", "") for it in self.items])


def collate_pad_audio(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 길이 wav → batch 최장 길이 padding. (wavs, padding_mask, labels)."""
    wavs, labels = zip(*batch)
    max_len = max(w.size(0) for w in wavs)
    bsz = len(wavs)

    padded = torch.zeros(bsz, max_len, dtype=torch.float32)
    padding_mask = torch.ones(bsz, max_len, dtype=torch.bool)
    for i, w in enumerate(wavs):
        n = w.size(0)
        padded[i, :n] = w
        padding_mask[i, :n] = False
    return padded, padding_mask, torch.tensor(labels, dtype=torch.long)


# ---------- 공통 유틸 ----------

def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency 클래스 weight, 평균 1로 정규화."""
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    return counts.sum() / (num_classes * counts)


_PERSON_ADAPTER = None


def person_id_from_speaker(speaker: str) -> str:
    """speaker-ID → '사람' 식별자. 데이터셋별 규칙은 데이터셋 어댑터(person_id)로 위임.

    기본 데이터셋의 어댑터를 캐시해 사용. 다른 데이터셋의 split 생성은
    그 어댑터의 person_id 를 직접 호출한다 (scripts.make_*_split).
    """
    global _PERSON_ADAPTER
    if _PERSON_ADAPTER is None:
        from ser.datasets.emotion_style_speech import EmotionStyleSpeech
        _PERSON_ADAPTER = EmotionStyleSpeech()
    return _PERSON_ADAPTER.person_id(speaker)
