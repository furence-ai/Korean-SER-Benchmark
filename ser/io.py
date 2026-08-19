"""공유 I/O 유틸 — manifest/transcript 로드 + 체크포인트 메타 구성 (중복 제거 단일 소스).

이전엔 manifest 로드(여러 곳), transcript join, best.pt meta dict 구성(engine 2곳)이 흩어져 있었다.
여기 모아 한 곳에서만 바꾸면 되게 한다 (특히 build_ckpt_meta = best.pt 스키마의 단일 소스).
"""
from __future__ import annotations

import json
from pathlib import Path

_UNSET = object()


def load_split_jsonl(manifest_dir: str | Path, split: str) -> list[dict]:
    """{manifest_dir}/{split}.jsonl → dict 리스트."""
    path = Path(manifest_dir) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_transcript_map(transcript_dir: str | Path, split: str) -> dict[str, str]:
    """{transcript_dir}/{split}.jsonl ({audio,text}) → {audio: text} 매핑."""
    path = Path(transcript_dir) / f"{split}.jsonl"
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["audio"]] = r["text"]
    return out


def build_ckpt_meta(
    *,
    mode: str,
    head_type: str,
    embedding_dim: int,
    num_classes: int,
    keep_labels,
    hidden_dim: int,
    dropout: float,
    backbone,
    label_to_english: dict,
    modality=_UNSET,
    full_ft_max_audio_sec=_UNSET,
) -> dict:
    """best.pt/last.pt 메타 dict 단일 소스. build_model_from_ckpt 가 의존하는 스키마.

    modality/full_ft_max_audio_sec 는 전달된 경우에만 포함 (acoustic vs cached 모드 차이 보존).
    """
    meta = {
        "mode": mode, "head_type": head_type, "embedding_dim": embedding_dim,
        "num_classes": num_classes, "keep_labels": keep_labels,
        "hidden_dim": hidden_dim, "dropout": dropout, "backbone": backbone,
        "label_to_english": label_to_english,
    }
    if modality is not _UNSET:
        meta["modality"] = modality
    if full_ft_max_audio_sec is not _UNSET:
        meta["full_ft_max_audio_sec"] = full_ft_max_audio_sec
    return meta
