"""모든 entrypoint 공통 CLI/config 해석 — config(YAML) 단일 소스.

원칙: 공유 파라미터(dataset/keep_labels/manifest_dir/splits/device/seed)는 YAML 에서만 정의,
argparse 는 override(default=None)만. 이전엔 11개 스크립트가 config 를 무시하고 자체 기본값을
선언해(keep_labels 6곳, splits _si 하드코딩 등) config 변경이 전파되지 않았다 — 그 보일러플레이트를
여기로 모은다.

사용:
    p = argparse.ArgumentParser()
    add_common_overrides(p)
    # ... 스크립트 고유 인자 추가 ...
    cfg = load_config(p.parse_args())   # ser.config.load_config
    finalize_device(cfg)
    ctx = resolve_dataset_ctx(cfg)      # adapter + 라벨 해석 한 방에
    splits = config_splits(cfg)         # [train, val, test]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from ser.datasets import DatasetAdapter, resolve_dataset

from .config import add_config_arg
from .labels import resolve_label_subset


def add_common_overrides(p: argparse.ArgumentParser) -> None:
    """전 entrypoint 공통 override 인자 (전부 default=None → config 따름)."""
    add_config_arg(p)
    p.add_argument("--dataset", default=None, help="데이터셋 어댑터 id (config common.dataset)")
    p.add_argument("--keep-labels", dest="keep_labels", type=int, nargs="+", default=None,
                   help="사용할 원본 라벨 ID (config common.keep_labels). 미지정=config")
    p.add_argument("--manifest-dir", dest="manifest_dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=None,
                   help="평가/처리할 split 이름들. 미지정=config 의 split_{train,val,test}")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)


def finalize_device(cfg: dict) -> dict:
    """cuda 미가용 시 cpu 폴백 (이전 10곳 중복 로직 통합)."""
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def config_splits(cfg: dict) -> list[str]:
    """config 의 split_{train,val,test} → [train, val, test]. (CLI --splits 가 있으면 그게 우선)"""
    if cfg.get("splits"):
        return list(cfg["splits"])
    return [cfg.get("split_train", "train_di"), cfg.get("split_val", "val_di"),
            cfg.get("split_test", "test_di")]


@dataclass
class DatasetContext:
    """데이터셋 어댑터 + 라벨 해석 결과 묶음."""
    adapter: DatasetAdapter
    num_classes: int
    label_map: dict[int, int]      # 원본 라벨 ID → dense
    label_to_english: dict[int, str]


def resolve_dataset_ctx(cfg: dict) -> DatasetContext:
    """cfg → 데이터셋 어댑터 인스턴스 + keep_labels 해석 (모든 entrypoint 공용)."""
    ds = resolve_dataset(cfg.get("dataset") or "emotion_style_speech")()
    nc, remap, names = resolve_label_subset(cfg.get("keep_labels"), ds.label_space)
    return DatasetContext(ds, nc, remap, names)
