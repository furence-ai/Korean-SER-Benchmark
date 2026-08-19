"""Early fusion — 음향 임베딩 + 텍스트 임베딩을 path 정렬·concat → 결합 npz.
이후 ser.train_cached 로 head 학습 ((N, D_acoustic + D_text) 입력).

전제 (각각 {split}.npz: feats/labels/speakers/paths):
  음향: scripts.extract_acoustic_embeddings --backbone <X>
  텍스트: ser.text.extract_embeddings

실행:
    uv run python -m ser.fusion.early \\
        --acoustic-emb-dir data/embeddings/wavlm-large \\
        --text-emb-dir     data/embeddings/roberta-large \\
        --out-dir data/embeddings/early_wavlm_text
    uv run python -m ser.train_cached --emb-dir data/embeddings/early_wavlm_text --modality fusion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ser.cli import add_common_overrides, config_splits
from ser.config import load_config


def combine_split(acoustic_dir: Path, text_dir: Path, split: str) -> dict:
    """음향·텍스트 npz 를 path 교집합으로 정렬·concat. {feats, labels, speakers, paths} 반환."""
    a = np.load(acoustic_dir / f"{split}.npz", allow_pickle=True)
    t = np.load(text_dir / f"{split}.npz", allow_pickle=True)
    common = sorted(set(a["paths"].tolist()) & set(t["paths"].tolist()))
    if not common:
        raise ValueError(f"[{split}] 음향·텍스트 간 공통 path 없음 — 같은 split 임베딩인지 확인.")
    ia = {p: i for i, p in enumerate(a["paths"].tolist())}
    it = {p: i for i, p in enumerate(t["paths"].tolist())}
    fa = a["feats"][[ia[p] for p in common]]
    ft = t["feats"][[it[p] for p in common]]
    labels = np.array([a["labels"][ia[p]] for p in common])
    speakers = np.array([a["speakers"][ia[p]] for p in common])
    return {"feats": np.concatenate([fa, ft], axis=1).astype(np.float32),
            "labels": labels, "speakers": speakers, "paths": np.array(common)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--acoustic-emb-dir", type=Path, required=True)
    p.add_argument("--text-emb-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    splits = config_splits(load_config(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        d = combine_split(args.acoustic_emb_dir, args.text_emb_dir, split)
        np.savez(args.out_dir / f"{split}.npz", **d)
        print(f"[{split}] concat feats={d['feats'].shape} → {args.out_dir / f'{split}.npz'}", flush=True)
    print(f"\n학습: uv run python -m ser.train_cached --emb-dir {args.out_dir} --modality fusion")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
