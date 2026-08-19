"""캐싱 임베딩 head 학습 진입점 (텍스트 baseline / early-fusion / acoustic-cached 공용).

emb_dir 하위 {split}.npz 에 head 만 학습 (backbone 없음). modality 로 registry 구분.

실행:
    uv run python -m ser.train_cached --emb-dir data/embeddings/roberta-large --modality text --backbone klue/roberta-large
    uv run python -m ser.train_cached --emb-dir data/embeddings/early_emo_text --modality fusion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ser.config import add_config_arg, load_config, print_config
from ser.engine import run_head_cached


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--emb-dir", dest="emb_dir", type=str, required=True,
                   help="임베딩 npz 디렉토리 ({split}.npz 들).")
    p.add_argument("--modality", default="acoustic", help="registry 기록용 (acoustic|text|fusion).")
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--backbone", type=str, default=None, help="meta 기록용 인코더 id.")
    p.add_argument("--head-type", dest="head_type", type=str, default=None)
    p.add_argument("--keep-labels", dest="keep_labels", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-eval-on-test", dest="eval_on_test", action="store_false", default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args)
    cfg["emb_dir"] = args.emb_dir
    if not torch.cuda.is_available() and cfg.get("device") == "cuda":
        cfg["device"] = "cpu"
    print_config(cfg)
    run_head_cached(cfg, args.run_name, args.output, modality=args.modality, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
