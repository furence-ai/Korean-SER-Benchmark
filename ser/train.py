"""학습 진입점 (백본 무관). config + 백본만 갈아끼우면 어느 백본이든 학습.

실행 예시:
    uv run python -m ser.train                                    # full FT, default backbone(emotion2vec)
    uv run python -m ser.train --backbone microsoft/wavlm-large   # WavLM full FT (어댑터 필요)
    uv run python -m ser.train --full-ft-freeze-cnn               # CNN 동결 full FT
    uv run python -m ser.train --embed-on-the-fly                 # backbone 동결 + head만 (head_only_otf)
    uv run python -m ser.train --resume checkpoints/full_ft/<run>/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import ser.backbones  # noqa: F401  (어댑터 self-register)
from ser.config import add_config_arg, load_config, print_config
from ser.engine import run_acoustic


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--embed-on-the-fly", dest="freeze_backbone", action="store_true", default=False,
                   help="backbone 동결 + head만 학습 (head_only_otf). 없으면 full FT.")
    p.add_argument("--full-finetuning", action="store_true",
                   help="(명시용) full FT — 기본 동작. --embed-on-the-fly 와 배타적.")
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="체크포인트 저장 경로 직접 지정 ({mode}/{run_name} 자동추가 안 함).")
    p.add_argument("--resume", type=Path, default=None, help="last.pt 또는 run_dir 로 학습 재개.")
    # override (default=None → 값 줄 때만 YAML override)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--head-type", dest="head_type", type=str, default=None)
    p.add_argument("--keep-labels", dest="keep_labels", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=None)
    p.add_argument("--manifest-dir", dest="manifest_dir", type=Path, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--no-eval-on-test", dest="eval_on_test", action="store_false", default=None)
    # full FT 하이퍼파라미터
    p.add_argument("--full-ft-batch-size", dest="full_ft_batch_size", type=int, default=None)
    p.add_argument("--full-ft-lr", dest="full_ft_lr", type=float, default=None)
    p.add_argument("--head-lr", dest="head_lr", type=float, default=None,
                   help="head-only(--embed-on-the-fly) 시 head LR. full FT 엔 영향 없음.")
    p.add_argument("--full-ft-backbone-lr", dest="full_ft_backbone_lr", type=float, default=None)
    p.add_argument("--full-ft-warmup-steps", dest="full_ft_warmup_steps", type=int, default=None)
    p.add_argument("--full-ft-grad-accum", dest="full_ft_grad_accum", type=int, default=None)
    p.add_argument("--full-ft-max-audio-sec", dest="full_ft_max_audio_sec", type=float, default=None)
    p.add_argument("--full-ft-freeze-cnn", dest="full_ft_freeze_cnn", action="store_true", default=None,
                   help="CNN feature extractor만 동결, 트랜스포머는 학습.")
    p.add_argument("--full-ft-freeze-first-n-layers", dest="full_ft_freeze_first_n_layers",
                   type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args)
    if not torch.cuda.is_available() and cfg.get("device") == "cuda":
        cfg["device"] = "cpu"
    print_config(cfg)
    run_acoustic(cfg, args.run_name, args.output,
                 freeze_backbone=args.freeze_backbone, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
