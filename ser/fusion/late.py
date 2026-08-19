"""Late fusion — 음향 1개 + 텍스트 1개의 저장 logit(test_logits.npz)을 결합해 평가. 재학습 없음.

각 npz: logits(N,C), labels(N), paths(N). paths 로 정렬·교집합 후 결합:
  softmax_avg(기본) | logit_mean | vote → ser.metrics.evaluate_logits → registry(modality=fusion).

실행:
    uv run python -m ser.fusion.late \\
        --acoustic-logits checkpoints/full_ft/<음향run>/test_logits.npz \\
        --text-logits     checkpoints/head_cached/<텍스트run>/test_logits.npz \\
        --method softmax_avg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ser.cli import add_common_overrides, resolve_dataset_ctx
from ser.config import load_config
from ser.metrics import evaluate_logits, print_final_report
from ser.registry import log_fusion


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def combine(acoustic_npz: str, text_npz: str, method: str = "softmax_avg"):
    """음향·텍스트 logit npz 를 paths 교집합으로 정렬·결합 → (logits[N,C] tensor, labels[N] tensor, paths)."""
    a, t = np.load(acoustic_npz, allow_pickle=True), np.load(text_npz, allow_pickle=True)
    common = sorted(set(a["paths"].tolist()) & set(t["paths"].tolist()))
    if not common:
        raise ValueError("음향·텍스트 간 공통 path 없음 — 같은 test split 의 logit 인지 확인.")
    ia = {p: i for i, p in enumerate(a["paths"].tolist())}
    it = {p: i for i, p in enumerate(t["paths"].tolist())}
    la = a["logits"][[ia[p] for p in common]]
    lt = t["logits"][[it[p] for p in common]]
    labels = np.array([a["labels"][ia[p]] for p in common])

    if method == "vote":
        comb = np.zeros((len(common), la.shape[1]))
        for l in (la, lt):
            comb[np.arange(len(common)), l.argmax(1)] += 1
    elif method == "logit_mean":
        comb = (la + lt) / 2
    else:  # softmax_avg
        comb = (_softmax(la) + _softmax(lt)) / 2
    return torch.from_numpy(comb).float(), torch.from_numpy(labels).long(), common


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--acoustic-logits", required=True, help="음향 모델 test_logits.npz")
    p.add_argument("--text-logits", required=True, help="텍스트 모델 test_logits.npz")
    p.add_argument("--method", choices=["softmax_avg", "logit_mean", "vote"], default="softmax_avg")
    p.add_argument("--out-dir", type=Path, default=None, help="결합 logit npz 저장 폴더 (report CM 용)")
    p.add_argument("--no-registry", dest="registry", action="store_false", default=True)
    args = p.parse_args()
    ctx = resolve_dataset_ctx(load_config(args))

    comb, labels, common = combine(args.acoustic_logits, args.text_logits, args.method)
    m = evaluate_logits(comb, labels)
    nc, names = ctx.num_classes, ctx.label_to_english
    print(f"[late-fusion] n={len(common)} method={args.method}")
    print_final_report(f"Late fusion ({args.method})", m, nc, names)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_dir / "test_logits.npz", logits=comb.numpy(), labels=labels.numpy(), paths=np.array(common))
    if args.registry:
        log_fusion("late", Path(args.acoustic_logits).parent.name, Path(args.text_logits).parent.name,
                   labels=labels.numpy(), preds=comb.argmax(1).numpy(), run_dir=args.out_dir, extra={"method": args.method})
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
