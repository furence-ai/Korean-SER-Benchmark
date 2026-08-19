"""진단 — 저장된 logit에서 top-k / calibration / 오분류쌍 분석 (백본·모달리티 무관).

emotion2vec_probs.py 의 "top-2에 정답 95%" 분석을 일반화. test_logits.npz 만 있으면
어느 백본/모달리티든 돌아간다 → 어느 모델이 top-2 안에 정답을 잘 담는지(= LLM/top-2
arbitration 의 정당성)를 한눈에.
"""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def analyze_logits(logits: np.ndarray, labels: np.ndarray,
                   class_names: list[str] | None = None) -> dict:
    """(N,C) logits + (N,) labels → top-k acc, gt_rank 분포, margin, calibration, 오분류쌍."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n, c = logits.shape
    probs = _softmax(logits)
    order = np.argsort(-probs, axis=1)            # 내림차순 클래스 인덱스 (N, C)
    preds = order[:, 0]
    names = class_names or [str(i) for i in range(c)]

    # gt가 각 행에서 몇 등인지 (0-based → +1 = rank)
    gt_rank = np.array([int(np.where(order[i] == labels[i])[0][0]) + 1 for i in range(n)])
    topk = {f"top{k}": float((gt_rank <= k).mean()) for k in range(1, c + 1)}

    top1p = probs[np.arange(n), order[:, 0]]
    top2p = probs[np.arange(n), order[:, 1]] if c > 1 else np.zeros(n)
    margin = top1p - top2p
    correct = preds == labels

    # 오분류쌍 (gt→pred) 카운트
    pairs: dict[str, int] = {}
    for gt, pr in zip(labels[~correct], preds[~correct]):
        pairs[f"{names[gt]}→{names[pr]}"] = pairs.get(f"{names[gt]}→{names[pr]}", 0) + 1
    pairs = dict(sorted(pairs.items(), key=lambda kv: -kv[1]))

    return {
        "n": int(n), "num_classes": int(c), "acc": float(correct.mean()),
        "topk_acc": topk,                                   # ★ top-2 등
        "gt_rank_dist": {int(k): int((gt_rank == k).sum()) for k in range(1, c + 1)},
        "calibration": {
            "mean_top1_correct": float(top1p[correct].mean()) if correct.any() else None,
            "mean_top1_wrong": float(top1p[~correct].mean()) if (~correct).any() else None,
            "mean_margin_correct": float(margin[correct].mean()) if correct.any() else None,
            "mean_margin_wrong": float(margin[~correct].mean()) if (~correct).any() else None,
        },
        "wrong_pairs": pairs,
    }


def analyze_npz(npz_path: str, class_names: list[str] | None = None) -> dict:
    """test_logits.npz (logits/labels) → analyze_logits."""
    d = np.load(npz_path, allow_pickle=True)
    return analyze_logits(d["logits"], d["labels"], class_names)


def format_report(diag: dict) -> str:
    tk = diag["topk_acc"]
    lines = [
        f"n={diag['n']}  acc(top1)={diag['acc']:.4f}",
        "top-k 정확도: " + "  ".join(f"{k}={v:.4f}" for k, v in tk.items()),
        f"확신도(calibration): {diag['calibration']}",
        "주요 오분류쌍: " + ", ".join(f"{k}={v}" for k, v in list(diag["wrong_pairs"].items())[:8]),
    ]
    return "\n".join(lines)
