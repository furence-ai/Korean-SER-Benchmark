"""평가 지표 + 리포트 (백본/모달리티 무관). logit-in → metric-out."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_logits(all_logits: torch.Tensor, all_labels: torch.Tensor) -> dict:
    """(N,C) logits + (N,) labels → {loss, acc, macro_f1, weighted_f1, preds, labels}."""
    preds = all_logits.argmax(dim=-1)
    acc = (preds == all_labels).float().mean().item()
    macro_f1 = f1_score(all_labels.numpy(), preds.numpy(), average="macro")
    weighted_f1 = f1_score(all_labels.numpy(), preds.numpy(), average="weighted")
    loss = F.cross_entropy(all_logits, all_labels).item()
    return {
        "loss": loss, "acc": acc,
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "preds": preds.numpy(), "labels": all_labels.numpy(),
    }


def format_final_report(label: str, metrics: dict, num_classes: int,
                        label_to_english: dict[int, str]) -> str:
    target_names = [label_to_english[i] for i in range(num_classes)]
    return "\n".join([
        f"\n=== {label} ===",
        (f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
         f"macroF1={metrics['macro_f1']:.4f} weightedF1={metrics['weighted_f1']:.4f}"),
        classification_report(metrics["labels"], metrics["preds"], labels=list(range(num_classes)),
                              target_names=target_names, digits=4, zero_division=0),
        "confusion matrix (행=정답, 열=예측):",
        str(confusion_matrix(metrics["labels"], metrics["preds"], labels=list(range(num_classes)))),
    ])


def print_final_report(label: str, metrics: dict, num_classes: int,
                       label_to_english: dict[int, str], out_path: Path | None = None) -> None:
    text = format_final_report(label, metrics, num_classes, label_to_english)
    print(text)
    if out_path is not None:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def plot_history(history: list[dict], out_path: Path, title: str = "") -> None:
    """history → 학습 곡선 PNG (loss / acc / val F1 3-panel)."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train", linewidth=1.5)
    axes[0].plot(ep, [h["val_loss"] for h in history], label="internal_val", linewidth=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train", linewidth=1.5)
    axes[1].plot(ep, [h["val_acc"] for h in history], label="internal_val", linewidth=1.5)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[2].plot(ep, [h["val_macro_f1"] for h in history], label="macro F1", linewidth=1.5)
    axes[2].plot(ep, [h["val_weighted_f1"] for h in history], label="weighted F1", linewidth=1.5)
    best_idx = int(np.argmax([h["val_macro_f1"] for h in history]))
    axes[2].axvline(ep[best_idx], color="red", linestyle="--", alpha=0.5, label=f"best (ep {ep[best_idx]})")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("internal_val F1"); axes[2].set_title("Validation F1")
    axes[2].grid(True, alpha=0.3); axes[2].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
