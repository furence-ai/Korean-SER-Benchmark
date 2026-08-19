"""사전학습 emotion2vec(native 9-class zero-shot → 4) vs 학습된 모델 비교.

⚠️ emotion2vec 전용 — WavLM/wav2vec2 엔 native 감정 분류 head 가 없어 "사전학습 baseline" 개념이
   emotion2vec 에만 있다. 또 9→4 인덱스 매핑이 happy/sad/angry/neutral(keep_labels=[0,1,2,6])
   고정이라 그 4-class 에서만 동작 (다른 subset 이면 게이트).

학습 모델은 ser.engine.build_model_from_ckpt 로 로드(BackboneWithHead, bf16) — 다른 도구와 동일.
출력: <ckpt_dir>/comparison.txt + comparison.png.

실행:
    uv run python -m scripts.compare_pretrained --ckpt checkpoints/full_ft/<run>/best.pt
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from ser.audio import load_audio_16k_mono
from ser.engine import build_model_from_ckpt
from ser.heads import amp_autocast
from ser.labels import resolve_label_subset

# emotion2vec native 9-class: [0 angry,1 disgust,2 fear,3 happy,4 neutral,5 other,6 sad,7 surprise,8 unk]
# 우리 [0,1,2,6]=happy/sad/angry/neutral → native index [3,6,0,4]
STD_4CLASS = [0, 1, 2, 6]
NATIVE_IDX = [3, 6, 0, 4]


def load_items(manifest: Path, keep: list[int], remap: dict[int, int], max_per_class: int, seed: int):
    items = []
    for l in manifest.open(encoding="utf-8"):
        if not l.strip():
            continue
        import json
        d = __import__("json").loads(l)
        if int(d["label"]) in keep:
            items.append({"audio": d["audio"], "gt": remap[int(d["label"])]})
    if max_per_class and max_per_class > 0:
        by: dict[int, list] = {}
        for it in items:
            by.setdefault(it["gt"], []).append(it)
        rng = random.Random(seed)
        items = []
        for lst in by.values():
            rng.shuffle(lst)
            items.extend(lst[:max_per_class])
        rng.shuffle(items)
    return items


@torch.no_grad()
def predict_both(items, funasr_model, model, device, max_samples, amp_dtype, num_classes):
    gts, pre, our = [], [], []
    for it in tqdm(items, desc="predict", unit="utt"):
        try:
            wav = load_audio_16k_mono(it["audio"])
            res = funasr_model.generate(wav, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {it['audio']}: {e}")
            continue
        sub = np.asarray(res["scores"], dtype=np.float32)[NATIVE_IDX]   # (4,) happy/sad/angry/neutral
        pre.append(int(sub.argmax()))
        w = wav[:max_samples] if (max_samples and len(wav) > max_samples) else wav
        wt = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(device)
        with amp_autocast(device, amp_dtype):
            logits = model(wt)[0]
        our.append(int(logits.argmax().item()))
        gts.append(it["gt"])
    return np.array(gts), np.array(pre), np.array(our)


def _metrics(gt, pred):
    return {"acc": float((gt == pred).mean()),
            "macro_f1": float(f1_score(gt, pred, average="macro", zero_division=0))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/test_si.jsonl"))
    p.add_argument("--keep-labels", dest="keep_labels", nargs="+", type=int, default=STD_4CLASS)
    p.add_argument("--max-per-class", type=int, default=0, help="0=전체")
    p.add_argument("--backbone", default="emotion2vec/emotion2vec_plus_large")
    p.add_argument("--amp-dtype", default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    keep = sorted(set(args.keep_labels))
    if keep != STD_4CLASS:
        raise SystemExit(f"compare_pretrained 는 keep_labels={STD_4CLASS}(happy/sad/angry/neutral) 전용 "
                         f"— native 9→4 매핑이 그 클래스에 묶임. (받은 값: {keep})")
    if "emotion2vec" not in args.backbone:
        raise SystemExit("emotion2vec 전용 (WavLM/wav2vec2 엔 native 감정 head 없음).")
    num_classes, remap, names_map = resolve_label_subset(keep)
    names = [names_map[i] for i in range(num_classes)]

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = build_model_from_ckpt(ck, args.device)
    max_samples = int((ck.get("full_ft_max_audio_sec") or 10.0) * 16000)
    from funasr import AutoModel
    funasr_model = AutoModel(model=args.backbone, hub="hf", device=args.device, disable_update=True)

    items = load_items(args.manifest, keep, remap, args.max_per_class, args.seed)
    print(f"[test] {len(items)}개 from {args.manifest}", flush=True)
    t0 = time.time()
    gt, pre, our = predict_both(items, funasr_model, model, args.device, max_samples, args.amp_dtype, num_classes)
    pm, om = _metrics(gt, pre), _metrics(gt, our)

    blocks = [
        f"[test] n={len(gt)}  per-class={ {names[k]: int((gt==k).sum()) for k in range(num_classes)} }",
        f"[ckpt] {args.ckpt}",
        "\n=== Pre-trained (9→4 zero-shot) ===",
        f"acc={pm['acc']:.4f} macroF1={pm['macro_f1']:.4f}",
        classification_report(gt, pre, target_names=names, digits=4, zero_division=0),
        "\n=== Trained ===",
        f"acc={om['acc']:.4f} macroF1={om['macro_f1']:.4f}",
        classification_report(gt, our, target_names=names, digits=4, zero_division=0),
        f"\nΔacc={(om['acc']-pm['acc'])*100:+.2f}%p  ΔmacroF1={(om['macro_f1']-pm['macro_f1'])*100:+.2f}%p",
    ]
    text = "\n".join(blocks)
    print(text, flush=True)
    out_txt = args.ckpt.parent / "comparison.txt"
    out_txt.write_text(text, encoding="utf-8")
    _save_plot(gt, pre, our, names, args.ckpt.parent / "comparison.png")
    print(f"[time] {time.time()-t0:.1f}s\n[saved] {out_txt}", flush=True)
    return 0


def _save_plot(gt, pre, our, names, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, title in [(axes[0], pre, "Pre-trained 9→4"), (axes[1], our, "Trained")]:
        cm = confusion_matrix(gt, pred, labels=list(range(K)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(names, rotation=30); ax.set_yticklabels(names)
        ax.set_xlabel("pred"); ax.set_ylabel("GT")
        ax.set_title(f"{title}\nmF1={f1_score(gt, pred, average='macro', zero_division=0):.3f}")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.exit(main())
