"""emotion2vec 원본(native zero-shot) logits → test_logits.npz (late fusion 입력용).

late item①(em 원본 + RoBERTa) 용. funasr generate → native scores → 데이터셋 라벨공간에 맞춰
인덱싱 → logits. native↔데이터셋 라벨 매핑은 백본 어댑터의 NATIVE_LABELS 와 데이터셋의
label_to_english 를 영문명으로 매칭해 **동적 계산** (이전엔 [0,1,2,6] 4-class 하드코딩).

emotion2vec 처럼 NATIVE_LABELS 가 있는 백본 전용. keep_labels 의 감정명이 native 에 모두
존재해야 함(없으면 어느 라벨이 매핑 불가인지 에러).

실행:
    uv run python -m scripts.dump_pretrained_logits --out-dir <dir>            # config split_test
    uv run python -m scripts.dump_pretrained_logits --split test_di --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ser.backbones import resolve_adapter
from ser.audio import load_audio_16k_mono
from ser.cli import add_common_overrides, finalize_device, resolve_dataset_ctx
from ser.config import load_config
from ser.io import load_split_jsonl
from ser.registry import REGISTRY


def native_index_map(adapter_cls, label_to_english: dict[int, str]) -> list[int]:
    """dense 라벨 순서대로 native scores 인덱스 리스트. NATIVE_LABELS 와 영문명 매칭."""
    native = adapter_cls.NATIVE_LABELS
    if not native:
        raise SystemExit(f"{adapter_cls.__name__} 에 NATIVE_LABELS 없음 — native zero-shot 미지원 백본.")
    idx = []
    for dense in range(len(label_to_english)):
        eng = label_to_english[dense]
        if eng not in native:
            raise SystemExit(f"라벨 '{eng}' 가 native 라벨 {native} 에 없음 — 이 백본으로 매핑 불가.")
        idx.append(native.index(eng))
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--backbone", default="emotion2vec/emotion2vec_plus_large")
    p.add_argument("--split", default=None, help="평가 split (미지정=config split_test)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    cfg = finalize_device(load_config(args))
    ctx = resolve_dataset_ctx(cfg)
    split = args.split or cfg.get("split_test", "test_di")
    backbone = args.backbone
    if "emotion2vec" not in backbone:
        raise SystemExit("native zero-shot 은 현재 emotion2vec 전용.")

    adapter_cls = resolve_adapter(backbone)
    native_idx = native_index_map(adapter_cls, ctx.label_to_english)

    rows = [r for r in load_split_jsonl(cfg["manifest_dir"], split) if int(r["label"]) in ctx.label_map]
    from funasr import AutoModel
    m = AutoModel(model=backbone, hub="hf", device=cfg["device"], disable_update=True)

    logits, labels, paths = [], [], []
    for r in tqdm(rows, desc="em-원본", unit="utt"):
        try:
            wav = load_audio_16k_mono(r["audio"])
            res = m.generate(wav, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {r['audio']}: {e}"); continue
        logits.append(np.asarray(res["scores"], dtype=np.float32)[native_idx])
        labels.append(ctx.label_map[int(r["label"])]); paths.append(r["audio"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logits, labels = np.stack(logits), np.array(labels)
    np.savez(args.out_dir / "test_logits.npz", logits=logits, labels=labels, paths=np.array(paths))
    from sklearn.metrics import f1_score
    acc = float((logits.argmax(1) == labels).mean())
    mf1 = float(f1_score(labels, logits.argmax(1), average="macro", zero_division=0))
    per_class = f1_score(labels, logits.argmax(1), average=None, zero_division=0).tolist()
    (args.out_dir / "test_metrics.json").write_text(
        json.dumps({"acc": acc, "macro_f1": mf1, "per_class_f1": per_class, "n": len(labels)}, indent=2),
        encoding="utf-8")
    row = {"run_id": f"em-pretrained-{split}", "run_dir": str(args.out_dir), "modality": "acoustic",
           "mode": "pretrained", "backbone": backbone, "test_macro_f1": mf1, "test_acc": acc,
           "per_class_f1": per_class, "split_test": split, "ts": datetime.now().isoformat(timespec="seconds")}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[em-원본 zero-shot] n={len(labels)} acc={acc:.4f} macroF1={mf1:.4f}  → registry +1 row")
    print(f"[saved] {args.out_dir / 'test_logits.npz'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
