"""캐싱된 임베딩 .npz 를 클래스 subset 으로 필터 + dense remap (범용).

클래스는 config 와 동일하게 keep_labels 로 정의 (ser.labels.resolve_label_subset 단일 소스).
--keep-labels 미지정 시 전체 클래스 그대로(항등). 4-class 등 어떤 subset 도 가능.
출력 .npz 포맷은 원본과 동일(feats/labels/speakers/paths) + label_names.json
→ run_head_cached / ser.train_cached 가 그대로 학습.

실행:
    uv run python -m scripts.filter_subset --src data/embeddings/wavlm-large --keep-labels 0 1 2 6
    uv run python -m scripts.filter_subset --src data/embeddings/x --keep-labels 0 2   # 2-class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ser.labels import resolve_label_subset


def filter_npz(in_path: Path, out_path: Path, keep_labels: list[int], remap: dict[int, int]) -> dict[int, int]:
    """단일 .npz 필터 + remap. dense 클래스별 카운트 반환. (frame npz 도 지원: sizes/offsets 보존)"""
    d = np.load(in_path, allow_pickle=False)
    labels = d["labels"]
    mask = np.isin(labels, keep_labels)
    remapped = np.array([remap[int(l)] for l in labels[mask]], dtype=np.int64)

    out = {"labels": remapped, "speakers": d["speakers"][mask], "paths": d["paths"][mask]}
    if "sizes" in d.files:   # frame-level npz: feats 는 (sum_T, D), utterance 단위로 슬라이스 보존
        sizes, offsets = d["sizes"], d["offsets"]
        keep_idx = np.where(mask)[0]
        feats_parts, new_sizes = [], []
        for i in keep_idx:
            s = int(offsets[i]); e = s + int(sizes[i])
            feats_parts.append(d["feats"][s:e]); new_sizes.append(int(sizes[i]))
        feats = np.concatenate(feats_parts, axis=0) if feats_parts else d["feats"][:0]
        new_sizes = np.array(new_sizes, dtype=np.int64)
        out.update(feats=feats.astype(np.float32), sizes=new_sizes,
                   offsets=np.concatenate([[0], np.cumsum(new_sizes)[:-1]]).astype(np.int64))
    else:                    # utterance-level: (N, D)
        out["feats"] = d["feats"][mask].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    return {int(v): int((remapped == v).sum()) for v in sorted(remap.values())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="원본 임베딩 dir ({split}.npz 들)")
    p.add_argument("--dest", type=Path, default=None, help="출력 dir. 미지정 시 <src>_<k>class")
    p.add_argument("--keep-labels", dest="keep_labels", nargs="+", type=int, default=None,
                   help="원본 라벨 ID들 (config keep_labels 와 동일 의미). 미지정=전체 클래스.")
    p.add_argument("--splits", nargs="+", default=["train_si", "val_si", "test_si"])
    args = p.parse_args()

    keep = sorted(set(args.keep_labels)) if args.keep_labels else None
    num_classes, remap, names_map = resolve_label_subset(keep)
    keep_list = keep if keep else sorted(remap)          # remap 키 = 필터 대상 라벨
    label_names = [names_map[i] for i in range(num_classes)]
    dest = args.dest or Path(f"{args.src}_{num_classes}class")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[config] src={args.src} dest={dest}\n  keep={keep_list} → remap={remap}\n  names={label_names}\n")

    for split in args.splits:
        in_p, out_p = args.src / f"{split}.npz", dest / f"{split}.npz"
        if not in_p.exists():
            print(f"[skip] {in_p} 없음")
            continue
        counts = filter_npz(in_p, out_p, keep_list, remap)
        per = "  ".join(f"{label_names[i]}={counts[i]}" for i in range(num_classes))
        print(f"[done] {split}: {sum(counts.values())} samples ({per}) → {out_p}")

    (dest / "label_names.json").write_text(json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {dest / 'label_names.json'}")
    print(f"\n학습: uv run python -m ser.train_cached --emb-dir {dest} --modality acoustic")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
