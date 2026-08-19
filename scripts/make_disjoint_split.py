"""화자 + 대본(전사) **동시 분리** 3-way split — 텍스트 누수까지 차단.

각 split = (그 split 전용 화자) ∩ (그 split 전용 대본). 세 split 이 화자·대본 양쪽에서 서로 disjoint.
교차 셀(train화자가 test대본 읽은 것 등)은 **버린다** → 음향(새 화자)·텍스트(새 대본) 둘 다 누수 0.
대본은 감정별 stratify 분할 → 라벨 균형 유지. (데이터가 많아 교차 셀 폐기 감당 가능)

입력: data/manifests/{train_si,val_si,test_si}.jsonl (audio/label/speaker)  ← 전 데이터 풀
      data/transcripts/{...}.jsonl (audio→text)  ※ 없으면 ser.text.extract_transcripts 먼저
출력: data/manifests/{train_di,val_di,test_di}.jsonl

실행:
    uv run python -m scripts.make_disjoint_split --val-n-persons 4 --test-n-persons 4 \\
        --val-script-ratio 0.1 --test-script-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ser.loader import person_id_from_speaker

SRC_SPLITS = ("train_si", "val_si", "test_si")


def load_pool(manifest_dir: Path, transcript_dir: Path) -> list[dict]:
    """전 데이터: audio/label/speaker/person/text 레코드."""
    text: dict[str, str] = {}
    for sp in SRC_SPLITS:
        for l in (transcript_dir / f"{sp}.jsonl").open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l); text[d["audio"]] = d["text"]
    rows: list[dict] = []
    for sp in SRC_SPLITS:
        for l in (manifest_dir / f"{sp}.jsonl").open(encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d["audio"] not in text:
                continue
            rows.append({"audio": d["audio"], "label": int(d["label"]), "speaker": d["speaker"],
                         "person": person_id_from_speaker(d["speaker"]), "text": text[d["audio"]]})
    return rows


def partition_persons(rows, val_n, test_n, seed):
    persons = sorted({r["person"] for r in rows})
    random.Random(seed).shuffle(persons)
    test_p = set(persons[:test_n])
    val_p = set(persons[test_n:test_n + val_n])
    train_p = set(persons[test_n + val_n:])
    return train_p, val_p, test_p


def partition_scripts(rows, val_ratio, test_ratio, seed):
    """대본(text) 을 감정별 stratify 로 train/val/test 에 배분 (한 대본=한 감정)."""
    script_label = {r["text"]: r["label"] for r in rows}
    by_label: dict[int, list[str]] = defaultdict(list)
    for s, lab in script_label.items():
        by_label[lab].append(s)
    train_s, val_s, test_s = set(), set(), set()
    rng = random.Random(seed + 1)
    for lab, scripts in by_label.items():
        scripts = sorted(scripts); rng.shuffle(scripts)
        n = len(scripts); n_test = int(n * test_ratio); n_val = int(n * val_ratio)
        test_s.update(scripts[:n_test])
        val_s.update(scripts[n_test:n_test + n_val])
        train_s.update(scripts[n_test + n_val:])
    return train_s, val_s, test_s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    p.add_argument("--val-n-persons", type=int, default=4)
    p.add_argument("--test-n-persons", type=int, default=4)
    p.add_argument("--val-script-ratio", type=float, default=0.1)
    p.add_argument("--test-script-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = load_pool(args.manifest_dir, args.transcript_dir)
    train_p, val_p, test_p = partition_persons(rows, args.val_n_persons, args.test_n_persons, args.seed)
    train_s, val_s, test_s = partition_scripts(rows, args.val_script_ratio, args.test_script_ratio, args.seed)

    buckets: dict[str, list[dict]] = {"train_di": [], "val_di": [], "test_di": []}
    discarded = 0
    for r in rows:
        if r["person"] in test_p and r["text"] in test_s:
            buckets["test_di"].append(r)
        elif r["person"] in val_p and r["text"] in val_s:
            buckets["val_di"].append(r)
        elif r["person"] in train_p and r["text"] in train_s:
            buckets["train_di"].append(r)
        else:
            discarded += 1

    for name, recs in buckets.items():
        out = args.manifest_dir / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"audio": r["audio"], "label": r["label"], "speaker": r["speaker"]},
                                   ensure_ascii=False) + "\n")
        npers = len({r["person"] for r in recs}); nscr = len({r["text"] for r in recs})
        lab = dict(sorted(Counter(r["label"] for r in recs).items()))
        print(f"[{name}] {len(recs)} 발화 | 화자 {npers} | 대본 {nscr} | 라벨 {lab} → {out}", flush=True)

    # 누수 검증
    pe = {k: {r["person"] for r in v} for k, v in buckets.items()}
    se = {k: {r["text"] for r in v} for k, v in buckets.items()}
    print(f"[폐기] 교차 셀 {discarded} 발화 (전체의 {100*discarded/len(rows):.0f}%)")
    print(f"[검증] 화자 겹침 train∩test={len(pe['train_di']&pe['test_di'])} train∩val={len(pe['train_di']&pe['val_di'])}")
    print(f"[검증] 대본 겹침 train∩test={len(se['train_di']&se['test_di'])} train∩val={len(se['train_di']&se['val_di'])}")
    print("       (모두 0 이어야 화자·대본 동시 누수 0)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
