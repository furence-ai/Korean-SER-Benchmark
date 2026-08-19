"""Speaker-independent 3-way split 매니페스트 생성.

기존 train.jsonl + val.jsonl (AIHub Training/Validation, 감정 기준 분리)을 합친 뒤
사람(person) 단위로 train/val/test 를 분리한다. 한 사람은 한 split 에만 들어가
train/val/test 어디에도 화자가 겹치지 않는다 (speaker-independent).

핵심:
  - speaker 필드는 '사람×감정' 단위(감정마다 ID 다름)라, 감정 떼고 '번호_이니셜'로 묶는다
    (person_id_from_speaker).
  - val/test 는 '4클래스를 모두 보유한 사람'에서 우선 뽑아 클래스 균형을 보장한다.
  - 나머지 전원(부분 클래스 보유자 포함)은 train.

출력: <out-dir>/{train,val,test}_si.jsonl  (원본 train/val.jsonl 은 보존)

실행:
    unset PYTHONPATH
    uv run python -m scripts.make_speaker_split                 # 기본 4-class, val/test 각 10%
    uv run python -m scripts.make_speaker_split --n-val 4 --n-test 4
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ser.loader import person_id_from_speaker
from ser.labels import LABEL_TO_KOREAN as KO   # 데이터셋 라벨공간(전체 클래스 한글명)


def load_items(manifest_dir: Path, names: list[str], keep_labels: set[int]) -> list[dict]:
    items: list[dict] = []
    for name in names:
        path = manifest_dir / name
        if not path.is_file():
            print(f"[warn] 없음, 건너뜀: {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if int(d["label"]) in keep_labels:
                    items.append(d)
    return items


def assign_persons(
    items: list[dict], keep_labels: set[int], n_val: int, n_test: int, seed: int,
) -> dict[str, set[str]]:
    """사람을 train/val/test 로 배분. val/test 는 전 클래스 보유자에서 우선 추출."""
    per_classes: dict[str, set[int]] = defaultdict(set)
    for d in items:
        per_classes[person_id_from_speaker(d["speaker"])].add(int(d["label"]))

    n_classes = len(keep_labels)
    all_class = sorted(p for p, cs in per_classes.items() if len(cs) >= n_classes)
    partial = sorted(p for p, cs in per_classes.items() if len(cs) < n_classes)

    if len(all_class) < n_val + n_test:
        raise ValueError(
            f"전 클래스 보유 사람이 {len(all_class)}명뿐 → val({n_val})+test({n_test}) 확보 불가."
        )

    rng = np.random.RandomState(seed)
    pool = list(all_class)
    rng.shuffle(pool)
    val_p = set(pool[:n_val])
    test_p = set(pool[n_val : n_val + n_test])
    train_p = set(pool[n_val + n_test :]) | set(partial)
    return {"train": train_p, "val": val_p, "test": test_p}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--inputs", nargs="+", default=["train.jsonl", "val.jsonl"],
                   help="합칠 원본 매니페스트들.")
    p.add_argument("--keep-labels", nargs="+", type=int, default=[0, 1, 2, 6])
    p.add_argument("--n-val", type=int, default=4, help="val 로 뺄 사람 수 (전 클래스 보유자).")
    p.add_argument("--n-test", type=int, default=4, help="test 로 뺄 사람 수 (전 클래스 보유자).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="기본: --manifest-dir 와 동일.")
    p.add_argument("--suffix", default="_si", help="출력 파일 접미사 (기본 _si).")
    args = p.parse_args()

    keep = set(args.keep_labels)
    out_dir = args.out_dir or args.manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.manifest_dir, args.inputs, keep)
    if not items:
        print("[error] 합칠 항목이 없음.")
        return 1

    persons_by_split = assign_persons(items, keep, args.n_val, args.n_test, args.seed)
    person_to_split = {p: s for s, ps in persons_by_split.items() for p in ps}

    # 사람 배정에 따라 발화 분배
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for d in items:
        rows[person_to_split[person_id_from_speaker(d["speaker"])]].append(d)

    # 사람 겹침 검증
    pe = {s: {person_id_from_speaker(d["speaker"]) for d in rows[s]} for s in rows}
    assert not (pe["train"] & pe["val"]), "train/val 사람 겹침!"
    assert not (pe["train"] & pe["test"]), "train/test 사람 겹침!"
    assert not (pe["val"] & pe["test"]), "val/test 사람 겹침!"

    print(f"합친 발화 {len(items):,}개 / 사람 {len(person_to_split)}명 "
          f"(seed={args.seed}, n_val={args.n_val}, n_test={args.n_test})\n")
    total = len(items)
    for s in ["train", "val", "test"]:
        labc = Counter(int(d["label"]) for d in rows[s])
        out = out_dir / f"{s}{args.suffix}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for d in rows[s]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        cov = {KO.get(c, str(c)): labc.get(c, 0) for c in sorted(keep)}
        print(f"[{s:5}] 사람 {len(pe[s]):2}명 | 발화 {len(rows[s]):>7,}개 "
              f"({100*len(rows[s])/total:4.1f}%) | 클래스별 {cov}")
        print(f"         → {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
