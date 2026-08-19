"""_di split 에서 작은 subset(_smoke) 추출 — 파이프라인 빠른 검증용.

라벨 stratify 로 split당 per-class N개 샘플. disjoint 속성은 부분집합이라 그대로 유지.
매니페스트 + 전사 둘 다 생성 (train_smoke/val_smoke/test_smoke).

실행:
    uv run python -m scripts.make_smoke_subset --per-class 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    p.add_argument("--per-class", type=int, default=50, help="split·class 당 샘플 수")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    for src, dst in [("train_di", "train_smoke"), ("val_di", "val_smoke"), ("test_di", "test_smoke")]:
        rows = [json.loads(l) for l in (args.manifest_dir / f"{src}.jsonl").open(encoding="utf-8") if l.strip()]
        txt = {json.loads(l)["audio"]: json.loads(l)["text"]
               for l in (args.transcript_dir / f"{src}.jsonl").open(encoding="utf-8") if l.strip()}
        by: dict[int, list] = defaultdict(list)
        for r in rows:
            by[int(r["label"])].append(r)
        sel: list = []
        for rs in by.values():
            rng.shuffle(rs)
            sel.extend(rs[:args.per_class])
        rng.shuffle(sel)
        with (args.manifest_dir / f"{dst}.jsonl").open("w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (args.transcript_dir / f"{dst}.jsonl").open("w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps({"audio": r["audio"], "text": txt[r["audio"]]}, ensure_ascii=False) + "\n")
        labs = sorted({int(r["label"]) for r in sel})
        print(f"[{dst}] {len(sel)} 발화 (per-class {args.per_class}, labels {labs})", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
