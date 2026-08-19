"""raw 데이터셋 디렉토리 스캔 → manifest jsonl ({audio, label, speaker, ...}).

스캔 규칙은 데이터셋 어댑터(ser/datasets/<id>.py: iter_manifest)에 위임 — 새 데이터셋은 어댑터 파일
하나만 추가하면 이 스크립트 변경 없이 동작. 경로/태스크 같은 비코드 값은 configs/datasets/<id>.yaml.
※ 화자+대본 분리 split 은 이후 scripts.make_disjoint_split 로 생성.

실행:
    uv run python -m scripts.build_manifest --dataset emotion_style_speech --splits train val
    uv run python -m scripts.build_manifest --dataset emotion_style_speech --data-root /path --task 1.감정
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ser.config import load_yaml
from ser.datasets import resolve_dataset

_DATASET_CFG_DIR = Path(__file__).resolve().parent.parent / "configs" / "datasets"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="emotion_style_speech", help="데이터셋 어댑터 id")
    p.add_argument("--dataset-config", type=Path, default=None,
                   help="비코드 설정 yaml (기본 configs/datasets/<dataset>.yaml)")
    p.add_argument("--data-root", type=Path, default=None, help="raw 루트 (미지정: $SER_DATA_ROOT → configs/datasets/<id>.yaml)")
    p.add_argument("--task", default=None, help="서브태스크 (미지정=dataset-config 또는 어댑터 기본)")
    p.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="raw split (train/val 등)")
    args = p.parse_args()

    # 데이터셋 비코드 설정 로드 (configs/datasets/<id>.yaml)
    ds_cfg_path = args.dataset_config or (_DATASET_CFG_DIR / f"{args.dataset}.yaml")
    ds_cfg = load_yaml(ds_cfg_path) if ds_cfg_path.is_file() else {}
    root = args.data_root or os.environ.get("SER_DATA_ROOT") or ds_cfg.get("data_root")
    if not root:
        p.error("데이터셋 루트를 찾을 수 없습니다. --data-root 를 주거나 SER_DATA_ROOT 환경변수 "
                f"또는 {ds_cfg_path} 의 data_root 를 설정하세요.")
    data_root = Path(root)
    if not data_root.is_dir():
        p.error(f"데이터셋 루트가 없습니다: {data_root}")
    task = args.task or ds_cfg.get("task")

    adapter = resolve_dataset(args.dataset)()
    if task is not None and hasattr(adapter, "task"):
        adapter.task = task   # 데이터셋-무관: task 개념 있는 어댑터만 반영

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        items = list(adapter.iter_manifest(data_root, split))
        out = args.out_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        counts: dict[int, int] = {}
        for it in items:
            counts[it["label"]] = counts.get(it["label"], 0) + 1
        print(f"[{split}] {len(items)} utterances per-label={counts} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
