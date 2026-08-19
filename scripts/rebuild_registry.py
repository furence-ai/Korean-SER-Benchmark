"""기존 결과 폴더들을 훑어 runs/registry.jsonl 을 재생성한다.

registry.jsonl 은 run 마다 append 되는 소형 파일로, 원래 레포의 results/ 에만 남는다
(NAS 산출물엔 안 따라옴). 학습 폴더만 다른 곳으로 옮겨오면 registry 가 없어 scripts.report
가 못 도는데, 이 스크립트가 각 폴더의 test_logits.npz(+resolved_config.yaml)로부터 registry
를 통째로 복원한다. 재학습/재평가 불필요 — 이미 저장된 logits 로 지표만 다시 계산.

대상: <root> 아래에서 test_logits.npz 가 있는 모든 폴더.
    acoustic/* → modality=acoustic,  text/* → text,
    early|late|cross|llm/* → modality=fusion (fusion_method = 상위 폴더명)
backbone/mode/keep_labels 등 메타는 resolved_config.yaml 있으면 거기서, 없으면 경로로 추론.

예:
    uv run python -m scripts.rebuild_registry --root runs/all
    # 그 후:
    uv run python -m scripts.report --registry runs/all/registry.jsonl --cm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ser.config import load_yaml

# 상위 폴더명 → (modality, fusion_method)
_FUSION_DIRS = {"early": "early", "late": "late", "cross": "cross", "llm": "llm"}


def _row_from_dir(run_dir: Path) -> dict | None:
    """test_logits.npz 로 지표 계산 + 메타 수집 → registry 1행. npz 없으면 None."""
    npz = run_dir / "test_logits.npz"
    if not npz.is_file():
        return None
    d = np.load(npz, allow_pickle=True)
    y, preds = d["labels"], d["logits"].argmax(1)

    cfg = load_yaml(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").is_file() else {}
    parent = run_dir.parent.name
    name = run_dir.name

    row = {
        "run_id": name,
        "run_dir": str(run_dir),
        "transcript": "stt" if name.endswith("_stt") else "gt",   # 텍스트 전사 출처(보고서 GT/STT 구분)
        "backbone": cfg.get("backbone"),
        "mode": cfg.get("mode"),
        "keep_labels": cfg.get("keep_labels"),
        "head_type": cfg.get("head_type"),
        "split_test": cfg.get("split_test", "test_di"),
        "seed": cfg.get("seed"),
        "test_macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "test_acc": float(accuracy_score(y, preds)),
        "per_class_f1": f1_score(y, preds, average=None, zero_division=0).tolist(),
        "n": int(len(y)),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if parent in _FUSION_DIRS:                       # early/late/cross/llm
        row["modality"] = "fusion"
        row["fusion_method"] = _FUSION_DIRS[parent]
        row["acoustic"] = name                       # _describe 가 음향 소스로 표기
        row["text"] = "roberta-large"
    elif parent == "text":
        row["modality"] = "text"
    else:                                            # acoustic (또는 기타)
        row["modality"] = cfg.get("modality") or "acoustic"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="결과 루트 (예: runs/all)")
    p.add_argument("--out", type=Path, default=None, help="기본 <root>/registry.jsonl")
    args = p.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"폴더 없음: {root}")
    out = args.out or (root / "registry.jsonl")

    dirs = sorted({npz.parent for npz in root.rglob("test_logits.npz")})
    rows = []
    for rd in dirs:
        row = _row_from_dir(rd)
        if row is None:
            continue
        rows.append(row)
        disp = "/".join(rd.parts[-2:])
        print(f"[+] {disp:32s} mF1={row['test_macro_f1']:.4f} acc={row['test_acc']:.4f} "
              f"({row['modality']})", flush=True)

    rows.sort(key=lambda r: r["test_macro_f1"], reverse=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[saved] {out}  ({len(rows)} run)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
