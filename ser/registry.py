"""결과 레지스트리 — 모든 런의 지표를 runs/registry.jsonl 한 곳에 모은다.

(레포의 results/ 는 공개된 리더보드 — 학습 산출물이 아니다. $SER_REGISTRY 로 변경 가능.)

모든 "macro-F1 비교"·"best 음향 선택"·"fusion gain"은 이 파일 쿼리로 끝낸다 (재실행 X).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import load_yaml

# SER_REGISTRY 환경변수로 경로 override (스모크는 별도 파일로 → 본 registry 안 더럽힘)
REGISTRY = Path(os.environ.get("SER_REGISTRY", "runs/registry.jsonl"))


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def log_run(run_dir: str | Path, *, modality: str = "acoustic",
            registry_path: str | Path = REGISTRY, extra: dict | None = None) -> dict:
    """run_dir 의 resolved_config + test/val_metrics + test_logits 로 1행 작성 → registry append."""
    run_dir = Path(run_dir)
    cfg = load_yaml(run_dir / "resolved_config.yaml") if (run_dir / "resolved_config.yaml").is_file() else {}
    test = _read_json(run_dir / "test_metrics.json")
    val = _read_json(run_dir / "val_metrics.json")

    per_class_f1 = None
    logit_npz = run_dir / "test_logits.npz"
    if logit_npz.is_file():
        from sklearn.metrics import f1_score
        d = np.load(logit_npz, allow_pickle=True)
        preds = d["logits"].argmax(1)
        per_class_f1 = f1_score(d["labels"], preds, average=None).tolist()

    mode = cfg.get("mode") or run_dir.parent.name   # checkpoints/<mode>/<run>
    row = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "modality": modality,
        "backbone": cfg.get("backbone"),
        "mode": mode,
        "freeze_cnn": cfg.get("full_ft_freeze_cnn"),
        "head_type": cfg.get("head_type"),
        "num_classes": cfg.get("num_classes") or test.get("num_classes"),
        "keep_labels": cfg.get("keep_labels"),
        "val_macro_f1": val.get("macro_f1"),
        "test_macro_f1": test.get("macro_f1"),
        "test_acc": test.get("acc"),
        "per_class_f1": per_class_f1,
        "split_test": cfg.get("split_test", "test_si"),
        "seed": cfg.get("seed"),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        row.update(extra)

    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[registry] +1 row → {registry_path} (test macroF1={row['test_macro_f1']})", flush=True)
    return row


def log_fusion(method: str, acoustic: str, text: str, *, labels, preds, run_dir: str | Path | None = None,
               registry_path: str | Path = REGISTRY, extra: dict | None = None) -> dict:
    """fusion 결과 1행 → registry append. labels/preds 로 지표 계산.

    method: late|early|cross|llm,  acoustic/text: 사용한 음향/텍스트 run_id.
    run_dir: test_logits.npz 가 있는 폴더(있으면 report 가 CM 그림 생성).
    """
    import numpy as _np
    from sklearn.metrics import accuracy_score, f1_score
    labels, preds = _np.asarray(labels), _np.asarray(preds)
    row = {
        "run_id": f"fusion-{method}-{acoustic}+{text}",
        "run_dir": str(run_dir) if run_dir else None,
        "modality": "fusion",
        "fusion_method": method,
        "acoustic": acoustic,
        "text": text,
        "test_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "test_acc": float(accuracy_score(labels, preds)),
        "per_class_f1": f1_score(labels, preds, average=None, zero_division=0).tolist(),
        "n": int(len(labels)),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        row.update(extra)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[registry] +1 fusion row → {registry_path} "
          f"({method} {acoustic}+{text} macroF1={row['test_macro_f1']:.4f})", flush=True)
    return row


def load_registry(registry_path: str | Path = REGISTRY) -> list[dict]:
    p = Path(registry_path)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def best_by(metric: str = "test_macro_f1", *, modality: str | None = None,
            registry_path: str | Path = REGISTRY) -> dict | None:
    """레지스트리에서 metric 최대 행 반환 (modality 필터 가능). fusion gain 기준선 조회용."""
    rows = [r for r in load_registry(registry_path)
            if r.get(metric) is not None and (modality is None or r.get("modality") == modality)]
    return max(rows, key=lambda r: r[metric]) if rows else None
