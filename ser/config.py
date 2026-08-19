"""YAML config 로더 + argparse override 머지 (백본 무관).

사용 패턴:
    parser = build_parser()         # 모든 override 인자는 default=None
    args = parser.parse_args()
    cfg = load_config(args)         # YAML 기본값, None이 아닌 CLI 인자만 override
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def backbone_dirname(backbone_id: str) -> str:
    """백본 ID에서 파일시스템 안전 디렉토리명 추출 (emotion2vec/x → x, microsoft/wavlm-large → wavlm-large)."""
    name = backbone_id.rstrip("/").split("/")[-1]
    return name or backbone_id.replace("/", "_")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"YAML config 경로 (default: {DEFAULT_CONFIG_PATH}).",
    )


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    skip: tuple[str, ...] = ("config",),
) -> dict[str, Any]:
    """value가 None이 아닌 argparse 인자만 config[key] override."""
    for key, value in vars(args).items():
        if key in skip or value is None:
            continue
        config[key] = value
    return config


# YAML이 지수표기(3e-5)를 str로 파싱하는 함정 방지: 아래 키는 항상 float 강제.
_FLOAT_KEYS = (
    "lr", "weight_decay", "label_smoothing", "dropout",
    "full_ft_lr", "full_ft_backbone_lr", "full_ft_max_audio_sec", "head_lr",
)


def _coerce_numeric(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in _FLOAT_KEYS:
        v = cfg.get(key)
        if v is None or isinstance(v, float):
            continue
        try:
            cfg[key] = float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"config '{key}'={v!r} 를 float으로 변환 실패") from e
    return cfg


# 모드별 네임스페이스 YAML 섹션 (config.py 가 엔진이 쓰는 평면 키로 변환).
_NESTED_SECTIONS = ("common", "acoustic", "cached")


def _flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """모드별 섹션(common/acoustic/cached) → 평면 키. 이미 평면이면 그대로 (하위호환).

    매핑: common.* → 그대로(+split.{train,val,test}→split_*).
          acoustic.<k> → full_ft_<k> (단 full_ft_*/head_lr 는 그대로) — ser.train(run_acoustic).
          cached.* → 그대로(lr, batch_size) — ser.train_cached(run_head_cached).
    """
    if not any(s in cfg for s in _NESTED_SECTIONS):
        return cfg
    flat: dict[str, Any] = {}
    common = dict(cfg.get("common") or {})
    split = common.pop("split", None)
    flat.update(common)
    if isinstance(split, dict):
        flat["split_train"], flat["split_val"], flat["split_test"] = (
            split.get("train"), split.get("val"), split.get("test"))
    for k, v in (cfg.get("acoustic") or {}).items():
        flat[k if (k.startswith("full_ft_") or k == "head_lr") else f"full_ft_{k}"] = v
    flat.update(cfg.get("cached") or {})
    for k, v in cfg.items():            # 섹션 밖 평면 키(있으면) 보존
        if k not in _NESTED_SECTIONS:
            flat.setdefault(k, v)
    return flat


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _flatten_config(load_yaml(args.config))
    cfg = apply_cli_overrides(cfg, args)
    cfg = _coerce_numeric(cfg)
    return cfg


def dump_config(cfg: dict[str, Any], path: Path) -> None:
    """머지된 최종 config를 체크포인트와 함께 저장 (재현용)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(v: Any) -> Any:
        return str(v) if isinstance(v, Path) else v

    serializable = {k: _coerce(v) for k, v in cfg.items()}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False, allow_unicode=True)


def print_config(cfg: dict[str, Any]) -> None:
    print("=" * 60)
    print("Resolved config:")
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                     indent=2, ensure_ascii=False))
    print("=" * 60)
