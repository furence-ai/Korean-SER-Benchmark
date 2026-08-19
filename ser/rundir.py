"""run-dir / resume / seed / LR schedule 헬퍼 (백본 무관)."""
from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_run_dir(out_dir: Path, mode: str, run_name: str | None, suffix: str | None = None) -> Path:
    """{out_dir}/{mode}/{run_name or timestamp}[_{suffix}]/ 생성 (실행마다 unique)."""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        run_name = f"{run_name}_{suffix}"
    run_dir = out_dir / mode / run_name
    if run_dir.exists():
        for i in range(1, 100):
            alt = out_dir / mode / f"{run_name}_{i}"
            if not alt.exists():
                run_dir = alt
                break
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_run_dir(cfg: dict, mode: str, run_name: str | None, output: Path | None,
                    suffix: str | None = None) -> Path:
    if output is not None:
        run_dir = Path(output)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir(Path(cfg["out_dir"]), mode, run_name, suffix)


def resolve_resume(resume: Path | None, cfg: dict, mode: str, run_name: str | None,
                   output: Path | None, suffix: str, device: str) -> tuple[dict | None, Path]:
    """--resume 처리. 주어지면 (last.pt 로드, 같은 run_dir 재사용), 아니면 (None, 새 run_dir)."""
    if resume is None:
        return None, resolve_run_dir(cfg, mode, run_name, output, suffix=suffix)
    path = Path(resume)
    if path.is_dir():
        path = path / "last.pt"
    if not path.exists():
        raise FileNotFoundError(f"--resume 경로에 last.pt 없음: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("mode") != mode:
        raise ValueError(f"--resume mode 불일치: ckpt {ckpt.get('mode')!r} != 현재 {mode!r}.")
    print(f"[resume] {path} — epoch {ckpt['epoch']}까지 완료 → {ckpt['epoch'] + 1}부터 재개 "
          f"(best macroF1={ckpt.get('best_macro_f1', float('nan')):.4f})", flush=True)
    return ckpt, path.parent


def warmup_cosine_lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
