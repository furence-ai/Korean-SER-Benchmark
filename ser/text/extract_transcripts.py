"""GT(정답) 전사 추출 → data/transcripts/{split}.jsonl ({audio, text}).

전사 추출 규칙은 데이터셋 어댑터(ser/datasets/<id>.py: transcript_for)에 위임 — AIHub 라벨 json 스키마
같은 데이터셋 세부는 어댑터 안에만. GT 는 "텍스트가 도움 되나" 확인용 낙관적 상한.
다른 전사 소스(예: ASR 출력)를 쓰려면 같은 포맷으로 재생성하면 downstream 은 그대로다.

실행:
    uv run python -m ser.text.extract_transcripts                       # config 의 split들
    uv run python -m ser.text.extract_transcripts --splits test_di
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ser.cli import add_common_overrides, config_splits, resolve_dataset_ctx
from ser.config import load_config
from ser.io import load_split_jsonl

# ---- 하위호환 shim (구 import: from ser.text.extract_transcripts import org_label_text) ----
_GT_ADAPTER = None


def org_label_text(audio: str) -> str | None:
    """기본 데이터셋 GT 전사. 데이터셋별 규칙은 어댑터.transcript_for 로 위임."""
    global _GT_ADAPTER
    if _GT_ADAPTER is None:
        from ser.datasets.emotion_style_speech import EmotionStyleSpeech
        _GT_ADAPTER = EmotionStyleSpeech()
    return _GT_ADAPTER.transcript_for(audio)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--out-dir", type=Path, default=Path("data/transcripts"))
    args = p.parse_args()
    cfg = load_config(args)
    ctx = resolve_dataset_ctx(cfg)
    splits = config_splits(cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = load_split_jsonl(cfg["manifest_dir"], split)
        out = args.out_dir / f"{split}.jsonl"
        n_ok = n_miss = 0
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                t = ctx.adapter.transcript_for(r["audio"])
                if t is None:
                    n_miss += 1
                    continue
                f.write(json.dumps({"audio": r["audio"], "text": t}, ensure_ascii=False) + "\n")
                n_ok += 1
        print(f"[{split}] 전사 {n_ok} / 누락 {n_miss} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
