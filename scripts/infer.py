"""학습된 음향 모델로 단일(또는 여러) 오디오 파일 감정 추론.

테스트셋/매니페스트 없이 **wav 파일 경로만** 받아 감정 1개 + 클래스별 확률을 출력한다.
배포/데모/빠른 확인용. (test_logits.npz 같은 산출물은 안 만든다 — 그건 scripts.evaluate.)

음향 raw-audio 모델 전용: ckpt mode 가 full_ft / head_only_otf 여야 한다.
    head_cached(text/early)·fusion_cross 는 전사/임베딩/듀얼인코더가 필요해 단일 wav 만으론 추론 불가.

예:
    uv run python -m scripts.infer --ckpt runs/all/acoustic/w2v_full a.wav
    uv run python -m scripts.infer --ckpt .../acoustic/em_head a.wav b.wav c.wav     # 여러 개
    uv run python -m scripts.infer --ckpt .../acoustic/w2v_full --glob "clips/*.wav"  # glob
    uv run python -m scripts.infer --ckpt .../acoustic/w2v_full a.wav --json          # 기계 파싱용 JSON
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ser.audio import SAMPLE_RATE, load_audio_16k_mono
from ser.labels import resolve_label_subset


def _label_names(ckpt: dict) -> dict[int, str]:
    """dense index → 영어 라벨명. ckpt meta 우선, 없으면 keep_labels 로 재구성."""
    l2e = ckpt.get("label_to_english")
    if l2e:
        return {int(k): v for k, v in l2e.items()}
    _, _, names = resolve_label_subset(ckpt.get("keep_labels"))
    return names


@torch.no_grad()
def _predict_one(model, wav_path: str, device: str, max_samples: int, amp_dtype: str) -> tuple:
    """wav 1개 → (probs[C] numpy, n_samples). 모델 forward 는 학습과 동일 경로."""
    from ser.heads import amp_autocast

    wav = load_audio_16k_mono(wav_path)
    n = len(wav)
    if max_samples and n > max_samples:        # 학습과 동일하게 앞부분 truncate
        wav = wav[:max_samples]
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)          # [1, T]
    pad_mask = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=device)   # 단일이라 pad 없음
    with amp_autocast(device, amp_dtype):
        logits = model(wav_t, padding_mask=pad_mask)
    probs = F.softmax(logits.float(), dim=-1)[0].cpu().numpy()
    return probs, n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="단일 오디오 파일 감정 추론 (음향 모델)")
    p.add_argument("audio", nargs="*", help="오디오 파일 경로 (여러 개 가능)")
    p.add_argument("--ckpt", type=Path, required=True, help="best.pt 또는 run 폴더")
    p.add_argument("--glob", type=str, default=None, help='glob 패턴 (예: "clips/*.wav")')
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (기본: 가능하면 cuda)")
    p.add_argument("--max-sec", dest="max_sec", type=float, default=None,
                   help="앞에서부터 사용할 최대 길이(초). 기본: ckpt 학습값(없으면 10s)")
    p.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    return p


def main() -> int:
    args = build_parser().parse_args()

    paths = list(args.audio)
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        raise SystemExit("오디오 파일 경로가 없습니다 (인자 또는 --glob).")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt / "best.pt" if args.ckpt.is_dir() else args.ckpt
    if not ckpt_path.is_file():
        raise SystemExit(f"체크포인트 없음: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode = ckpt.get("mode")
    if mode not in ("full_ft", "head_only_otf"):
        raise SystemExit(
            f"이 스크립트는 음향 raw-audio 모델 전용인데 mode={mode!r} 입니다.\n"
            f"  head_cached(text/early)·fusion_cross 는 전사/임베딩이 필요해 단일 wav 추론 불가."
        )

    from ser.engine import build_model_from_ckpt
    model = build_model_from_ckpt(ckpt, device)
    names = _label_names(ckpt)
    max_sec = args.max_sec if args.max_sec is not None else ckpt.get("full_ft_max_audio_sec", 10.0)
    max_samples = int(max_sec * SAMPLE_RATE) if max_sec else 0
    amp_dtype = ckpt.get("amp_dtype", "bf16")

    results = []
    for ap in paths:
        try:
            probs, n = _predict_one(model, ap, device, max_samples, amp_dtype)
        except Exception as e:  # noqa: BLE001 — 파일 하나 실패해도 나머지 진행
            print(f"[error] {ap}: {e!r}", flush=True)
            results.append({"audio": ap, "error": repr(e)})
            continue
        top = int(probs.argmax())
        rec = {
            "audio": ap,
            "pred": names.get(top, str(top)),
            "pred_index": top,
            "confidence": round(float(probs[top]), 4),
            "duration_sec": round(n / SAMPLE_RATE, 2),
            "probs": {names.get(i, str(i)): round(float(probs[i]), 4) for i in range(len(probs))},
        }
        results.append(rec)
        if not args.json:
            dist = "  ".join(f"{k}={v:.3f}" for k, v in rec["probs"].items())
            print(f"{ap}\n  → {rec['pred']}  (conf={rec['confidence']:.3f}, {rec['duration_sec']}s)  | {dist}",
                  flush=True)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
