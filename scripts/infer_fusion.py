"""wav + 전사(텍스트) 로 음향+텍스트 **앙상블** 감정 추론.

    wav ─┬─ 음향 모델(adapter+head) ─────────┐
         └─ 전사 → RoBERTa ── 텍스트 분기 ───┴─ 결합(late|early) → 감정 1개 + 확률

전사는 `--text` 로 직접 준다. 이 레포는 ASR 을 포함하지 않는다 — 배포에서는 쓰던 STT 의
출력을 그대로 넘기면 된다. (음향만으로 충분하면 `scripts.infer` 가 wav 하나로 끝난다.)

결합 방법:
  late  (기본) : 음향 logits + 텍스트 logits 를 softmax 평균. 필요한 ckpt = 음향 head + 텍스트 head.
  early        : 음향 emb ⊕ 텍스트 emb → early head. 필요한 ckpt = 음향(emb 추출용) + early head.

음향 ckpt 종류:
  --acoustic-ckpt   full-FT/head-only best.pt (그 백본으로 forward). late=logits, early=emb.
  --acoustic-backbone <id>   (early 전용 대안) 원본 동결 백본으로 emb 추출.

예 (early, full-FT 음향):
  uv run python -m scripts.infer_fusion --method early \\
      --acoustic-ckpt runs/all/acoustic/w2v_full --fusion-ckpt runs/all/early/w2v_full \\
      a.wav --text "오늘 정말 기분이 좋아"
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ser.audio import SAMPLE_RATE, load_audio_16k_mono
from ser.labels import resolve_label_subset


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _label_names(ckpt: dict) -> dict[int, str]:
    l2e = ckpt.get("label_to_english")
    if l2e:
        return {int(k): v for k, v in l2e.items()}
    _, _, names = resolve_label_subset(ckpt.get("keep_labels"))
    return names


# ---------- 텍스트 분기 (전사 → RoBERTa 임베딩 → 텍스트 head) ----------

class TextBranch:
    """RoBERTa 임베딩 lazy 로더. extract_embeddings 와 동일한 mean-pool."""

    def __init__(self, roberta_model: str, device: str, max_len: int = 64,
                 language: str = "korean"):
        self.device, self.max_len, self.language = device, max_len, language
        self.roberta_model = roberta_model
        self._tok = self._enc = None


    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """텍스트 1개 → mean-pool 임베딩 [D] (attention mask 가중; extract_embeddings 와 동일)."""
        self._ensure_roberta()
        enc = self._tok([text], padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt").to(self.device)
        h = self._enc(**enc).last_hidden_state                    # (1, T, D)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return pooled[0].float().cpu().numpy()


@torch.no_grad()
def _text_head_logits(text_ckpt: dict, emb: np.ndarray, device: str) -> np.ndarray:
    """RoBERTa 임베딩 → 텍스트 head(custom-utterance-mlp) → logits [C]."""
    from ser.heads import build_head
    head = build_head(text_ckpt["head_type"], text_ckpt["embedding_dim"], text_ckpt["num_classes"],
                      text_ckpt["hidden_dim"], text_ckpt["dropout"]).to(device).eval()
    head.load_state_dict(text_ckpt["model_state"])
    x = torch.from_numpy(emb).unsqueeze(0).to(device)
    return head(x)[0].float().cpu().numpy()


# ---------- 음향 분기 ----------

@torch.no_grad()
def _acoustic_logits(model, wav: np.ndarray, device: str, amp_dtype: str) -> np.ndarray:
    from ser.heads import amp_autocast
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=device)
    with amp_autocast(device, amp_dtype):
        logits = model(wav_t, padding_mask=pad)
    return logits[0].float().cpu().numpy()


@torch.no_grad()
def _acoustic_embed(adapter, wav: np.ndarray, device: str, amp_dtype: str) -> np.ndarray:
    """학습/추출과 동일한 frames_and_mask + masked_mean_pool → utterance 임베딩 [D]."""
    from ser.engine import frames_and_mask, masked_mean_pool
    from ser.heads import amp_autocast
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    pad = torch.zeros(1, wav_t.size(1), dtype=torch.bool, device=device)
    with amp_autocast(device, amp_dtype):
        f, m = frames_and_mask(adapter, wav_t, pad, adapter.normalize_input)
        pooled = masked_mean_pool(f, m)
    return pooled[0].float().cpu().numpy()


@torch.no_grad()
def _early_head_probs(fusion_ckpt: dict, ac_emb: np.ndarray, txt_emb: np.ndarray, device: str) -> np.ndarray:
    from ser.heads import build_head
    head = build_head(fusion_ckpt["head_type"], fusion_ckpt["embedding_dim"], fusion_ckpt["num_classes"],
                      fusion_ckpt["hidden_dim"], fusion_ckpt["dropout"]).to(device).eval()
    head.load_state_dict(fusion_ckpt["model_state"])
    x = torch.from_numpy(np.concatenate([ac_emb, txt_emb])).unsqueeze(0).to(device)
    return F.softmax(head(x)[0].float(), dim=-1).cpu().numpy()


def _resolve_ckpt(path: Path | None) -> Path | None:
    if path is None:
        return None
    p = path / "best.pt" if path.is_dir() else path
    if not p.is_file():
        raise SystemExit(f"체크포인트 없음: {p}")
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="wav + 전사 → 음향+텍스트 앙상블 감정 추론")
    p.add_argument("audio", nargs="*", help="오디오 파일 경로 (여러 개 가능)")
    p.add_argument("--glob", type=str, default=None)
    p.add_argument("--method", choices=["late", "early"], default="late")
    p.add_argument("--acoustic-ckpt", dest="acoustic_ckpt", type=Path, default=None,
                   help="음향 best.pt/run폴더 (late=logits, early=emb 추출)")
    p.add_argument("--acoustic-backbone", dest="acoustic_backbone", type=str, default=None,
                   help="(early 대안) 원본 동결 백본 id — early/*_pre 용")
    p.add_argument("--text-ckpt", dest="text_ckpt", type=Path, default=None, help="(late) 텍스트 head best.pt")
    p.add_argument("--fusion-ckpt", dest="fusion_ckpt", type=Path, default=None, help="(early) early head best.pt")
    p.add_argument("--roberta-model", dest="roberta_model", default="klue/roberta-large")
    p.add_argument("--late-method", dest="late_method", choices=["softmax_avg", "logit_mean"], default="softmax_avg")
    p.add_argument("--text", required=True, help="발화 전사 (이 레포는 ASR 을 포함하지 않는다)")
    p.add_argument("--device", default=None)
    p.add_argument("--max-sec", dest="max_sec", type=float, default=None)
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    paths = list(args.audio)
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        raise SystemExit("오디오 파일 경로가 없습니다 (인자 또는 --glob).")
    if len(paths) > 1:
        raise SystemExit("--text 가 발화마다 달라야 하므로 한 번에 wav 1개만 처리한다.")

    # ---- 음향 모델 로드 ----
    ac_ckpt_path = _resolve_ckpt(args.acoustic_ckpt)
    ac_model = ac_adapter = ac_meta = None
    if ac_ckpt_path is not None:
        from ser.engine import build_model_from_ckpt
        ac_meta = torch.load(ac_ckpt_path, map_location=device, weights_only=False)
        if ac_meta.get("mode") not in ("full_ft", "head_only_otf"):
            raise SystemExit(f"--acoustic-ckpt mode={ac_meta.get('mode')!r} 는 음향 raw-audio 가 아님")
        m = build_model_from_ckpt(ac_meta, device)
        ac_model, ac_adapter = m, m.adapter
    elif args.method == "early" and args.acoustic_backbone:
        import ser.backbones  # self-register  # noqa: F401
        from ser.backbones import resolve_adapter
        print(f"[load] 음향 원본 백본(동결): {args.acoustic_backbone}", flush=True)
        ac_adapter = resolve_adapter(args.acoustic_backbone).load(
            args.acoustic_backbone, freeze_backbone=True).to(device).eval()
    else:
        raise SystemExit("--acoustic-ckpt (또는 early 에서 --acoustic-backbone) 가 필요합니다.")

    # ---- 결합용 head ckpt + 라벨명 ----
    if args.method == "late":
        tk = _resolve_ckpt(args.text_ckpt)
        if tk is None:
            raise SystemExit("late 는 --text-ckpt (텍스트 head) 가 필요합니다.")
        text_ckpt = torch.load(tk, map_location=device, weights_only=False)
        names = _label_names(ac_meta or text_ckpt)
    else:  # early
        fk = _resolve_ckpt(args.fusion_ckpt)
        if fk is None:
            raise SystemExit("early 는 --fusion-ckpt (early head) 가 필요합니다.")
        fusion_ckpt = torch.load(fk, map_location=device, weights_only=False)
        names = _label_names(fusion_ckpt)

    text_branch = TextBranch(args.roberta_model, device)
    max_samples = int((args.max_sec if args.max_sec is not None
                       else (ac_meta or {}).get("full_ft_max_audio_sec", 10.0)) * SAMPLE_RATE)
    amp_dtype = (ac_meta or {}).get("amp_dtype", "bf16")

    results = []
    for ap in paths:
        try:
            wav = load_audio_16k_mono(ap)
            wav_ac = wav[:max_samples] if (max_samples and len(wav) > max_samples) else wav
            text = args.text
            txt_emb = text_branch.embed(text)

            if args.method == "late":
                ac_p = _softmax_np(_acoustic_logits(ac_model, wav_ac, device, amp_dtype))
                tx_p = _softmax_np(_text_head_logits(text_ckpt, txt_emb, device))
                probs = (ac_p + tx_p) / 2 if args.late_method == "softmax_avg" else None
                if probs is None:  # logit_mean
                    al = _acoustic_logits(ac_model, wav_ac, device, amp_dtype)
                    tl = _text_head_logits(text_ckpt, txt_emb, device)
                    probs = _softmax_np((al + tl) / 2)
            else:  # early
                ac_emb = _acoustic_embed(ac_adapter, wav_ac, device, amp_dtype)
                probs = _early_head_probs(fusion_ckpt, ac_emb, txt_emb, device)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {ap}: {e!r}", flush=True)
            results.append({"audio": ap, "error": repr(e)})
            continue

        top = int(probs.argmax())
        rec = {
            "audio": ap, "method": args.method,
            "transcript": text,
            "pred": names.get(top, str(top)), "pred_index": top,
            "confidence": round(float(probs[top]), 4),
            "probs": {names.get(i, str(i)): round(float(probs[i]), 4) for i in range(len(probs))},
        }
        results.append(rec)
        if not args.json:
            dist = "  ".join(f"{k}={v:.3f}" for k, v in rec["probs"].items())
            print(f"{ap}\n  STT: {text!r}\n  → {rec['pred']}  (conf={rec['confidence']:.3f}, {args.method})  | {dist}",
                  flush=True)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
