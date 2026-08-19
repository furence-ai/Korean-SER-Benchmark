"""KLUE-RoBERTa 로 전사 텍스트 → frozen 임베딩 .npz (EmbeddingDataset 포맷).

transcript(data/transcripts/{split}.jsonl: {audio,text}) + manifest({audio,label,speaker}) 를
audio 경로로 join → keep_labels 필터/dense remap → RoBERTa mean-pool 임베딩 → npz 저장.
출력 포맷은 acoustic 캐시와 동일(feats/labels/speakers/paths) → ser.engine.run_head_cached 가 학습.

공유 파라미터(keep_labels/manifest_dir/splits/device/dataset)는 config 단일 소스 (CLI override).

실행:
    uv run python -m ser.text.extract_embeddings                       # config 의 split들
    uv run python -m ser.text.extract_embeddings --transcript-dir data/transcripts_stt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ser.cli import add_common_overrides, config_splits, finalize_device, resolve_dataset_ctx
from ser.config import load_config
from ser.io import load_split_jsonl, load_transcript_map


def _load_split(manifest_dir: Path, transcript_dir: Path, split: str, label_map: dict[int, int]):
    """manifest(label/speaker) + transcript(text) join → (texts, labels(dense), speakers, paths)."""
    txt = load_transcript_map(transcript_dir, split)
    texts, labels, speakers, paths = [], [], [], []
    for r in load_split_jsonl(manifest_dir, split):
        lab = int(r["label"])
        if lab not in label_map:        # keep_labels 필터 (label_map 키 = 유지할 원본 라벨)
            continue
        if r["audio"] not in txt:
            continue
        texts.append(txt[r["audio"]])
        labels.append(label_map[lab])
        speakers.append(r.get("speaker", ""))
        paths.append(r["audio"])
    return texts, np.array(labels, dtype=np.int64), np.array(speakers), np.array(paths)


@torch.no_grad()
def _embed(texts: list[str], model, tok, device: str, max_len: int, batch_size: int) -> np.ndarray:
    """mean-pool (attention mask 가중) 임베딩 (N, D)."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed", unit="batch"):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state                       # (B, T, D)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)      # (B, T, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--model", default="klue/roberta-large")
    p.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    p.add_argument("--out-dir", type=Path, default=None, help="기본 data/embeddings/{model_basename}")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    cfg = finalize_device(load_config(args))
    device = cfg["device"]
    ctx = resolve_dataset_ctx(cfg)

    from transformers import AutoModel, AutoTokenizer
    out_dir = args.out_dir or (Path("data/embeddings") / args.model.split("/")[-1])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    for split in config_splits(cfg):
        texts, labels, speakers, paths = _load_split(Path(cfg["manifest_dir"]), args.transcript_dir,
                                                     split, ctx.label_map)
        print(f"[{split}] {len(texts)}개 텍스트 임베딩 추출 (D={model.config.hidden_size})", flush=True)
        feats = _embed(texts, model, tok, device, args.max_len, args.batch_size)
        np.savez(out_dir / f"{split}.npz", feats=feats.astype(np.float32),
                 labels=labels, speakers=speakers, paths=paths)
        print(f"  → {out_dir / f'{split}.npz'}  feats={feats.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
