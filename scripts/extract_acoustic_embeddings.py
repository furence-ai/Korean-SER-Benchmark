"""음향 백본(adapter)으로 raw audio → utterance 임베딩 .npz (어느 백본이든 + keep_labels).

ser.engine 의 frames_and_mask + masked_mean_pool 를 재사용 → 학습/평가와 동일한 pooling.
출력 npz(feats/labels(dense)/speakers/paths) → ser.train_cached / ser.fusion.early 입력.

공유 파라미터(keep_labels/manifest_dir/splits/device/dataset)는 config 단일 소스 (CLI override).

실행:
    uv run python -m scripts.extract_acoustic_embeddings --backbone microsoft/wavlm-large
    uv run python -m scripts.extract_acoustic_embeddings --ckpt <run>/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ser.backbones  # 어댑터 self-register  # noqa: F401
from ser.backbones import resolve_adapter
from ser.cli import add_common_overrides, config_splits, finalize_device, resolve_dataset_ctx
from ser.config import backbone_dirname, load_config
from ser.loader import RawAudioDataset, collate_pad_audio
from ser.engine import build_model_from_ckpt, frames_and_mask, masked_mean_pool
from ser.heads import amp_autocast


@torch.no_grad()
def extract_split(adapter, manifest: Path, max_sec: float, label_map, bs: int, nw: int,
                  device: str, amp_dtype: str) -> dict:
    ds = RawAudioDataset(manifest, max_sec=max_sec, random_crop=False, label_map=label_map)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_pad_audio, pin_memory=True)
    feats = []
    for wav, pad_mask, _ in tqdm(loader, desc=manifest.stem, unit="batch"):
        wav, pad_mask = wav.to(device), pad_mask.to(device)
        with amp_autocast(device, amp_dtype):
            f, m = frames_and_mask(adapter, wav, pad_mask, adapter.normalize_input)
            pooled = masked_mean_pool(f, m)
        feats.append(pooled.float().cpu().numpy())
    return {"feats": np.concatenate(feats).astype(np.float32),
            "labels": ds.labels_tensor.numpy(), "speakers": ds.speakers,
            "paths": np.array([it["audio"] for it in ds.items])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--backbone", default=None, help="원본(동결) 백본 id. --ckpt 와 택1.")
    p.add_argument("--ckpt", type=Path, default=None, help="full-FT best.pt — 그 백본(동결)으로 임베딩 추출.")
    p.add_argument("--out-dir", type=Path, default=None, help="기본 data/embeddings/{backbone_basename}")
    p.add_argument("--max-sec", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=8)
    p.add_argument("--amp-dtype", default="bf16")
    args = p.parse_args()
    cfg = finalize_device(load_config(args))
    device = cfg["device"]

    if not args.backbone and not args.ckpt:
        p.error("--backbone (원본) 또는 --ckpt (full-FT) 중 하나는 필요")
    ctx = resolve_dataset_ctx(cfg)
    label_map = ctx.label_map
    if args.ckpt:   # full-FT 백본 (동결)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        bb = backbone_dirname(ck.get("backbone")) + "-ckpt"
        print(f"[load] adapter from ckpt {args.ckpt} (backbone={ck.get('backbone')})", flush=True)
        adapter = build_model_from_ckpt(ck, device).adapter
    else:           # 원본(동결) 백본
        bb = backbone_dirname(args.backbone)
        print(f"[load] adapter for {args.backbone} (원본 동결)", flush=True)
        adapter = resolve_adapter(args.backbone).load(args.backbone, freeze_backbone=True)
    adapter = adapter.to(device).eval()
    out_dir = args.out_dir or (Path("data/embeddings") / bb)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in config_splits(cfg):
        d = extract_split(adapter, Path(cfg["manifest_dir"]) / f"{split}.jsonl", args.max_sec, label_map,
                          args.batch_size, args.num_workers, device, args.amp_dtype)
        np.savez(out_dir / f"{split}.npz", **d)
        print(f"[{split}] feats={d['feats'].shape} → {out_dir / f'{split}.npz'}", flush=True)
    print(f"\n학습: uv run python -m ser.train_cached --emb-dir {out_dir} --backbone {args.backbone}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
