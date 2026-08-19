"""학습된 체크포인트로 '테스트만' 수행 (재학습 없음).

run_all.sh 는 학습 후 자동으로 test 평가까지 하지만(각 run 폴더의 test_metrics.json/
test_logits.npz/final_report.txt 가 그 결과), 이 스크립트는 이미 학습된 best.pt 만 받아
held-out 테스트 split 추론 → 동일 산출물(test_metrics.json / test_logits.npz / final_report.txt)
을 저장한다. 다른 split·manifest 로 재평가하거나, 산출물이 유실됐을 때 복구하는 용도.

ckpt 의 mode 로 평가 경로 자동 분기:
    head_only_otf / full_ft   → 음향 raw-audio (backbone+head 재구성, build_model_from_ckpt)
    head_cached               → 캐시 임베딩 head (--emb-dir 필요; text / early-fusion)
    fusion_cross              → 미지원 (듀얼 인코더 전용 구조 → ser.fusion.train_cross 로 평가)

예:
    # 음향 모델 (raw audio) — 한 개 / 여러 개
    uv run python -m scripts.evaluate --ckpt runs/all/acoustic/em_head
    uv run python -m scripts.evaluate --ckpt .../acoustic/em_head .../acoustic/w2v_full
    # 캐시-임베딩 head (text / early-fusion) — 해당 임베딩 폴더 지정
    uv run python -m scripts.evaluate --ckpt .../text/roberta --emb-dir .../emb/roberta-large
    # 다른 split 으로 평가
    uv run python -m scripts.evaluate --ckpt .../acoustic/em_full --split test_di --manifest-dir data/manifests

--ckpt 는 best.pt 파일 또는 그 run 폴더(폴더면 best.pt 자동) 둘 다 허용.
기본 저장 위치는 ckpt 폴더(학습 때 산출물 자리에 덮어씀). --out-dir 로 따로 뺄 수 있다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ser.config import add_config_arg, load_config
from ser.labels import resolve_label_subset
from ser.metrics import print_final_report


def _resolve_ckpt(path: Path) -> Path:
    """파일이면 그대로, 폴더면 best.pt 로 해석."""
    p = Path(path)
    if p.is_dir():
        p = p / "best.pt"
    if not p.is_file():
        raise SystemExit(f"체크포인트 없음: {p}")
    return p


def _save_outputs(out_dir: Path, tm: dict, paths, num_classes: int,
                  label_to_english: dict, split: str) -> None:
    """eval 결과 → final_report.txt(append) / test_metrics.json / test_logits.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "final_report.txt"
    print_final_report(f"Held-out TEST ({split}) — eval-only", tm, num_classes,
                       label_to_english, out_path=report_path)
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in tm.items() if k not in ("preds", "labels", "logits")}, f, indent=2)
    np.savez(out_dir / "test_logits.npz", logits=tm["logits"], labels=tm["labels"],
             paths=np.array(paths))
    print(f"[saved] {out_dir}/test_metrics.json · test_logits.npz · final_report.txt", flush=True)


def _eval_acoustic(ckpt: dict, cfg: dict, split: str) -> tuple:
    """raw-audio 백본+head 재구성 → 테스트 추론. (logits/labels/paths 포함 dict, paths) 반환."""
    from ser.loader import RawAudioDataset, collate_pad_audio
    from ser.engine import build_model_from_ckpt, eval_loader

    device = cfg["device"]
    keep = ckpt.get("keep_labels", cfg.get("keep_labels"))
    num_classes, label_map, label_to_english = resolve_label_subset(keep)
    md = Path(cfg["manifest_dir"])
    ds = RawAudioDataset(md / f"{split}.jsonl", max_sec=cfg["full_ft_max_audio_sec"],
                         random_crop=False, label_map=label_map)
    print(f"[data] {split}: {len(ds)}개 | classes={list(label_to_english.values())}", flush=True)
    loader = DataLoader(ds, batch_size=cfg["full_ft_batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_pad_audio, pin_memory=True)
    model = build_model_from_ckpt(ckpt, device)
    tm = eval_loader(model, loader, device, cfg.get("amp_dtype", "bf16"), return_logits=True)
    paths = [it["audio"] for it in ds.items]
    return tm, paths, num_classes, label_to_english


def _eval_cached(ckpt: dict, cfg: dict, split: str, emb_dir: Path) -> tuple:
    """캐시 임베딩(.npz) → head 추론. emb_dir/{split}.npz 필요."""
    from ser.loader import EmbeddingDataset, FrameEmbeddingDataset, collate_pad_frames, is_frame_level_npz
    from ser.engine import eval_head_loader
    from ser.heads import build_head

    device = cfg["device"]
    npz = Path(emb_dir) / f"{split}.npz"
    if not npz.is_file():
        raise SystemExit(f"임베딩 없음: {npz}  (--emb-dir 가 맞는지 확인)")
    keep = ckpt.get("keep_labels", cfg.get("keep_labels"))
    num_classes, _, label_to_english = resolve_label_subset(keep)
    is_frame = is_frame_level_npz(npz)
    ds = (FrameEmbeddingDataset if is_frame else EmbeddingDataset)(npz)
    print(f"[data] {split}: {len(ds)}개 | embedding_dim={ds.embedding_dim} (frame={is_frame})", flush=True)
    head = build_head(ckpt["head_type"], ckpt["embedding_dim"], ckpt["num_classes"],
                      ckpt["hidden_dim"], ckpt["dropout"]).to(device)
    head.load_state_dict(ckpt["model_state"])
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 256), shuffle=False,
                        num_workers=cfg["num_workers"],
                        collate_fn=collate_pad_frames if is_frame else None, pin_memory=True)
    tm = eval_head_loader(head, loader, device, is_frame, return_logits=True)
    return tm, ds.paths, num_classes, label_to_english


def _evaluate_one(ckpt_path: Path, cfg: dict, split: str, emb_dir: Path | None,
                  out_dir: Path | None) -> bool:
    ckpt = torch.load(ckpt_path, map_location=cfg["device"], weights_only=False)
    mode = ckpt.get("mode")
    print(f"\n######## eval: {ckpt_path}  (mode={mode}) ########", flush=True)

    if mode in ("full_ft", "head_only_otf"):
        tm, paths, nc, l2e = _eval_acoustic(ckpt, cfg, split)
    elif mode == "head_cached":
        if emb_dir is None:
            raise SystemExit(f"head_cached 모델은 --emb-dir 필요 ({ckpt_path}). "
                             f"예: text→emb/roberta-large, early→emb/early_<src>")
        tm, paths, nc, l2e = _eval_cached(ckpt, cfg, split, emb_dir)
    elif mode == "fusion_cross":
        print(f"[skip] cross-attn 듀얼 인코더는 이 스크립트로 평가 불가 → "
              f"ser.fusion.train_cross 의 전용 평가 경로 사용 ({ckpt_path})", flush=True)
        return False
    else:
        raise SystemExit(f"알 수 없는 mode={mode!r} ({ckpt_path})")

    target = out_dir or ckpt_path.parent
    _save_outputs(target, tm, paths, nc, l2e, split)
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="학습된 체크포인트 테스트-only 평가")
    add_config_arg(p)
    p.add_argument("--ckpt", type=Path, nargs="+", required=True,
                   help="best.pt 파일 또는 run 폴더 (여러 개 가능)")
    p.add_argument("--split", type=str, default=None, help="평가 split (기본 config split_test=test_di)")
    p.add_argument("--emb-dir", dest="emb_dir", type=Path, default=None,
                   help="head_cached(text/early) 모델용 임베딩 폴더 ({split}.npz 있는 곳)")
    p.add_argument("--out-dir", dest="out_dir_cli", type=Path, default=None,
                   help="결과 저장 폴더 (기본: 각 ckpt 폴더). 여러 ckpt 면 무시하고 각 폴더에 저장 권장")
    p.add_argument("--manifest-dir", dest="manifest_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args)
    if not torch.cuda.is_available() and cfg.get("device") == "cuda":
        cfg["device"] = "cpu"
        print("[device] cuda 없음 → cpu 폴백", flush=True)
    split = args.split or cfg.get("split_test", "test_di")

    ckpts = [_resolve_ckpt(c) for c in args.ckpt]
    if args.out_dir_cli and len(ckpts) > 1:
        print("[warn] --out-dir 는 ckpt 1개일 때만 사용 — 여러 개면 각 ckpt 폴더에 저장", flush=True)
    out_dir = args.out_dir_cli if (args.out_dir_cli and len(ckpts) == 1) else None

    ok = 0
    for cp in ckpts:
        if _evaluate_one(cp, cfg, split, args.emb_dir, out_dir):
            ok += 1
    print(f"\n######## 완료 — {ok}/{len(ckpts)} 평가됨 (split={split}) ########", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
