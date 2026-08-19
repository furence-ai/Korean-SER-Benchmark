"""Cross-attention fusion 학습 — on-the-fly 듀얼 인코더 (음향 frame ↔ RoBERTa token).

음향 백본(원본 동결 or full-FT ckpt 백본, 동결) + RoBERTa(동결)를 매 배치 forward(no_grad) →
시퀀스(프레임/토큰)를 CrossAttnFusion 에 넣어 그 모듈만 학습. 별도 추출/캐시 없음(시퀀스 on-the-fly).
결과 → registry(modality=fusion, method=cross) + test_logits.npz (late 비교용).

실행:
    uv run python -m ser.fusion.train_cross --backbone facebook/wav2vec2-large-xlsr-53     # 원본(동결) 음향
    uv run python -m ser.fusion.train_cross --acoustic-ckpt checkpoints/full_ft/<run>/best.pt  # full-FT 음향 백본
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import ser.backbones  # noqa: F401  어댑터 self-register
from ser.backbones import resolve_adapter
from ser.audio import SAMPLE_RATE, load_audio_16k_mono
from ser.config import add_config_arg, backbone_dirname, dump_config, load_config, print_config
from ser.loader import class_weights
from ser.engine import build_model_from_ckpt, frames_and_mask
from ser.heads import amp_autocast
from ser.labels import resolve_label_subset
from ser.metrics import evaluate_logits, print_final_report
from ser.registry import log_fusion
from ser.rundir import make_run_dir, set_seed
from ser.fusion.cross_attn import CrossAttnFusion


class CrossModalDataset(Dataset):
    """발화별 (wav, text, dense_label, audio_path) — manifest + transcript join."""

    def __init__(self, manifest: Path, transcript: Path, max_sec: float, label_map: dict | None):
        txt = {json.loads(l)["audio"]: json.loads(l)["text"]
               for l in transcript.open(encoding="utf-8") if l.strip()}
        self.items = []
        for l in manifest.open(encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            lab = int(d["label"])
            if label_map is not None and lab not in label_map:
                continue
            if d["audio"] not in txt:
                continue
            self.items.append({"audio": d["audio"], "text": txt[d["audio"]],
                               "label": label_map[lab] if label_map else lab})
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.labels = [it["label"] for it in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav = load_audio_16k_mono(it["audio"])
        if len(wav) > self.max_samples:
            wav = wav[:self.max_samples]
        return torch.from_numpy(np.asarray(wav, dtype=np.float32)), it["text"], it["label"], it["audio"]


def collate_cross(batch):
    wavs, texts, labels, paths = zip(*batch)
    T = max(w.shape[0] for w in wavs)
    wav = torch.zeros(len(wavs), T)
    mask = torch.ones(len(wavs), T, dtype=torch.bool)   # True=pad
    for i, w in enumerate(wavs):
        wav[i, :w.shape[0]] = w
        mask[i, :w.shape[0]] = False
    return wav, mask, list(texts), torch.tensor(labels), list(paths)


@torch.no_grad()
def _encode(ctx, wav, pad_mask, texts):
    """on-the-fly: 음향→frames(B,Ta,Da)+mask, 텍스트→tokens(B,Tt,Dt)+mask. (both True=pad)

    ctx = (adapter, normalize, roberta, tok, device, amp_dtype, max_len)
    """
    adapter, normalize, roberta, tok, device, amp_dtype, max_len = ctx
    with amp_autocast(device, amp_dtype):
        frames, ac_mask = frames_and_mask(adapter, wav.to(device), pad_mask.to(device), normalize)
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with amp_autocast(device, amp_dtype):
        tokens = roberta(**enc).last_hidden_state
    tx_mask = enc["attention_mask"] == 0
    return frames.float(), ac_mask, tokens.float(), tx_mask


def _eval(model, loader, ctx, return_logits=False):
    model.eval()
    ls, lb, ps = [], [], []
    with torch.no_grad():
        for wav, pad_mask, texts, labels, paths in loader:
            f, am, t, tm = _encode(ctx, wav, pad_mask, texts)
            logits = model(f, am, t, tm)
            ls.append(logits.float().cpu()); lb.append(labels); ps += list(paths)
    all_logits = torch.cat(ls)
    out = evaluate_logits(all_logits, torch.cat(lb))
    if return_logits:
        out["logits"] = all_logits.numpy(); out["paths"] = ps
    return out


def run_cross_fusion(cfg: dict, acoustic_ckpt: Path | None, run_name: str | None, output: Path | None) -> Path:
    set_seed(cfg["seed"])
    device = cfg["device"]
    num_classes, label_map, label_to_eng = resolve_label_subset(cfg.get("keep_labels"))

    # 음향 인코더 (동결): 원본 or full-FT ckpt 백본
    if acoustic_ckpt:
        ck = torch.load(acoustic_ckpt, map_location=device, weights_only=False)
        adapter = build_model_from_ckpt(ck, device).adapter
        acoustic_id = f"{backbone_dirname(ck.get('backbone'))}-ckpt"
    else:
        adapter = resolve_adapter(cfg["backbone"]).load(cfg["backbone"], freeze_backbone=True)
        acoustic_id = f"{backbone_dirname(cfg['backbone'])}-원본"
    adapter = adapter.to(device).eval()
    for p in adapter.parameters():
        p.requires_grad = False
    normalize = adapter.normalize_input

    # 텍스트 인코더 (동결): RoBERTa
    from transformers import AutoModel, AutoTokenizer
    text_model = cfg.get("text_model", "klue/roberta-large")
    tok = AutoTokenizer.from_pretrained(text_model)
    roberta = AutoModel.from_pretrained(text_model).to(device).eval()
    for p in roberta.parameters():
        p.requires_grad = False
    max_len = int(cfg.get("text_max_len", 64))
    amp = cfg["amp_dtype"]
    enc_args = (adapter, normalize, roberta, tok, device, amp, max_len)  # _encode 앞부분 고정 인자

    model = CrossAttnFusion(adapter.embedding_dim, roberta.config.hidden_size,
                            hidden=cfg["hidden_dim"], num_classes=num_classes, dropout=cfg["dropout"]).to(device)
    run_dir = make_run_dir(Path(cfg["out_dir"]), "fusion_cross", run_name,
                           f"{acoustic_id}_x_{backbone_dirname(text_model)}") if output is None else output
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cross] 음향={acoustic_id}(동결) ↔ 텍스트={text_model}(동결) → {run_dir}", flush=True)

    mdir, st, sv, ste = Path(cfg["manifest_dir"]), cfg["split_train"], cfg["split_val"], cfg["split_test"]
    tdir = Path("data/transcripts")
    mk = lambda s, sh: DataLoader(  # noqa: E731
        CrossModalDataset(mdir / f"{s}.jsonl", tdir / f"{s}.jsonl", cfg["full_ft_max_audio_sec"], label_map),
        batch_size=cfg["full_ft_batch_size"], shuffle=sh, num_workers=cfg["num_workers"],
        collate_fn=collate_cross, pin_memory=True)
    train_loader, val_loader, test_loader = mk(st, True), mk(sv, False), mk(ste, False)
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)

    weights = (class_weights(torch.tensor(train_loader.dataset.labels), num_classes).to(device)
               if cfg["use_class_weights"] else None)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("head_lr", 5e-4), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    dump_config(cfg, run_dir / "resolved_config.yaml")   # 다른 run 과 동일 산출물 일관성
    best_f1, no_imp, patience = -1.0, 0, cfg.get("early_stop_patience", 0)
    best_path = run_dir / "best.pt"
    history: list[dict] = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot, corr, n = 0.0, 0, 0
        for wav, pad_mask, texts, labels, _ in tqdm(train_loader, desc=f"ep {epoch:03d}", leave=False, mininterval=2.0):
            f, am, t, tm = _encode(enc_args, wav, pad_mask, texts)
            labels = labels.to(device)
            with amp_autocast(device, amp):
                logits = model(f, am, t, tm)
                loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * labels.size(0); corr += (logits.argmax(-1) == labels).sum().item(); n += labels.size(0)
        sched.step()
        val = _eval(model, val_loader, enc_args)
        history.append({"epoch": epoch, "train_loss": tot / n, "train_acc": corr / n,
                        "val_acc": val["acc"], "val_macro_f1": val["macro_f1"]})
        print(f"[ep {epoch:03d}] train loss={tot/n:.4f} acc={corr/n:.4f} | val macroF1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, no_imp = val["macro_f1"], 0
            torch.save({"mode": "fusion_cross", "model_state": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_f1, "acoustic_id": acoustic_id, "text_model": text_model,
                        "num_classes": num_classes}, best_path)
        else:
            no_imp += 1
        if patience and no_imp >= patience:
            print(f"[early-stop] epoch {epoch} (best={best_f1:.4f})", flush=True); break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    report_path = run_dir / "final_report.txt"
    report_path.write_text(f"[best] val macroF1={best_f1:.4f}\n", encoding="utf-8")
    final_val = _eval(model, val_loader, enc_args)
    print_final_report("Internal Val (best ckpt)", final_val, num_classes, label_to_eng, out_path=report_path)
    (run_dir / "val_metrics.json").write_text(
        json.dumps({k: v for k, v in final_val.items() if k not in ("preds", "labels", "logits")}, indent=2), encoding="utf-8")
    if cfg["eval_on_test"]:
        tm = _eval(model, test_loader, enc_args, return_logits=True)
        print_final_report(f"Cross-attn fusion TEST ({cfg.get('split_test', 'test')})", tm, num_classes, label_to_eng,
                           out_path=report_path)
        (run_dir / "test_metrics.json").write_text(
            json.dumps({k: v for k, v in tm.items() if k not in ("preds", "labels", "logits", "paths")}, indent=2), encoding="utf-8")
        np.savez(run_dir / "test_logits.npz", logits=tm["logits"], labels=tm["labels"], paths=np.array(tm["paths"]))
        log_fusion("cross", acoustic_id, backbone_dirname(text_model), labels=tm["labels"],
                   preds=tm["logits"].argmax(1), run_dir=run_dir)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[saved] {best_path}", flush=True)
    return run_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_arg(p)
    p.add_argument("--acoustic-ckpt", type=Path, default=None, help="full-FT 음향 best.pt (백본 동결 사용). 없으면 --backbone 원본")
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--text-model", dest="text_model", type=str, default=None)
    p.add_argument("--run-name", dest="run_name", type=str, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--head-lr", dest="head_lr", type=float, default=None)
    p.add_argument("--keep-labels", dest="keep_labels", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    cfg = load_config(args)
    if not torch.cuda.is_available() and cfg.get("device") == "cuda":
        cfg["device"] = "cpu"
    print_config(cfg)
    run_cross_fusion(cfg, args.acoustic_ckpt, args.run_name, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
