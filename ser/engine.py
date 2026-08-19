"""학습/평가 엔진 (백본 무관).

BackboneWithHead: adapter(백본) + head 를 하나로 묶어 forward — 파형정규화 → adapter.extract_features
→ mask resample/None → frame-head(pool 내부) 또는 utterance(masked mean pool) → head.
어느 백본이든 동일. 백본-특정 부분은 전부 adapter 안에 있다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ser.backbones import resolve_adapter

from .config import backbone_dirname, dump_config
from .loader import (
    EmbeddingDataset,
    FrameEmbeddingDataset,
    RawAudioDataset,
    class_weights,
    collate_pad_audio,
    collate_pad_frames,
    is_frame_level_npz,
    person_id_from_speaker,
)
from .heads import HEAD_CUSTOM_UTTERANCE, HEAD_OFFICIAL_FRAME, amp_autocast, build_head, is_frame_head
from .io import build_ckpt_meta
from .labels import resolve_label_subset
from .metrics import evaluate_logits, plot_history, print_final_report
from .rundir import (
    capture_rng,
    resolve_resume,
    restore_rng,
    set_seed,
    warmup_cosine_lr,
)


def normalize_waveform(wav: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """공식 F.layer_norm(source) 의 batch/padding 안전판 — valid 구간 평균/분산으로 표준화."""
    if padding_mask is None:
        mean = wav.mean(dim=1, keepdim=True)
        var = wav.var(dim=1, keepdim=True, unbiased=False)
    else:
        valid = (~padding_mask).to(wav.dtype)
        n = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (wav * valid).sum(dim=1, keepdim=True) / n
        var = (((wav - mean) ** 2) * valid).sum(dim=1, keepdim=True) / n
    return (wav - mean) / torch.sqrt(var + 1e-5)


def masked_mean_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,T,D) + mask(True=pad) → (B,D) masked mean over T."""
    valid = (~mask).unsqueeze(-1).to(features.dtype)
    return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def frames_and_mask(adapter, wav: torch.Tensor, padding_mask: torch.Tensor | None,
                    normalize_input: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """파형정규화 → adapter.extract_features → mask resample/None 처리. (frames (B,T,D), mask (B,T) True=pad).

    BackboneWithHead.forward 와 임베딩 추출(scripts.extract_acoustic_embeddings)이 공유 — 동일 로직.
    """
    if normalize_input:
        wav = normalize_waveform(wav, padding_mask)
    features, mask = adapter.extract_features(wav, padding_mask)
    if mask is not None and mask.shape[1] != features.shape[1]:   # audio rate → frame rate resample
        T_f = features.shape[1]
        idx = (torch.arange(T_f, device=mask.device) * (mask.shape[1] / T_f)).long().clamp(max=mask.shape[1] - 1)
        mask = mask[:, idx]
    if mask is None:
        mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=features.device)
    return features, mask


class BackboneWithHead(nn.Module):
    """adapter(extract_features) + head. 백본 무관 forward."""

    def __init__(self, adapter, head: nn.Module, head_is_frame: bool, normalize_input: bool):
        super().__init__()
        self.adapter = adapter
        self.head = head
        self.head_is_frame = head_is_frame
        self.normalize_input = normalize_input

    def forward(self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        features, mask = frames_and_mask(self.adapter, wav, padding_mask, self.normalize_input)
        if self.head_is_frame:
            return self.head(features, padding_mask=mask)
        return self.head(masked_mean_pool(features, mask))


def build_model_from_ckpt(ckpt: dict, device: str) -> BackboneWithHead:
    """ckpt 메타로 BackboneWithHead 재구성 + weight 로드. (full_ft/head_only_otf 체크포인트용)

    구 Emotion2vecFullFT 체크포인트(`backbone.*`/`head.*`)도 `adapter.backbone.*` 로 remap 해 로드.
    """
    backbone_id = ckpt["backbone"]
    head_type = ckpt.get("head_type", "official-frame-mlp")
    mode = ckpt.get("mode", "full_ft")
    adapter_cls = resolve_adapter(backbone_id)
    adapter = adapter_cls.load(backbone_id, freeze_backbone=(mode == "head_only_otf"))
    head = build_head(head_type, ckpt.get("embedding_dim", adapter.embedding_dim),
                      ckpt["num_classes"], ckpt["hidden_dim"], ckpt["dropout"])
    model = BackboneWithHead(adapter, head, is_frame_head(head_type), adapter.normalize_input).to(device)

    state = ckpt["model_state"]
    if any(k.startswith("backbone.") for k in state) and not any(k.startswith("adapter.") for k in state):
        # 구 Emotion2vecFullFT → 새 BackboneWithHead 키 remap
        state = {(f"adapter.{k}" if k.startswith("backbone.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def eval_loader(model: nn.Module, loader, device: str, amp_dtype: str = "bf16",
                return_logits: bool = False) -> dict:
    """raw-audio loader → 전체 logit 모아 evaluate_logits. backbone forward 는 amp_dtype(기본 bf16).

    return_logits=True 면 결과에 'logits'(N,C numpy) 추가 (test_logits.npz / fusion 용).
    """
    model.eval()
    ls, lb = [], []
    for wav, pad_mask, labels in loader:
        wav = wav.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        with amp_autocast(device, amp_dtype):
            logits = model(wav, padding_mask=pad_mask)
        ls.append(logits.float().cpu())
        lb.append(labels)
    all_logits = torch.cat(ls)
    out = evaluate_logits(all_logits, torch.cat(lb))
    if return_logits:
        out["logits"] = all_logits.numpy()
    return out


def _dataloader(ds, bs, nw, shuffle, prefetch=False):
    persistent = nw > 0
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=nw, collate_fn=collate_pad_audio,
        pin_memory=True, persistent_workers=persistent,
        prefetch_factor=(4 if (persistent and prefetch) else None),
    )


def run_acoustic(cfg: dict, run_name: str | None = None, output: Path | None = None,
                 freeze_backbone: bool = False, resume: Path | None = None) -> Path:
    """raw audio → backbone(adapter) → head 학습 (백본 무관). run_dir 반환.

    freeze_backbone=False: backbone+head 전부 학습 (full FT, mode=full_ft).
    freeze_backbone=True : backbone 동결 + head만 학습 (head_only_otf).
    """
    set_seed(cfg["seed"])
    device = cfg["device"]
    mode = "head_only_otf" if freeze_backbone else "full_ft"
    head_type = cfg.get("head_type", HEAD_OFFICIAL_FRAME)
    suffix = f"{backbone_dirname(cfg['backbone'])}_{head_type}"   # 폴더명에 백본 포함
    resume_ckpt, run_dir = resolve_resume(resume, cfg, mode, run_name, output, suffix, device)
    cfg["mode"] = mode   # registry 가 resolved_config 에서 읽도록 (run_dir 폴더명에 의존 X)
    dump_config(cfg, run_dir / "resolved_config.yaml")
    print(f"[run] mode={mode} backbone={cfg['backbone']} → {run_dir}", flush=True)

    num_classes, label_map, label_to_english = resolve_label_subset(cfg.get("keep_labels"))
    print(f"[classes] num_classes={num_classes} names={list(label_to_english.values())} "
          f"keep_labels={cfg.get('keep_labels')}", flush=True)

    md = Path(cfg["manifest_dir"])
    st, sv, ste = cfg.get("split_train", "train_si"), cfg.get("split_val", "val_si"), cfg.get("split_test", "test_si")
    msec = cfg["full_ft_max_audio_sec"]
    train_ds = RawAudioDataset(md / f"{st}.jsonl", max_sec=msec, random_crop=True, label_map=label_map)
    internal_val_ds = RawAudioDataset(md / f"{sv}.jsonl", max_sec=msec, random_crop=False, label_map=label_map)
    test_ds = RawAudioDataset(md / f"{ste}.jsonl", max_sec=msec, random_crop=False, label_map=label_map)
    train_labels = train_ds.labels_tensor

    def _np(ds):
        return len(np.unique([person_id_from_speaker(s) for s in ds.speakers]))
    print(f"[data] train={len(train_ds)}({_np(train_ds)}명) val={len(internal_val_ds)}({_np(internal_val_ds)}명) "
          f"test={len(test_ds)}({_np(test_ds)}명) | speaker-independent ({st}/{sv}/{ste})", flush=True)

    bs, nw = cfg["full_ft_batch_size"], cfg["num_workers"]
    train_loader = _dataloader(train_ds, bs, nw, shuffle=True, prefetch=True)
    internal_val_loader = _dataloader(internal_val_ds, bs, nw, shuffle=False)
    test_loader = _dataloader(test_ds, bs, nw, shuffle=False)

    adapter = resolve_adapter(cfg["backbone"]).load(
        cfg["backbone"], freeze_backbone=freeze_backbone,
        freeze_cnn=cfg.get("full_ft_freeze_cnn", False),
        freeze_first_n_layers=cfg.get("full_ft_freeze_first_n_layers", 0),
        gradient_checkpointing=cfg.get("full_ft_gradient_checkpointing", False),
    )
    head = build_head(head_type, adapter.embedding_dim, num_classes, cfg["hidden_dim"], cfg["dropout"])
    model = BackboneWithHead(adapter, head, is_frame_head(head_type), adapter.normalize_input).to(device)
    print(f"[model] head={head.__class__.__name__} embedding_dim={adapter.embedding_dim} "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    weights = None
    if cfg["use_class_weights"]:
        weights = class_weights(train_labels, num_classes).to(device)
        print(f"[loss] class weights: {weights.tolist()}", flush=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])

    # adapter 는 백본만 감싸므로 adapter.parameters() == 백본 파라미터 (속성명 self.backbone/self.model 무관)
    backbone_params = [p for p in model.adapter.parameters() if p.requires_grad]
    # head LR: full FT 면 full_ft_lr, head-only(백본 동결)면 head_lr (없으면 full_ft_lr 폴백)
    head_lr = cfg.get("head_lr", cfg["full_ft_lr"]) if freeze_backbone else cfg["full_ft_lr"]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg["full_ft_backbone_lr"]})
    param_groups.append({"params": model.head.parameters(), "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    has_bb = bool(backbone_params)
    print(f"[optim] head_lr={head_lr}" + (f" backbone_lr={cfg['full_ft_backbone_lr']}" if has_bb else " (head-only)"),
          flush=True)

    grad_accum = cfg["full_ft_grad_accum"]
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = cfg["full_ft_warmup_steps"]
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    amp_dtype = cfg.get("amp_dtype", "bf16")

    meta = build_ckpt_meta(
        mode=mode, head_type=head_type, embedding_dim=adapter.embedding_dim,
        num_classes=num_classes, keep_labels=cfg.get("keep_labels"),
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"], backbone=cfg["backbone"],
        label_to_english=label_to_english, full_ft_max_audio_sec=msec,
    )
    best_macro_f1, epochs_no_improve = -1.0, 0
    patience = cfg.get("early_stop_patience", 0)
    history: list[dict] = []
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    global_step, start_epoch = 0, 1

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        global_step = resume_ckpt["global_step"]
        start_epoch = resume_ckpt["epoch"] + 1
        best_macro_f1 = resume_ckpt["best_macro_f1"]
        history = resume_ckpt["history"]
        restore_rng(resume_ckpt["rng"])

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"ep {epoch:03d}/{cfg['epochs']}", leave=False,
                    dynamic_ncols=True, mininterval=2.0)
        for batch_idx, (wav, pad_mask, labels) in enumerate(pbar):
            wav = wav.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_autocast(device, amp_dtype):
                logits = model(wav, padding_mask=pad_mask)
                loss = loss_fn(logits, labels) / grad_accum
            loss.backward()
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr_scale = warmup_cosine_lr(global_step, warmup, total_steps)
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * lr_scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            n = labels.size(0)
            tr_loss += loss.item() * grad_accum * n
            tr_correct += (logits.argmax(-1) == labels).sum().item()
            tr_n += n
            pbar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}", acc=f"{tr_correct/tr_n:.3f}",
                             lr=f"{optimizer.param_groups[-1]['lr']:.1e}")
        pbar.close()

        val = eval_loader(model, internal_val_loader, device, amp_dtype)
        log = {
            "epoch": epoch, "lr_head": optimizer.param_groups[-1]["lr"],
            "lr_backbone": optimizer.param_groups[0]["lr"] if has_bb else None,
            "train_loss": tr_loss / tr_n, "train_acc": tr_correct / tr_n,
            "val_loss": val["loss"], "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"],
        }
        history.append(log)
        lr_str = (f"lr_bb={log['lr_backbone']:.2e} lr_hd={log['lr_head']:.2e}"
                  if has_bb else f"lr_hd={log['lr_head']:.2e}")
        print(f"[ep {epoch:03d}] train loss={log['train_loss']:.4f} acc={log['train_acc']:.4f} | "
              f"internal_val loss={val['loss']:.4f} acc={val['acc']:.4f} macroF1={val['macro_f1']:.4f} | {lr_str}",
              flush=True)

        if val["macro_f1"] > best_macro_f1:
            best_macro_f1, epochs_no_improve = val["macro_f1"], 0
            torch.save({**meta, "model_state": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best_macro_f1}, best_path)
        else:
            epochs_no_improve += 1
        torch.save({**meta, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step, "epoch": epoch, "val_macro_f1": val["macro_f1"],
                    "best_macro_f1": best_macro_f1, "history": history, "rng": capture_rng()}, last_path)
        if patience and epochs_no_improve >= patience:
            print(f"[early-stop] internal_val macroF1 {patience} epoch 무개선 → epoch {epoch} 종료 "
                  f"(best={best_macro_f1:.4f})", flush=True)
            break

    # best ckpt 재로드 → 최종 평가
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    report_path = run_dir / "final_report.txt"
    best_ep = torch.load(best_path, map_location=device)["epoch"]
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"\n[best] epoch={best_ep} internal_val macroF1={best_macro_f1:.4f}\n")
    print(f"\n[best] epoch={best_ep} internal_val macroF1={best_macro_f1:.4f}", flush=True)

    final_val = eval_loader(model, internal_val_loader, device, amp_dtype)
    print_final_report("Internal Val (best ckpt)", final_val, num_classes, label_to_english, out_path=report_path)
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in final_val.items() if k not in ("preds", "labels", "logits")}, f, indent=2)

    if cfg["eval_on_test"]:
        tm = eval_loader(model, test_loader, device, amp_dtype, return_logits=True)
        print_final_report(f"Held-out TEST ({cfg.get('split_test', 'test')})", tm, num_classes, label_to_english,
                           out_path=report_path)
        with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in tm.items() if k not in ("preds", "labels", "logits")}, f, indent=2)
        # fusion/diagnostics 용 logit 저장 (paths 는 test_ds 순서)
        np.savez(run_dir / "test_logits.npz", logits=tm["logits"], labels=tm["labels"],
                 paths=np.array([it["audio"] for it in test_ds.items]))

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, run_dir / "history.png", title=f"{mode} — {run_dir.name}")
    print(f"\n[saved] {best_path}\n[saved] {report_path}", flush=True)

    # 결과 레지스트리 1행 (test_si 지표 + per-class F1). 실패해도 학습 산출물엔 영향 없음.
    try:
        from .registry import log_run
        log_run(run_dir, modality="acoustic")
    except Exception as e:  # noqa: BLE001
        print(f"[registry] 기록 실패 (학습엔 영향 없음): {e!r}", flush=True)
    return run_dir


@torch.no_grad()
def eval_head_loader(model, loader, device, is_frame: bool, return_logits: bool = False) -> dict:
    """캐싱 임베딩 loader → head logit 모아 evaluate_logits (backbone 없음, fp32)."""
    model.eval()
    ls, lb = [], []
    for batch in loader:
        if is_frame:
            feats, pad_mask, labels = batch
            logits = model(feats.to(device), pad_mask.to(device))
        else:
            feats, labels = batch
            logits = model(feats.to(device))
        ls.append(logits.float().cpu())
        lb.append(labels)
    all_logits = torch.cat(ls)
    out = evaluate_logits(all_logits, torch.cat(lb))
    if return_logits:
        out["logits"] = all_logits.numpy()
    return out


def run_head_cached(cfg: dict, run_name: str | None = None, output: Path | None = None,
                    modality: str = "acoustic", resume: Path | None = None) -> Path:
    """캐싱된 임베딩(.npz)에 head 만 학습 (backbone 없음). 텍스트 baseline·acoustic-cached 공용.

    cfg["emb_dir"] 하위 {split_train/val/test}.npz (feats/labels/speakers/paths) 를 읽는다.
    modality: registry 기록용 ("acoustic" | "text"). run_dir 반환.
    """
    set_seed(cfg["seed"])
    device = cfg["device"]
    mode = "head_cached"
    emb_dir = Path(cfg["emb_dir"])
    st, sv, ste = cfg.get("split_train", "train_si"), cfg.get("split_val", "val_si"), cfg.get("split_test", "test_si")
    is_frame = is_frame_level_npz(emb_dir / f"{st}.npz")
    head_name = cfg.get("head_type", HEAD_OFFICIAL_FRAME if is_frame else HEAD_CUSTOM_UTTERANCE)
    # npz 형태에 맞춰 head 자동 보정 (config head_type 이 안 맞아도)
    if is_frame and not is_frame_head(head_name):
        head_name = HEAD_OFFICIAL_FRAME
    elif not is_frame and is_frame_head(head_name):   # utterance 임베딩엔 frame head 못 씀 → utterance head
        head_name = HEAD_CUSTOM_UTTERANCE

    suffix = f"{backbone_dirname(cfg.get('backbone') or 'cached')}_{head_name}"   # 폴더명에 인코더 포함
    resume_ckpt, run_dir = resolve_resume(resume, cfg, mode, run_name, output, suffix, device)
    cfg["mode"] = mode   # registry 가 resolved_config 에서 읽도록
    dump_config(cfg, run_dir / "resolved_config.yaml")
    num_classes, _, label_to_english = resolve_label_subset(cfg.get("keep_labels"))
    print(f"[run] mode={mode} modality={modality} encoder={cfg.get('backbone')} → {run_dir}", flush=True)

    ds_cls = FrameEmbeddingDataset if is_frame else EmbeddingDataset
    collate = collate_pad_frames if is_frame else None
    train_ds, val_ds, test_ds = (ds_cls(emb_dir / f"{s}.npz") for s in (st, sv, ste))
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"embedding_dim={train_ds.embedding_dim} (frame={is_frame})", flush=True)
    bs, nw = cfg.get("batch_size", 256), cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True)

    model = build_head(head_name, train_ds.embedding_dim, num_classes, cfg["hidden_dim"], cfg["dropout"]).to(device)
    print(f"[head] {model.__class__.__name__} (head_type={head_name})", flush=True)

    weights = class_weights(train_ds.labels, num_classes).to(device) if cfg["use_class_weights"] else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    meta = build_ckpt_meta(
        mode=mode, modality=modality, head_type=head_name, embedding_dim=train_ds.embedding_dim,
        num_classes=num_classes, keep_labels=cfg.get("keep_labels"), hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"], backbone=cfg.get("backbone"), label_to_english=label_to_english,
    )
    best_macro_f1, epochs_no_improve = -1.0, 0
    patience = cfg.get("early_stop_patience", 0)
    history: list[dict] = []
    best_path, last_path, start_epoch = run_dir / "best.pt", run_dir / "last.pt", 1
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"]); optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"]); start_epoch = resume_ckpt["epoch"] + 1
        best_macro_f1 = resume_ckpt["best_macro_f1"]; history = resume_ckpt["history"]; restore_rng(resume_ckpt["rng"])

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"ep {epoch:03d}/{cfg['epochs']}", leave=False, mininterval=2.0)
        for batch in pbar:
            if is_frame:
                feats, pad_mask, labels = batch
                logits = model(feats.to(device), pad_mask.to(device)); labels = labels.to(device)
            else:
                feats, labels = batch
                logits = model(feats.to(device)); labels = labels.to(device)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            n = labels.size(0); tr_loss += loss.item() * n
            tr_correct += (logits.argmax(-1) == labels).sum().item(); tr_n += n
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{tr_correct/tr_n:.3f}")
        pbar.close(); scheduler.step()
        val = eval_head_loader(model, val_loader, device, is_frame)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tr_loss/tr_n,
                        "train_acc": tr_correct/tr_n, "val_loss": val["loss"], "val_acc": val["acc"],
                        "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"]})
        print(f"[ep {epoch:03d}] train loss={tr_loss/tr_n:.4f} acc={tr_correct/tr_n:.4f} | "
              f"val acc={val['acc']:.4f} macroF1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_macro_f1:
            best_macro_f1, epochs_no_improve = val["macro_f1"], 0
            torch.save({**meta, "model_state": model.state_dict(), "epoch": epoch, "val_macro_f1": best_macro_f1}, best_path)
        else:
            epochs_no_improve += 1
        torch.save({**meta, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "epoch": epoch, "val_macro_f1": val["macro_f1"],
                    "best_macro_f1": best_macro_f1, "history": history, "rng": capture_rng()}, last_path)
        if patience and epochs_no_improve >= patience:
            print(f"[early-stop] epoch {epoch} 종료 (best={best_macro_f1:.4f})", flush=True); break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    report_path = run_dir / "final_report.txt"
    report_path.write_text(f"[best] internal_val macroF1={best_macro_f1:.4f}\n", encoding="utf-8")
    final_val = eval_head_loader(model, val_loader, device, is_frame)
    print_final_report("Internal Val (best ckpt)", final_val, num_classes, label_to_english, out_path=report_path)
    with (run_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in final_val.items() if k not in ("preds", "labels", "logits")}, f, indent=2)
    if cfg["eval_on_test"]:
        tm = eval_head_loader(model, test_loader, device, is_frame, return_logits=True)
        print_final_report(f"Held-out TEST ({cfg.get('split_test', 'test')})", tm, num_classes, label_to_english, out_path=report_path)
        with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in tm.items() if k not in ("preds", "labels", "logits")}, f, indent=2)
        np.savez(run_dir / "test_logits.npz", logits=tm["logits"], labels=tm["labels"], paths=test_ds.paths)
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, run_dir / "history.png", title=f"{mode}/{modality} — {run_dir.name}")
    try:
        from .registry import log_run
        log_run(run_dir, modality=modality)
    except Exception as e:  # noqa: BLE001
        print(f"[registry] 기록 실패: {e!r}", flush=True)
    print(f"[saved] {best_path}", flush=True)
    return run_dir
