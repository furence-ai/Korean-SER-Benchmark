"""emotion2vec 백본 어댑터 (funasr). Emotion2vecFullFT 의 백본-특정 부분만 이식.

pooling·head·파형정규화·mask resample 은 ser.engine.BackboneWithHead 가 담당.
이 파일은 (1) funasr 로 백본 로드 (2) frame feature 추출 (3) CNN/블록 동결만 책임진다.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from .base import BackboneAdapter, register_adapter


def _detect_embedding_dim(backbone: nn.Module) -> int:
    """funasr 버전마다 다른 backbone embed_dim 위치를 best-effort로 찾음."""
    candidates = [
        "encoder_embed_dim", "embed_dim", "cfg.encoder_embed_dim", "cfg.embed_dim",
        "encoder.embed_dim", "encoder.embedding_dim", "config.hidden_size",
    ]
    for path in candidates:
        obj, ok = backbone, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, int):
            return obj
    raise RuntimeError("백본 embedding dim 자동 감지 실패. embedding_dim 을 명시적으로 넘기세요.")


def _backbone_normalize_flag(backbone: nn.Module) -> bool:
    """backbone.cfg.normalize (입력 파형 표준화 여부). 못 찾으면 True (emotion2vec 기본)."""
    cfg = getattr(backbone, "cfg", None)
    if cfg is None:
        return True
    val = getattr(cfg, "normalize", None)
    if val is None and hasattr(cfg, "get"):
        val = cfg.get("normalize", True)
    return True if val is None else bool(val)


@register_adapter("emotion2vec")
class Emotion2vecAdapter(BackboneAdapter):
    """funasr emotion2vec 백본 래퍼. extract_features/freeze 만 제공."""

    # native zero-shot 9-class 분류기의 라벨 순서 (m.generate(...)["scores"] 순서).
    # dump_pretrained_logits 가 데이터셋 라벨공간(label_to_english)과 매칭해 인덱스를 동적 계산.
    NATIVE_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "other", "sad", "surprise", "unk"]

    def __init__(self, backbone: nn.Module, embedding_dim: int, normalize_input: bool,
                 freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.normalize_input = normalize_input
        self.freeze_backbone = freeze_backbone

    @classmethod
    def load(cls, backbone_id, *, freeze_backbone=False, freeze_cnn=False,
             freeze_first_n_layers=0, gradient_checkpointing=False) -> "Emotion2vecAdapter":
        from funasr import AutoModel

        wrapper = AutoModel(model=backbone_id, hub="hf", disable_update=True)
        backbone = wrapper.model
        emb = _detect_embedding_dim(backbone)
        norm = _backbone_normalize_flag(backbone)
        self = cls(backbone, emb, norm, freeze_backbone)
        print(f"[adapter:emotion2vec] embedding_dim={emb} normalize_input={norm}", flush=True)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in backbone.parameters())
            print(f"[freeze] backbone 전체 동결 ({total/1e6:.1f}M) — head만 학습 (on-the-fly)")
        elif freeze_cnn or freeze_first_n_layers > 0:
            self.freeze(freeze_cnn, freeze_first_n_layers)
        if gradient_checkpointing and not freeze_backbone:
            self._try_enable_gradient_checkpointing()
        return self

    def train(self, mode: bool = True) -> "Emotion2vecAdapter":
        """동결 backbone은 항상 eval 유지 (deterministic feature)."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def freeze(self, cnn: bool, n_layers: int) -> None:
        """CNN(local_encoder) + 하위 n개 트랜스포머 블록(blocks.{i}) 동결."""
        n_cnn = n_blk = 0
        for name, p in self.backbone.named_parameters():
            if cnn and "local_encoder" in name:
                p.requires_grad = False
                n_cnn += 1
            elif any(name.startswith(f"blocks.{i}.") for i in range(n_layers)):
                p.requires_grad = False
                n_blk += 1
        if cnn and n_cnn == 0:
            raise RuntimeError(
                "freeze_cnn=True 인데 'local_encoder' 파라미터를 못 찾음 — 백본 이름 규약 확인 필요."
            )
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[freeze] CNN={cnn}(params {n_cnn}), first_{n_layers}_blocks(params {n_blk}) "
              f"→ frozen {frozen/1e6:.1f}M / {total/1e6:.1f}M", flush=True)

    def _try_enable_gradient_checkpointing(self) -> None:
        for attr in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
            fn = getattr(self.backbone, attr, None)
            if callable(fn):
                fn()
                print(f"[grad-ckpt] enabled via backbone.{attr}()")
                return
        if hasattr(self.backbone, "gradient_checkpointing"):
            self.backbone.gradient_checkpointing = True
            print("[grad-ckpt] set backbone.gradient_checkpointing=True")
            return
        print("[grad-ckpt] WARN: backbone이 checkpointing 미지원 — 무시됨")

    def extract_features(self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """raw wav → (frames [B,T,D], mask [B,T] or None). 동결이면 no_grad."""
        grad_ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with grad_ctx:
            if hasattr(self.backbone, "extract_features"):
                try:
                    out = self.backbone.extract_features(source=wav, padding_mask=padding_mask)
                except TypeError:
                    out = self.backbone.extract_features(wav)
            else:
                try:
                    out = self.backbone(source=wav, padding_mask=padding_mask, features_only=True)
                except TypeError:
                    out = self.backbone(wav)
        if isinstance(out, dict):
            x = out.get("x", out.get("features", out.get("last_hidden_state")))
            mask = out.get("padding_mask", padding_mask)
        elif isinstance(out, tuple):
            x = out[0]
            mask = out[1] if len(out) > 1 else padding_mask
        else:
            x, mask = out, padding_mask
        if x is None:
            raise RuntimeError("백본 출력 형태 인식 실패 — extract_features 출력 확인 필요.")
        return x, mask
