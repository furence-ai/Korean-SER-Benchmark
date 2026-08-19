"""HF wav2vec2-family(WavLM / wav2vec2) 공통 어댑터 베이스.

WavLM/wav2vec2 는 HF에서 같은 API: feature_extractor(CNN) + encoder.layers(트랜스포머),
freeze_feature_encoder(), model(input_values, attention_mask).last_hidden_state, config.hidden_size.
emotion2vec 어댑터와 동일한 BackboneAdapter 규격을 구현 → ser.engine 가 그대로 학습/평가.
"""
from __future__ import annotations

import contextlib

import torch

from .base import BackboneAdapter


def _allow_trusted_bin_load() -> None:
    """transformers 4.5x + torch<2.6 는 .bin(pickle) 로드를 CVE-2025-32434 로 막는다.

    torch 는 funasr 때문에 2.5.1 고정이고, WavLM/wav2vec2 는 신뢰된 공식 HF 모델(safetensors
    없는 경우 .bin)이라 그 체크만 우회한다. funasr/emotion2vec 경로엔 영향 없음.
    """
    try:
        import transformers.modeling_utils as _mu
        _mu.check_torch_load_is_safe = lambda *a, **k: None
    except Exception:  # noqa: BLE001
        pass


class HFWav2Vec2FamilyAdapter(BackboneAdapter):
    """WavLM/wav2vec2 공통. subclass 는 _model_cls() 만 구현(lazy import)."""

    @classmethod
    def _model_cls(cls):
        """HF 모델 클래스 반환 (subclass에서 lazy import). 예: transformers.WavLMModel."""
        raise NotImplementedError

    def __init__(self, model, embedding_dim: int, freeze_backbone: bool):
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        # wav2vec2-large / WavLM-large 는 do_normalize=True → core 가 per-utterance 파형 표준화.
        self.normalize_input = True
        self.freeze_backbone = freeze_backbone

    @classmethod
    def load(cls, backbone_id, *, freeze_backbone=False, freeze_cnn=False,
             freeze_first_n_layers=0, gradient_checkpointing=False):
        _allow_trusted_bin_load()
        model = cls._model_cls().from_pretrained(backbone_id)
        emb = model.config.hidden_size
        self = cls(model, emb, freeze_backbone)
        print(f"[adapter:{cls.__name__}] {backbone_id} embedding_dim={emb}", flush=True)
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            total = sum(p.numel() for p in model.parameters())
            print(f"[freeze] backbone 전체 동결 ({total/1e6:.1f}M) — head만 학습 (on-the-fly)")
        elif freeze_cnn or freeze_first_n_layers > 0:
            self.freeze(freeze_cnn, freeze_first_n_layers)
        if gradient_checkpointing and not freeze_backbone:
            model.gradient_checkpointing_enable()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def freeze(self, cnn: bool, n_layers: int) -> None:
        """cnn → freeze_feature_encoder (CNN), n_layers → encoder.layers[:n] 동결."""
        if cnn:
            self.model.freeze_feature_encoder()
        for i, layer in enumerate(self.model.encoder.layers):
            if i < n_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[freeze] CNN={cnn}, first_{n_layers}_layers → frozen {frozen/1e6:.1f}M / {total/1e6:.1f}M",
              flush=True)

    def extract_features(self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """raw wav (B,T) → (frames (B,T',D), frame_mask (B,T') True=pad).

        HF가 시간축을 다운샘플하므로, 입력 padding_mask 로부터 프레임 길이를 계산해
        프레임-레벨 mask 를 직접 만든다 (core 의 pooling 이 padding 을 제외하도록).
        """
        attn = None if padding_mask is None else (~padding_mask).long()  # 1=valid
        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            out = self.model(input_values=wav, attention_mask=attn).last_hidden_state  # (B,T',D)

        if padding_mask is None:
            return out, None
        input_lengths = (~padding_mask).sum(dim=-1)                          # (B,) valid samples
        out_lengths = self.model._get_feat_extract_output_lengths(input_lengths).long()  # (B,) frames
        T = out.shape[1]
        idx = torch.arange(T, device=out.device).unsqueeze(0)               # (1,T')
        frame_mask = idx >= out_lengths.unsqueeze(1)                         # (B,T') True=pad
        return out, frame_mask
