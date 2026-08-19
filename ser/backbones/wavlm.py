"""WavLM 백본 어댑터 (HF). backbone_id 예: microsoft/wavlm-large."""
from __future__ import annotations

from ._hf import HFWav2Vec2FamilyAdapter
from .base import register_adapter


@register_adapter("wavlm")
class WavLMAdapter(HFWav2Vec2FamilyAdapter):
    @classmethod
    def _model_cls(cls):
        from transformers import WavLMModel
        return WavLMModel
