"""wav2vec2 백본 어댑터 (HF). backbone_id 예: facebook/wav2vec2-large-xlsr-53."""
from __future__ import annotations

from ._hf import HFWav2Vec2FamilyAdapter
from .base import register_adapter


@register_adapter("wav2vec2")
class Wav2Vec2Adapter(HFWav2Vec2FamilyAdapter):
    @classmethod
    def _model_cls(cls):
        from transformers import Wav2Vec2Model
        return Wav2Vec2Model
