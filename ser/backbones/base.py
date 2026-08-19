"""백본 어댑터 공통 인터페이스 (콘센트 규격).

학습/평가 루프(ser.engine)는 이 인터페이스로만 백본을 다룬다 — funasr/HF 같은 구현 세부는 몰라도 됨.
어댑터는 2가지만 책임진다:
  - extract_features(wav) -> (frames[B,T,D], mask)   # raw wav → 프레임 특징
  - freeze(cnn, n_layers)                            # CNN/하위 N블록 동결
pooling·head·파형 정규화·mask resample 은 전부 ser.engine.BackboneWithHead 가 담당.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# keyword(예: "emotion2vec") → adapter class
ADAPTERS: dict[str, type["BackboneAdapter"]] = {}


def register_adapter(*keywords: str):
    """어댑터 클래스를 keyword 들로 등록. backbone_id 안에 keyword 가 들어있으면 매칭."""
    def deco(cls: type["BackboneAdapter"]) -> type["BackboneAdapter"]:
        for kw in keywords:
            ADAPTERS[kw.lower()] = cls
        return cls
    return deco


def resolve_adapter(backbone_id: str) -> type["BackboneAdapter"]:
    """backbone_id(예: 'microsoft/wavlm-large') 안의 keyword 로 어댑터 클래스 반환."""
    bid = backbone_id.lower()
    for kw, cls in ADAPTERS.items():
        if kw in bid:
            return cls
    raise KeyError(
        f"backbone_id={backbone_id!r} 에 맞는 어댑터 없음. 등록된 keyword: {list(ADAPTERS)} "
        "(어댑터 모듈을 import 했는지 확인 — register_adapter 가 실행돼야 등록됨)."
    )


class BackboneAdapter(nn.Module, ABC):
    """raw wav → frame feature 추출 + freeze 제어를 제공하는 백본 래퍼.

    구현체 필수: 클래스메서드 load(), extract_features(), freeze(), 속성 embedding_dim/normalize_input.
    """

    embedding_dim: int
    normalize_input: bool
    # 선택적: 백본 native zero-shot 감정 분류기의 라벨 순서(영문). emotion2vec 처럼 native 분류기가
    # 있는 백본만 채운다. dump_pretrained_logits 가 데이터셋 라벨공간과 매칭해 동적 인덱싱에 사용.
    NATIVE_LABELS: list[str] | None = None

    @classmethod
    @abstractmethod
    def load(
        cls,
        backbone_id: str,
        *,
        freeze_backbone: bool = False,
        freeze_cnn: bool = False,
        freeze_first_n_layers: int = 0,
        gradient_checkpointing: bool = False,
    ) -> "BackboneAdapter":
        """백본 가중치 로드 + 동결 적용 후 어댑터 인스턴스 반환."""
        ...

    @abstractmethod
    def extract_features(
        self, wav: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """wav (B, T) → (frames (B, T', D), mask (B, T') True=pad or None).

        mask=None 이면 core 가 전부 valid 로 처리(프레임레이트 차이는 core 가 resample).
        """
        ...

    @abstractmethod
    def freeze(self, cnn: bool, n_layers: int) -> None:
        """cnn=True → CNN feature extractor 동결, n_layers>0 → 하위 N개 트랜스포머 블록 동결."""
        ...
