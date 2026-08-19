"""분류 head (백본 무관) + forward 정밀도 헬퍼.

3종 head — canonical id(문자열)는 config head_type / 체크포인트 head_type / 폴더 suffix 공용:
  custom-utterance-mlp : CustomUtteranceMLP — utterance-level pre-pooled (B, D) 입력
  custom-frame-mlp     : CustomFrameMLP     — frame-level (B, T, D), 깊은 head
  official-frame-mlp   : OfficialFrameMLP   — frame-level (B, T, D), 공식 emotion2vec 구조
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

HEAD_CUSTOM_UTTERANCE = "custom-utterance-mlp"
HEAD_CUSTOM_FRAME = "custom-frame-mlp"
HEAD_OFFICIAL_FRAME = "official-frame-mlp"


def amp_autocast(device: str, amp_dtype: str = "bf16"):
    """raw-audio backbone forward용 autocast (train/eval/compare 공용).

    amp_dtype='bf16' 이고 cuda면 bf16 autocast, 아니면 no-op(fp32).
    """
    if amp_dtype == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class CustomUtteranceMLP(nn.Module):
    """[custom-utterance-mlp] Utterance-level (B, D) → (B, num_classes). pre-pooled 전용."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OfficialFrameMLP(nn.Module):
    """[official-frame-mlp] 공식 emotion2vec/iemocap_downstream 그대로.

    Frame (B, T, D) + padding_mask → Linear → ReLU → masked mean pool → Linear. dropout 무시.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.pre_net = nn.Linear(in_dim, hidden_dim)
        self.post_net = nn.Linear(hidden_dim, num_classes)
        self.activate = nn.ReLU()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.activate(self.pre_net(x))                      # (B, T, hidden)
        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)       # (B, T, 1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (x * valid).sum(dim=1) / denom                 # (B, hidden)
        return self.post_net(pooled)


class CustomFrameMLP(nn.Module):
    """[custom-frame-mlp] Frame (B, T, D) + padding_mask → (B, num_classes).

    pool 전 비선형(official과 같은 family)이되 custom-utterance와 깊이/regularization 동일.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.frame_proj(x)                                  # (B, T, hidden)
        valid = (~padding_mask).unsqueeze(-1).to(h.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (h * valid).sum(dim=1) / denom
        return self.post_pool(pooled)


# ---------- head 팩토리 ----------
FRAME_HEADS: dict[str, type[nn.Module]] = {
    HEAD_CUSTOM_FRAME: CustomFrameMLP,
    HEAD_OFFICIAL_FRAME: OfficialFrameMLP,
}
HEAD_CLASSES: dict[str, type[nn.Module]] = {
    HEAD_CUSTOM_UTTERANCE: CustomUtteranceMLP,
    **FRAME_HEADS,
}


def is_frame_head(head_type: str) -> bool:
    """frame-level head인지 (forward가 padding_mask 받고 pool 내부)."""
    return head_type in FRAME_HEADS


def build_head(head_type: str, in_dim: int, num_classes: int,
               hidden_dim: int = 256, dropout: float = 0.3) -> nn.Module:
    """head_type id로 head 인스턴스 생성 (utterance/frame 모두)."""
    try:
        cls = HEAD_CLASSES[head_type]
    except KeyError:
        raise ValueError(f"unknown head_type={head_type!r}, choices: {list(HEAD_CLASSES)}") from None
    return cls(in_dim=in_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)


def build_frame_head(head_type: str, in_dim: int, num_classes: int,
                     hidden_dim: int = 256, dropout: float = 0.3) -> nn.Module:
    """frame-level head 전용 빌더 (cached frame 임베딩 경로용)."""
    if not is_frame_head(head_type):
        raise ValueError(f"frame head 필요한데 head_type={head_type!r}. choices: {list(FRAME_HEADS)}")
    return build_head(head_type, in_dim, num_classes, hidden_dim, dropout)
