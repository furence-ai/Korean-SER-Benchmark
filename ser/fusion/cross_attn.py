"""Cross-attention fusion 모델 — 음향 프레임 ↔ 텍스트 토큰 교차 어텐션 → pool → head.

⚠️ 입력이 *시퀀스* 임베딩(음향 frame (B,Ta,Da) + 텍스트 token (B,Tt,Dt))이라,
   pooled 캐시(현재 ser.text.extract_embeddings / acoustic on-the-fly)와 달리
   **프레임/토큰 시퀀스 캐시 추출이 선행**돼야 학습 가능 (extractor 는 후속 추가).
   여기서는 모델 정의 + forward 규격만 제공 (다른 fusion 과 동일하게 ser.metrics 로 평가).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttnFusion(nn.Module):
    """음향(query) ↔ 텍스트(key/value) cross-attention. mask: True=pad."""

    def __init__(self, dim_acoustic: int, dim_text: int, hidden: int = 256,
                 num_classes: int = 4, nhead: int = 4, dropout: float = 0.3):
        super().__init__()
        self.proj_a = nn.Linear(dim_acoustic, hidden)
        self.proj_t = nn.Linear(dim_text, hidden)
        self.attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, ac: torch.Tensor, ac_mask: torch.Tensor,
                tx: torch.Tensor, tx_mask: torch.Tensor) -> torch.Tensor:
        """ac (B,Ta,Da), tx (B,Tt,Dt), *_mask (B,T) True=pad → logits (B,num_classes)."""
        q, kv = self.proj_a(ac), self.proj_t(tx)
        out, _ = self.attn(q, kv, kv, key_padding_mask=tx_mask)   # 음향이 텍스트를 attend
        out = self.norm(q + out)
        valid = (~ac_mask).unsqueeze(-1).to(out.dtype)            # 음향 프레임 masked mean pool
        pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)
