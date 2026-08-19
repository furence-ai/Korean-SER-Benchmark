"""masked_mean_pool 수치 동작 가드 (엔진 핵심 공유 로직 — 학습/추출/추론 공용)."""
import torch

from ser.engine import masked_mean_pool


def test_pool_ignores_padded_frames():
    feats = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])   # (1,3,2)
    mask = torch.tensor([[False, False, True]])                       # True=pad → 3번째 무시
    out = masked_mean_pool(feats, mask)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


def test_pool_all_valid():
    feats = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    mask = torch.tensor([[False, False]])
    out = masked_mean_pool(feats, mask)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


def test_pool_all_pad_no_nan():
    feats = torch.tensor([[[5.0, 5.0]]])
    mask = torch.tensor([[True]])      # 전부 pad → clamp(min=1) 로 0 division 방지
    out = masked_mean_pool(feats, mask)
    assert torch.isfinite(out).all()
