"""emotion2vec native 9-class → 데이터셋 라벨 동적 매핑이 옛 하드코딩 [3,6,0,4] 와 일치하는지."""
import pytest

from ser.backbones import resolve_adapter
from ser.cli import resolve_dataset_ctx
from scripts.dump_pretrained_logits import native_index_map


def test_native_idx_4class_regression():
    ctx = resolve_dataset_ctx({"dataset": "emotion_style_speech", "keep_labels": [0, 1, 2, 6]})
    ad = resolve_adapter("emotion2vec/emotion2vec_plus_large")
    assert native_index_map(ad, ctx.label_to_english) == [3, 6, 0, 4]


def test_native_idx_unmappable_labels_error_clearly():
    # 라벨공간 전체에는 native 에 없는 hurt/embarrassed 가 있어 매핑 불가 → 명확한 에러
    ctx = resolve_dataset_ctx({"dataset": "emotion_style_speech", "keep_labels": None})
    ad = resolve_adapter("emotion2vec/emotion2vec_plus_large")
    with pytest.raises(SystemExit):
        native_index_map(ad, ctx.label_to_english)
