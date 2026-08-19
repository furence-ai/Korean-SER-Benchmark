"""resolve_label_subset 의 dense remap 회귀 고정 (라벨 plugin 이주 후에도 동일 동작)."""
from ser.labels import resolve_label_subset
from ser.datasets.emotion_style_speech import EmotionStyleSpeech


def test_4class_remap():
    nc, remap, names = resolve_label_subset([0, 1, 2, 6], EmotionStyleSpeech.label_space)
    assert nc == 4
    assert remap == {0: 0, 1: 1, 2: 2, 6: 3}
    assert names == {0: "happy", 1: "sad", 2: "angry", 3: "neutral"}


def test_none_is_full_identity():
    nc, remap, names = resolve_label_subset(None, EmotionStyleSpeech.label_space)
    assert nc == 7
    assert remap == {i: i for i in range(7)}
    assert names[3] == "fear" and names[6] == "neutral"


def test_default_label_space_equals_explicit():
    # label_space 인자 생략 시 기본 데이터셋과 동일해야 (하위호환 shim)
    assert resolve_label_subset([0, 1, 2, 6]) == resolve_label_subset([0, 1, 2, 6], EmotionStyleSpeech.label_space)
