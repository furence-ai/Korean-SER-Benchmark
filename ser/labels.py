"""라벨 subset 해석 (dense remap). 라벨 공간 정의는 데이터셋 어댑터(ser/datasets/<id>.py)로 이동.

resolve_label_subset 은 LabelSpace 를 받아 (num_classes, orig→dense remap, dense→english) 를 만든다.
하위호환: label_space 를 안 주면 기본 데이터셋의 라벨 공간을 사용하고,
구 상수(EMOTION_DIR_TO_LABEL / LABEL_TO_ENGLISH / LABEL_TO_KOREAN / NUM_CLASSES)도 re-export 한다.
"""
from __future__ import annotations

from ser.datasets.emotion_style_speech import EmotionStyleSpeech as _DEFAULT_DS

# 구 import 호환 (ser.labels.EMOTION_DIR_TO_LABEL 등) — 기본 데이터셋의 라벨 공간으로 재노출
EMOTION_DIR_TO_LABEL = _DEFAULT_DS.label_space.dir_to_label
LABEL_TO_ENGLISH = _DEFAULT_DS.label_space.label_to_english
LABEL_TO_KOREAN = _DEFAULT_DS.label_space.label_to_korean
NUM_CLASSES = _DEFAULT_DS.label_space.num_classes


def resolve_label_subset(
    keep: list[int] | None,
    label_space=None,
) -> tuple[int, dict[int, int], dict[int, str]]:
    """클래스 subset 스펙을 (num_classes, orig→dense remap, dense→english) 으로 해석.

    keep: 원본 라벨 ID 리스트 (예: [0, 1, 2, 6]). None/빈 리스트면 전체 클래스 (remap=항등).
    label_space: 데이터셋의 LabelSpace. None 이면 기본 데이터셋.
    """
    if label_space is None:
        label_space = _DEFAULT_DS.label_space
    l2e = label_space.label_to_english
    if not keep:
        n = label_space.num_classes
        return n, {i: i for i in range(n)}, dict(l2e)
    remap = {orig: dense for dense, orig in enumerate(keep)}
    names = {dense: l2e.get(orig, f"class_{orig}") for dense, orig in enumerate(keep)}
    return len(keep), remap, names
