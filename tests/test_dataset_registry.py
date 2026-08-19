"""데이터셋 어댑터 레지스트리 (resolve_dataset) + 화자ID 파싱."""
import pytest

from ser.datasets import resolve_dataset
from ser.datasets.emotion_style_speech import EmotionStyleSpeech


def test_resolve_known_ids():
    assert resolve_dataset("emotion_style_speech") is EmotionStyleSpeech
    assert resolve_dataset("emotion_style") is EmotionStyleSpeech
    assert resolve_dataset("emotion_style_speech") is EmotionStyleSpeech   # 대소문자 무관


def test_resolve_unknown_raises_with_listing():
    with pytest.raises(KeyError) as e:
        resolve_dataset("does_not_exist")
    assert "emotion_style_speech" in str(e.value)   # 등록 목록을 메시지에 노출


def test_person_id_groups_by_person():
    ad = EmotionStyleSpeech()
    # 같은 사람(9999_..._XYZ)인데 감정 E필드만 다른 두 speaker → 같은 person 키
    assert ad.person_id("9999_G1A3E1S0C0_XYZ") == "9999_XYZ"
    assert ad.person_id("9999_G1A3E3S0C0_XYZ") == "9999_XYZ"
