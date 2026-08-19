"""build_ckpt_meta 가 best.pt 스키마(build_model_from_ckpt 가 의존)를 정확히 재현하는지."""
from ser.io import build_ckpt_meta

ACOUSTIC_KEYS = {"mode", "head_type", "embedding_dim", "num_classes", "keep_labels",
                 "hidden_dim", "dropout", "backbone", "label_to_english", "full_ft_max_audio_sec"}
CACHED_KEYS = {"mode", "modality", "head_type", "embedding_dim", "num_classes", "keep_labels",
               "hidden_dim", "dropout", "backbone", "label_to_english"}


def test_acoustic_meta_keys():
    m = build_ckpt_meta(mode="full_ft", head_type="official-frame-mlp", embedding_dim=1024,
                        num_classes=4, keep_labels=[0, 1, 2, 6], hidden_dim=256, dropout=0.2,
                        backbone="emotion2vec/x", label_to_english={0: "happy"},
                        full_ft_max_audio_sec=10.0)
    assert set(m) == ACOUSTIC_KEYS
    assert "modality" not in m   # acoustic 메타엔 modality 키 없음 (옛 동작 보존)


def test_cached_meta_keys():
    m = build_ckpt_meta(mode="head_cached", modality="text", head_type="custom-utterance-mlp",
                        embedding_dim=1024, num_classes=4, keep_labels=[0, 1, 2, 6], hidden_dim=256,
                        dropout=0.2, backbone="klue/roberta-large", label_to_english={0: "happy"})
    assert set(m) == CACHED_KEYS
    assert "full_ft_max_audio_sec" not in m
