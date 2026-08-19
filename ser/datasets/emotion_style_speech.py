"""「감성 및 발화 스타일별 음성합성 데이터」(AI-Hub) 어댑터.

기존에 흩어져 있던 이 데이터셋 종속 로직을 한 파일로 모음:
  - 라벨 공간      (구 core/labels.py 상수)
  - manifest 빌드  (구 scripts/build_manifest.py 본문)
  - 화자ID 파싱    (구 ser/loader.py: person_id_from_speaker)
  - GT 전사 추출   (구 text/extract_transcripts.py: org_label_text)

다른 데이터셋은 이 파일을 본떠 ser/datasets/<id>.py 하나만 추가하면 된다.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .base import DatasetAdapter, LabelSpace, register_dataset


@register_dataset("emotion_style_speech", "emotion_style")
class EmotionStyleSpeech(DatasetAdapter):
    label_space = LabelSpace(
        dir_to_label={
            "1.기쁨": 0, "2.슬픔": 1, "3.분노": 2, "4.불안": 3,
            "5.상처": 4, "6.당황": 5, "7.중립": 6,
        },
        label_to_english={
            0: "happy", 1: "sad", 2: "angry", 3: "fear",
            4: "hurt", 5: "embarrassed", 6: "neutral",
        },
    )

    SPLIT_DIRS = {"train": "Training", "val": "Validation"}
    DEFAULT_TASK = "1.감정"

    def __init__(self, task: str | None = None):
        # resolve_dataset(id)() 처럼 무인자 생성 가능. task 는 configs/datasets/emotion_style_speech.yaml 로 override 가능.
        self.task = task or self.DEFAULT_TASK

    def iter_manifest(self, data_root: Path, split: str, task: str | None = None) -> Iterator[dict]:
        """원천데이터/{task}/<emotion_dir>/<speaker>/*.wav → {audio,label,speaker,emotion_dir}."""
        task = task or self.task
        root = Path(data_root) / self.SPLIT_DIRS[split] / "원천데이터" / task
        if not root.is_dir():
            raise FileNotFoundError(f"디렉토리 없음: {root}")
        d2l = self.label_space.dir_to_label
        for emo_dir in sorted(d for d in root.iterdir() if d.name in d2l):
            label = d2l[emo_dir.name]
            for spk_dir in sorted(p for p in emo_dir.iterdir() if p.is_dir()):
                for wav in sorted(spk_dir.glob("*.wav")):
                    yield {"audio": str(wav), "label": label,
                           "speaker": spk_dir.name, "emotion_dir": emo_dir.name}

    def person_id(self, speaker: str) -> str:
        """speaker 는 '사람×감정' 단위(0001_G1A3E{n}S0C0_PSB) → 감정 E필드 무시, 번호+이니셜로 묶음."""
        parts = str(speaker).split("_")
        if len(parts) < 2:
            return str(speaker)
        return f"{parts[0]}_{parts[-1]}"

    def transcript_for(self, audio: str) -> str | None:
        """병렬 라벨링데이터 json 의 전사정보.OrgLabelText (정답 전사). 없으면 None."""
        jp = Path(audio.replace("/원천데이터/", "/라벨링데이터/")).with_suffix(".json")
        if not jp.is_file():
            return None
        try:
            d = json.load(jp.open(encoding="utf-8"))
            return ((d.get("전사정보", {}) or {}).get("OrgLabelText") or "").strip() or None
        except (json.JSONDecodeError, OSError):
            return None
