"""데이터셋 어댑터 공통 인터페이스 (콘센트 규격) — backbones/base.py 와 대칭.

백본 어댑터가 "raw wav → frames" 를 추상화하듯, 데이터셋 어댑터는
"raw 디렉토리 → manifest / 화자그룹 / GT전사 / 라벨공간" 을 추상화한다.
새 데이터셋 = 이 인터페이스를 구현한 파일 하나(ser/datasets/<id>.py) + config `dataset: <id>`.

전처리/학습 코드는 이 인터페이스로만 데이터셋을 다룬다 — AIHub 디렉토리 규칙·라벨 json
스키마·화자ID 포맷 같은 데이터셋 세부는 어댑터 안에만 있다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# dataset id(예: "emotion_style_speech") → adapter class
DATASETS: dict[str, type["DatasetAdapter"]] = {}


def register_dataset(*keys: str):
    """어댑터 클래스를 dataset id 들로 등록 (정확 일치 매칭)."""
    def deco(cls: type["DatasetAdapter"]) -> type["DatasetAdapter"]:
        for k in keys:
            DATASETS[k.lower()] = cls
        return cls
    return deco


def resolve_dataset(dataset_id: str) -> type["DatasetAdapter"]:
    """dataset id 로 어댑터 클래스 반환 (정확 일치). backbones.resolve_adapter 와 같은 역할."""
    cls = DATASETS.get((dataset_id or "").lower())
    if cls is None:
        raise KeyError(
            f"dataset={dataset_id!r} 에 맞는 어댑터 없음. 등록된 id: {list(DATASETS)} "
            "(datasets 패키지를 import 했는지 확인 — register_dataset 가 실행돼야 등록됨)."
        )
    return cls


@dataclass(frozen=True)
class LabelSpace:
    """데이터셋의 라벨 공간 정의(백본 무관). 기존 ser.labels 상수들을 데이터로 들고 다님."""

    dir_to_label: dict[str, int]       # 디렉토리명→원본 라벨 ID (manifest 빌드용). 예: {"1.기쁨": 0}
    label_to_english: dict[int, str]   # 원본 라벨 ID→영어명 (report/meta 용). 예: {0: "happy"}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_english)

    @property
    def label_to_korean(self) -> dict[int, str]:
        """디렉토리명에서 한국어 감정명 유도 ('1.기쁨'→'기쁨'). 없으면 영어명 폴백."""
        rev = {v: k for k, v in self.dir_to_label.items()}
        return {lab: (rev[lab].split(".", 1)[-1] if lab in rev else eng)
                for lab, eng in self.label_to_english.items()}


class DatasetAdapter(ABC):
    """데이터셋별 raw 처리 규칙. 인스턴스는 resolve_dataset(id)() 로 생성 (무인자 생성 가능해야 함)."""

    label_space: LabelSpace

    @abstractmethod
    def iter_manifest(self, data_root: Path, split: str) -> Iterator[dict]:
        """raw 디렉토리 스캔 → manifest 레코드 스트림 {audio, label, speaker, ...}.

        label 은 원본 라벨 ID (dense remap 전). split 은 'train'|'val' 등 raw split.
        """
        ...

    @abstractmethod
    def person_id(self, speaker: str) -> str:
        """speaker 필드 → '사람' 단위 그룹 키 (speaker-independent split 용)."""
        ...

    @abstractmethod
    def transcript_for(self, audio: str) -> str | None:
        """GT(정답) 전사 추출. 없거나 STT 만 쓸 거면 None 반환 가능."""
        ...

    def one_script_one_emotion(self) -> bool:
        """make_disjoint_split: 한 대본=한 감정 가정 여부 (대본 누수 분리에 사용)."""
        return True
