"""데이터셋 어댑터 패키지. import 시 어댑터들을 self-register (backbones/__init__.py 와 동일 패턴).

새 데이터셋 추가: ser/datasets/<id>.py 작성(@register_dataset) 후 아래에 import 한 줄.
"""
from . import emotion_style_speech  # noqa: F401  (register_dataset 실행 → DATASETS 등록)
from .base import DatasetAdapter, LabelSpace, register_dataset, resolve_dataset

__all__ = ["DatasetAdapter", "LabelSpace", "register_dataset", "resolve_dataset"]
