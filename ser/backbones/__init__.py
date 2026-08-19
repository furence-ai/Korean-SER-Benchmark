"""백본 어댑터 — 백본(emotion2vec/wavlm/wav2vec2)을 이름으로 아는 유일한 곳.

adapter import 시 register_adapter 로 keyword(예: "emotion2vec","wavlm","wav2vec2")가 등록되고,
resolve_adapter(backbone_id) 가 id 안의 keyword 로 어댑터 클래스를 찾는다.
"""
from .base import ADAPTERS, BackboneAdapter, register_adapter, resolve_adapter  # noqa: F401

# 어댑터 self-register: 모듈 import 시 register_adapter 가 실행돼 keyword 등록.
# 무거운 dep(funasr/transformers)는 어댑터 load() 안에서 lazy import 라 여기선 안 끌려옴.
from . import emotion2vec  # noqa: F401,E402

for _mod in ("wavlm", "wav2vec2"):   # 아직 없을 수 있음 (Phase 1에서 추가)
    try:
        __import__(f"backbones.{_mod}")
    except ImportError:
        pass
