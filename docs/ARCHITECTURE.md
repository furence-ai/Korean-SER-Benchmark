# 기술 문서

프로젝트 개요와 결과는 최상위 [README.md](../README.md) 를 참고하세요.

## 개요

여러 음향 백본 + 텍스트 + 멀티모달 결합을 **하나의 공유 엔진**으로 공정 비교합니다.
백본마다 코드를 복붙하지 않고, 모든 백본·모달리티가 **같은 split · 같은 지표
(macro-F1) · 같은 평가 루프**를 씁니다.

```
ser/            프레임워크 (패키지 하나로 통합)
  train.py        음향 학습 진입점 (full fine-tune / head-only)
  train_cached.py 캐시 임베딩 head 학습 진입점
  engine.py       BackboneWithHead · 학습/평가 루프
  loader.py       Dataset · collate · 라벨 remap 필터
  heads.py · metrics.py · diagnostics.py · registry.py · rundir.py
  config.py · cli.py · io.py · labels.py · audio.py
  backbones/      백본 어댑터: base(인터페이스) · emotion2vec(funasr) · wavlm · wav2vec2(HF)
  datasets/       데이터셋 어댑터: base(register/resolve) · emotion_style_speech
  text/           텍스트 분기: 전사 추출 · KLUE-RoBERTa 임베딩
  fusion/         결합: late · early · cross_attn(+train_cross)
scripts/        데이터 준비 · 평가 · 추론 · 리포트 · split_fingerprint
configs/        default.yaml · smoke.yaml · datasets/<id>.yaml
```

## 설계 원칙

SER 모델 = **백본(소리→특징) + head(특징→감정)**. head·dataset·학습루프·지표·split
은 백본이 뭐든 동일하고 **백본만 다릅니다.** 그래서 학습/평가 루프는 한 번만 짜고,
백본은 **어댑터** 하나로 갈아끼웁니다.

- `extract_features(wav) -> (frames[B,T,D], mask)`: raw wav → 프레임 특징
- `freeze(cnn, n_layers)`: CNN / 하위 N블록 동결

pooling·파형정규화·mask resample·head 는 전부 `ser.engine.BackboneWithHead` 가
담당합니다. → **새 백본 추가 = 어댑터 파일 하나.**

## split 구성

「감성 및 발화 스타일별 음성합성 데이터」는 **대본 9,182개가 감정별 1:1 고정**이고 각 대본을 ~32명이 낭독합니다.
따라서 두 가지를 동시에 held-out 해야 합니다.

- **화자 분리**: 음향 모델이 학습한 목소리로 평가되지 않도록
- **대본 분리**: test 문장이 train 에 없도록. 대본이 감정에 1:1로 묶여 있어서,
  문장을 본 적 있는 텍스트 모델은 소리를 안 듣고도 거의 만점을 받습니다

`scripts/make_disjoint_split.py` 가 화자와 대본을 각각 분할한 뒤 **대각선만** 남깁니다
(각 split = 그 split 전용 화자 ∩ 그 split 전용 대본). 교차 셀 ~33% 는 폐기하며,
train 190,837 / val 2,933 / test 2,451 이 남습니다.

**이 split 은 고정입니다.** 공개된 19개 run 이 전부 같은 split 위에 있고, 그래서
서로 비교 가능합니다. `scripts/split_fingerprint.py` 로 동일성을 해시 검증합니다.

쓸 감정 클래스는 `make_speaker_split --keep-labels` 로 고르고, 선택된 라벨은 0부터
연속 번호로 dense remap 됩니다(`ser.labels.resolve_label_subset`). 학습 시
`keep_labels` 와 같은 값이어야 하며, 기본은 `[0, 1, 2, 6]`(기쁨·슬픔·분노·중립)
입니다. 자세한 내용은 [DATA.md](DATA.md#3-클래스-선택).

## 학습 모드

세 모드는 **학습 루프에서 백본을 어떻게 다루는가**로 갈립니다. head 는 세 경우 모두
학습됩니다.

| 모드 | 백본 가중치 | 백본 forward 시점 | 학습 루프 입력 |
|---|---|---|---|
| **full fine-tune** | 갱신됨 | 매 배치 | raw wav |
| **head-only (on-the-fly)** | 동결 | 매 배치 | raw wav |
| **cached head** | 동결 | 학습 전 1회 (별도 스크립트) | 저장된 임베딩 벡터 |

`cached head` 의 백본은 없는 것이 아니라 **학습 시작 전에 이미 다 돌아간 상태**입니다.
별도 추출 스크립트가 전 발화를 백본에 한 번 통과시켜 발화당 벡터 하나를 `.npz` 로
저장해 두고, 학습 루프는 그 벡터만 읽습니다. 백본이 GPU 에 올라가지 않으므로 세 모드
중 가장 빠르고, 같은 임베딩을 여러 head 실험에 재사용할 수 있습니다.

```
full fine-tune / head-only        cached head
  wav ─► 백본 ─► head ─► loss       [사전 추출] wav ─► 백본 ─► vec.npz
        (매 배치)                    [학습]     vec.npz ─► head ─► loss
```

텍스트 분기와 early 결합이 `cached head` 를 쓰는 이유는, 두 경우 모두 백본을 학습할
일이 없기 때문입니다. RoBERTa 는 항상 동결이고, early 결합은 이미 뽑아 둔 음향 벡터와
텍스트 벡터를 이어 붙인 것을 입력으로 받습니다.

### 실행

```bash
# ① full fine-tune: 백본까지 학습 (CNN 특징추출기는 동결)
uv run python -m ser.train --backbone facebook/wav2vec2-large-xlsr-53 --full-ft-freeze-cnn
uv run python -m ser.train --backbone emotion2vec/emotion2vec_plus_large --full-ft-freeze-cnn
uv run python -m ser.train --backbone microsoft/wavlm-large --full-ft-freeze-cnn

# ② head-only: 백본 동결, 매 배치 forward
uv run python -m ser.train --backbone facebook/wav2vec2-large-xlsr-53 --embed-on-the-fly

# ③ cached head: 임베딩을 먼저 뽑고, 그 벡터로 head 만 학습
#    텍스트
uv run python -m ser.text.extract_transcripts
uv run python -m ser.text.extract_embeddings                      # → data/embeddings/roberta-large/*.npz
uv run python -m ser.train_cached --emb-dir data/embeddings/roberta-large \
    --modality text --backbone klue/roberta-large
#    음향 (early 결합용 벡터 추출)
uv run python -m scripts.extract_acoustic_embeddings --backbone <X> --splits train_di val_di test_di
```

산출물은 `<run_dir>/` 에 `best.pt` · `test_metrics.json` · `test_logits.npz` ·
`history.png`, 그리고 `runs/registry.jsonl` 에 1행(test macro-F1 + 클래스별 F1).

## 결합 방식

세 방법은 **어느 단계에서 합치느냐**가 다릅니다.

| 방법 | 합치는 대상 | 방법 | 학습 | 비용 |
|---|---|---|---|---|
| **late** | 최종 클래스 확률 | softmax 평균 | 없음 | 0 |
| **early** | 발화당 임베딩 1벡터씩 | concat → head | head만 | 낮음 |
| **cross-attn** | 프레임·토큰 시퀀스 | 교차 어텐션 → pool → head | head+어텐션 | 최고 |

```bash
# late: 저장된 logits 두 개를 결합 (재학습 없음)
uv run python -m ser.fusion.late --acoustic-logits <A>/test_logits.npz --text-logits <T>/test_logits.npz

# early: 임베딩 concat 후 head 학습
uv run python -m ser.fusion.early --acoustic-emb-dir <A> --text-emb-dir <T> --out-dir data/embeddings/early
uv run python -m ser.train_cached --emb-dir data/embeddings/early --modality fusion

# cross-attention: on-the-fly 듀얼 인코더 (임베딩 캐시 불가, GPU 부담 큼)
uv run python -m ser.fusion.train_cross --backbone facebook/wav2vec2-large-xlsr-53
```

`cross_attn.py` 는 음향 프레임을 query, 텍스트 토큰을 key/value 로 두어 각 음향
프레임이 관련 단어를 직접 attend 합니다. 유일하게 시간 정렬을 쓰는 방식입니다.

## 추론

```bash
# 음향 단독: wav 만 있으면 됨
uv run python -m scripts.infer --ckpt <run_dir>/acoustic/w2v_full a.wav
uv run python -m scripts.infer --ckpt <run_dir>/acoustic/w2v_full --glob "clips/*.wav" --json

# 앙상블: 전사를 직접 준다 (이 레포는 ASR 을 포함하지 않는다)
uv run python -m scripts.infer_fusion --method early \
    --acoustic-ckpt <run_dir>/acoustic/w2v_full \
    --fusion-ckpt   <run_dir>/early/w2v_full a.wav --text "오늘 정말 기분이 좋아"
```

## 체크포인트 재평가

```bash
# 음향 raw-audio 모델 (여러 개 한 번에 가능)
uv run python -m scripts.evaluate --ckpt <run_dir>/acoustic/em_head <run_dir>/acoustic/w2v_full
# 캐시-임베딩 head (text / early-fusion): 해당 임베딩 폴더 필요
uv run python -m scripts.evaluate --ckpt <run_dir>/text/roberta --emb-dir <run_dir>/emb/roberta-large
```

ckpt 의 `mode` 로 평가 경로가 자동 분기합니다. `full_ft`/`head_only_otf` 는 음향
raw-audio(`build_model_from_ckpt` 로 백본+head 재구성), `head_cached` → 캐시 임베딩
head(`--emb-dir` 필요), `fusion_cross` → `ser.fusion.train_cross` 전용.

**비교표 복원**: 학습 폴더만 옮겨와 registry 가 없으면 각 폴더의 `test_logits.npz`
로부터 재계산만으로 복원됩니다.

```bash
uv run python -m scripts.rebuild_registry --root <results_root>
uv run python -m scripts.report --registry <results_root>/registry.jsonl --cm
```

## 설정

`config.py` 가 중첩 섹션을 엔진용 평면 키로 변환합니다. **3 섹션:**

| 섹션 | 누가 쓰나 | 주요 키 |
|---|---|---|
| `common` | 전 모드 | `backbone` · `head_type` · `keep_labels` · `epochs` · `weight_decay` · `early_stop_patience` · `split` |
| `acoustic` | `ser.train` | `full_ft_lr` / `full_ft_backbone_lr`(full FT) · `head_lr`(head-only) · `batch_size` · `freeze_cnn` · `warmup_steps` |
| `cached` | `ser.train_cached` | `lr` · `batch_size` |

LR 키가 모드별로 분리돼 있습니다. **full FT=`full_ft_lr`, on-the-fly
head-only=`head_lr`, cached head=`lr`**. CLI 인자가 최종 override 입니다.

**설정 단일 소스**: 공유 파라미터(`dataset` · `keep_labels` · `split_*` ·
`manifest_dir` · `device` · `seed`)는 `configs/*.yaml` 에서만 정의하고, 모든
entrypoint 가 `ser.cli.add_common_overrides` 로 같은 인자를 받습니다.

## 진단

저장된 `test_logits.npz` 로 백본/모달리티 무관 분석: **top-k 정확도**, gt_rank 분포,
margin, calibration(정답 vs 오답 confidence), 오분류쌍. `scripts.report --cm` 이
전 run 비교표에 이것들을 붙여 줍니다.

## 백본 추가

`ser/backbones/<name>.py` 에 `BackboneAdapter` 구현 + `@register_adapter("<keyword>")`.
HF wav2vec2 계열(WavLM/wav2vec2)은 `ser/backbones/_hf.py` 의 공통 베이스를 상속해
`_model_cls()` 만 지정하면 됩니다. `backbone` id 안에 keyword 가 들어가면 자동
resolve 됩니다.

## 데이터셋 추가

데이터셋 종속(라벨공간·디렉토리 규칙·화자ID·GT전사)은 전부 **데이터셋 어댑터**에만
있습니다. 새 데이터셋 = **파일 하나 + config 한 줄**:

1. `ser/datasets/<id>.py` 에 `DatasetAdapter` 구현 + `@register_dataset("<id>")`
   - `label_space = LabelSpace(dir_to_label={...}, label_to_english={...})`
   - `iter_manifest(data_root, split)` → `{audio, label, speaker}` 스트림
   - `person_id(speaker)` → 사람 단위 그룹키 (화자 분리용)
   - `transcript_for(audio)` → GT 전사 (없으면 `None`)
2. `ser/datasets/__init__.py` 에 `from . import <id>` 한 줄 (self-register)
3. `configs/datasets/<id>.yaml` 에 비코드 값(data_root 등)
4. config `common.dataset: <id>` 로 교체

`ser/backbones/` 의 `register_adapter`/`resolve_adapter` 와 완전히 대칭입니다.
이 데이터셋 구현은 `ser/datasets/emotion_style_speech.py` 를 참고하세요.

## 데이터 준비

```bash
export SER_DATA_ROOT=/path/to/emotion-style-speech
uv run python -m scripts.build_manifest --splits train val            # ① raw 스캔 → manifest
uv run python -m scripts.make_speaker_split --keep-labels 0 1 2 6     # ② 클래스 선택 + 화자 분리
uv run python -m ser.text.extract_transcripts                         #    (③ 이 전사를 사용)
uv run python -m scripts.make_disjoint_split --seed 42                # ③ 대본까지 분리 (최종)
uv run python -m scripts.extract_acoustic_embeddings --backbone <X> --splits train_di val_di test_di
uv run python -m scripts.dump_pretrained_logits --split test_di # emotion2vec 원본 zero-shot
uv run python -m scripts.compare_pretrained --backbone <X>      # emotion2vec 사전학습(9→4) vs 학습
```