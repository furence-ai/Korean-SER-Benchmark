# 한국어 음성 감정 인식(SER) 벤치마크

**[모델](https://huggingface.co/furence-ai/korean-ser-wav2vec2-xlsr-4class)** ·
**[결과](results/)** ·
[기술 문서](docs/ARCHITECTURE.md) ·
[데이터 준비](docs/DATA.md)

## 개요

여러 음향 백본과 텍스트 분기, 멀티모달 결합을 하나의 공유 엔진으로 비교하는 한국어
SER 벤치마크 및 파인튜닝 프레임워크. 모든 구성이 같은 split · 같은 지표 · 같은
평가 루프를 쓰므로, run 사이의 차이는 비교 대상으로 삼은 요소에서만 발생합니다.

| 항목 | 내용 |
|---|---|
| 데이터셋 | [「감성 및 발화 스타일별 음성합성 데이터」](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=466) (AI-Hub) |
| 클래스 | 4종 (기쁨 · 슬픔 · 분노 · 중립) |
| split | 화자 + 대본 동시 분리 (train 190,837 / val 2,933 / test 2,451) |
| 지표 | test macro-F1 |
| 백본 | emotion2vec · wav2vec2-XLSR |
| 결합 | late · early · cross-attention |
| 최고 성능 | macro-F1 **0.8880** (early 결합, wav2vec2 full-FT + RoBERTa) |

## split 규격

이 코퍼스는 낭독체이고 **대본 9,182개가 감정과 1:1로 묶여 있으며 각 대본을 약 32명이
낭독합니다.** 그래서 화자만 분리하면 test 문장이 train 에 그대로 남고, 텍스트 모델이
소리를 듣지 않고 문장→감정 암기만으로 점수를 올립니다.

split 은 고정이며 공개된 19개 run 이 전부 같은 split 위에 있습니다. 더 느슨한 split 으로
보고된 다른 수치와는 직접 비교할 수 없습니다.

## 저장소 구성

```
ser/          프레임워크 (패키지 하나로 통합, 최상위 이름 충돌 없음)
  train.py · train_cached.py    학습 진입점
  engine.py · loader.py · heads.py · metrics.py · diagnostics.py
  config.py · cli.py · io.py · labels.py · registry.py · rundir.py · audio.py
  backbones/  어댑터: emotion2vec(funasr) · WavLM · wav2vec2(HF)
  datasets/   어댑터: base(register/resolve) · emotion_style_speech
  text/       전사 추출 · KLUE-RoBERTa 임베딩
  fusion/     late · early · cross-attention
scripts/      데이터 준비 · 평가 · 추론 · 리포트 · split 지문 검증
configs/      default.yaml · smoke.yaml · datasets/<id>.yaml
results/      벤치마크: 19-run 리더보드 · confusion matrix · registry
docs/         기술 문서 · 데이터 조건
tests/        단위 테스트
data/         의도적으로 비어 있음 (data/README.md 참고)
```

## 데이터 준비

**[AI-Hub 데이터셋 페이지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=466)**
에서 이용을 신청해 받은 뒤, 아래로 split 을 재생성합니다. 디렉토리 배치와 각 단계
설명은 **[docs/DATA.md](docs/DATA.md)** 에 있습니다.

```bash
uv sync
export SER_DATA_ROOT=/path/to/emotion-style-speech

uv run python -m scripts.build_manifest --splits train val              # ① raw 스캔
uv run python -m scripts.make_speaker_split --keep-labels 0 1 2 6       # ② 클래스 선택 + 화자 분리
uv run python -m ser.text.extract_transcripts                           #    (③ 이 전사를 사용)
uv run python -m scripts.make_disjoint_split --seed 42                  # ③ 대본까지 분리

# 공개 실험과 동일한 split 을 얻었는지 검증
uv run python -m scripts.split_fingerprint \
    --manifest-dir data/manifests --data-root "$SER_DATA_ROOT" \
    --check data/SPLIT_CHECKSUMS.json
#   ✓  train_di: matches (190837 items)
#   ✓  val_di:   matches (2933 items)
#   ✓  test_di:  matches (2451 items)
```

②의 `--keep-labels` 가 쓸 감정 클래스를 고릅니다. 학습 시 `keep_labels` 와 같은 값이어야
합니다. 지문 세 줄이 모두 ✓ 여야 [리더보드](results/LEADERBOARD.md) 수치와 비교할 수 있습니다.

## 학습과 평가

```bash
# 음향: 어느 백본이든 명령은 같고 backbone 만 교체
uv run python -m ser.train --backbone emotion2vec/emotion2vec_plus_large --full-ft-freeze-cnn
uv run python -m ser.train --backbone microsoft/wavlm-large --full-ft-freeze-cnn
uv run python -m ser.train --backbone facebook/wav2vec2-large-xlsr-53 --embed-on-the-fly

# 텍스트 분기: 전사 → RoBERTa 임베딩 → head
uv run python -m ser.text.extract_transcripts
uv run python -m ser.text.extract_embeddings
uv run python -m ser.train_cached --emb-dir data/embeddings/roberta-large \
    --modality text --backbone klue/roberta-large

# 결합 (음향 × 텍스트)
uv run python -m ser.fusion.late  --acoustic-logits <A>/test_logits.npz --text-logits <T>/test_logits.npz
uv run python -m ser.fusion.early --acoustic-emb-dir <A> --text-emb-dir <T> --out-dir data/embeddings/early
uv run python -m ser.fusion.train_cross --backbone facebook/wav2vec2-large-xlsr-53

# 비교표 + confusion matrix + top-k 분포
uv run python -m scripts.report --cm
```

`bash run_all.sh` 로 전체 스윕(음향 + late + early + cross-attention)을 돌리고 비교표를
생성합니다. 체크포인트와 캐시 임베딩이 수백 GB 나오므로 `$ROOT` 를 큰 볼륨으로 먼저
지정합니다.

## 공개 모델

성능 상위 두 구성을 HuggingFace 에 공개했습니다.

| 모델 | macro-F1 | 입력 |
|---|---:|---|
| [korean-ser-wav2vec2-xlsr-4class](https://huggingface.co/furence-ai/korean-ser-wav2vec2-xlsr-4class) | 0.8588 | 오디오 |
| [korean-ser-wav2vec2-xlsr-4class-fusion-head](https://huggingface.co/furence-ai/korean-ser-wav2vec2-xlsr-4class-fusion-head) | **0.8880** | 오디오 + 전사 |

두 모델은 각자 베이스 모델의 라이선스를 따릅니다. 각 모델 카드 확인.

```bash
# wav 하나 넣으면 감정이 나옴 (ASR 불필요)
uv run python -m scripts.infer --ckpt korean_ser_wav2vec2_xlsr_4class.pt sample.wav

# +2.9pt, 대신 전사를 직접 줘야 함
uv run python -m scripts.infer_fusion --method early \
    --acoustic-ckpt korean_ser_wav2vec2_xlsr_4class.pt \
    --fusion-ckpt   korean_ser_wav2vec2_xlsr_4class_fusion_head.pt \
    sample.wav --text "오늘 정말 기분이 좋아"
```

## 결과

19개 구성 전체와 run 별 confusion matrix · 진단은 [results/](results/) 에 있습니다.

| # | 구성 | macro-F1 | acc |
|---:|---|---:|---:|
| 1 | early 결합 · wav2vec2 full-FT + RoBERTa | **0.8880** | 0.8947 |
| 2 | cross-attention · wav2vec2 full-FT + RoBERTa | 0.8858 | 0.8927 |
| 3 | late 결합 · wav2vec2 full-FT + RoBERTa | 0.8763 | 0.8821 |
| 4 | 음향 단독 · wav2vec2 full-FT | 0.8588 | 0.8658 |
| 8 | 음향 단독 · emotion2vec full-FT | 0.7981 | 0.8152 |
| 13 | 음향 단독 · emotion2vec head-only (동결) | 0.6538 | 0.6801 |
| 16 | 음향 단독 · wav2vec2 head-only (동결) | 0.5999 | 0.6589 |
| 17 | 텍스트 단독 · KLUE-RoBERTa | 0.5961 | 0.6047 |
| 18 | 음향 단독 · emotion2vec zero-shot | 0.5877 | 0.6255 |
