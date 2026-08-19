# 데이터 준비

## 1. 데이터셋 받기

**「감성 및 발화 스타일별 음성합성 데이터」** (AI-Hub)
<https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=466>

대본 9,182개를 각각 약 32명이 낭독한 한국어 낭독체 코퍼스입니다. 감정 라벨이 달려
있고, 이 벤치마크는 그중 **기쁨 · 슬픔 · 분노 · 중립** 4종을 씁니다.

## 2. 디렉토리 배치

압축을 풀면 아래 구조가 됩니다. 이 구조 **그대로** 두어야 합니다. 어댑터가 경로
규칙으로 라벨과 전사를 찾습니다.

```
/path/to/emotion-style-speech/                      ← 이 경로를 SER_DATA_ROOT 로 지정
├── Training/
│   ├── 원천데이터/
│   │   └── 1.감정/                        ← task (configs/datasets/emotion_style_speech.yaml)
│   │       ├── 1.기쁨/
│   │       │   ├── 0001_G1A3E1S0C0_ABC/  ← 화자 폴더
│   │       │   │   ├── 0001_G1A3E1S0C0_ABC_000001.wav
│   │       │   │   └── ...
│   │       │   └── ...
│   │       ├── 2.슬픔/  3.분노/  4.불안/
│   │       └── 5.상처/  6.당황/  7.중립/
│   └── 라벨링데이터/
│       └── 1.감정/                        ← 원천데이터와 같은 트리 구조
│           └── 1.기쁨/0001_G1A3E1S0C0_ABC/0001_G1A3E1S0C0_ABC_000001.json
└── Validation/
    ├── 원천데이터/1.감정/...
    └── 라벨링데이터/1.감정/...
```

핵심 규칙 세 가집니다.

- **`원천데이터`(wav)와 `라벨링데이터`(json)가 같은 트리를 이룹니다.** 전사를 찾을 때
  wav 경로에서 `/원천데이터/` 를 `/라벨링데이터/` 로 바꾸고 확장자를 `.json` 으로
  바꾸는 방식이라([`ser/datasets/emotion_style_speech.py`](../ser/datasets/emotion_style_speech.py)의
  `transcript_for`), 한쪽만 옮기면 전사를 못 찾습니다.
- **감정 폴더 이름이 라벨입니다.** `1.기쁨` ~ `7.중립` 이름을 그대로 둡니다. 알려진
  이름이 아닌 폴더는 스캔에서 제외됩니다.
- **화자 폴더 이름이 화자 ID 입니다.** `0001_G1A3E1S0C0_ABC` 에서 앞 4자리와 뒤
  이니셜이 사람을 식별하고(`0001_ABC`), 가운데 `E1` 은 감정 코드라 같은 사람이 감정별로
  다른 폴더를 갖습니다. 화자 분리 split 은 이 사람 단위로 묶습니다.

경로를 알려주는 방법은 세 가지이고 위쪽이 우선입니다.

```bash
uv run python -m scripts.build_manifest --data-root /path/to/emotion-style-speech   # ① CLI
export SER_DATA_ROOT=/path/to/emotion-style-speech                                  # ② 환경변수
# ③ configs/datasets/emotion_style_speech.yaml 의 data_root 수정
```

### 라벨 json 에서 읽는 필드

```json
{
  "전사정보": { "OrgLabelText": "<발화 전사 텍스트>" },
  "화자정보": { "Emotion": "Happy" }
}
```

`전사정보.OrgLabelText` 만 씁니다(GT 전사). 감정은 json 이 아니라 **폴더 이름**에서
가져옵니다.

## 3. 클래스 선택

코퍼스에는 감정 7종이 있지만 이 벤치마크는 4종만 씁니다. 어떤 클래스를 쓸지는
**원본 라벨 ID 목록**으로 지정합니다.

| 원본 ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 감정 | 기쁨 | 슬픔 | 분노 | 불안 | 상처 | 당황 | 중립 |

기본값은 `keep_labels: [0, 1, 2, 6]`(기쁨·슬픔·분노·중립)입니다. 선택된 클래스는
**0부터 연속 번호로 다시 매겨집니다**(dense remap).

```
원본 0 기쁨 → 0        학습·추론에서 쓰는 번호는 항상 0..N-1 이고,
원본 1 슬픔 → 1        체크포인트의 label_to_english 에 이 매핑이 저장됩니다.
원본 2 분노 → 2        (선택 안 된 라벨의 발화는 학습 시 자동으로 버려진다)
원본 6 중립 → 3
```

지정하는 곳은 두 군데이고 CLI 가 우선입니다.

```yaml
# configs/default.yaml
common:
  keep_labels: [0, 1, 2, 6]   # null 이면 데이터셋 라벨공간 전체
```

```bash
uv run python -m ser.train --keep-labels 0 1 2 6      # CLI 로 덮어쓰기
```

**중요:** 클래스 선택은 split 을 만들 때(`make_speaker_split --keep-labels`)와 학습할
때(`keep_labels`) **같은 값이어야** 합니다. split 단계에서 이미 해당 클래스만 남기고
사람별 클래스 균형을 맞추기 때문입니다. 기본값과 다른 조합을 쓰면 공개된
`SPLIT_CHECKSUMS.json` 과 지문이 달라지고, 리더보드 수치와도 비교할 수 없게 됩니다.

## 4. 준비 파이프라인

```bash
uv sync
export SER_DATA_ROOT=/path/to/emotion-style-speech
```

전체 흐름은 세 단계입니다.

```
raw 디렉토리
   │  ① build_manifest              라벨 필터 없이 전체 스캔
   ▼
train.jsonl · val.jsonl
   │  ② make_speaker_split          클래스 선택 + 사람 단위 홀드아웃
   ▼
train_si.jsonl · val_si.jsonl · test_si.jsonl
   │  ③ make_disjoint_split         대본 분리 추가
   ▼
train_di.jsonl · val_di.jsonl · test_di.jsonl      ← 학습에 쓰는 최종 split
```

### ① manifest 생성

```bash
uv run python -m scripts.build_manifest --splits train val
```

`원천데이터/1.감정/<감정>/<화자>/*.wav` 를 `sorted()` 순서로 훑어
`data/manifests/{train,val}.jsonl` 을 만듭니다.

```json
{"audio": "/path/to/.../1.기쁨/0001_.../0001_..._000001.wav",
 "label": 0, "speaker": "0001_G1A3E1S0C0_ABC"}
```

### ② 화자 분리 + 클래스 선택

```bash
uv run python -m scripts.make_speaker_split --keep-labels 0 1 2 6 --n-val 4 --n-test 4
```

두 가지를 동시에 합니다.

- **클래스 선택**: `--keep-labels` 에 없는 라벨의 발화를 버립니다.
- **사람 단위 홀드아웃**: 한 사람이 한 split 에만 들어가도록 나눕니다. `speaker`
  필드는 '사람 × 감정' 단위라 감정 코드를 떼고 `번호_이니셜`로 묶다
  (`0001_G1A3E1S0C0_ABC` → `0001_ABC`). val/test 는 선택된 클래스를 모두 보유한
  사람에서 4명씩 뽑아 클래스 균형을 보장하고, 나머지는 전부 train 입니다.

출력은 `data/manifests/{train,val,test}_si.jsonl` 이며 `si` = speaker-independent
입니다. 원본 `train.jsonl`/`val.jsonl` 은 보존됩니다.

### ③ 대본 분리

```bash
uv run python -m ser.text.extract_transcripts     # ③ 이 전사를 필요로 함
uv run python -m scripts.make_disjoint_split --seed 42
```

`_si` 3개를 전체 풀로 삼아 **대본까지 겹치지 않게** 다시 나눕니다. 각 split =
(그 split 전용 화자) ∩ (그 split 전용 대본)이고, 교차 셀에 떨어지는 약 33% 는
버립니다. 대본은 감정별 stratify 로 나눠 라벨 균형을 유지합니다. 왜 둘 다 분리해야
하는지는 최상위 [README](../README.md#split-규격) 를 참고하세요.

대본을 전사 텍스트로 식별하므로 **`ser.text.extract_transcripts` 를 먼저 돌려야**
합니다(`라벨링데이터` 가 제자리에 있어야 함). 결과는
`data/manifests/{train_di,val_di,test_di}.jsonl` 이고 규모는 **train 190,837 /
val 2,933 / test 2,451** 입니다.

### ④ split 검증

```bash
uv run python -m scripts.split_fingerprint \
    --manifest-dir data/manifests --data-root "$SER_DATA_ROOT" \
    --check data/SPLIT_CHECKSUMS.json
#   ✓  train_di: matches (190837 items)
#   ✓  val_di:   matches (2933 items)
#   ✓  test_di:  matches (2451 items)
```

### ⑤ 텍스트 임베딩 캐시

```bash
uv run python -m ser.text.extract_embeddings
```

KLUE-RoBERTa 로 발화당 1벡터를 뽑아 `data/embeddings/roberta-large/{split}.npz` 에
저장합니다. 백본은 동결이라 한 번만 돌리면 됩니다. 음향 단독 모델만 돌릴 거면
건너뛰어도 됩니다.

## 5. 산출물

```
data/
  manifests/      train.jsonl · val.jsonl          ← ①
                  train_si.jsonl · val_si.jsonl · test_si.jsonl   ← ②
                  train_di.jsonl · val_di.jsonl · test_di.jsonl   ← ③ (학습에 쓰는 것)
  transcripts/    {split}.jsonl                    ← ③ 이 필요로 함
  embeddings/     roberta-large/*.npz              ← ⑤ (텍스트/결합을 쓸 때만)
```