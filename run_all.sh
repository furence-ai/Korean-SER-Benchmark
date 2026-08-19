#!/usr/bin/env bash
# ============================================================================
#   - split: 화자+대본 동시분리 (*_di)
#   - 모든 디스크 산출물(체크포인트/임베딩/logits) → $ROOT (기본 runs/all, 대용량이니 큰 디스크 권장)
#   - registry(runs/registry.jsonl, 소형)만 레포에 남음 → 전 결과 비교는 그 파일로
#
# 실행:  bash run_all.sh         (오류 시 멈춤: set -e. 매우 오래 걸림 — 음향 학습이 각 수시간)
#   재개: 이미 끝난 단계의 --output 폴더를 지우거나, 해당 줄을 주석 처리하고 다시 실행.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH

ROOT="${ROOT:-runs/all}"          # 산출물 루트. 수십~수백 GB 나오므로 큰 볼륨을 가리키게 하세요.
EM=emotion2vec/emotion2vec_plus_large
W2V=facebook/wav2vec2-large-xlsr-53
TEXT=klue/roberta-large
SPLITS="train_di val_di test_di"
TXT_NPZ=$ROOT/text/roberta/test_logits.npz
mkdir -p "$ROOT"
R(){ echo; echo "######## $* ########"; uv run python -m "$@"; }

# ============================================================================
# B. 음향 단독 4개  (각 수십분~수시간)
# ============================================================================
R ser.train --backbone "$EM"  --embed-on-the-fly   --output "$ROOT/acoustic/em_head"   # ① em head-only
R ser.train --backbone "$EM"  --full-ft-freeze-cnn --output "$ROOT/acoustic/em_full"   # ② em cnn-frozen full-FT
R ser.train --backbone "$W2V" --embed-on-the-fly   --output "$ROOT/acoustic/w2v_head"  # ③ w2v head-only
R ser.train --backbone "$W2V" --full-ft-freeze-cnn --output "$ROOT/acoustic/w2v_full"  # ④ w2v cnn-frozen full-FT
# em 원본 zero-shot logits (late①·used only as fusion input)
R scripts.dump_pretrained_logits --split test_di --out-dir "$ROOT/acoustic/em_pre"

# ============================================================================
# C. 텍스트 1개 (RoBERTa) — _di 라 누수 없음
# ============================================================================
R ser.text.extract_embeddings --splits $SPLITS --out-dir "$ROOT/emb/roberta-large"
R ser.train_cached --emb-dir "$ROOT/emb/roberta-large" --modality text --backbone "$TEXT" --output "$ROOT/text/roberta"

# ============================================================================
# D-late (5)  — 음향 logits + 텍스트 logits 평균 (재학습 없음, 공짜)
# ============================================================================
for A in em_pre em_head em_full w2v_head w2v_full; do                                              # ⑤⑥⑦⑧⑨
  R ser.fusion.late --acoustic-logits "$ROOT/acoustic/$A/test_logits.npz" --text-logits "$TXT_NPZ" --out-dir "$ROOT/late/$A"
done

# ============================================================================
# D-early (4)  — 음향 emb ⊕ 텍스트 emb → head 학습
#   먼저 음향 임베딩 추출 (원본=--backbone, full-FT=--ckpt). train_di 19만이라 추출도 다소 걸림.
# ============================================================================
R scripts.extract_acoustic_embeddings --backbone "$EM"  --splits $SPLITS --out-dir "$ROOT/emb/em_pre"
R scripts.extract_acoustic_embeddings --ckpt "$ROOT/acoustic/em_full/best.pt"  --splits $SPLITS --out-dir "$ROOT/emb/em_full"
R scripts.extract_acoustic_embeddings --backbone "$W2V" --splits $SPLITS --out-dir "$ROOT/emb/w2v_pre"
R scripts.extract_acoustic_embeddings --ckpt "$ROOT/acoustic/w2v_full/best.pt" --splits $SPLITS --out-dir "$ROOT/emb/w2v_full"
for A in em_pre em_full w2v_pre w2v_full; do                                                      # ⑩⑪⑫⑬
  R ser.fusion.early --acoustic-emb-dir "$ROOT/emb/$A" --text-emb-dir "$ROOT/emb/roberta-large" \
                 --splits $SPLITS --out-dir "$ROOT/emb/early_$A"
  R ser.train_cached --emb-dir "$ROOT/emb/early_$A" --modality fusion --output "$ROOT/early/$A"
done


# ============================================================================
# D-cross (4)  — on-the-fly 듀얼 인코더 (GPU 부담 가장 큼) — 맨 마지막
# ============================================================================
R ser.fusion.train_cross --backbone "$EM"  --output "$ROOT/cross/em_pre"                       # ⑭ em 원본
R ser.fusion.train_cross --acoustic-ckpt "$ROOT/acoustic/em_full/best.pt" --output "$ROOT/cross/em_full"   # ⑮ em full
R ser.fusion.train_cross --backbone "$W2V" --output "$ROOT/cross/w2v_pre"                      # ⑯ w2v 원본
R ser.fusion.train_cross --acoustic-ckpt "$ROOT/acoustic/w2v_full/best.pt" --output "$ROOT/cross/w2v_full" # ⑰ w2v full

# E. 종합 비교 표 + confusion matrix + top-k 분포
R scripts.report --cm
echo; echo "######## 완료 — 비교표: results/report.md  (registry: runs/registry.jsonl) ########"
