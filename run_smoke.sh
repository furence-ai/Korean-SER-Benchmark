#!/usr/bin/env bash
# ============================================================================
# SMOKE — run_all.sh 와 동일한 19단계를 미니 데이터(_smoke)+epochs 1 로 빠르게 한 바퀴.
#   목적: "어디서 깨지나"만 확인 (숫자는 무의미). 산출물 → $ROOT (기본 runs/smoke)
#   본 실험(run_all.sh, 기본 runs/all)과 안 섞임.
# 실행:  bash run_smoke.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONPATH

ROOT="${ROOT:-runs/smoke}"
export SER_REGISTRY="$ROOT/registry.jsonl"   # 스모크 결과는 별도 registry (본 results/ 안 더럽힘)
C=configs/smoke.yaml                 # 학습 cmd 는 이 config (split=_smoke, epochs=1)
EM=emotion2vec/emotion2vec_plus_large
W2V=facebook/wav2vec2-large-xlsr-53
TEXT=klue/roberta-large
SPLITS="train_smoke val_smoke test_smoke"
TXT_NPZ=$ROOT/text/roberta/test_logits.npz
mkdir -p "$ROOT"
R(){ echo; echo "######## $* ########"; uv run python -m "$@"; }

# 0. 미니 subset 생성 (_di → _smoke, per-class 50)
R scripts.make_smoke_subset --per-class 50

# B. 음향 단독 4 (config=smoke → epochs 1, split _smoke)
R ser.train --config "$C" --backbone "$EM"  --embed-on-the-fly   --output "$ROOT/acoustic/em_head"
R ser.train --config "$C" --backbone "$EM"  --full-ft-freeze-cnn --output "$ROOT/acoustic/em_full"
R ser.train --config "$C" --backbone "$W2V" --embed-on-the-fly   --output "$ROOT/acoustic/w2v_head"
R ser.train --config "$C" --backbone "$W2V" --full-ft-freeze-cnn --output "$ROOT/acoustic/w2v_full"
R scripts.dump_pretrained_logits --split test_smoke --out-dir "$ROOT/acoustic/em_pre"

# C. 텍스트
R ser.text.extract_embeddings --splits $SPLITS --out-dir "$ROOT/emb/roberta-large"
R ser.train_cached --config "$C" --emb-dir "$ROOT/emb/roberta-large" --modality text --backbone "$TEXT" --output "$ROOT/text/roberta"

# D-late (5)
for A in em_pre em_head em_full w2v_head w2v_full; do
  R ser.fusion.late --acoustic-logits "$ROOT/acoustic/$A/test_logits.npz" --text-logits "$TXT_NPZ" --out-dir "$ROOT/late/$A"
done

# D-early (4)
R scripts.extract_acoustic_embeddings --backbone "$EM"  --splits $SPLITS --out-dir "$ROOT/emb/em_pre"
R scripts.extract_acoustic_embeddings --ckpt "$ROOT/acoustic/em_full/best.pt"  --splits $SPLITS --out-dir "$ROOT/emb/em_full"
R scripts.extract_acoustic_embeddings --backbone "$W2V" --splits $SPLITS --out-dir "$ROOT/emb/w2v_pre"
R scripts.extract_acoustic_embeddings --ckpt "$ROOT/acoustic/w2v_full/best.pt" --splits $SPLITS --out-dir "$ROOT/emb/w2v_full"
for A in em_pre em_full w2v_pre w2v_full; do
  R ser.fusion.early --acoustic-emb-dir "$ROOT/emb/$A" --text-emb-dir "$ROOT/emb/roberta-large" \
                 --splits $SPLITS --out-dir "$ROOT/emb/early_$A"
  R ser.train_cached --config "$C" --emb-dir "$ROOT/emb/early_$A" --modality fusion --output "$ROOT/early/$A"
done


# D-cross (4)
R ser.fusion.train_cross --config "$C" --backbone "$EM"  --output "$ROOT/cross/em_pre"
R ser.fusion.train_cross --config "$C" --acoustic-ckpt "$ROOT/acoustic/em_full/best.pt" --output "$ROOT/cross/em_full"
R ser.fusion.train_cross --config "$C" --backbone "$W2V" --output "$ROOT/cross/w2v_pre"
R ser.fusion.train_cross --config "$C" --acoustic-ckpt "$ROOT/acoustic/w2v_full/best.pt" --output "$ROOT/cross/w2v_full"

# 종합 비교 표 + CM + top-k
R scripts.report --registry "$ROOT/registry.jsonl" --cm
echo; echo "######## SMOKE 완료 — 깨진 단계 없으면 run_all.sh OK. 비교표: $ROOT/report.md ########"
