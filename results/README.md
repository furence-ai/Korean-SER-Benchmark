# 결과

4-class 한국어 SER(기쁨 · 슬픔 · 분노 · 중립) 19개 구성. 전부 동일한 화자+대본 분리
split(`*_di`, seed 42) 위에서 같은 엔진, 같은 지표로 측정했습니다.

- **[LEADERBOARD.md](LEADERBOARD.md)**: 19개 run 전체. 클래스별 F1 · confusion
  matrix · top-k 정확도 · calibration · 주요 오분류쌍
- **[registry.jsonl](registry.jsonl)**: 원본 레코드

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

## 유의사항

- 낭독체 연기 음성 기준입니다. 자발 발화·대화체에서 성립한다고 가정하면 안 됩니다.
- 모든 run 이 seed 42 단일 시드입니다. 1포인트 안쪽 차이는 실재한다고 읽으면 안 됩니다.
- 4개 클래스만 다룹니다.
