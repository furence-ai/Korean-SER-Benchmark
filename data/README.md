# data/

이 디렉토리에는 **공개 레포 기준으로 데이터가 없습니다.** 설계상 그렇습니다.

기준 실험은 [AI-Hub 「감성 및 발화 스타일별 음성합성 데이터」][aihub] 를 씁니다.
AI-Hub 는 승인받은 개인에게만 이용을 허가하며 **데이터 및 그로부터 파생된 자료를
제3자에게 제공하는 것을 금지합니다.** manifest 는 오디오 경로와 화자 식별자를,
전사 파일은 원문 대본을 그대로 담으므로 둘 다 파생물에 해당합니다. 그래서 어느
쪽도 공개하지 않습니다.

대신 들어 있는 것:

| 파일 | 용도 |
|---|---|
| `SPLIT_CHECKSUMS.json` | 기준 split 의 SHA-256 지문 (동일한 split 을 재현했는지 증명용) |

로컬에서 데이터를 다시 만드는 데 필요한 것은 전부 레포 안에 있습니다. 최상위
[README](../README.md) 의 **split 재현** 을 참고하세요.

재생성 후 이 디렉토리는 다음을 담게 됩니다 (전부 git 무시 대상):

```
data/
  manifests/      train.jsonl · val.jsonl                        (① build_manifest)
                  train_si.jsonl · val_si.jsonl · test_si.jsonl  (② make_speaker_split)
                  train_di.jsonl · val_di.jsonl · test_di.jsonl  (③ 학습에 쓰는 최종 split)
  transcripts/    라벨 json 에서 추출한 GT 전사
  embeddings/     캐시된 백본 임베딩
```

[aihub]: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=466
