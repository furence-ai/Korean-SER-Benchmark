"""실험 결과 종합 비교 — registry 표 + 클래스별 F1 + confusion matrix + top-k 분포(diagnostics).

registry 의 모든 run 을 test_macro_f1 내림차순 표로. test_logits.npz 가 있는 run 은
confusion matrix + top-k 정확도(정답이 top-2 안에 들 확률 등)까지. md 파일로 저장.

실행:
    uv run python -m scripts.report                                                  # runs/registry.jsonl
    uv run python -m scripts.report --registry runs/all/registry.jsonl --cm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from ser.cli import add_common_overrides, resolve_dataset_ctx
from ser.config import backbone_dirname, load_config
from ser.diagnostics import analyze_npz, format_report
from ser.labels import resolve_label_subset


def _f(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


_MODE_KO = {"head_only_otf": "head-only", "full_ft": "full-FT", "pretrained": "원본 zero-shot", "head_cached": "head"}


def _pretty_src(s: str) -> str:
    """음향 소스 식별자(em_full, ...-원본, ...-ckpt 등) → 사람이 읽는 이름."""
    s = (s or "").lower()
    bb = "emotion2vec" if ("em" in s or "emotion2vec" in s) else ("wav2vec2" if ("w2v" in s or "wav2vec2" in s) else (s or "?"))
    mode = ("head-only" if "head" in s else "full-FT" if ("full" in s or "ckpt" in s)
            else "원본" if ("pre" in s or "원본" in s) else "")
    return f"{bb} {mode}".strip()


def _describe(r: dict) -> str:
    """registry 행 → 사람이 읽는 실험 설명 (+ 텍스트 전사 출처 GT/STT 표기)."""
    base = _describe_base(r)
    if r.get("transcript") == "stt" and r.get("modality") in ("text", "fusion"):
        base += " 〔STT〕"
    return base


def _describe_base(r: dict) -> str:
    """registry 행 → 사람이 읽는 실험 설명 (메타 없으면 run_dir 경로로 추론)."""
    mod = r.get("modality")
    rd = r.get("run_dir")
    name = Path(rd).name if rd else ""
    parent = Path(rd).parent.name if rd else ""
    if mod == "acoustic":
        bb = backbone_dirname(r.get("backbone") or "")
        bbk = "emotion2vec" if "emotion2vec" in bb else ("wav2vec2" if ("wav2vec2" in bb or "xlsr" in bb) else bb)
        mode = _MODE_KO.get(r.get("mode"))
        if not mode:   # mode 메타 없으면 run 이름으로 추론
            mode = ("head-only" if "head" in name else "full-FT" if "full" in name
                    else "원본 zero-shot" if "pre" in name else (r.get("mode") or ""))
        return f"음향단독 · {bbk} {mode}".strip()
    if mod == "text":
        return f"텍스트단독 · {backbone_dirname(r.get('backbone') or 'RoBERTa')}"
    if mod == "fusion":
        method = r.get("fusion_method") or parent   # early 는 log_run 경유 → run_dir parent(early)
        src = r.get("acoustic") or name
        meth = {"late": "late", "early": "early", "cross": "cross-attn", "llm": "LLM top-2"}.get(method, method or "")
        return f"{meth} · 음향[{_pretty_src(src)}] + RoBERTa"
    return r.get("run_id", "")


def _macro_pr(run_dir) -> tuple:
    """test_logits.npz 에서 macro precision / recall (없으면 None)."""
    npz = Path(run_dir) / "test_logits.npz" if run_dir else None
    if not (npz and npz.is_file()):
        return None, None
    from sklearn.metrics import precision_score, recall_score
    d = np.load(npz, allow_pickle=True)
    y, p = d["labels"], d["logits"].argmax(1)
    return (float(precision_score(y, p, average="macro", zero_division=0)),
            float(recall_score(y, p, average="macro", zero_division=0)))


def _save_cm_png(cm, names, title, out_png) -> None:
    """confusion matrix → matplotlib PNG (정규화 색 + 카운트 주석)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(names)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("pred"); ax.set_ylabel("GT"); ax.set_title(title, fontsize=9)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_overrides(p)
    p.add_argument("--registry", type=Path, default=Path("runs/registry.jsonl"))
    p.add_argument("--out", type=Path, default=None, help="기본 <registry 폴더>/report.md")
    p.add_argument("--cm", action="store_true", help="run별 confusion matrix + top-k 분포 포함")
    args = p.parse_args()

    ctx = resolve_dataset_ctx(load_config(args))
    nc, nm = ctx.num_classes, ctx.label_to_english
    names = [nm[i] for i in range(nc)]
    if not args.registry.is_file():
        raise SystemExit(f"registry 없음: {args.registry}")
    rows = [json.loads(l) for l in args.registry.open(encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: (r.get("test_macro_f1") if r.get("test_macro_f1") is not None else -1), reverse=True)

    L = [f"# 실험 결과 비교  ({args.registry})", "", f"총 {len(rows)} run · test_macro_f1 내림차순",
         "(mF1/macroP/macroR = macro 평균, 클래스별 F1 은 우측). precision/recall 은 test_logits.npz 있는 run 만.", "",
         "| # | 구성 | mF1 | acc | macroP | macroR | " + " | ".join(names) + " |",
         "|---|---|---|---|---|---|" + "---|" * nc]
    for i, r in enumerate(rows, 1):
        pcf = r.get("per_class_f1") or []
        pcs = "".join(f" {pcf[j]:.3f} |" if j < len(pcf) else " - |" for j in range(nc))
        pr, rc = _macro_pr(r.get("run_dir"))
        L.append(f"| {i} | {_describe(r)} | {_f(r.get('test_macro_f1'))} | {_f(r.get('test_acc'))} | "
                 f"{_f(pr)} | {_f(rc)} |{pcs}")

    if args.cm:
        n_png = 0
        for r in rows:
            rd = r.get("run_dir")
            npz = Path(rd) / "test_logits.npz" if rd else None
            if not (npz and npz.is_file()):
                continue
            disp = "/".join(Path(rd).parts[-2:])
            d = np.load(npz, allow_pickle=True)
            cm = confusion_matrix(d["labels"], d["logits"].argmax(1), labels=list(range(nc)))
            png = Path(rd) / "confusion_matrix.png"
            _save_cm_png(cm, names, f"{disp}  mF1={_f(r.get('test_macro_f1'))}", png)
            n_png += 1
            L += ["", f"## {disp}  (test mF1={_f(r.get('test_macro_f1'))})",
                  f"![cm]({png})", "",
                  "```", format_report(analyze_npz(str(npz), names)),
                  "confusion (행=정답, 열=예측) " + " ".join(names) + ":", str(cm), "```"]
        print(f"[cm] confusion_matrix.png {n_png}개 저장 (각 run_dir)", flush=True)

    report = "\n".join(L)
    print(report)
    out = args.out or (args.registry.parent / "report.md")
    out.write_text(report, encoding="utf-8")
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
