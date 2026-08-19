"""Split fingerprint — verify you reproduced the *exact* published split.

The published repo ships no manifests (the the dataset terms forbid redistributing
the data or anything derived from it). Instead it ships a fingerprint: a hash of
the split's content that carries no audio path, no transcript and no speaker id.

Regenerate the manifests locally (see README "Reproducing the split"), then run

    uv run python -m scripts.split_fingerprint --manifest-dir data/manifests

and compare against data/SPLIT_CHECKSUMS.json. Identical hashes == identical
split, so your numbers are comparable to the published ones.

The hash is taken over sorted "<relpath>\t<label>" lines, where <relpath> is the
audio path relative to the dataset root. Absolute paths therefore do not matter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train_di", "val_di", "test_di")


def _relpath(audio: str, root: str | None) -> str:
    """Path relative to the dataset root, so local mount points don't change the hash."""
    if root and audio.startswith(root):
        return audio[len(root):].lstrip("/")
    # Fall back to the last 4 components (Training/원천데이터/<emotion>/<speaker>/<file>.wav)
    return "/".join(Path(audio).parts[-4:])


def fingerprint(path: Path, root: str | None) -> dict:
    keys, labels = [], Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.append(f"{_relpath(rec['audio'], root)}\t{rec['label']}")
            labels[rec["label"]] += 1
    keys.sort()
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return {"n": len(keys), "sha256": digest, "label_counts": dict(sorted(labels.items()))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--data-root", default=None,
                   help="Dataset root to strip from audio paths (default: keep last 4 components)")
    p.add_argument("--out", type=Path, default=None, help="Write JSON here instead of stdout")
    p.add_argument("--check", type=Path, default=None,
                   help="Compare against a published SPLIT_CHECKSUMS.json and exit non-zero on mismatch")
    args = p.parse_args()

    out = {}
    for split in args.splits:
        f = args.manifest_dir / f"{split}.jsonl"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        out[split] = fingerprint(f, args.data_root)

    blob = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(blob + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(blob)

    if args.check:
        doc = json.loads(args.check.read_text(encoding="utf-8"))
        try:
            ref = doc["configs"]["4class"]["splits"]
        except KeyError:                       # tolerate a flat {split: {...}} file
            ref = doc.get("splits", doc)
        print(f"\nchecking against {args.check}")
        bad = [s for s, v in out.items()
               if s in ref and v["sha256"] != ref[s]["sha256"]]
        for split in out:
            if split not in ref:
                print(f"  ?  {split}: not in reference")
            elif split in bad:
                print(f"  ✗  {split}: MISMATCH (expected {ref[split]['sha256'][:16]}…, "
                      f"got {out[split]['sha256'][:16]}…)")
            else:
                print(f"  ✓  {split}: matches ({out[split]['n']} items)")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
