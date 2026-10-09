"""
D1 — regression gate: should a candidate model/config replace the current one?

  python src/gate.py --baseline C-vae-bf16-4gb-rerun --candidate C-trt-vae-4gb

Reads both rows from bench/results.md (written by bench.py) and checks the candidate:
- speed:   sec/image at most --max-slowdown slower than the baseline   (default 50%)
- quality: CLIP at most --max-clip-drop lower than the baseline         (default 15%)
- memory:  peak VRAM within the edge budget --vram-budget-gb            (default 4 GB)

A row that never produced numbers (OOM, no CLIP score) fails. Exit code 0 = PASS, 1 = FAIL,
so the gate can sit in a script or CI step in front of a model swap. Compare rows measured on the
same day — this laptop's speed varies between sessions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results.md"


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    """results.md table → {label: {column: value}}."""
    lines = [l for l in path.read_text().splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {}
    for line in lines[2:]:  # skip header + |---| separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows[cells[0]] = dict(zip(header, cells))
    return rows


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # "OOM", "n/a", "OOM (~3.83 in use at failure)", ...


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass/fail a candidate run against the current best")
    ap.add_argument("--baseline", required=True, help="label of the current model's results row")
    ap.add_argument("--candidate", required=True, help="label of the new model's results row")
    ap.add_argument("--max-slowdown", type=float, default=0.50, help="0.50 = up to 50%% slower")
    ap.add_argument("--max-clip-drop", type=float, default=0.15, help="0.15 = up to 15%% lower CLIP")
    ap.add_argument("--vram-budget-gb", type=float, default=4.0)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()

    rows = load_rows(args.results)
    for label in (args.baseline, args.candidate):
        if label not in rows:
            raise SystemExit(f"no row labelled '{label}' in {args.results}")
    base, cand = rows[args.baseline], rows[args.candidate]

    base_sec, cand_sec = number(base["sec/image"]), number(cand["sec/image"])
    base_clip, cand_clip = number(base["clip"]), number(cand["clip"])
    cand_vram = number(cand["peak_vram_gb"])
    if base_sec is None or base_clip is None:
        raise SystemExit(f"baseline '{args.baseline}' has no usable sec/image or CLIP")

    max_sec = base_sec * (1 + args.max_slowdown)
    min_clip = base_clip * (1 - args.max_clip_drop)
    checks = [
        ("speed", cand["sec/image"], f"<= {max_sec:.2f} s",
         cand_sec is not None and cand_sec <= max_sec),
        ("quality (CLIP)", cand["clip"], f">= {min_clip:.2f}",
         cand_clip is not None and cand_clip >= min_clip),
        ("peak VRAM", cand["peak_vram_gb"], f"<= {args.vram_budget_gb:.1f} GB",
         cand_vram is not None and cand_vram <= args.vram_budget_gb),
    ]

    print(f"baseline:  {args.baseline}  ({base_sec} s, CLIP {base_clip}, {base['peak_vram_gb']} GB)")
    print(f"candidate: {args.candidate}\n")
    for name, value, rule, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<15} {value:<32} (needs {rule})")
    passed = all(ok for *_, ok in checks)
    print(f"\n{'PASS — candidate may replace the baseline' if passed else 'FAIL — keep the baseline'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
