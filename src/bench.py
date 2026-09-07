"""
Benchmark harness: run a fixed prompt set, average the metrics, append a labeled
row to bench/results.md. This produces the A->B->C comparison table.

  Phase A:  python src/bench.py --label "A-baseline"
  Phase B:  python src/bench.py --label "B-edge-4gb" --max-vram-gb 4
  Phase C:  python src/bench.py --label "C-offload+16steps" --max-vram-gb 4 --steps 16

CLIP score (quality proxy) is computed if torchmetrics is installed; otherwise skipped.
"""
from __future__ import annotations

import argparse
import os
import statistics
from datetime import date
from pathlib import Path

import torch

from pipeline import cap_vram, load_pipeline, generate, DEFAULT_MODEL

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results.md"

# Fixed, comparable prompt set — keep this stable so rows stay comparable across phases.
PROMPTS = [
    "a small autonomous delivery robot on a city street at dusk, cinematic",
    "a red lawn-mowing robot on a green suburban lawn, golden hour",
    "a warehouse robot arm sorting parcels, industrial lighting",
    "a quadruped robot walking through a forest trail, morning fog",
    "a close-up of a circuit board with glowing traces, macro photography",
]


def maybe_clip_score(images, prompts) -> float | None:
    """Mean CLIP score of images vs their prompts; None if torchmetrics unavailable."""
    try:
        from torchmetrics.multimodal.clip_score import CLIPScore
    except Exception:
        return None
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    import numpy as np
    scores = []
    for img, prompt in zip(images, prompts):
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC->CHW uint8
        scores.append(float(metric(arr, prompt)))
    return round(statistics.mean(scores), 2)


def run(label: str, max_vram_gb: float | None, steps: int, guidance: float,
        model: str, size: int, vae_native_fp32: bool = False,
        quantize_text_encoder: bool = False) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — see CLAUDE.md §2 (reboot / --gpus all).")

    cap_vram(max_vram_gb)
    pipe = load_pipeline(model, vae_native_fp32=vae_native_fp32,
                         quantize_text_encoder=quantize_text_encoder)

    # Warm-up (discarded).
    generate(pipe, PROMPTS[0], steps=steps, guidance=guidance,
             height=size, width=size, vram_cap_gb=max_vram_gb, save=False)

    secs, ms, peaks, images = [], [], [], []
    for i, prompt in enumerate(PROMPTS):
        r = generate(pipe, prompt, steps=steps, guidance=guidance,
                     height=size, width=size, seed=i,
                     vram_cap_gb=max_vram_gb, save=True)
        secs.append(r.sec_per_image)
        ms.append(r.ms_per_step)
        peaks.append(r.peak_vram_gb)
        from PIL import Image
        images.append(Image.open(r.image_path))
        print(f"  [{i+1}/{len(PROMPTS)}] {r.sec_per_image}s  peak {r.peak_vram_gb} GB")

    clip = maybe_clip_score(images, PROMPTS)
    row = (f"| {label} | {model.split('/')[-1]} | {size}px | {steps} | "
           f"{max_vram_gb or 'none'} | {round(statistics.mean(secs),3)} | "
           f"{round(statistics.mean(ms),1)} | {round(max(peaks),2)} | "
           f"{clip if clip is not None else 'n/a'} | {date.today()} |")

    RESULTS.parent.mkdir(exist_ok=True)
    _insert_row(row)
    print(f"\nAppended to {RESULTS}:\n{row}")


def _insert_row(row: str) -> None:
    """Insert a row into the results table (not blind EOF-append — the file has a
    narrative section after the table, so a plain append would land in the wrong place)."""
    if not RESULTS.exists():
        RESULTS.write_text(row + "\n")
        return

    lines = RESULTS.read_text().splitlines()
    sep_idx = next((i for i, l in enumerate(lines) if l.startswith("|---")), None)
    if sep_idx is None:
        RESULTS.write_text(RESULTS.read_text() + row + "\n")
        return

    table_end = sep_idx + 1
    while table_end < len(lines) and lines[table_end].startswith("|"):
        table_end += 1

    if table_end == sep_idx + 1 or "no runs yet" in lines[sep_idx + 1]:
        lines[sep_idx + 1:table_end] = [row]
    else:
        lines[table_end:table_end] = [row]

    RESULTS.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SANA across a fixed prompt set")
    ap.add_argument("--label", required=True, help='e.g. "A-baseline", "B-edge-4gb"')
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-vram-gb", type=float,
                    default=float(os.environ.get("EDGE_VRAM_GB", 0)) or None)
    ap.add_argument("--vae-native-fp32", action="store_true",
                    help="EXPERIMENT: load VAE at its true on-disk fp32 precision instead of "
                         "bf16-then-upcast. Same VRAM either way; see CLAUDE.md §5 Phase C.")
    ap.add_argument("--quantize-text-encoder", action="store_true",
                    help="PHASE C: load the Gemma-2 text encoder in 8-bit (bitsandbytes) to cut "
                         "VRAM. Transformer left untouched. See CLAUDE.md §5 Phase C.")
    args = ap.parse_args()
    run(args.label, args.max_vram_gb, args.steps, args.guidance, args.model, args.size,
        vae_native_fp32=args.vae_native_fp32,
        quantize_text_encoder=args.quantize_text_encoder)


if __name__ == "__main__":
    main()
