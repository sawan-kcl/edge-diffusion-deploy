"""
One-off VRAM profiler: run a single generation with the GpuTelemetry sampler
running, mark the encode / denoise / decode phase boundaries, and report which
phase the VRAM peak actually falls in.

This is the "profile it instead of guessing" step before picking the next Phase-C
lever (attention slicing / VAE tiling target the denoise+decode peak; text-encoder
work only helps the encode phase).

  python src/profile_vram.py --steps 20
  python src/profile_vram.py --steps 20 --quantize-text-encoder
  python src/profile_vram.py --steps 20 --quantize-text-encoder --max-vram-gb 4

Notes:
- Telemetry VRAM is pynvml's *whole-process* GPU usage (CUDA context + allocator
  reserve + everything), so it reads a few hundred MB higher than bench.py's
  `torch.cuda.max_memory_allocated()`. Compare shapes over time, not absolutes;
  the allocator peak is printed too for cross-reference.
- The sampler thread adds a tiny amount of overhead — don't use these timings as
  official bench rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch

from pipeline import cap_vram, load_pipeline, DEFAULT_MODEL, OUTPUT_DIR
from telemetry import GpuTelemetry


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile VRAM over one generation, by phase")
    ap.add_argument("--prompt", default="a red robot mowing a lawn, golden hour")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.2, help="Telemetry sample period (s)")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "vram_profile.csv"))
    ap.add_argument("--max-vram-gb", type=float,
                    default=float(os.environ.get("EDGE_VRAM_GB", 0)) or None)
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--vae-native-fp32", action="store_true")
    ap.add_argument("--quantize-text-encoder", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — see CLAUDE.md §2.")

    cap_vram(args.max_vram_gb)
    pipe = load_pipeline(args.model, offload=not args.no_offload,
                         vae_native_fp32=args.vae_native_fp32,
                         quantize_text_encoder=args.quantize_text_encoder)

    common = dict(prompt=args.prompt, height=args.size, width=args.size,
                  num_inference_steps=args.steps, guidance_scale=args.guidance)

    # Warm-up (discarded, no telemetry) — same rationale as CLAUDE.md §6.
    print("[warmup] discarded run…")
    pipe(generator=torch.Generator("cuda").manual_seed(args.seed), **common)

    step_times: list[float] = []       # perf_counter of each denoise-step-end, rel. to wall0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with GpuTelemetry(args.out, interval_s=args.interval) as tel:
        wall0 = time.perf_counter()

        def on_step_end(pipe_, i, t, kw):
            step_times.append(time.perf_counter() - wall0)
            return kw

        image = pipe(
            generator=torch.Generator("cuda").manual_seed(args.seed),
            callback_on_step_end=on_step_end,
            **common,
        ).images[0]
        torch.cuda.synchronize()
        t_end = time.perf_counter() - wall0

    alloc_peak = torch.cuda.max_memory_allocated() / 1e9
    img_path = OUTPUT_DIR / f"vram_profile_{int(time.time())}.png"
    image.save(img_path)

    # Phase boundaries. First callback fires after step 0 finishes, so its timestamp
    # is (encode + one step); last callback ≈ end of denoising, before VAE decode.
    encode_end = step_times[0] if step_times else t_end
    denoise_end = step_times[-1] if step_times else t_end
    phases = [
        ("encode (prompt → embeds)", 0.0, encode_end),
        ("denoise (%d steps)" % args.steps, encode_end, denoise_end),
        ("decode (VAE → image)", denoise_end, t_end),
    ]

    rows = list(csv.DictReader(open(args.out)))
    for r in rows:
        r["t_s"] = float(r["t_s"])
        r["vram_used_gb"] = float(r["vram_used_gb"])

    def window_peak(lo: float, hi: float):
        xs = [r["vram_used_gb"] for r in rows if lo <= r["t_s"] <= hi]
        return max(xs) if xs else None

    overall = max(rows, key=lambda r: r["vram_used_gb"]) if rows else None

    print("\n=== VRAM profile ===")
    print(f"  cap: {args.max_vram_gb or 'none'}   "
          f"quantize_text_encoder: {args.quantize_text_encoder}   "
          f"vae_native_fp32: {args.vae_native_fp32}")
    print(f"  samples: {len(rows)} @ {args.interval}s   total wall: {t_end:.2f}s")
    print(f"  allocator peak (torch.cuda.max_memory_allocated): {alloc_peak:.2f} GB")
    if overall:
        print(f"  telemetry peak (nvml, whole process): {overall['vram_used_gb']:.2f} GB "
              f"at t={overall['t_s']:.1f}s")
    print(f"\n  {'phase':<26} {'window':>14}   {'dur':>7}   {'nvml peak':>10}")
    for name, lo, hi in phases:
        pk = window_peak(lo, hi)
        print(f"  {name:<26} {f'{lo:.1f}–{hi:.1f}s':>14}   {hi-lo:>6.2f}s   "
              f"{(f'{pk:.2f} GB' if pk is not None else '—'):>10}")

    # which phase owns the peak
    if overall:
        for name, lo, hi in phases:
            if lo <= overall["t_s"] <= hi:
                print(f"\n  → peak is in the '{name.split(' (')[0]}' phase")
                break

    print(f"\n  csv:   {args.out}")
    print(f"  image: {img_path}")


if __name__ == "__main__":
    main()
