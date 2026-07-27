"""
SANA 0.6B text-to-image pipeline with an optional edge-style VRAM cap.

Phase A (baseline):   python src/pipeline.py --prompt "..." --steps 20
Phase B (edge cap):   python src/pipeline.py --prompt "..." --steps 20 --max-vram-gb 4

See CLAUDE.md §4 for why the cap is done in-process (docker can't hard-limit VRAM)
and §6 for the metric definitions this file emits.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

DEFAULT_MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


@dataclass
class GenResult:
    """One generation's metrics — see CLAUDE.md §6."""
    prompt: str
    steps: int
    height: int
    width: int
    guidance: float
    dtype: str
    vram_cap_gb: float | None
    sec_per_image: float
    ms_per_step: float
    peak_vram_gb: float
    image_path: str


def cap_vram(max_vram_gb: float | None) -> None:
    """Simulate an edge memory budget by capping PyTorch's allocator (CLAUDE.md §4, method 1).

    Exceeding the cap raises CUDA OOM on purpose — that OOM is the forcing function
    for the Phase-C optimization work, not a bug to paper over.
    """
    if not max_vram_gb:
        return
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    frac = min(max_vram_gb / total_gb, 1.0)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"[cap] VRAM budget ~{max_vram_gb:.1f} GB of {total_gb:.1f} GB "
          f"(fraction {frac:.3f})")


def load_pipeline(model_id: str = DEFAULT_MODEL,
                  dtype: torch.dtype = torch.bfloat16,
                  offload: bool = True):
    """Load SANA. On 6 GB, model CPU offload keeps the Gemma-2 text encoder from OOMing.

    If the text encoder is gated on HuggingFace, run `huggingface-cli login` first
    (see CLAUDE.md §5, Phase A note).
    """
    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype)
    # SANA's DC-AE VAE is more stable in fp32; text encoder stays in the compute dtype.
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(dtype)

    if offload:
        pipe.enable_model_cpu_offload()      # streams modules on/off GPU to fit small VRAM
    else:
        pipe.to("cuda")
    return pipe


def generate(pipe, prompt: str, steps: int = 20, guidance: float = 4.5,
             height: int = 512, width: int = 512, seed: int | None = 0,
             vram_cap_gb: float | None = None, save: bool = True) -> GenResult:
    """Run one generation and return its metrics. Assumes the first (warm-up) call is discarded upstream."""
    torch.cuda.reset_peak_memory_stats()
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    img_path = ""
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        img_path = str(OUTPUT_DIR / f"gen_{int(time.time())}.png")
        image.save(img_path)

    dtype_str = str(pipe.dtype).replace("torch.", "")
    return GenResult(
        prompt=prompt, steps=steps, height=height, width=width, guidance=guidance,
        dtype=dtype_str, vram_cap_gb=vram_cap_gb,
        sec_per_image=round(sec, 3), ms_per_step=round(sec / steps * 1000, 1),
        peak_vram_gb=round(peak_gb, 2), image_path=img_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SANA text-to-image with optional edge VRAM cap")
    ap.add_argument("--prompt", default="a small autonomous robot on a city street at dusk")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-vram-gb", type=float,
                    default=float(os.environ.get("EDGE_VRAM_GB", 0)) or None,
                    help="Edge memory budget in GB (e.g. 4). Omit for uncapped baseline.")
    ap.add_argument("--no-offload", action="store_true", help="Disable CPU offload (uses more VRAM).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — did nvidia-smi work and is the container --gpus all? See CLAUDE.md §2.")

    cap_vram(args.max_vram_gb)
    pipe = load_pipeline(args.model, offload=not args.no_offload)

    # Warm-up run (compile/JIT/allocations) — discarded, per CLAUDE.md §6.
    print("[warmup] first run (discarded)…")
    generate(pipe, args.prompt, steps=args.steps, guidance=args.guidance,
             height=args.height, width=args.width, seed=args.seed,
             vram_cap_gb=args.max_vram_gb, save=False)

    res = generate(pipe, args.prompt, steps=args.steps, guidance=args.guidance,
                   height=args.height, width=args.width, seed=args.seed,
                   vram_cap_gb=args.max_vram_gb, save=True)

    print("\n=== metrics ===")
    for k, v in asdict(res).items():
        print(f"{k:>14}: {v}")


if __name__ == "__main__":
    main()
