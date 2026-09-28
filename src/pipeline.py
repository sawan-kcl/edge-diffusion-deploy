"""
SANA 0.6B text-to-image pipeline with an optional edge-style VRAM cap.

Phase A (baseline):   python src/pipeline.py --prompt "..." --steps 20
Phase B (edge cap):   python src/pipeline.py --prompt "..." --steps 20 --max-vram-gb 4

The VRAM cap is done in-process because plain `docker run` can't hard-limit GPU memory
(`--memory` is CPU RAM only, `--gpus` is visibility not memory, MIG is data-center-only).
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
    """One generation's metrics: sec/image, ms/step, peak VRAM, plus the run params."""
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
    """Simulate an edge memory budget by capping PyTorch's allocator.

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
                  offload: bool = True,
                  vae_native_fp32: bool = False,
                  quantize_text_encoder: bool = False,
                  quantize_text_encoder_4bit: bool = False,
                  pin_text_encoder: bool = False,
                  vae_bf16: bool = False):
    """Load SANA. On 6 GB, model CPU offload keeps the Gemma-2 text encoder from OOMing.

    If the text encoder is gated on HuggingFace, run `huggingface-cli login` first.

    vae_native_fp32: EXPERIMENT (off by default). The saved
    VAE checkpoint is true fp32 on disk. The default path below loads the whole pipeline
    in `dtype` (bf16), which downcasts the VAE to bf16 too, then upcasts it back to fp32 —
    that upcast keeps the fp32 storage size but not the detail already rounded away in the
    downcast. This flag skips the round trip and loads the VAE at its real on-disk
    precision from the start, so no detail is ever lost. Same final VRAM either way (the
    VAE ends up fp32 in both cases) — this is a quality experiment, not a memory trade-off.

    quantize_text_encoder / quantize_text_encoder_4bit: PHASE C optimization #1 (off by
    default, mutually exclusive). Loads the Gemma-2 text encoder in
    8-bit or 4-bit (NF4) via bitsandbytes instead of bf16 — cuts its weight memory (8-bit
    ~halves it; 4-bit roughly halves it again). This is the biggest single module and the
    exact thing that OOM'd in Phase B, but it only runs once per image (prompt encoding),
    so it tolerates precision loss better than the transformer (which runs once per
    denoising step, left untouched here). Needs CUDA + `bitsandbytes` installed.

    The two modes offload differently. 8-bit is always GPU-resident ("pinned"): `transformers`
    blocks `.to()` on 8-bit bnb models, so it can't be streamed. 4-bit CAN be moved, so by
    default it's streamed CPU→GPU on every call like the other modules (small peak VRAM, but
    ~4 s encode spent copying weights).

    pin_text_encoder: 4-bit only. Keep the encoder on the GPU instead of streaming it —
    faster encode, but its ~2.3 GB stays resident during denoise/decode.

    vae_bf16: keep the VAE in `dtype` (bf16) instead of upcasting it to fp32. Tests whether
    the VAE survives reduced precision (a TensorRT prerequisite); halves its memory.
    """
    if vae_bf16 and vae_native_fp32:
        raise ValueError("vae_bf16 and vae_native_fp32 are mutually exclusive")
    if quantize_text_encoder and quantize_text_encoder_4bit:
        raise ValueError("quantize_text_encoder and quantize_text_encoder_4bit are mutually exclusive")
    if pin_text_encoder and not quantize_text_encoder_4bit:
        raise ValueError("pin_text_encoder requires quantize_text_encoder_4bit (8-bit is always pinned)")
    quantized_te = quantize_text_encoder or quantize_text_encoder_4bit

    from diffusers import SanaPipeline

    load_kwargs = {}
    if vae_native_fp32:
        from diffusers import AutoencoderDC
        load_kwargs["vae"] = AutoencoderDC.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32)
    if quantized_te:
        from transformers import Gemma2Model, BitsAndBytesConfig
        if quantize_text_encoder_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=dtype)
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["text_encoder"] = Gemma2Model.from_pretrained(
            model_id, subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=dtype)

    pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs)
    # SANA's DC-AE VAE is more stable in fp32; text encoder stays in the compute dtype.
    # (No-op when vae_native_fp32=True — the VAE is already fp32 at this point.)
    if not vae_bf16:
        pipe.vae.to(torch.float32)
    if not quantized_te:
        pipe.text_encoder.to(dtype)   # can't .to(dtype) a bitsandbytes quantized module — it raises

    if offload:
        if quantized_te:
            # A pinned (hookless) text encoder makes diffusers 0.32.2's _execution_device fall
            # back to self.device (CPU, post-offload), since the encoder is first in the
            # component list — token ids go to CPU while weights sit on GPU → device-mismatch
            # RuntimeError. Excluding it makes _execution_device resolve off the transformer's hook.
            # assign (not append) — the default is a shared class-level list
            pipe._exclude_from_cpu_offload = pipe._exclude_from_cpu_offload + ["text_encoder"]
        if pin_text_encoder:
            # The exclude list is ignored for modules in model_cpu_offload_seq, so the text
            # encoder must also be dropped from the sequence or it still gets an offload hook.
            pipe.model_cpu_offload_seq = "->".join(
                m for m in pipe.model_cpu_offload_seq.split("->") if m != "text_encoder")
        pipe.enable_model_cpu_offload()      # streams modules on/off GPU to fit small VRAM
        if pin_text_encoder:
            pipe.text_encoder.to("cuda")
            if type(getattr(pipe.text_encoder, "_hf_hook", None)).__name__ == "CpuOffload":
                raise RuntimeError("pin_text_encoder: text encoder still has a CPU-offload hook")
    else:
        pipe.to("cuda")
    return pipe


def generate(pipe, prompt: str, steps: int = 20, guidance: float = 4.5,
             height: int = 512, width: int = 512, seed: int | None = 0,
             vram_cap_gb: float | None = None, save: bool = True,
             prompt_instruction: bool = True) -> GenResult:
    """Run one generation and return its metrics. Assumes the first (warm-up) call is discarded upstream."""
    torch.cuda.reset_peak_memory_stats()
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)
    # SANA prepends a ~250-token built-in instruction to every prompt by default; None skips it.
    extra = {} if prompt_instruction else {"complex_human_instruction": None}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        **extra,
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
    ap.add_argument("--vae-native-fp32", action="store_true",
                    help="EXPERIMENT: load VAE at its true on-disk fp32 precision instead of "
                         "bf16-then-upcast. Same VRAM either way.")
    ap.add_argument("--quantize-text-encoder", action="store_true",
                    help="PHASE C: load the Gemma-2 text encoder in 8-bit (bitsandbytes) to cut "
                         "VRAM. Transformer left untouched.")
    ap.add_argument("--quantize-text-encoder-4bit", action="store_true",
                    help="PHASE C: load the Gemma-2 text encoder in 4-bit NF4 (bitsandbytes) "
                         "instead of 8-bit — smaller pinned footprint. Mutually exclusive with "
                         "--quantize-text-encoder.")
    ap.add_argument("--no-prompt-instruction", action="store_true",
                    help="PHASE C: skip SANA's built-in instruction text prepended to every prompt "
                         "(~550 → ~300 encoder tokens). Faster encode; may cost quality.")
    ap.add_argument("--pin-text-encoder", action="store_true",
                    help="PHASE C: keep the 4-bit text encoder on the GPU instead of streaming it "
                         "from CPU every call. Requires --quantize-text-encoder-4bit.")
    ap.add_argument("--vae-bf16", action="store_true",
                    help="PHASE C: keep the VAE in bf16 instead of fp32 (precision check before TensorRT).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — did nvidia-smi work and is the container started with --gpus all?")

    cap_vram(args.max_vram_gb)
    pipe = load_pipeline(args.model, offload=not args.no_offload,
                         vae_native_fp32=args.vae_native_fp32,
                         quantize_text_encoder=args.quantize_text_encoder,
                         quantize_text_encoder_4bit=args.quantize_text_encoder_4bit,
                         pin_text_encoder=args.pin_text_encoder,
                         vae_bf16=args.vae_bf16)

    # Warm-up run (compile/JIT/allocations) — discarded so reported numbers are always warm.
    print("[warmup] first run (discarded)…")
    generate(pipe, args.prompt, steps=args.steps, guidance=args.guidance,
             height=args.height, width=args.width, seed=args.seed,
             vram_cap_gb=args.max_vram_gb, save=False,
             prompt_instruction=not args.no_prompt_instruction)

    res = generate(pipe, args.prompt, steps=args.steps, guidance=args.guidance,
                   height=args.height, width=args.width, seed=args.seed,
                   vram_cap_gb=args.max_vram_gb, save=True,
                   prompt_instruction=not args.no_prompt_instruction)

    print("\n=== metrics ===")
    for k, v in asdict(res).items():
        print(f"{k:>14}: {v}")


if __name__ == "__main__":
    main()
