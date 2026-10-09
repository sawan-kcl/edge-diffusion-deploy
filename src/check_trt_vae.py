"""
C3.2 — check the TensorRT VAE engine against PyTorch: same picture? how fast?

  python src/check_trt_vae.py      # needs models/trt/vae_decoder_bf16.engine + models/onnx/vae_ref.pt

- Feeds the engine the fixed latent saved by export_vae_onnx.py and compares its image to the
  saved fp32 PyTorch image (pixel diff on the 0–255 scale).
- Also runs the PyTorch bf16 VAE (what the pipeline uses since C3.0) on the same latent, as a
  yardstick: the engine's error should be no worse than bf16 PyTorch's.
- Times both GPU-only (weights already on the GPU — no CPU-offload transfer), so it's a fair
  engine-vs-PyTorch comparison of the decode math itself.
- Saves reference | bf16 PyTorch | TensorRT side by side to outputs/trt_vae_check.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pipeline import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "models" / "onnx" / "vae_ref.pt"
DEFAULT_ENGINE = ROOT / "models" / "trt" / "vae_decoder_bf16.engine"


def to_pixels(image: torch.Tensor) -> torch.Tensor:
    """VAE output ([-1, 1]) → 0–255 floats, the scale the saved PNG ends up on."""
    return ((image.float() / 2 + 0.5).clamp(0, 1) * 255).cpu()


def diff_report(name: str, image: torch.Tensor, ref: torch.Tensor) -> None:
    d = (to_pixels(image) - to_pixels(ref)).abs()
    print(f"{name:<14} vs fp32 ref: max {d.max():6.2f}  mean {d.mean():.3f}  (0–255 scale)")


def time_ms(fn, runs: int) -> float:
    fn()  # warm-up, discarded
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the TensorRT VAE engine vs PyTorch")
    ap.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=10, help="timed runs (after one warm-up)")
    args = ap.parse_args()

    import tensorrt as trt
    from diffusers import AutoencoderDC
    from PIL import Image

    ref = torch.load(REF_PATH)
    latent = ref["latent"].to("cuda")
    ref_image = ref["image"]

    # --- TensorRT engine ---
    free_before, _ = torch.cuda.mem_get_info()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    context = engine.create_execution_context()
    trt_image = torch.empty(tuple(engine.get_tensor_shape("image")), dtype=torch.float32, device="cuda")
    context.set_tensor_address("latent", latent.data_ptr())
    context.set_tensor_address("image", trt_image.data_ptr())
    torch.cuda.synchronize()
    engine_gb = (free_before - torch.cuda.mem_get_info()[0]) / 1e9

    stream = torch.cuda.Stream()  # TensorRT warns (and adds syncs) on the default stream

    def run_trt():
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)

    with torch.cuda.stream(stream):
        trt_ms = time_ms(run_trt, args.runs)

    # --- PyTorch bf16 VAE (the pipeline's current decode) ---
    vae = AutoencoderDC.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
    vae = vae.to("cuda").eval()
    with torch.no_grad():
        def run_torch():
            return vae.decode(latent.to(torch.bfloat16), return_dict=False)[0]
        torch_ms = time_ms(run_torch, args.runs)
        torch_image = run_torch()

    print()
    diff_report("PyTorch bf16", torch_image, ref_image)
    diff_report("TensorRT", trt_image, ref_image)
    print(f"\ndecode time (GPU only, mean of {args.runs}):  "
          f"PyTorch bf16 {torch_ms:.1f} ms  |  TensorRT {trt_ms:.1f} ms  "
          f"({torch_ms / trt_ms:.1f}x)")
    print(f"engine GPU memory (weights + workspace): ~{engine_gb:.2f} GB")

    row = torch.cat([to_pixels(ref_image), to_pixels(torch_image), to_pixels(trt_image)], dim=3)
    out = ROOT / "outputs" / "trt_vae_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(row[0].permute(1, 2, 0).round().byte().numpy()).save(out)
    print(f"saved fp32 ref | PyTorch bf16 | TensorRT → {out}")


if __name__ == "__main__":
    main()
