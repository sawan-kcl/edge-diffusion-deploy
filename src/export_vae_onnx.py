"""
C3.1 — export SANA's VAE decoder (DC-AE) to ONNX, the input format TensorRT builds engines from.

  python src/export_vae_onnx.py            # → models/onnx/vae_decoder.onnx (+ vae_ref.pt)

- Fixed shape (one 512px image) so TensorRT can optimize for exactly that size.
- Exported in fp32; TensorRT converts to bf16 when it builds the engine (C3.2).
- Saves a fixed test input + the PyTorch output (vae_ref.pt) so C3.2 can check the engine
  against the same numbers (no onnxruntime in the container to check the ONNX file directly).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pipeline import DEFAULT_MODEL

ONNX_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"


class VaeDecoder(torch.nn.Module):
    """latent → image, exactly `vae.decode` (the pipeline's latent scaling stays outside)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent, return_dict=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the SANA VAE decoder to ONNX")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=512, help="image height/width in pixels")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from diffusers import AutoencoderDC
    import onnx

    vae = AutoencoderDC.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to("cuda").eval()

    # Same formula SanaPipeline uses to get from image size to latent size.
    scale = 2 ** (len(vae.config.encoder_block_out_channels) - 1)
    shape = (1, vae.config.latent_channels, args.size // scale, args.size // scale)
    print(f"latent shape {shape} → image {args.size}x{args.size}")

    latent = torch.randn(shape, generator=torch.Generator("cuda").manual_seed(0),
                         device="cuda", dtype=torch.float32)
    model = VaeDecoder(vae)
    with torch.no_grad():
        ref = model(latent)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ONNX_DIR / "vae_decoder.onnx"
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(model, (latent,), str(onnx_path),
                          input_names=["latent"], output_names=["image"],
                          opset_version=args.opset, do_constant_folding=True)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB) "
          f"in {time.perf_counter() - t0:.1f}s")

    onnx.checker.check_model(str(onnx_path))
    print("onnx.checker: OK")

    ref_path = ONNX_DIR / "vae_ref.pt"
    torch.save({"latent": latent.cpu(), "image": ref.cpu()}, ref_path)
    print(f"reference input/output saved → {ref_path} (image shape {tuple(ref.shape)})")


if __name__ == "__main__":
    main()
