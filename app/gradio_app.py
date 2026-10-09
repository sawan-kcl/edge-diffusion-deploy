"""
Live demo: type a prompt -> get an image, with latency + peak VRAM shown.
This is the Month-2 "run it live" deliverable.

  python app/gradio_app.py                 # best config, under the 4 GB edge cap
  EDGE_VRAM_GB=0 python app/gradio_app.py  # same config, uncapped

Best config = 4-bit text encoder + no prompt instruction + bf16 VAE + TensorRT VAE engine
(falls back to the PyTorch VAE if the engine file isn't built).

Then open the printed local URL in a browser.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr

from pipeline import cap_vram, load_pipeline, generate  # noqa: E402

_CAP = float(os.environ.get("EDGE_VRAM_GB", 4)) or None
_TRT_VAE = Path(__file__).resolve().parent.parent / "models" / "trt" / "vae_decoder_bf16_strict.engine"
_PIPE = None


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        cap_vram(_CAP)
        _PIPE = load_pipeline(quantize_text_encoder_4bit=True, vae_bf16=True,
                              trt_vae=str(_TRT_VAE) if _TRT_VAE.exists() else None)
    return _PIPE


def infer(prompt: str, steps: int, guidance: float):
    pipe = _get_pipe()
    r = generate(pipe, prompt, steps=int(steps), guidance=float(guidance),
                 vram_cap_gb=_CAP, save=True, prompt_instruction=False)
    stats = (f"{r.sec_per_image:.2f} s/image · {r.ms_per_step:.0f} ms/step · "
             f"peak {r.peak_vram_gb:.2f} GB · {r.dtype}"
             + (" · TensorRT VAE" if hasattr(pipe, "trt_vae_resident_gb") else " · PyTorch VAE")
             + (f" · cap {_CAP} GB" if _CAP else " · uncapped"))
    return r.image_path, stats


with gr.Blocks(title="SANA edge demo") as demo:
    gr.Markdown("# SANA 0.6B — edge-simulated text-to-image\nType a prompt; latency and peak VRAM are shown below.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="a small autonomous robot on a city street at dusk", scale=4)
        go = gr.Button("Generate", variant="primary", scale=1)
    with gr.Row():
        steps = gr.Slider(4, 30, value=20, step=1, label="Denoising steps")
        guidance = gr.Slider(1.0, 8.0, value=4.5, step=0.5, label="Guidance scale")
    image = gr.Image(label="Output")
    stats = gr.Markdown()
    go.click(infer, inputs=[prompt, steps, guidance], outputs=[image, stats])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
