# edge-diffusion-deploy

Running a **transformer-based text-to-image diffusion model** (NVIDIA **SANA 0.6B**) end-to-end like a
product: from R&D baseline → a **memory-constrained "edge" deployment** → an optimization pass, with real
before/after numbers at every step.

> **Scope note.** This validates the full edge-optimization + MLOps toolchain (quantization, TensorRT,
> monitoring, OTA model-swap, regression gating) on a **simulated** edge memory budget on a laptop GPU
> (RTX 3060, 6 GB). It is not claimed to run on real edge silicon.

## The story

1. **R&D / baseline** — get SANA generating on the full GPU, measure it.
2. **Edge deployment** — containerize and impose a **4 GB VRAM ceiling**, measure under constraint.
3. **Optimization** — CPU offload, fewer steps, `torch.compile`, **TensorRT**, INT8/FP8 quantization; measure the delta.

## Results

See [`bench/results.md`](bench/results.md) for the A→B→C numbers table (latency, peak VRAM, quality).

## Quickstart

```bash
docker run --gpus all -it -v "$PWD":/work -w /work \
  -e HF_HOME=/work/models \
  --name edge-diffusion \
  nvcr.io/nvidia/pytorch:24.10-py3
pip install -r requirements.txt
python src/pipeline.py --prompt "a red robot mowing a lawn, golden hour" --steps 20 --max-vram-gb 4
# next time: docker start -ai edge-diffusion   (packages already installed, no --rm)
```

## Metrics tracked

seconds/image · ms/denoising-step · peak VRAM (GB) · CLIP score (prompt adherence).

## Stack

SANA 0.6B · HuggingFace `diffusers` · PyTorch · TensorRT · CUDA MPS (memory cap) · Docker (NGC) · Gradio · pynvml/Prometheus/Grafana.
