# edge-diffusion-deploy

Text-to-image generation with NVIDIA's [SANA 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
under a fixed GPU memory budget (default **4 GB**), with a benchmark harness, a TensorRT-accelerated VAE,
a regression gate, and a Gradio demo.

The default diffusers setup needs ~5.3 GB of VRAM and fails under a 4 GB budget. The optimized config
runs in **~2.5–2.8 GB** with the same image quality (CLIP score).

## Features

- **VRAM budget** — `--max-vram-gb 4` caps PyTorch's allocator; anything that doesn't fit raises CUDA OOM.
- **4-bit text encoder** — Gemma-2 encoder loaded in NF4 via bitsandbytes (the main fix for the budget).
- **TensorRT VAE** — the VAE decoder exported to ONNX and built as a bf16 TensorRT engine.
- **Benchmark harness** — fixed 5-prompt set → sec/image, ms/step, peak VRAM, CLIP score → `bench/results.md`.
- **Regression gate** — compares a candidate run against the current one; exit code 0/1.
- **Profiler** — per-phase timing and VRAM (prompt encode / denoise / decode).
- **Gradio demo** — prompt in, image out, with latency and peak VRAM.

## Results

Benchmarked on an NVIDIA RTX 3060 (6 GB), 512×512, 20 steps, 4 GB budget unless noted.

| Config | sec/image | Peak VRAM | CLIP | Fits 4 GB |
|---|---|---|---|---|
| diffusers default (bf16, CPU offload), no budget | 11.17 | 5.33 GB | 33.42 | ✗ |
| diffusers default, 4 GB budget | OOM | — | — | ✗ |
| + 4-bit text encoder | 7.66 | 2.51 GB | 33.19 | ✓ |
| + no prompt instruction | 7.58 | 2.47 GB | 33.73 | ✓ |
| + bf16 VAE | **7.25** | 2.47 GB | 33.74 | ✓ |
| + TensorRT VAE | −6% vs bf16 VAE\* | 2.80 GB | 33.73 | ✓ |

\* Measured as a same-session pair (10.75 → 10.08 s); that session ran slower overall, so only the
relative change is comparable with the rows above. The extra 0.33 GB is the engine's weights.

Full table, per-phase profiles, and notes on each change: [`bench/results.md`](bench/results.md).

## Requirements

- NVIDIA GPU with ≥ 6 GB VRAM (Ampere or newer for bf16)
- NVIDIA driver, Docker, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- ~10 GB disk for model weights (downloaded on first run into `models/`), +1 GB for the TensorRT files

Everything else comes from the NGC PyTorch container (CUDA, PyTorch, TensorRT 10.5).

## Setup

```bash
git clone <this repo> && cd edge-diffusion-deploy

docker run --gpus all -it \
  -p 7860:7860 \
  -e HF_HOME=/work/models \
  -v "$PWD":/work -w /work \
  --name edge-diffusion \
  nvcr.io/nvidia/pytorch:24.10-py3

# inside the container
pip install -r requirements.txt
echo 'export PIP_CONSTRAINT=/work/constraints.txt' >> ~/.bashrc && source ~/.bashrc
```

Later sessions: `docker start -ai edge-diffusion` (packages stay installed).

`constraints.txt` pins the container's PyTorch build so that a later `pip install` can't replace it with a
PyPI wheel, which breaks the container's prebuilt `flash_attn`. Alternatively, build the image in
`docker/Dockerfile`.

If the model download asks for authentication, run `huggingface-cli login`.

## Usage

All commands run inside the container from `/work`.

**Generate an image (recommended config):**
```bash
python src/pipeline.py --prompt "a red robot mowing a lawn, golden hour" --max-vram-gb 4 \
  --quantize-text-encoder-4bit --no-prompt-instruction --vae-bf16
```
Images are saved to `outputs/`.

**Benchmark** (appends a row to `bench/results.md`):
```bash
python src/bench.py --max-vram-gb 4 --quantize-text-encoder-4bit --no-prompt-instruction --vae-bf16 \
  --label "my-run"
```

**Regression gate** (compare two rows by label):
```bash
python src/gate.py --baseline C-vae-bf16-4gb-rerun --candidate C-trt-vae-4gb
```
Passes if the candidate is at most 50% slower, has at most 15% lower CLIP, and peaks within 4 GB.
Thresholds: `--max-slowdown`, `--max-clip-drop`, `--vram-budget-gb`.

**Profile one generation by phase:**
```bash
python src/profile_vram.py --max-vram-gb 4 --quantize-text-encoder-4bit --no-prompt-instruction \
  --vae-bf16 --split-decode
```

**Demo:**
```bash
python app/gradio_app.py          # 4 GB budget; EDGE_VRAM_GB=0 for no budget
```
Open http://localhost:7860. It loads the recommended config, plus the TensorRT VAE if the engine is built.

## TensorRT VAE

Build the engine once per machine (engines are tied to the GPU and TensorRT version):

```bash
python src/export_vae_onnx.py            # fp32 ONNX + reference input/output (models/onnx/vae_ref.pt)
python src/export_vae_onnx.py --bf16     # bf16 ONNX, used for the engine
mkdir -p models/trt
trtexec --onnx=models/onnx/vae_decoder_bf16.onnx \
        --saveEngine=models/trt/vae_decoder_bf16_strict.engine --stronglyTyped
python src/check_trt_vae.py --engine models/trt/vae_decoder_bf16_strict.engine   # accuracy + timing
```

Then add `--trt-vae models/trt/vae_decoder_bf16_strict.engine` to `pipeline.py` / `bench.py` /
`profile_vram.py`.

Why this recipe:
- `trtexec --bf16` on the fp32 ONNX only *allows* bf16; TensorRT kept most layers in fp32 and the engine was
  slower than PyTorch (284 vs 173 ms).
- `trtexec --fp16` is 2.5× faster but the VAE overflows fp16 and outputs NaN.
- A bf16 ONNX + `--stronglyTyped` forces bf16. TensorRT 10.5 has no bf16 `Resize`, so the export runs the
  decoder's upsampling in fp32. Result: 148 ms, and slightly closer to the fp32 reference than PyTorch bf16.

The engine only accepts one 512×512 image. Its weights are allocated outside PyTorch's allocator, so
they aren't limited by `--max-vram-gb`; they are added to the reported peak VRAM instead.

## Options

Shared by `pipeline.py`, `bench.py`, and `profile_vram.py`:

| Flag | Effect |
|---|---|
| `--max-vram-gb N` | Cap PyTorch's GPU memory at N GB (env: `EDGE_VRAM_GB`) |
| `--quantize-text-encoder-4bit` | Gemma-2 text encoder in 4-bit NF4 |
| `--quantize-text-encoder` | Text encoder in 8-bit (faster, but doesn't fit 4 GB) |
| `--no-prompt-instruction` | Skip SANA's built-in instruction prepended to each prompt (fewer tokens, better CLIP here) |
| `--vae-bf16` | Keep the VAE in bf16 instead of upcasting to fp32 |
| `--trt-vae ENGINE` | Decode with a TensorRT engine |
| `--pin-text-encoder` | Keep the 4-bit encoder on the GPU (faster encode, ~4.4 GB peak) |
| `--no-offload` | Disable CPU offload |
| `--vae-native-fp32` | Load the VAE at its on-disk fp32 precision |

## How it works

- **Memory budget:** `torch.cuda.set_per_process_memory_fraction()` limits what PyTorch may allocate
  (`cap_vram()` in `src/pipeline.py`).
- **Offload:** `enable_model_cpu_offload()` keeps one model (text encoder, transformer, or VAE) on the
  GPU at a time. The 4-bit encoder is moved to the GPU on every call, which is most of the encode time.
- **Where the time goes** (recommended config): prompt encode ~4 s, denoise ~1 s, decode ~1.8 s — and
  ~90% of decode is moving weights between CPU and GPU, not the VAE's computation.

## Project layout

```
src/
  pipeline.py        core: cap_vram(), load_pipeline(), generate(); also a CLI
  bench.py           5-prompt benchmark → bench/results.md
  gate.py            regression gate over results.md rows
  profile_vram.py    per-phase timing + VRAM profile
  telemetry.py       pynvml sampler (VRAM, temperature, power, utilization) → CSV
  export_vae_onnx.py VAE decoder → ONNX
  check_trt_vae.py   TensorRT engine vs PyTorch: pixel error + timing
app/gradio_app.py    web demo
bench/results.md     benchmark table and notes
docker/Dockerfile    NGC PyTorch + requirements
```

`pipeline.py` is the single source of the loading and generation logic; the CLI, benchmark, profiler,
and demo all import from it. New options go into `load_pipeline()` / `generate()`, with the flag added to
the three CLIs.

## Benchmarking notes

- Every script discards one warm-up generation before measuring.
- sec/image is wall-clock for one `pipe()` call; ms/step is sec/image ÷ steps (approximate).
- Peak VRAM is `torch.cuda.max_memory_allocated()` (plus TensorRT engine weights when used).
- CLIP score uses `openai/clip-vit-base-patch16` via torchmetrics.
- Compare rows measured in the same session — GPU speed varies with power and thermal state.

## Known issues

- `Setting clean_caption=True requires the ftfy library` — harmless warning; prompts are used as written.
- The 8-bit text encoder can't be offloaded (`transformers` blocks `.to()` on 8-bit bitsandbytes models).
- VAE tiling isn't implemented for SANA's VAE (`AutoencoderDC`) in diffusers 0.32.2.
- No long-running monitoring yet; `telemetry.py` is only used by the profiler.

## License

Code: not yet licensed. Model weights (SANA, Gemma-2) are subject to their own licenses.
