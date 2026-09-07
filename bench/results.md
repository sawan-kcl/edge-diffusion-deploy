# Benchmark results — the A→B→C story

Each row = mean over the fixed 5-prompt set in `src/bench.py` (peak VRAM is the max across prompts).
`src/bench.py --label ...` appends rows automatically. Fill the narrative notes by hand.

Metric definitions: sec/image and ms/step are warm-run wall-clock timings (GPU-synchronized);
peak VRAM is `torch.cuda.max_memory_allocated()`; CLIP is a prompt-adherence proxy score.

| label | model | size | steps | vram_cap_gb | sec/image | ms/step | peak_vram_gb | clip | date |
|-------|-------|------|-------|-------------|-----------|---------|--------------|------|------|
| A-baseline | Sana_600M_512px_diffusers | 512px | 20 | none | 11.172 | 558.6 | 5.33 | 33.42 | 2026-07-27 |
| B-edge-4gb | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | OOM | OOM | OOM (~3.83 in use at failure) | n/a | 2026-07-28 |
| C-vae-native-fp32 | Sana_600M_512px_diffusers | 512px | 20 | none | 11.232 | 561.6 | 5.33 | 33.39 | 2026-07-30 |

## Narrative (fill as you go)

**Phase A — baseline (native, uncapped):**
- What worked / what the numbers were: SANA loads and generates cleanly with `enable_model_cpu_offload()`, no special handling needed. 11.17 sec/image, 558.6 ms/step, 5.33 GB peak VRAM at 512px/20 steps.
- Surprises: peak VRAM (5.33 GB) is already close to the 6 GB card limit even uncapped — little headroom before hitting a wall.

**Phase B — edge deployment (4 GB cap):**
- Did it run, OOM, or slow down? Immediate OOM — failed during the warm-up run, before any of the 5 timed prompts (0/5 completed).
- What broke: the crash is inside `accelerate`'s CPU-offload hook (`pre_forward` → `module.to(execution_device)`), while it tries to move the **Gemma-2 text encoder** onto the GPU to encode the prompt. So under a 4 GB cap, the text encoder alone doesn't fit when its turn comes to be swapped onto the GPU — this happens before the diffusion transformer or VAE are even touched. `enable_model_cpu_offload()` streams modules one at a time, but "one module" here is still too big for the remaining budget.

**Phase C — optimization (per change, keep the deltas):**
- offload/slicing/tiling → VRAM before/after:
- fewer steps (20 → ?) → latency before/after, quality cost:
- torch.compile → latency before/after:
- TensorRT → latency/VRAM before/after:
- quantization (INT8/FP8) → VRAM/latency before/after, quality cost:

- **[opt #1] Text-encoder 8-bit quantization (`--quantize-text-encoder`, bitsandbytes on Gemma-2 only):**
  Profiled with `src/profile_vram.py` (phase-marked VRAM trace), 512px / 20 steps, uncapped:

  | phase | no quant | 8-bit encoder |
  |---|---|---|
  | total wall | 11.23 s | **4.73 s** (2.4× faster) |
  | encode | ~7.6 s | ~1.1 s |
  | denoise (20 steps) | ~1.0 s | ~1.0 s (~49 ms/step) |
  | decode (VAE) | ~2.7 s | ~2.7 s |
  | allocator peak VRAM | 5.33 GB | 5.22 GB (≈unchanged) |
  | peak falls in phase | encode | decode |

  - **This is a latency win, not a memory win (uncapped).** Default `enable_model_cpu_offload()` streams the
    full ~5 GB bf16 Gemma-2 encoder CPU→GPU→CPU on *every* generation (~7 s). 8-bit shrinks it enough to
    keep it **pinned resident on the GPU** (excluded from offload) — encode drops from ~7.6 s to ~1.1 s.
    The bf16 encoder can't be pinned (5 GB won't fit alongside the transformer + VAE on 6 GB); 8-bit is
    what *enables* the pin, and the pin is what removes the per-generation PCIe round-trip.
  - **Peak VRAM barely moved** and shifted encode→decode: the pinned encoder now coincides with the fp32
    VAE decode, which is the new bottleneck.
  - 8-bit Gemma-2-2B pins at **~3.5 GB**, not ~2.7 — bitsandbytes leaves the embedding table unquantized
    and Gemma-2's vocab is 256k tokens (~1.2 GB fp16 embedding).
  - **Still OOMs under the 4 GB cap** (`C-te-8bit-4gb`, not recorded as a row — died in warm-up): the
    original Phase B *encode* OOM is cleared, but a new *denoise* OOM appears — the offload hook can't fit
    the transformer (~1.2 GB) onto the GPU on top of the ~3.5 GB pinned encoder (3.73 GiB ceiling).
  - Bug fixed en route: `enable_model_cpu_offload()` + a bnb-8-bit encoder hit a device-mismatch in
    `_execution_device` (returns CPU because the pinned/hookless encoder is first in the component list);
    fix is to add `text_encoder` to `pipe._exclude_from_cpu_offload` before enabling offload.
  - **Next:** try 4-bit (NF4) — should pin at ~2 GB and clear the 4 GB cap. Then vae-tiling / attention-slicing.
  - Metric note: `bench.py`'s ms/step is `total_pipe_time / steps`, dominated by encode + decode — not real
    denoising (~49 ms/step). The step callback now lives in `profile_vram.py` and could feed `bench.py`.
- [experiment] VAE native fp32 vs. bf16-then-upcast (`--vae-native-fp32`) → CLIP score before/after (expect same VRAM, testing quality only): VRAM identical (5.33 GB both), confirming the "no memory cost" prediction. CLIP 33.39 vs. 33.42 baseline — a 0.03 difference, within normal run-to-run noise, not a meaningful change. Conclusion: skipping the bf16 round-trip is theoretically more precise but produces no measurable quality difference here — the simpler default code isn't actually costing anything in practice for this model.

**One-line summary:** e.g. *"Cut latency X→Y and VRAM 6→<4 GB with <Z CLIP-point quality cost."*
