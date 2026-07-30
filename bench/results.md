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
- [experiment] VAE native fp32 vs. bf16-then-upcast (`--vae-native-fp32`) → CLIP score before/after (expect same VRAM, testing quality only): VRAM identical (5.33 GB both), confirming the "no memory cost" prediction. CLIP 33.39 vs. 33.42 baseline — a 0.03 difference, within normal run-to-run noise, not a meaningful change. Conclusion: skipping the bf16 round-trip is theoretically more precise but produces no measurable quality difference here — the simpler default code isn't actually costing anything in practice for this model.

**One-line summary:** e.g. *"Cut latency X→Y and VRAM 6→<4 GB with <Z CLIP-point quality cost."*
