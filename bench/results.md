# Benchmark results — the A→B→C story

Each row = mean over the fixed 5-prompt set in `src/bench.py` (peak VRAM is the max across prompts).
`src/bench.py --label ...` appends rows automatically. Fill the narrative notes by hand.

Metric definitions: sec/image and ms/step are warm-run wall-clock timings (GPU-synchronized);
peak VRAM is `torch.cuda.max_memory_allocated()`; CLIP is a prompt-adherence proxy score.

| label | model | size | steps | vram_cap_gb | sec/image | ms/step | peak_vram_gb | clip | date |
|-------|-------|------|-------|-------------|-----------|---------|--------------|------|------|
| A-baseline | Sana_600M_512px_diffusers | 512px | 20 | none | 11.172 | 558.6 | 5.33 | 33.42 | 2026-07-27 |

## Narrative (fill as you go)

**Phase A — baseline (native, uncapped):**
- What worked / what the numbers were:
- Surprises (e.g. Gemma text-encoder memory, needed offload?):

**Phase B — edge deployment (4 GB cap):**
- Did it run, OOM, or slow down? By how much?
- What broke:

**Phase C — optimization (per change, keep the deltas):**
- offload/slicing/tiling → VRAM before/after:
- fewer steps (20 → ?) → latency before/after, quality cost:
- torch.compile → latency before/after:
- TensorRT → latency/VRAM before/after:
- quantization (INT8/FP8) → VRAM/latency before/after, quality cost:

**One-line summary:** e.g. *"Cut latency X→Y and VRAM 6→<4 GB with <Z CLIP-point quality cost."*
