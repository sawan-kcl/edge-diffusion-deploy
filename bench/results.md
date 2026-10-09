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
| C-te-4bit | Sana_600M_512px_diffusers | 512px | 20 | none | 7.652 | 382.6 | 2.51 | 33.19 | 2026-09-18 |
| C-te-4bit-4gb | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 7.662 | 383.1 | 2.51 | 33.19 | 2026-09-18 |
| C-te-4bit-noinstr-4gb | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 7.581 | 379.1 | 2.47 | 33.73 | 2026-09-28 |
| C-vae-bf16-4gb | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 7.254 | 362.7 | 2.47 | 33.74 | 2026-09-28 |
| C-trt-vae-4gb-battery | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 10.416 | 520.8 | 2.8 | 33.73 | 2026-10-09 |
| C-vae-bf16-4gb-rerun | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 10.746 | 537.3 | 2.47 | 33.74 | 2026-10-09 |
| C-trt-vae-4gb | Sana_600M_512px_diffusers | 512px | 20 | 4.0 | 10.076 | 503.8 | 2.8 | 33.73 | 2026-10-09 |

*No 8-bit rows above:* opt #1 (8-bit text encoder) was measured with `src/profile_vram.py` (one prompt), not
`bench.py` — uncapped 4.73 s / 5.22 GB; capped at 4 GB it OOMs. Full numbers in the opt #1 note below.

## Narrative (fill as you go)

**Phase A — baseline (native, uncapped):**
- What worked / what the numbers were: SANA loads and generates cleanly with `enable_model_cpu_offload()`, no special handling needed. 11.17 sec/image, 558.6 ms/step, 5.33 GB peak VRAM at 512px/20 steps.
- Surprises: peak VRAM (5.33 GB) is already close to the 6 GB card limit even uncapped — little headroom before hitting a wall.

**Phase B — edge deployment (4 GB cap):**
- Did it run, OOM, or slow down? Immediate OOM — failed during the warm-up run, before any of the 5 timed prompts (0/5 completed).
- What broke: the crash is inside `accelerate`'s CPU-offload hook (`pre_forward` → `module.to(execution_device)`), while it tries to move the **Gemma-2 text encoder** onto the GPU to encode the prompt. So under a 4 GB cap, the text encoder alone doesn't fit when its turn comes to be swapped onto the GPU — this happens before the diffusion transformer or VAE are even touched. `enable_model_cpu_offload()` streams modules one at a time, but "one module" here is still too big for the remaining budget.

**Phase C — optimization (per change, keep the deltas):**
- faster encode → done, see [opt #2a], [finding], [opt #2b] and the C2 result below.
- bf16 VAE (precision check before TensorRT) → done, see [C3.0] below.
- ONNX → TensorRT (VAE, then transformer) → latency/VRAM before/after:
- *(optional)* fewer steps (20 → ?) → latency before/after, quality cost:

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
  - Metric note: `bench.py`'s ms/step is `total_pipe_time / steps`, dominated by encode + decode — not real
    denoising (~49 ms/step). The step callback now lives in `profile_vram.py` and could feed `bench.py`.

- **[opt #1b] Text-encoder 4-bit NF4 quantization (`--quantize-text-encoder-4bit`, bitsandbytes on
  Gemma-2 only) — the actual Phase B fix.** Before trying this, we investigated whether the 8-bit
  encoder could stay *dynamically* offloaded (moved CPU↔GPU per call, like every other module)
  instead of pinned — `transformers` hard-blocks this: `ValueError: .to is not supported for 8-bit
  bitsandbytes models`, a deliberate library guard, not a bug we could work around. So we tried a
  smaller footprint instead, which is what 4-bit gives.

  > **Correction (2026-09-28):** we assumed 4-bit was pinned like 8-bit. It isn't — the `.to()` block is
  > 8-bit-only, so `enable_model_cpu_offload()` gives the 4-bit encoder a normal offload hook and
  > streams it CPU→GPU on every call. Confirmed in the container (encoder params on `cpu` at rest,
  > `_hf_hook` attached). The bullets below are corrected; see the [finding] entry further down.

  | | baseline (bf16) | 8-bit (pinned) | 4-bit (streamed) |
  |---|---|---|---|
  | peak VRAM | 5.33 GB | 5.22 GB | **2.51 GB** |
  | fits under 4 GB cap? | n/a | **OOMs** (denoise phase) | **yes — comfortably** |
  | sec/image (uncapped) | 11.17 | 4.73 | 7.65 |
  | sec/image (4 GB cap) | n/a | n/a (OOM) | 7.66 (~same as uncapped) |
  | CLIP | 33.42 | n/a | 33.19 |

  - **This is the first optimization that produces a working image under the 4 GB budget** — clears
    both the original Phase B failure (encoder OOMing while loading onto the GPU) and the new one
    8-bit introduced (encoder + transformer not fitting together during denoise).
  - **Capped and uncapped runs are nearly identical** (7.66s/2.51GB vs 7.65s/2.51GB) — there's real
    headroom under 4 GB, not a knife-edge pass. The low peak is because the streamed encoder has left
    the GPU before the transformer and VAE run.
  - **Slower than 8-bit** (7.65s vs 4.73s total) because 8-bit is pinned and 4-bit is streamed: the
    ~2.9 s gap is 4-bit's weights being copied CPU→GPU on every generation — the same round-trip that
    made the bf16 baseline slow, just with smaller weights. *(Originally written up as "4-bit's
    dequantization math is slower" — wrong; see the correction above.)*
  - **Quality cost is small but real**: CLIP 33.19 vs. 33.42 baseline (−0.23), a slightly bigger drop
    than the VAE experiment's noise-level ±0.03, consistent with 4-bit being a more aggressive
    quantization than 8-bit.
  - Not the same mechanics as 8-bit: `text_encoder` is added to `_exclude_from_cpu_offload`, but diffusers
    ignores that list for modules in `model_cpu_offload_seq` — so it still gets hooked and streamed.

- **[profile] Where the 4-bit run's time goes** (`profile_vram.py --quantize-text-encoder-4bit
  --max-vram-gb 4`, one prompt, 512px / 20 steps, 2026-09-28):

  | phase | 8-bit (opt #1 profile) | 4-bit | share of 4-bit total |
  |---|---|---|---|
  | encode | ~1.1 s | **4.02 s** | 52% |
  | denoise (20 steps) | ~1.0 s | 0.97 s | 13% |
  | decode (VAE) | ~2.7 s | 2.71 s | 35% |
  | total wall | 4.73 s | 7.70 s | |
  | allocator peak | 5.22 GB | 2.51 GB | |
  | nvml peak (whole process) | — | 3.10 GB, in encode | |

  - **The 4-bit-vs-8-bit slowdown is entirely in encode** (+2.9 s ≈ the 7.65 − 4.73 gap); denoise and
    decode are unchanged. (Cause found two experiments later — see [finding] below.)
  - **Encode is now the biggest target**, then VAE decode. Denoise is <1 s total, so fewer steps can
    save ~0.5 s at most; VAE tiling saves VRAM, not time, and VRAM (2.51 of 4 GB) isn't the constraint.
    → Plan reordered: speed up encode first, then TensorRT on the VAE; tiling + fewer steps demoted.

- **[opt #2a] Drop SANA's built-in prompt instruction (`--no-prompt-instruction`)** — `SanaPipeline`
  prepends a ~1,050-char instruction (`complex_human_instruction`) to every prompt, padding the encoder
  input to ~550 tokens instead of ~300. Hypothesis: fewer tokens → faster encode.

  | | 4-bit | 4-bit, no instruction |
  |---|---|---|
  | encode (profile) | 4.02 s | 3.92 s |
  | sec/image (bench, 4 GB cap) | 7.66 | 7.58 |
  | peak VRAM | 2.51 GB | 2.47 GB |
  | CLIP | 33.19 | **33.73** |

  - **Hypothesis wrong on speed:** ~45% fewer tokens bought only ~2.5% faster encode — the cost is
    per call, not per token. That clue led to the [finding] below.
  - **Unexpected quality gain:** CLIP +0.54, above the bf16 baseline (33.42). CLIP is only a proxy and
    this is 5 prompts — eyeball the images before trusting it.

- **[finding] The 4-bit encoder is streamed, not pinned.** A quick check in the container: encoder
  parameters sit on `cpu` after load (`hf_device_map` said GPU at load time), an accelerate `_hf_hook`
  is attached, and layers are `Linear4bit`. `transformers` blocks `.to()` only for **8-bit** bnb models;
  4-bit can move, so `enable_model_cpu_offload()` hooks it like any other module. The ~4 s encode is
  mostly copying ~2 GB of weights CPU→GPU per call — explains why token count barely mattered.
  **Why `_exclude_from_cpu_offload` didn't help:** diffusers only checks that list for modules *outside*
  `model_cpu_offload_seq` (`text_encoder->transformer->vae`); the text encoder is inside it.
  → Next: `--pin-text-encoder` drops it from the sequence so it's placed on the GPU once and stays.

- **[opt #2b] Pin the 4-bit encoder (`--pin-text-encoder`)** — profile, uncapped, one prompt:

  | phase | 4-bit streamed | 4-bit pinned |
  |---|---|---|
  | encode | 4.02 s | **1.08 s** |
  | denoise | 0.97 s | 0.97 s |
  | decode | 2.71 s | **5.64 s** |
  | total | 7.70 s | 7.70 s |
  | allocator peak | 2.51 GB | **4.38 GB** (nvml 5.38 of 6 GB) |

  - Pinning works — encode drops ~3 s, confirming the [finding] above.
  - **But decode doubles**, so total is unchanged, and the 4.38 GB peak **doesn't fit the 4 GB cap**.
    Likely cause (unconfirmed): with ~5.4 of 6 GB in use, the allocator has to free/re-reserve memory
    during the fp32 VAE decode. Under this budget, pinning moves the cost from encode to decode rather
    than removing it — a real speed-vs-memory tradeoff. Parked: only viable with a 5 GB budget.
  - Tried to cut the decode peak with VAE tiling: `AutoencoderDC.enable_tiling` exists but `tiled_decode`
    raises `NotImplementedError` in diffusers 0.32.2 — the API has the switch, the implementation isn't
    there. Dropped.

**C2 result:** best config under 4 GB = 4-bit streamed encoder + `--no-prompt-instruction` →
**7.58 s/img, 2.47 GB, CLIP 33.73** (vs A-baseline 11.17 s / 5.33 GB / 33.42, which doesn't fit at all).

- **[C3.0] VAE in bf16 instead of fp32 (`--vae-bf16`)** — precision check before TensorRT, on top of the
  C2 best config, 4 GB cap:

  | | fp32 VAE (C2 best) | bf16 VAE |
  |---|---|---|
  | decode (profile) | 2.71 s | **1.84 s** (−32%) |
  | decode nvml peak | 2.99 GB | 2.02 GB |
  | sec/image (bench) | 7.58 | **7.25** |
  | allocator peak | 2.47 GB | 2.47 GB (peak is in encode) |
  | CLIP | 33.73 | 33.74 |

  - **The VAE survives bf16:** same-seed images are visually identical to fp32, CLIP unchanged. The fp32
    upcast in the original code ("for stability") cost ~0.9 s of decode for no visible benefit here.
  - Caveat for TensorRT: bf16 keeps fp32's numeric range; **fp16 does not**, so fp16 overflow is still
    untested — build the TRT engine in bf16 first.
  - Denoise read 1.26 s vs 0.97 s in this single profile; bench total still dropped — treated as noise.
  - Adopted: new best config = 4-bit streamed encoder + no prompt instruction + bf16 VAE.

- **[C3.2] VAE → TensorRT engine (`trtexec --bf16`)** — built from the C3.1 ONNX file; checked with
  `src/check_trt_vae.py` on a fixed latent (0–255 pixel scale, vs the fp32 PyTorch reference):

  | | PyTorch bf16 | TensorRT `--bf16` |
  |---|---|---|
  | max pixel diff | 14.17 | 18.85 |
  | mean pixel diff | 0.253 | **0.233** |
  | decode, GPU only (mean of 10) | **172.7 ms** | 283.6 ms (0.6×) |
  | GPU memory | ~0.3 GB weights | ~1.46 GB (weights + workspace) |

  - **Correct:** images look identical; mean error is slightly *lower* than bf16 PyTorch.
  - **But slower and bigger.** The engine file (626 MB) is about the size of the fp32 ONNX (637 MB) —
    `--bf16` only *allows* bf16 kernels, it doesn't force them, so TensorRT likely kept most layers in fp32.
    Lesson: a TensorRT engine is not automatically faster than a well-tuned PyTorch bf16 model; check the
    precision it actually picked.

- **[C3.2a] Where the decode time goes** (`profile_vram.py --split-decode`, best config, 4 GB cap):

  | part of the decode phase | time |
  |---|---|
  | move models (offload hooks: transformer off GPU, VAE on) | **1.56 s** |
  | VAE compute | 0.18 s |
  | finish (→ PIL image, VAE back to CPU) | 0.56 s |
  | gap after denoising | 0.00 s |
  | allocator retries (cache flush under the cap) | **0** |

  - **~90% of decode is moving weights between CPU and GPU, not computing.** CPU offload keeps one model
    on the GPU at a time, so every image pays for the shuffle. TensorRT can only speed up the 0.18 s.
  - The "4 GB cap causes allocator pressure" guess was wrong — zero retries.
  - Whole run was slow (10.4 s vs ~7.7 s usual, likely background load), so read the split as proportions.
  - Not pursued (scope): keeping the VAE/transformer resident on the GPU would remove most of it —
    decode peaks at ~2 GB, so there is room under 4 GB. Good "what would you do next?" answer.

- **[C3.3] TensorRT VAE in the pipeline (`--trt-vae`)** — same-day, plugged-in pair (2026-10-09):

  | | sec/image | peak VRAM | CLIP |
  |---|---|---|---|
  | PyTorch bf16 VAE (`C-vae-bf16-4gb-rerun`) | 10.75 | 2.47 GB | 33.74 |
  | TensorRT bf16 VAE (`C-trt-vae-4gb`) | **10.08** (−0.67 s, −6%) | 2.80 GB | 33.73 |

  - **Faster, same quality, +0.33 GB** — the engine's weights stay on the GPU (counted in peak VRAM,
    since TensorRT allocates them outside PyTorch's cap). Engine: strict bf16 (`export_vae_onnx.py --bf16`
    + `trtexec --stronglyTyped`) — the third try, after `--bf16` (stayed fp32, 0.6×) and `--fp16` (NaN).
  - Small win, as expected: decode compute was only ~0.2 s; most of the saving is likely the VAE no longer
    being moved CPU→GPU each image (see [C3.2a]).
  - Compare only the two rows above: **this machine ran ~45% slower on 2026-10-09** than on 2026-09-28
    (same config: 7.25 → 10.75 s), cause not found. Rows from different days aren't comparable.

- **[power] Battery vs plugged in** — `C-trt-vae-4gb-battery` (10.42 s) ran on battery; the same config
  plugged in gave 10.08 s — only **3% slower on battery**. So battery was *not* the main cause of the
  slow day (the plugged-in rerun was slow too). Edge lesson anyway: record power state with every
  benchmark, and compare runs from the same session.

- [experiment] VAE native fp32 vs. bf16-then-upcast (`--vae-native-fp32`) → CLIP score before/after (expect same VRAM, testing quality only): VRAM identical (5.33 GB both), confirming the "no memory cost" prediction. CLIP 33.39 vs. 33.42 baseline — a 0.03 difference, within normal run-to-run noise, not a meaningful change. Conclusion: skipping the bf16 round-trip is theoretically more precise but produces no measurable quality difference here — the simpler default code isn't actually costing anything in practice for this model.

**One-line summary:** e.g. *"Cut latency X→Y and VRAM 6→<4 GB with <Z CLIP-point quality cost."*
