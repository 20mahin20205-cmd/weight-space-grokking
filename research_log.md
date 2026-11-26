# Research Log

## 2025 — Session 1: baseline reproduction (sandbox CPU pilot)

**Hypothesis (written before runs):** 1-layer transformer, no LayerNorm, AdamW
(wd=1.0, lr=1e-3, full-batch) on (a+b) mod 53 with 40% train fraction will
memorize within ~1k steps and grok between 5k–30k steps; seed variance in
delay will be large (expect > 2x spread).

**Setup:** `train.py` (this repo). Definitions per `DEFINITIONS.md`. 3 seeds,
data_seed fixed at 1000, init_seed varied. snapshots float16 every 1000 steps.

**Results (all 3 grokked):**

| seed | T_mem | T_grok | delay | wall (CPU, 2 thr) |
|------|-------|--------|-------|-------------------|
| 0 | 400 | 9700 | 9300 | 624 s |
| 1 | 400 | 11100 | 10700 | 600 s |
| 2 | 300 | 15300 | 15000 | 707 s |

- Classic grokking curve reproduced: train acc hits 100% by step ~400; test
  acc stays near 0 for thousands of steps, then climbs to ~100% within ~2–3k
  steps.
- Delay varies 9,300–15,000 across seeds (~1.6x) — confirming per-seed
  statistics are mandatory for any RQ2 claim.
- Fourier sanity check on final-snapshot embeddings: top-4 frequencies hold
  72–87% of embedding energy → sparse Fourier representation present, matches
  Nanda-style circuit expectations. Detailed per-key-frequency analysis still
  pending (top-freq lists currently over all channels; needs per-frequency
  key identification next).

**Timing extrapolation:** ~24 ms/step on 5-thread CPU at p=53. On the RTX
Pro 6000 (96 GB), expect <1 min per 30k-step run; measure before committing.

**Failures / lessons:**
- Background process logs empty while running (Python buffering) — read
  `state.pt` progress instead; use `python -u` next time.
- Seed 0 metrics kept a duplicate eval at resume-check; harmless but fix.

**Next:** 20-run timing pilot on GPU → zoo v1 grid (wd x frac x lr, 10+ seeds).
