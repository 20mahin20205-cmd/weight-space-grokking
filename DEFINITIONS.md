# Definitions (v1, 2025 — pilot)

- **T_mem**: first step where train accuracy >= 99%.
- **T_grok**: first step where test accuracy >= 95% AND holds for 3 consecutive evaluations.
- **Delay**: T_grok - T_mem.
- **Phases (per snapshot)**:
  - Pre-grok: train high, test low, future grokking.
  - Grok: test rapidly rising (window around T_grok).
  - Post-grok: test high, stable.
  - Anti-grok: grokked earlier, test now falling while train high.
  - Never-grok: test never rose (separate class).
- **Targets**: (a) phase class, (b) log(T_grok - t).

## Pilot scope (sandbox, CPU-only)
- Task: (a+b) mod p, p=97, train fraction 0.4
- Model: 1-layer transformer, d_model=128, 4 heads, MLP 4x d_model, no LayerNorm
- Optimizer: AdamW, wd=1.0 (also 3.0 variant), lr=1e-3, full-batch
- Device: CPU (real runs go on RTX Pro 6000 96GB)

Change log:
- 2025: created.
