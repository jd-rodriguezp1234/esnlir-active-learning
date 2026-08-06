# Experiment Comments

This document provides short, practical notes for each Active Learning experiment configuration under `params/experiments/`.

Legend

- Strategy: Random (baseline), NegE (Negative Energy uncertainty), Rem (remove low-utility, then select)
- K: acquisitions per iteration; L: AL iterations
- B: training batch size; W: warmup steps (optimizer schedule)
- Model short names: `bertin` = bertin-project/bertin-roberta-base-spanish, `xlm` = FacebookAI/xlm-roberta-base

---

## Random (baseline)

- model-bertin**strategy-Random**K200**L4**B16\_\_W0.json
  - K=200, L=4 → modest budget; small batch (B16) and no warmup (W0) emphasize quick iterations. Good low-cost baseline.
- model-bertin**strategy-Random**K200**L6**B32\_\_W500.json
  - More feedback cycles (L6) with moderate batch (B32) and warmup (W500) for smoother optimization; baseline at mid budget.
- model-bertin**strategy-Random**K200**L8**B64\_\_W0.json
  - Many iterations (L8) with large batch (B64), no warmup; checks if bigger batch helps baseline stability.
- model-xlm**strategy-Random**K500**L4**B16\_\_W500.json
  - Larger K (500) per iteration; fewer cycles (L4); small batch with warmup. Tests faster budget spend on XLM-R.
- model-xlm**strategy-Random**K500**L6**B32\_\_W0.json
  - Mid cycles, larger K, moderate batch without warmup; baseline to compare against uncertainty-based methods.
- model-xlm**strategy-Random**K500**L8**B64\_\_W500.json
  - High L with large batch and warmup; stress-tests training throughput with frequent feedback.

## Negative Energy (NegE)

- model-bertin**strategy-NegE**K500**L4**B16\_\_W0.json
  - Strong per-iter acquisition (K500) with few cycles; NegE targets high-uncertainty samples quickly.
- model-bertin**strategy-NegE**K500**L6**B32\_\_W500.json
  - Balanced cycles (L6) and batch (B32) with warmup; common stable setting for uncertainty sampling.
- model-bertin**strategy-NegE**K500**L8**B64\_\_W0.json
  - Many cycles with large batch, no warmup; probes whether throughput offsets lack of warmup under NegE.
- model-xlm**strategy-NegE**K200**L4**B16\_\_W500.json
  - Smaller K (200) and warmup for steadier updates; early comparison of NegE vs Random in low-K.
- model-xlm**strategy-NegE**K200**L6**B32\_\_W0.json
  - Mid cycles, mid batch, no warmup; isolation of NegE effect without schedule warmup.
- model-xlm**strategy-NegE**K200**L8**B64\_\_W500.json
  - Frequent feedback (L8) with warmup and large batch; aims at stable yet fast improvement.

## REM (remove-then-select)

- model-bertin**strategy-Rem**K200**L4**B16**W0**R200.json
  - Remove_k=200 equals K; aggressive pruning then selection; tests noise/redundancy reduction at small budget.
- model-xlm**strategy-Rem**K200**L6**B32**W500**R200.json
  - Balanced L with moderate batch and warmup; evaluates if removal improves mid-horizon learning.
- model-bertin**strategy-Rem**K200**L8**B64**W0**R200.json
  - Many cycles with large batch; removal may accelerate curation of the pool across more iterations.
- model-xlm**strategy-Rem**K500**L6**B32**W0**R200.json
  - Larger K with moderate cycles; removal avoids saturating with low-utility samples as budget spends faster.
- model-xlm**strategy-Rem**K500**L4**B16**W500**R300.json
  - Remove_k> K; stronger pruning with fewer cycles; tests if aggressive discard helps precision early.
- model-bertin**strategy-Rem**K500**L6**B32**W0**R300.json
  - Larger K and removal (R300) mid-horizon; checks robustness to heavier pruning on bertin.
- model-xlm**strategy-Rem**K500**L8**B64**W500**R300.json
  - High L and large batch with strong pruning; aims at compounding gains via sustained curation.
- model-bertin**strategy-Rem**K500**L4**B16**W0**R300.json
  - Few cycles but heavier removal; explores early-stage denoising effect before budget is consumed.

---

## General expectations

- Random: sets the lower bound; expect slower gains but robust comparisons.
- NegE: prioritizes uncertain samples; typically faster improvements per labeled budget.
- Rem: counters redundancy/noise; removal too aggressive can discard rare but informative samples—compare R200 vs R300 carefully.
