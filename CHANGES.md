# Changes

Work done while investigating why XLM-R appeared to predict a single class
(`contrasting`) on every test row, and preparing the pipeline to run on Colab.

## Context: what the original symptom turned out to be

The artefact under `models/xlmroberta/test/` was **not** evidence of class bias. It came
from a 40-row smoke run (10 per class) that trained for 3 optimizer steps: `train_loss`
1.4013 against `ln(4) = 1.3863`, `eval_loss` flat at 1.3963. The model never left random
initialisation, so `argmax` returned whichever output unit had the largest random bias —
`contrasting` is index 0 of `sorted(classes)`.

The data is not imbalanced either: train, val and test are 25% per class. Real-run
collapse remained a hypothesis, and a later smoke run on Colab at `2e-5` / batch 32 /
`warmup_steps: 0` reached ~0.6, showing XLM-R trains fine at those settings. The earlier
XLM-R experiment configs used `3e-5` and `5e-5` at batch 16, which is the more likely
source of any real instability.

## New configuration keys

All optional; defaults preserve previous behaviour unless noted.

| key | default | effect |
|---|---|---|
| `eval_batch_size` | `batch_size * 2` | was unset, so HF defaulted to **8** |
| `bf16` / `fp16` | `false` | mixed precision; ~2x on Ampere |
| `dataloader_num_workers` | `0` | `BERTDataset` tokenises in `__getitem__`, so 0 workers starves a fast GPU. Applies to AL pool scoring too |
| `al_seed_size` | `0` | class-balanced initial labeled set (see below) |
| `al_report_n` | `2000` | rows in the per-cycle validation report; `0` disables, `null` uses all of validation |
| `al_eval_splits` | `null` (all) | which test splits to evaluate per cycle |
| `al_rem_select_mode` | `"random"` | REM acquisition: `random` or `uncertainty` |
| `al_rem_utility` | `"energy"` | REM utility: `energy` or `entropy` |

CLI equivalents exist for `--al_rem_select_mode`, `--al_rem_utility`, `--al_seed_size`.

## Fixes

**`esnlir/active_learning/rem.py`**

- RNG was `random.Random()` with no seed, constructed fresh inside `select()` on every
  cycle. REM was the one non-reproducible strategy despite `random_seed: 42` elsewhere.
  Now a seeded instance on the strategy, wired to `random_seed`.
- `set(to_remove)` and `set(remaining)` were rebuilt *inside* comprehensions over the
  whole pool — O(n²). Since `scores` is already sorted, the removed and kept slices are
  contiguous, so both are now slices. This was tolerable at `remove_k: 1000` and would
  not have finished under `select_mode: "uncertainty"`.
- Behavioural note: under *exact* utility ties, tie-breaking now favours high indices
  rather than low. Exact float ties are effectively impossible, but the pool is sorted by
  class (`sort_values("connector_type")`), so index order is class order.

**`esnlir/training/train.py`**

- Test split keys used `.replace(".json", "")`, which turned `test_full.jsonl` into
  **`test_fulll`**. Now `os.path.splitext`. Output directories rename accordingly.
- `EarlyStoppingCallback` was constructed alongside `eval_strategy='no'`, which trips its
  own `assert args.eval_strategy != IntervalStrategy.NO` — the non-AL path could not run.
  `eval_strategy` is now `"epoch"` on the plain path and `"no"` under AL, and
  `load_best_model_at_end` follows, so `monitor` and `patience` are finally live.
- `RemStrategy` was built without `select_mode`, so it silently used the `"random"`
  default: energy chose only which rows to *discard*, and acquisition was uniform random.
  With `remove_k: 1000` against a 1M pool that made REM ≈ Random minus 1% of the data.
  Both knobs are now explicit in the configs rather than defaulted.

**`esnlir/dataset_utils/dataset.py`**

- `train_test_split(stratify=connector_type__dataset)` raises when any stratum has fewer
  than 2 rows, which happens on small or truncated files. Now degrades to stratifying on
  `connector_type` alone (class balance is what must be preserved), then to no
  stratification, warning either way. Never triggers at 1M+ rows.

**`esnlir/evaluation/metric_generation.py`**

- Added `classification_report` per split, with `labels=`/`target_names=`/`zero_division=0`
  so every class appears even when the model never predicts it. Written to
  `<split>/total/classification_report.csv`.

## New behaviour

**Seed set (`al_seed_size`).** `PoolManager` previously started empty, so cycle 0's
acquisition was scored by a randomly initialised classification head — energy ranked rows
by a random projection of sentence embeddings, not by uncertainty, and committed `al_K`
labels to that. With `al_seed_size` set, a class-balanced sample (via the new
`stratified_indices` in `esnlir/utils/utils.py`, seeded from `random_seed`) is labeled
first and trained on, activating the previously dead `if self.pool.labeled_indices:`
branch in `run()`. Every strategy draws the identical seed set.

**Per-cycle validation report.** After each cycle, a per-class precision/recall/F1 table
over a fixed class-balanced subset of validation, plus the class distribution of the
acquired pool. Printed and saved to `val_report_iter_{N}.csv`. The subset is sampled once
and reused so per-class curves are not contaminated by resampling noise. The pool
distribution is the direct measurement of whether NegE skews acquisition — relevant
because class weights are deliberately *not* recomputed per cycle.

**Best-model snapshotting.** Improvement tracking was gated behind
`if self.alc.iter_patience is not None`, which is `null` in every config — so
`best_iter_metric` never updated. Tracking is now unconditional and only the `break`
remains gated. On improvement the weights are saved to `best_model/` with
`best_iteration.json` recording iteration, metric and `n_labeled`.

**Evaluation layout.**

| output | model | splits | when |
|---|---|---|---|
| `iter_{N}/` | that cycle | per `al_eval_splits` | every cycle |
| `val_report_iter_{N}.csv` | that cycle | validation subset | every cycle |
| `final/` | `final_model` | all | end of run |
| `best/` | `best_model` | all | end of run, only if best ≠ last |

`best/` reloads the saved checkpoint rather than re-evaluating every improvement — the
latter would have cost ~8x more, since a growing labeled set improves nearly every cycle.
When the best cycle *is* the last, `final/` already describes those weights and the extra
pass is skipped.

## New files

- `params/experiments/model-xlm__strategy-{NegE,Rem}__E3__B32__LR2e-5.json` — XLM-R at
  `2e-5` / batch 32, matching the bertin configs. Both strategies at the same LR, so
  strategy is no longer confounded with learning rate. Rem at `remove_k: 25000` (20% of
  the pool over the run) instead of 1000, which was too small to register.
- `colab/` — notebook, `prepare_colab_data.py`, six configs, README. See `colab/README.md`.

## Known issues, not addressed

- **`confusion_matrix` is still called without `labels=`** at
  `metric_generation.py:65` and `:111`. When a model predicts one class the matrix is
  smaller than 4x4 and `classes[ix]` mislabels columns; this is the source of the
  `invalid value encountered in divide` warnings. Only the new `classification_report`
  passes `labels=`.
- **`warmup_steps: 0` everywhere.** Kept to match the existing bertin runs. The AL loop
  builds a fresh `Trainer` each cycle, so the optimizer state and LR schedule restart
  every cycle — the risky transient recurs 9-11 times per run, not once. `warmup_ratio`
  would be the right knob (cycle lengths span ~100x) but is not plumbed; only
  `warmup_steps` is read.
- **The bertin configs are not aligned** with the updated XLM-R ones: `al_seed_size: 0`,
  and Rem at `lr: 3e-5` / `remove_k: 1000`. Any XLM-R-vs-bertin gap currently absorbs
  those differences. Re-running bertin with matched settings is required for a clean
  model comparison.
- **`train.py:244` builds a model unconditionally** which the AL path never uses — it
  holds ~1.1 GB of VRAM for the whole run. Harmless on a 40 GB GPU.
- **`out_root` double-nests**: `models/<config-name>/<model>_active_<strategy>/`, so model
  and strategy appear twice while hyperparameters appear only in the outer name.
- **HF checkpoints land outside `out_root`** (`training_args.output_dir` is
  `output_folder`), and `global_step` restarts each cycle so they overwrite each other.
  Treat them as scratch; `best_model/` and `final_model/` are the real artefacts.
- **No AL resume.** A killed run restarts from zero; completed cycles keep their metrics.
- **`requirements.txt` conflicts with `pyproject.toml`** — it pins `accelerate==1.8.1` and
  `tensorboard==2.19.0` against requirements of `>=1.11.0` and `>=2.20.0`. The two files
  describe incompatible environments; the Poetry-to-uv migration is also uncommitted.

## Verification status

All changes are syntax-checked and the selection-logic rewrite in `rem.py` was verified
equivalent on distinct scores. The Colab smoke run exercised dataset loading, the
stratify fallback, the seed set, and the start of training. The `best/` reload path is
untested — it only executes when a cycle regresses.
