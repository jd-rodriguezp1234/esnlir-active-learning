# Running the AL pipeline on Colab

The repo lives in Google Drive; the notebook mounts Drive, `cd`s into it, and runs
`esnlir/training/train.py` from there. Everything the run produces — metrics,
`best_model/`, `final_model/` — is written into `models/` inside that same Drive
repo, so it survives a disconnect.

## Which data route?

**Option B (original `data/`, High-RAM) is preferred if you have it.** `BERTDataset`
then draws the 1M subset with its own seeded `train_test_split`, so the run is
row-for-row comparable to an HPC run. Option A exists for standard-RAM runtimes and
for keeping the Drive upload small.

| | A: `data_colab/` | B: original `data/` |
|---|---|---|
| Drive upload | ~700 MB | ~3.1 GB |
| peak RAM | ~2 GB | **~12.3 GB** |
| runtime needed | any | **High-RAM** |
| comparable to HPC runs | no (different subset) | **yes** |
| configs | `*_colab.json` | `*_colab_fulldata.json` |

Peak for B breaks down as: 4.1 GB DataFrame, ~3 GB raw text held during `read_json`,
doubled again by `sort_values` in `BERTDataset`. Standard Colab RAM is 12.7 GB — too
close to the line. High-RAM (51 GB / 83.5 GB) is comfortable.

For B, **stage the data to local disk first** (notebook cell 4b). Reading 3 GB through
the Drive FUSE mount is slow and stalls; `/content` is real disk. The
`*_colab_fulldata.json` configs already point `dataset_folder` at `/content/data`.
Outputs still go to `models/` in the Drive repo.

## One-time setup

**1. (Option A only) Build the Colab-sized dataset, locally.**

```bash
python colab/prepare_colab_data.py            # 1M train rows, 20k val, full test splits
```

`data/train.json` is ~3 GB and `pandas.read_json` loads it whole before `max_samples`
ever trims it — that is a slow upload and an OOM risk on a standard runtime. The
script streams the file twice and writes a proportional stratified sample over
`(connector_type, dataset)` to `data_colab/`, roughly 700 MB.

**2. Put the repo in Drive**, e.g. `MyDrive/AL-NLI/nli-training-example`, including
`data_colab/`. Either upload it or `git clone` into Drive from the notebook.

**3. Open `colab/AL_NLI_Colab.ipynb`** in Colab and set `REPO_DIR` in cell 2 if your
path differs.

## Evaluating pretrained baselines

`Baseline_Eval_Colab.ipynb` scores a finished model from the HuggingFace Hub — no
training. Set `MODEL_ID` in its config cell and `Runtime -> Run all`:

```python
MODEL_ID = 'Flaglab/ESNLIR-RoBERTa'        # or 'Flaglab/ESNLIR-XLM-RoBERTa'
```

It writes to `models/baseline_<name>/` using the same `MetricGenerator` as the AL runs,
so the CSVs line up column for column. It also writes
`<split>/total/predictions.csv` with per-class probabilities, so metrics can be
recomputed later without re-running inference.

Before predicting it compares the checkpoint's `id2label` against the dataset's
alphabetical class order and permutes the logit columns if they differ. A silent
mismatch there is indistinguishable from a model that predicts a single class, which is
worth guarding because it looks exactly like a real failure.

## Runtime

Runtime → Change runtime type → **A100 GPU** + **High-RAM**. Estimates for the full
config (1M pool, `al_K` 100k, 3 epochs, ~10 cycles ≈ 16.8M training passes):

| GPU | estimate | verdict |
|---|---|---|
| A100 40GB, bf16 | ~15 h | fits a 24 h session |
| L4, bf16 | ~35 h | too slow |
| T4, fp32 | ~128 h | not viable |

Two settings do most of that work, and both are new config keys:

- `"bf16": true` — roughly 2× on Ampere. Set `"fp16": true` instead on pre-Ampere.
- `"dataloader_num_workers": 4` — `BERTDataset` tokenises inside `__getitem__`, so with
  the default of 0 workers the CPU starves the GPU. This applies to AL pool scoring too.

## Configs

| file | pool | `al_K` | epochs | approx |
|---|---|---|---|---|
| `model-xlm__strategy-NegE__smoke.json` | 20k | 4k | 1 | ~10 min |
| `model-xlm__strategy-Rem__smoke.json` | 20k | 4k | 1 | ~10 min |
| `model-xlm__strategy-NegE__colab.json` | 1M | 100k | 3 | ~15 h |
| `model-xlm__strategy-Rem__colab.json` | 1M | 100k | 3 | ~12 h |
| `model-xlm__strategy-NegE__colab_fulldata.json` | 1M | 100k | 3 | ~15 h + ~10 min load |
| `model-xlm__strategy-Rem__colab_fulldata.json` | 1M | 100k | 3 | ~12 h + ~10 min load |

The `*_colab.json` pair uses `dataset_folder: "data_colab"` with `max_samples: null`
(already subsampled). The `*_colab_fulldata.json` pair uses `/content/data` with
`max_samples: 1000000`, letting `BERTDataset` subsample. The extra ~10 min is the
one-time `read_json` parse of 4.4M rows, not a per-cycle cost.

**Run a smoke config first** — none of this pipeline has been
executed end to end, and the smoke run exercises every path the full run uses.

## How to run

Open `colab/AL_NLI_Colab.ipynb`, edit **one cell** (`REPO_DIR` and `CONFIG_PATH`), then
`Runtime -> Run all`. Everything downstream is derived from the config file: data
staging, mixed precision, output paths, plots. Switching between a smoke test and a
full run is a one-line change.

The notebook patches `bf16 -> fp16` on pre-Ampere GPUs and clamps
`dataloader_num_workers` to the available CPUs, writing a resolved copy to
`logs/<run>.resolved.json`. Your config file is never modified.

Training runs in the foreground and streams output live, so `Run all` completes end to
end and you can watch progress in the cell. tqdm bars are throttled to one line every
30 s. The same output goes to `logs/<run>.log` in Drive, so you can inspect it from
another session while a long run is in flight.

## Monitoring

The notebook launches training under `nohup`, so the cell returns immediately and the
run survives closing the tab (Pro+ background execution). The monitor cells are
re-runnable and read straight from the Drive outputs:

- log tail, process liveness, `nvidia-smi`
- per-cycle table from `metrics_iter_*.json`, with measured minutes-per-cycle
- learning curve, with the collapse thresholds drawn on
- latest per-class report from `val_report_iter_*.csv`
- final: `best_iteration.json` plus the `best/` and `final/` classification reports

## What to watch in the first report

`warmup_steps` is `0`, matching the bertin runs. If XLM-R destabilises it shows up
immediately:

- eval loss flat at **1.386** (`ln 4`) → not learning
- accuracy **0.250**, macro-F1 **0.100**, every row predicted `contrasting` → collapsed

Kill it and set `warmup_steps` to ~6% of the first cycle's steps. A weak-but-healthy
start looks like macro-F1 around 0.30–0.45 with all four classes represented — low is
fine, *one class* is not.

## Limits

- **No resume.** A dead runtime loses in-flight training; completed cycles keep their
  metrics in Drive but training restarts from zero. Size runs to finish in one session.
- **Drive writes.** `best_model/` is ~1.1 GB and is rewritten on every improvement,
  which on a monotonic curve means every cycle. If that becomes the bottleneck, point
  `output_folder` at `/content/` and copy results to Drive at the end.
- **`batch_size` is 32** to stay comparable with the bertin runs. 64 is ~20–30% faster
  on an A100 but changes the effective learning rate.
- **Option A's subset is not the HPC subset.** `prepare_colab_data.py` draws its own 1M
  rows, so those runs are comparable to each other, not row-for-row to cluster results.
  Option B (`*_colab_fulldata.json`) does not have this problem.
- **Stage to `/content`, don't read from Drive.** Training reads every example on every
  epoch — `BERTDataset` tokenises in `__getitem__` — so a dataset on the FUSE mount is
  read repeatedly, not once. Cell 4b copies it to local disk.
