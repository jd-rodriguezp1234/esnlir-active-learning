# Active Learning Module

This folder implements a modular, pool-based Active Learning (AL) framework for NLI classification models based on Transformers. It includes a common strategy interface, three strategies (Random, Negative Energy, REM), a pool manager, and an orchestration trainer that runs iterative acquisition-and-train cycles while reusing the project’s Hugging Face Trainer and evaluation stack.

The goal is to reduce labeling cost by iteratively selecting the most informative unlabeled samples for annotation and model retraining.

- Module path: `esnlir/active_learning/`
- Strategies:
  - Random baseline: `random_sel.py`
  - Negative Energy (uncertainty): `neg_energy.py`
  - REM (remove low-utility, then select): `rem.py`
- Pool management: `pool_manager.py`
- Orchestration trainer: `active_trainer.py`
- Integration flag in training script: `esnlir/training/train.py` via `--active-learning`

## Conceptual background

We consider a pool-based AL setting with:

- Labeled set $\mathcal{L}$
- Unlabeled pool $\mathcal{U}$
- Acquisition size per iteration $K$ and number of iterations $L$ (here $L$ denotes number of AL cycles)

Loop per iteration $t = 0, \dots, L-1$:

1. Score all $x \in \mathcal{U}$ using the current model $f_\theta$.
2. Select $K$ samples according to a strategy (e.g., uncertainty sampling).
3. Move selected samples to $\mathcal{L}$ (as if newly labeled) and optionally remove low-utility samples from $\mathcal{U}$ (REM).
4. Retrain or continue fine-tuning the model with the updated $\mathcal{L}$.

This module implements this loop efficiently with batching and `torch.no_grad()` during scoring.

## Data and model assumptions

- The dataset class is `esnlir.dataset_utils.dataset.BERTDataset`, returning inputs for Transformer encoders and one-hot label vectors (shape: C). The `WeightedTrainer` converts them to suitable targets for cross-entropy.
- The model is a Hugging Face `AutoModelForSequenceClassification` with logits output `logits \in \mathbb{R}^{B \times C}`.
- GPU is used if available and specified in the config (device string), otherwise CPU.

## Strategy interface

File: `esnlir/active_learning/base_strategy.py`

- Abstract base class:
  - `select(self, model, unlabeled_loader, k)` → returns a list of global indices (or a tuple `(selected, removed)` for strategies like REM).
- The `unlabeled_loader` must iterate over a `torch.utils.data.Subset` whose `.indices` map to global indices in the full training dataset. This preserves a consistent index space for pool updates and logging.

## Random Strategy (baseline)

File: `esnlir/active_learning/random_sel.py`

- Idea: Pick $K$ samples uniformly at random from the unlabeled pool $\mathcal{U}$. This is a **baseline** to compare against more informed strategies.
- Algorithm:
  1. Read the pool’s global indices from `unlabeled_loader.dataset.indices`.
  2. Sample $K$ without replacement.
  3. Return the selected indices.
- Complexity: $\mathcal{O}(|\mathcal{U}|)$ to read indices and $\mathcal{O}(K)$ for sampling.
- Pros/Cons: Trivially fast and unbiased; does not exploit model knowledge, may be inefficient.

## Negative Energy Strategy (uncertainty sampling)

File: `esnlir/active_learning/neg_energy.py`

- Motivation: Prefer samples on which the model is most uncertain. We use an **energy-based** uncertainty score derived from logits.
- For a sample with logits vector $z \in \mathbb{R}^C$ (pre-softmax):

  $$E(x) = -\log\sum_{c=1}^C e^{z_c} = -\mathrm{logsumexp}(z).$$

  Higher $E(x)$ implies lower model confidence (higher uncertainty).

- Implementation details:
  - Inference under `model.eval()` and `torch.no_grad()`.
  - Batch over the unlabeled pool using a configurable scoring batch size (default 64).
  - Compute energy per sample, sort in descending order of $E(x)$, and select top-$K$.
  - Return global indices by slicing `Subset.indices` consistently with batch positions.
- Complexity: Scoring is $\mathcal{O}(|\mathcal{U}|)$ forward passes; selection requires a sort $\mathcal{O}(|\mathcal{U}|\log|\mathcal{U}|)$.
- Notes: Energy avoids softmax calibration issues and is efficient to compute from logits.

## REM Strategy (remove then select)

File: `esnlir/active_learning/rem.py`

- Motivation: **R**emove low-utility samples from the pool, then **em**ploy either random selection or uncertainty selection from the remainder. This can mitigate noisy or redundant data in $\mathcal{U}$.
- Utility scoring options:

  - Energy (as above): $E(x) = -\mathrm{logsumexp}(z)$.
  - Entropy of softmax probabilities $p = \mathrm{softmax}(z)$:

    $$H(p) = -\sum_{c=1}^C p_c \log p_c.$$

- Algorithm per iteration:
  1. Score all $x \in \mathcal{U}$ with chosen utility (energy or entropy).
  2. Remove the $\texttt{remove\_k}$ samples with the **lowest** utility (least informative).
  3. Select $K$ samples from the remaining pool:
     - Mode “random”: uniform random among remaining
     - Mode “uncertainty”: highest utility (e.g., largest energy or entropy)
  4. Return `(selected_indices, removed_indices)`.
- Complexity: A full sort $\mathcal{O}(|\mathcal{U}|\log|\mathcal{U}|)$; forward scoring is $\mathcal{O}(|\mathcal{U}|)$.
- Notes: Setting `remove_k = 0` reduces REM to a standard acquisition strategy.

## Pool management

File: `esnlir/active_learning/pool_manager.py`

- Maintains two disjoint, exhaustive lists of global indices:
  - `labeled_indices` (initially empty by default)
  - `unlabeled_indices` (covers the remainder)
- Methods:
  - `get_unlabeled_subset(dataset)`: `Subset(dataset, unlabeled_indices)`
  - `get_labeled_subset(dataset)`: `Subset(dataset, labeled_indices)`
  - `acquire(indices)`: move indices from unlabeled → labeled, preserving stable order
  - `remove(indices)`: drop indices from unlabeled pool (discard)
- Invariants: `labeled_indices \cap unlabeled_indices = \varnothing` and union equals all dataset indices.

## Orchestration trainer

File: `esnlir/active_learning/active_trainer.py`

- Runs $L$ cycles of acquisition and training. The workflow per iteration $t$:

  1. Build an unlabeled `DataLoader` over `Subset(unlabeled_indices)`.
  2. Strategy scoring and selection on the subset (batched, `torch.no_grad()`).
  3. Pool update: `remove(removed)` (if strategy returns removals), then `acquire(selected)`.
  4. Train a `WeightedTrainer` (Hugging Face Trainer subclass) on the labeled subset.
  5. Evaluate on test splits using the project’s `MetricGenerator` and write CSV reports.
  6. Log per-iteration artifacts:
     - `selected_indices_iter_t.csv`
     - `removed_indices_iter_t.csv` (if applicable)
     - `metrics_iter_t.json`

- Warm-start (default): keep model weights across iterations (continued fine-tuning). If disabled, reinitialize from the base pretrained checkpoint each iteration.
- Output location: `models/{base_output_dir}/{model_short}_active_{strategy}/`
- Configuration dataclass:
  - `ActiveLearningConfig(L=500, K=8, remove_k=None, scoring_batch_size=64, warm_start=True)`

## Usage and configuration

The standard training script `esnlir/training/train.py` is extended with AL flags and config keys. CLI flags override JSON.

- JSON keys (optional):

  - `"active_learning"`: true | false
  - `"al_strategy"`: "NegE" | "Random" | "Rem"
  - `"al_L"`: integer (iterations)
  - `"al_K"`: integer (acquisitions per iteration)
  - `"al_remove_k"`: integer (for REM)
  - `"al_scoring_batch_size"`: integer (default 64)
  - `"warm_start"`: true | false (default true)
  - `"al_iter_patience"`: integer (optional) early stopping across AL iterations based on validation metric
  - `"al_iter_min_delta"`: float (optional) minimum improvement to reset patience
  - `"al_iter_metric"`: string (optional) which metric to monitor across iterations (default uses training `monitor`)
  - `"al_top_p"`: float in (0,1] (optional, NegE) restrict acquisitions to the top p proportion by energy
  - `"al_min_energy"`: float (optional, NegE) minimum energy threshold to be eligible for acquisition

- CLI examples (PowerShell on Windows):

```powershell
# Normal training (unchanged behavior)
python esnlir\training\train.py --config-file params\train_xlmroberta.json

# Active Learning: Negative Energy, 1 iteration, K=2
python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy NegE --al_L 1 --al_K 2

# Active Learning: Random baseline
python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy Random --al_L 1 --al_K 2

# Active Learning: REM (remove 5 lowest-utility, then select 2)
python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy Rem --al_remove_k 5 --al_L 1 --al_K 2

# Active Learning with iteration-level early stopping and thresholds
python esnlir\training\train.py --config-file params\train_xlmroberta.json --active-learning --al_strategy NegE --al_L 50 --al_K 50 --al_iter_patience 3 --al_iter_min_delta 0.001 --al_iter_metric f1_score --al_top_p 0.2 --al_min_energy 1.5
```

After the first iteration ($t=0$), expect at least:

- `selected_indices_iter_0.csv`
- `metrics_iter_0.json`

under:

```
models/{configured_output_folder}/{model_short}_active_{strategy}/
```

## Practical considerations

- Batching and device: scoring uses the configured batch size (default 64) under `torch.no_grad()` to minimize memory. Inputs and model are moved to the configured `device`.
- Label format: the dataset provides one-hot labels; `WeightedTrainer` uses a cross-entropy loss compatible with this format.
- Class imbalance: class weights from the dataset are passed to the `WeightedTrainer` to mitigate imbalance.
- Reproducibility: seeding is configured at the start of `train.py` using `seed_everything`.

## Complexity and cost

- Let $N = |\mathcal{U}|$, $C$ classes, batch size $B$.
  - Random selection: negligible scoring cost; $\mathcal{O}(N)$ indexing + $\mathcal{O}(K)$ sampling.
  - Negative Energy: one forward pass over $\mathcal{U}$, $\approx \lceil N/B \rceil$ batches; sorting $\mathcal{O}(N\log N)$.
  - REM: same forward pass + sorting; additional set operations for removal.
- Training dominates runtime depending on $|\mathcal{L}|$ and epochs.

## Limitations and extensions

- Initial labeled set: currently defaults to empty; a small stratified seed could stabilize early iterations.
- Budget scheduling: fixed $K$ per iteration; could adapt based on validation metrics.
- Diversity: add coverage/diversity-aware selection (e.g., k-Medoids in embedding space) to complement uncertainty.
- Calibration: energy/entropy rely on model calibration; temperature scaling could be used to improve entropy-based utility.

## File map

- `base_strategy.py`: abstract interface
- `random_sel.py`: RandomStrategy (baseline)
- `neg_energy.py`: NegativeEnergyStrategy (uncertainty via energy)
- `rem.py`: RemStrategy (remove low-utility then select)
- `pool_manager.py`: management of labeled/unlabeled pools
- `active_trainer.py`: orchestration of AL cycles with HF Trainer

## References (informal)

- Energy-based confidence: scoring via `-logsumexp(logits)` is a common uncertainty proxy derivable from the partition function.
- Entropy-based uncertainty: classic criterion for classification uncertainty.
