# ESNLIR — Active Learning

Pool-based **active learning for Spanish Natural Language Inference**: two energy-based acquisition
strategies on the ESNLIR corpus, evaluated on a newly annotated 1,695-pair test set.

> **Active Learning for Spanish Natural Language Inference on a Heterogeneous Multi-Domain Corpus**
> Diego Ortiz, Johan R. Portela, Ruben Manrique — Universidad de los Andes, Bogotá
> *Advances in Artificial Intelligence — IBERAMIA 2026* (to appear)

Annotation is expensive. The question here is how much of it you can skip: if a model chooses which
examples to label next, how close does it get to training on everything?

## Main result

**Active learning gets most of the way there, but energy-based selection is not why.**

| encoder | method | iter. | labels | acc. | macro F1 | contr. | entail. | neutr. | reas. |
|---|---|---|---|---|---|---|---|---|---|
| XLM-R | full supervision | — | 1,000k | 0.829 | *0.822* | 0.813 | 0.824 | 0.876 | 0.776 |
| XLM-R | **NegE (best)** | 5 | 610k | 0.801 | **0.796** | 0.809 | 0.792 | 0.845 | 0.740 |
| XLM-R | NegE (final) | 9 | 1,000k | 0.794 | 0.789 | 0.788 | 0.802 | 0.833 | 0.735 |
| XLM-R | LER (final) | 7 | 800k | 0.795 | 0.789 | 0.787 | 0.777 | 0.846 | 0.746 |
| Bertin | full supervision | — | 1,000k | 0.818 | *0.811* | 0.796 | 0.812 | 0.864 | 0.773 |
| Bertin | **NegE (best)** | 6 | 710k | 0.771 | **0.764** | 0.749 | 0.767 | 0.819 | 0.722 |
| Bertin | NegE (final) | 9 | 1,000k | 0.758 | 0.752 | 0.726 | 0.755 | 0.803 | 0.722 |
| Bertin | LER (final) | 7 | 800k | 0.760 | 0.754 | 0.752 | 0.740 | 0.816 | 0.706 |

Three findings:

* **AL reaches 95.5% of full supervision with 61% of the labels** — 0.796 vs 0.822 macro F1 — but
  never closes the remaining 2.6-point gap. Performance **peaks near 610k labels and then declines**;
  more labels stop helping.
* **NegE and LER are indistinguishable** (≤0.7 points apart within an encoder). Since LER acquires
  **uniformly at random** from a filtered pool, energy-based acquisition buys nothing over
  near-random selection. That is the paper's central negative result.
* **Cross-lingual beats monolingual everywhere.** XLM-RoBERTa leads Bertin-RoBERTa by 3.2–4.2 points
  in every configuration, and under full supervision too, so it is a property of the encoders rather
  than an interaction with acquisition.

Web news, testimonial, literary and clinical text stay hardest throughout; no acquisition strategy
narrows the gap between the easiest and hardest genre.

## Acquisition strategies

Both score instances by **energy** over the class logits,
$a(x) = -\log \sum_k \exp(z_k(x))$ — high energy means the model gives no class a large logit.

**NegE** ranks the unlabeled pool by energy and acquires the highest-scoring instances. Unlike
softmax entropy, energy is not distorted by probability normalization.

**LER (Low-Energy Removal)** filters instead of ranking: each round it permanently discards the
$r$ lowest-energy instances — the ones the model already handles — then draws the acquisition batch
**uniformly at random from the survivors**. Energy decides only what is removed, never what is
selected. This makes LER a near-random control: in early rounds it samples uniformly from more than
90% of the pool.

> In the code and config filenames LER is called `Rem`. Same strategy, older name.

The **full-supervision reference** is the pair of publicly released ESNLIR models
([`Flaglab/ESNLIR-XLM-RoBERTa`](https://huggingface.co/Flaglab/ESNLIR-XLM-RoBERTa),
[`Flaglab/ESNLIR-RoBERTa`](https://huggingface.co/Flaglab/ESNLIR-RoBERTa)), trained conventionally
over the whole 1M pool. They are a ceiling, not an AL arm.

## Artifacts

Everything is on the Hub — 🤗 [**ESNLIR Active Learning collection**](https://huggingface.co/collections/Flaglab/esnlir-active-learning-6a74c197d0a463e76d2dea82).

| artifact | what it is |
|---|---|
| [`Flaglab/esnlir-al-annotated-test`](https://huggingface.co/datasets/Flaglab/esnlir-al-annotated-test) | the 1,695-pair annotated test set |
| [`Flaglab/esnlir-annotation-candidates`](https://huggingface.co/datasets/Flaglab/esnlir-annotation-candidates) | the 2,664 confidence-stratified candidates sent to annotators |
| [`Flaglab/esnlir-al-trajectories`](https://huggingface.co/datasets/Flaglab/esnlir-al-trajectories) | per-round metrics and acquired/removed pool indices for every run |
| [`Flaglab/ESNLIR-AL-XLM-RoBERTa-NegE`](https://huggingface.co/Flaglab/ESNLIR-AL-XLM-RoBERTa-NegE) | best checkpoint, iter 5, 610k labels |
| [`Flaglab/ESNLIR-AL-XLM-RoBERTa-LER`](https://huggingface.co/Flaglab/ESNLIR-AL-XLM-RoBERTa-LER) | best checkpoint, iter 7, 800k labels |
| [`Flaglab/ESNLIR-AL-BERTIN-NegE`](https://huggingface.co/Flaglab/ESNLIR-AL-BERTIN-NegE) | best checkpoint, iter 6, 710k labels |
| [`Flaglab/ESNLIR-AL-BERTIN-LER`](https://huggingface.co/Flaglab/ESNLIR-AL-BERTIN-LER) | best checkpoint, iter 7, 800k labels |

The training pool is the ESNLIR corpus,
[`Flaglab/ESNLIR-dataset`](https://huggingface.co/datasets/Flaglab/ESNLIR-dataset) — not duplicated
here.

### The annotated test set

1,695 instances across **24 domains and 8 genres**, spanning formal registers (legal, clinical) and
informal ones (web comments, tweets). Classes are uneven — `neutral` 37.4% (634) against
`contrasting` 18.4% (312) — which is why **macro F1 is the primary metric**, not accuracy.

Candidates were sampled by model confidence before annotation. Both ESNLIR baselines scored the test
pool, and instances were binned into four strata by their mean max-softmax — `high`, `medium` and
`low` for pairs both models got right, plus `no` for pairs they got wrong (which is why that
stratum's confidence spans the whole range: a model can be confidently mistaken). 2,664 candidates
went to annotators; 1,695 survived.

The full selection pipeline is in [`notebooks/annotation/`](notebooks/annotation/), and the candidate
pool itself is published as
[`Flaglab/esnlir-annotation-candidates`](https://huggingface.co/datasets/Flaglab/esnlir-annotation-candidates).

This has two consequences pulling in opposite directions: confidence stratification enriches the set
for instances the reference checkpoint finds *hard*, biasing its score down, while the
majority-must-match-connector retention rule keeps only pairs that are *easy to label consistently*,
biasing scores up. Absolute numbers here are therefore not comparable with the full ESNLIR test
split — **relative comparisons between systems are the intended use.**

## Configuration

All runs use identical settings; only the encoder and the acquisition strategy vary, so differences
are not confounded by tuning.

| parameter | value | | cost (1× A100 40GB) | |
|---|---|---|---|---|
| learning rate | 2e-5 | | AL rounds (NegE / LER) | 10 / 8 |
| batch size | 32 | | training passes (NegE) | 16.8M |
| epochs per round | 3 | | scoring passes (NegE) | 5.4M |
| acquisition size *b* | 100,000 | | wall-clock XLM-R NegE | 9.7 h |
| LER removal size *r* | 25,000 | | wall-clock XLM-R LER | 6.5 h |
| seed set | 10,000 | | throughput (train) | 523 ex/s |
| max sequence length | 256 | | throughput (scoring) | 2705 ex/s |
| warm start | yes | | | |

## Installation

Python 3.10+.

```bash
git clone https://github.com/jd-rodriguezp1234/esnlir-active-learning.git
cd esnlir-active-learning
uv sync                      # or: python -m venv .venv && . .venv/bin/activate
                             #     pip install -e . && pip install -r requirements.txt
```

## Running

Fetch the pool and the annotated test set into `data/`:

```python
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download("Flaglab/ESNLIR-dataset", repo_type="dataset", local_dir="data",
                  allow_patterns=["train.json", "val.json", "test.json"])
hf_hub_download("Flaglab/esnlir-al-annotated-test", "test_full.jsonl",
                repo_type="dataset", local_dir="data")
```

Then run a configuration — each is a JSON file under [`params/`](params/) or
[`colab/params/`](colab/params/):

```bash
python -m esnlir.training.train --config-file colab/params/model-xlm__strategy-NegE__colab_fulldata.json
```

Drop `--active-learning` handling by pointing at a non-AL config to train conventionally. A short
end-to-end check that runs in minutes:

```bash
python scripts/run_al_smoke_test.py
```

The paper's runs were executed on Colab — see [`colab/AL_NLI_Colab.ipynb`](colab/AL_NLI_Colab.ipynb)
for the AL runs and [`colab/Baseline_Eval_Colab.ipynb`](colab/Baseline_Eval_Colab.ipynb) for
evaluating the full-supervision references on the annotated test set.
[`docs/ACTIVE_LEARNING_RUN.md`](docs/ACTIVE_LEARNING_RUN.md) walks through a full run.

## Layout

```
esnlir/
  active_learning/     strategy interface, NegE, LER (rem.py), random baseline,
                       pool manager, and the orchestrating ActiveTrainer
  training/train.py    full training flow; --active-learning switches on the AL loop
  dataset_utils/       ESNLIR splits as a torch Dataset
  evaluation/          metrics per genre / domain / source dataset / total
  utils/               seeding
params/                hyperparameter configs; params/experiments/ holds the sweep
colab/                 notebooks and configs for the Colab runs, plus data prep
scripts/               smoke test and batch experiment runner
notebooks/             single-pair prediction against a published checkpoint
  annotation/          building the annotated test set: scoring, confidence stratification,
                       Label Studio round-trip (see its README)
docs/                  ACTIVE_LEARNING_RUN.md — end-to-end run guide
```

`CHANGES.md` records the debugging of an apparent single-class collapse in XLM-R — it turned out to
be a 40-row smoke run that never left random initialisation, not class bias. Worth reading before
interpreting any short run.

## Citation

```bibtex
@InProceedings{ortiz2026activelearningspanishnli,
  author    = {Ortiz, Diego and Portela, Johan R. and Manrique, Ruben},
  title     = {Active Learning for Spanish Natural Language Inference
               on a Heterogeneous Multi-Domain Corpus},
  booktitle = {Advances in Artificial Intelligence -- IBERAMIA 2026},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  note      = {To appear},
}
```

The corpus is ESNLIR ([Portela, Pérez-Terán & Manrique, 2026](https://doi.org/10.1007/978-3-032-07175-0_23)),
a separate project with its own [code](https://github.com/jd-rodriguezp1234/esnlir) and paper.
