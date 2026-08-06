# Building the annotated test set

How the 1,695-pair evaluation set was produced: score the ESNLIR test pool with the released
baselines, stratify candidates by model confidence, ship them to Label Studio, and fold the human
labels back in.

The output is published as
[`Flaglab/esnlir-al-annotated-test`](https://huggingface.co/datasets/Flaglab/esnlir-al-annotated-test);
the candidate pool that went to annotators is
[`Flaglab/esnlir-annotation-candidates`](https://huggingface.co/datasets/Flaglab/esnlir-annotation-candidates).

## Pipeline

| notebook | what it does |
|---|---|
| `0_custom_prediction.ipynb` | sanity check — run a fine-tuned checkpoint on a single pair |
| `1_proba_calculation.ipynb` | score the test pool with both baselines, saving logits to `y_pred_bertin.json` / `y_pred_xlm_roberta.json` |
| `2_1_…_high_confidence.ipynb` | select the **high** stratum |
| `2_2_…_medium_confidence.ipynb` | select the **medium** stratum |
| `2_3_…_low_confidence.ipynb` | select the **low** stratum |
| `2_4_…_no_confidence.ipynb` | select the **no-confidence** stratum |
| `2_5_merge_new_annotation_examples.ipynb` | merge the four strata into one candidate file |
| `3_jsonl_to_label_studio_format.ipynb` | JSONL → Label Studio task format |
| `process_dataset.ipynb` | the same conversion, standalone |
| `5_label_studio_format_to_jsonl.ipynb` | completed Label Studio annotations → JSONL |

Run them in numeric order. Each `2_x` notebook excludes the instances already taken by the earlier
strata, so the order matters.

## How the strata are defined

Confidence is the **mean of the maximum softmax probability across both baselines** —
`bertin-roberta-base-spanish` and `xlm-roberta-base` — not a single model's score.

The first three strata contain only instances **both models classified correctly**, split by that
mean confidence:

| stratum | criterion | n | confidence range |
|---|---|---|---|
| `high` | both correct, high confidence | 1,856 | 0.965 – 0.999 |
| `medium` | both correct, mid confidence | 284 | 0.600 – 0.749 |
| `low` | both correct, low confidence | 248 | 0.298 – 0.494 |
| `no` | **not** both correct | 276 | 0.337 – 0.983 |

`no` is the important one to read carefully: it is defined by the models being **wrong**, not by a
confidence band, which is why its confidence range spans almost the whole interval. A pair can sit
here with 0.98 confidence — the model was confidently mistaken. Those are the most diagnostic
instances in the set.

Total: **2,664 candidates**, balanced at 666 per class.

## Annotation

Candidates were loaded into a self-hosted Label Studio instance and assigned to annotators in
**overlapping ranges** — sliding windows offset by 250 instances, so consecutive annotators shared
half their workload. That overlap is what makes the paper's *overlap-weighted mean* Cohen's κ
computable, and it is why each instance carries roughly 3 independent labels.

Final labels were assigned by majority vote, and a pair was retained only when that majority matched
the connector-derived label — 151 disagreements were discarded, leaving 1,695.

> **The notebook that launched Label Studio is not included.** It contained annotator accounts,
> their credentials and the server address. The assignment design is described above; the mechanics
> of standing up the server are not reproducible from this repository by design.

## Inputs

These notebooks expect the ESNLIR splits and the two baseline checkpoints:

```python
from huggingface_hub import snapshot_download
snapshot_download("Flaglab/ESNLIR-dataset", repo_type="dataset", local_dir="data",
                  allow_patterns=["test.json"])
# baselines: Flaglab/ESNLIR-RoBERTa and Flaglab/ESNLIR-XLM-RoBERTa
```
