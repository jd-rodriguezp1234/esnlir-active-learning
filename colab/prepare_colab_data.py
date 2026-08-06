"""Build a Drive-sized copy of data/ for Colab.

Run this LOCALLY (where the full dataset lives), then upload only `data_colab/`
to Drive. train.json is ~3 GB and pandas loads it whole before `max_samples`
subsamples it, so shipping the full file to Colab is both a slow upload and an
out-of-memory risk on standard runtimes.

Sampling is proportional-stratified over (connector_type, dataset) using two
streaming passes, so nothing large is ever held in memory. The result is
distributionally equivalent to what BERTDataset's own `max_samples` subsample
would produce, but it is NOT the identical subset -- runs prepared this way are
comparable to each other, not row-for-row to an HPC run.

Usage
-----
    python colab/prepare_colab_data.py                     # 1M train, 20k val
    python colab/prepare_colab_data.py --train-n 100000    # smaller
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict

KEY_RE = re.compile(rb'"connector_type"\s*:\s*"([^"]+)".*?"dataset"\s*:\s*"([^"]+)"')


def stratum_of(raw_line: bytes):
    """(connector_type, dataset) via regex; falls back to json for odd field order."""
    match = KEY_RE.search(raw_line)
    if match:
        return match.group(1).decode(), match.group(2).decode()
    record = json.loads(raw_line)
    return record["connector_type"], record["dataset"]


def count_strata(path):
    counts = Counter()
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                counts[stratum_of(line)] += 1
    return counts


def sample_file(src, dst, target_n, seed):
    """Proportional stratified sample, streaming, using per-stratum reservoirs."""
    counts = count_strata(src)
    total = sum(counts.values())
    if target_n >= total:
        print(f"  {os.path.basename(src)}: {total:,} rows <= target, copying whole file")
        shutil.copyfile(src, dst)
        return total

    quota = {k: max(1, round(count * target_n / total)) for k, count in counts.items()}
    rng = random.Random(seed)
    seen = Counter()
    reservoir = defaultdict(list)
    with open(src, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            key = stratum_of(line)
            seen[key] += 1
            limit = quota[key]
            if len(reservoir[key]) < limit:
                reservoir[key].append(line)
            else:
                j = rng.randrange(seen[key])
                if j < limit:
                    reservoir[key][j] = line

    kept = [line for lines in reservoir.values() for line in lines]
    rng.shuffle(kept)
    with open(dst, "wb") as handle:
        handle.writelines(kept)
    classes = Counter(stratum_of(line)[0] for line in kept)
    print(f"  {os.path.basename(src)}: {total:,} -> {len(kept):,} rows  classes={dict(classes)}")
    return len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data", help="source data folder")
    parser.add_argument("--dst", default="data_colab", help="output folder to upload to Drive")
    parser.add_argument("--train-n", type=int, default=1_000_000)
    parser.add_argument("--val-n", type=int, default=20_000)
    parser.add_argument("--test-n", type=int, default=0, help="0 = copy test.json whole")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    print(f"Writing to {args.dst}/")

    # Names must stay train.json / val.json / test*.json -- train.py looks for those exactly
    sample_file(os.path.join(args.src, "train.json"),
                os.path.join(args.dst, "train.json"), args.train_n, args.seed)
    sample_file(os.path.join(args.src, "val.json"),
                os.path.join(args.dst, "val.json"), args.val_n, args.seed)

    test_src = os.path.join(args.src, "test.json")
    if os.path.exists(test_src):
        if args.test_n:
            sample_file(test_src, os.path.join(args.dst, "test.json"), args.test_n, args.seed)
        else:
            shutil.copyfile(test_src, os.path.join(args.dst, "test.json"))
            print("  test.json: copied whole")

    full_src = os.path.join(args.src, "test_full.jsonl")
    if os.path.exists(full_src):
        shutil.copyfile(full_src, os.path.join(args.dst, "test_full.jsonl"))
        print("  test_full.jsonl: copied whole")

    size = sum(
        os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst)
    )
    print(f"\nTotal {size/1e9:.2f} GB in {args.dst}/ -- upload this folder to Drive")
    print("Set \"max_samples\": null in the Colab config so it is not subsampled again.")


if __name__ == "__main__":
    main()
