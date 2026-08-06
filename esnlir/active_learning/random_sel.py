"""Random acquisition Active Learning strategy."""
from __future__ import annotations

import random
from typing import List

from .base_strategy import ActiveLearningStrategy


class RandomStrategy(ActiveLearningStrategy):
    """Select k random examples from the unlabeled pool.

    It assumes unlabeled_loader.dataset is a torch.utils.data.Subset with `.indices`
    mapping to global indices in the full dataset.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, model, unlabeled_loader, k: int) -> List[int]:
        # Build the list of global indices present in current unlabeled pool
        # The pool is the concatenation over batches. We rely on Subset.indices
        subset = unlabeled_loader.dataset
        pool_indices = getattr(subset, "indices", None)
        if pool_indices is None:
            # Fallback: accumulate via iteration (slower but safe)
            pool_indices = []
            start = 0
            for batch in unlabeled_loader:
                batch_size = next(iter(batch.values())).shape[0] if isinstance(batch, dict) else len(batch)
                pool_indices.extend(list(range(start, start + batch_size)))
                start += batch_size
        pool_indices = list(pool_indices)
        if k >= len(pool_indices):
            return pool_indices
        return self.rng.sample(pool_indices, k)
