"""Pool management utilities for Active Learning."""
from __future__ import annotations

from typing import Iterable, List

import random
from torch.utils.data import Subset


class PoolManager:
    """Tracks labeled and unlabeled indices of a dataset.

    Methods
    - get_unlabeled_subset(dataset)
    - get_labeled_subset(dataset)
    - acquire(indices): move indices from unlabeled to labeled
    - remove(indices): remove indices from unlabeled (discard)
    """

    def __init__(self, total_size: int, initial_labeled_indices: Iterable[int] | None = None):
        all_indices = list(range(total_size))
        if initial_labeled_indices is None:
            self.labeled_indices: List[int] = []
            self.unlabeled_indices: List[int] = all_indices
        else:
            il = sorted(set(int(i) for i in initial_labeled_indices))
            self.labeled_indices = il
            self.unlabeled_indices = [i for i in all_indices if i not in set(il)]

    def get_unlabeled_subset(self, dataset):
        return Subset(dataset, self.unlabeled_indices)

    def get_labeled_subset(self, dataset):
        return Subset(dataset, self.labeled_indices)

    def acquire(self, indices: Iterable[int]):
        s = set(self.unlabeled_indices)
        acquired = []
        for i in indices:
            if i in s:
                acquired.append(i)
        # Update lists while preserving stable order
        unlabeled_set = set(self.unlabeled_indices)
        for i in acquired:
            unlabeled_set.discard(i)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i in unlabeled_set]
        self.labeled_indices.extend(acquired)

    def remove(self, indices: Iterable[int]):
        removed_set = set(int(i) for i in indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in removed_set]
