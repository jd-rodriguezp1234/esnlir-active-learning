"""Base classes for Active Learning strategies.

Defines the abstract interface for an Active Learning strategy that selects
indices from an unlabeled pool to be labeled/acquired.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class ActiveLearningStrategy(ABC):
    """Abstract AL strategy.

    Contract
    - Inputs:
      - model: a torch.nn.Module compatible with HF models (with .eval(), .to())
      - unlabeled_loader: DataLoader over a torch.utils.data.Subset of the full train dataset
      - k: number of samples to select
    - Output:
      - A list of global indices of size k ordered by acquisition preference.
        Some strategies (e.g., REM) may return a tuple (selected_indices, removed_indices).
    """

    @abstractmethod
    def select(self, model, unlabeled_loader, k: int) -> Union[List[int], Tuple[List[int], List[int]]]:
        """Select k samples from the unlabeled pool.

        Parameters
        ----------
        model: torch.nn.Module
            Model used for scoring. Must be in eval() mode outside; strategy may also set it.
        unlabeled_loader: torch.utils.data.DataLoader
            DataLoader over a Subset; unlabeled_loader.dataset.indices must map to global indices.
        k: int
            Number of samples to acquire.

        Returns
        -------
        List[int] | Tuple[List[int], List[int]]
            Global indices to acquire; optionally also indices to remove (for REM strategy).
        """
        raise NotImplementedError
