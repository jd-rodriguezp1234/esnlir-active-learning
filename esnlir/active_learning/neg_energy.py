"""Negative Energy acquisition strategy.

Implements uncertainty sampling using energy-based score:
E(x) = - logsumexp(logits)
Selects top-k samples with highest energy (most uncertain).
"""
from __future__ import annotations

from typing import List

import torch

from .base_strategy import ActiveLearningStrategy


class NegativeEnergyStrategy(ActiveLearningStrategy):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        min_energy_threshold: float | None = None,
        top_p: float | None = None,
    ):
        """
        Parameters
        ----------
        device: torch device for forward passes.
        min_energy_threshold: if set, keep only samples with energy >= threshold before slicing top-k.
        top_p: if set in (0,1], keep only the top p proportion by energy before slicing top-k.
        """
        self.device = torch.device(device)
        self.min_energy_threshold = min_energy_threshold
        self.top_p = top_p

    @torch.no_grad()
    def select(self, model, unlabeled_loader, k: int) -> List[int]:
        model.eval()
        scores = []  # list of tuples (global_index, energy)

        subset = unlabeled_loader.dataset
        # Expect torch.utils.data.Subset with .indices
        pool_indices = getattr(subset, "indices", None)

        running_idx = 0
        for batch in unlabeled_loader:
            # Prepare inputs on device; ignore labels during scoring
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.get("logits")  # shape [B, C]

            # Energy score per sample: E(x) = - logsumexp(logits)
            energy = -torch.logsumexp(logits, dim=-1)

            bsz = energy.shape[0]
            if pool_indices is not None:
                batch_indices = pool_indices[running_idx: running_idx + bsz]
            else:
                batch_indices = list(range(running_idx, running_idx + bsz))
            running_idx += bsz

            for gi, e in zip(batch_indices, energy.detach().cpu().tolist()):
                scores.append((int(gi), float(e)))

        # Sort by energy descending (higher uncertainty first)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Optional filtering by top_p proportion
        cand = scores
        if self.top_p is not None and 0 < self.top_p <= 1:
            keep = max(1, int(len(scores) * float(self.top_p)))
            cand = scores[:keep]

        # Optional filtering by absolute threshold
        if self.min_energy_threshold is not None:
            cand = [t for t in cand if t[1] >= float(self.min_energy_threshold)]

        selected = [gi for gi, _ in cand[:k]]
        return selected
