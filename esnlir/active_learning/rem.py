"""REM strategy: Remove low-utility samples then acquire from remaining pool.

Utility can be entropy or energy; default uses energy for consistency.
Returns tuple (selected_indices, removed_indices) as global indices.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .base_strategy import ActiveLearningStrategy


def energy_scores(logits: torch.Tensor) -> torch.Tensor:
    # E(x) = - logsumexp(logits)
    return -torch.logsumexp(logits, dim=-1)


def entropy_scores(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    # Numerical stability: clamp
    probs = torch.clamp(probs, min=1e-12)
    return -(probs * probs.log()).sum(dim=-1)


class RemStrategy(ActiveLearningStrategy):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        remove_k: int = 0,
        utility: str = "energy",
        select_mode: str = "random",
        seed: int | None = None
    ):
        """Initialize REM.

        Parameters
        ----------
        device: torch device
        remove_k: number of samples to remove from pool (lowest utility)
        utility: 'energy' or 'entropy'
        select_mode: 'random' or 'uncertainty' (use high utility)
        seed: seed for the random selection stream (select_mode='random')
        """
        self.device = torch.device(device)
        self.remove_k = int(remove_k) if remove_k is not None else 0
        self.utility = utility
        self.select_mode = select_mode
        self.rng = random.Random(seed)

    @torch.no_grad()
    def select(self, model, unlabeled_loader, k: int) -> Tuple[List[int], List[int]]:
        model.eval()
        scores = []  # (global_idx, utility)

        subset = unlabeled_loader.dataset
        pool_indices = getattr(subset, "indices", None)
        running_idx = 0
        for batch in unlabeled_loader:
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = model(**inputs).get("logits")

            if self.utility == "entropy":
                util = entropy_scores(logits)
            else:
                util = energy_scores(logits)

            bsz = util.shape[0]
            if pool_indices is not None:
                batch_indices = pool_indices[running_idx: running_idx + bsz]
            else:
                batch_indices = list(range(running_idx, running_idx + bsz))
            running_idx += bsz

            for gi, u in zip(batch_indices, util.detach().cpu().tolist()):
                scores.append((int(gi), float(u)))

        # Sort by utility ascending to remove lowest utility
        scores.sort(key=lambda x: x[1])
        # `scores` is sorted, so the removed slice and the kept slice are contiguous
        to_remove = [gi for gi, _ in scores[: self.remove_k]] if self.remove_k > 0 else []
        kept = scores[self.remove_k:] if self.remove_k > 0 else scores

        remaining = [gi for gi, _ in kept]

        if k >= len(remaining):
            selected = remaining
        else:
            if self.select_mode == "uncertainty":
                # `kept` is ascending by utility, so the highest utility is at the tail
                selected = [gi for gi, _ in reversed(kept[-k:])]
            else:
                # Random selection from remaining
                selected = self.rng.sample(remaining, k)

        return selected, to_remove
