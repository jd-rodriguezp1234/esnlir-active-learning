"""Utils module"""
import os
import random
import numpy as np
import torch

def stratified_indices(targets, size, n_classes, seed):
    """Class-balanced sample of `size` row indices from one-hot `targets`.

    Indices are returned sorted so that Subsets built from them preserve dataset order.
    Drawing from a fixed seed keeps the sample identical across runs and strategies.
    """
    labels = np.asarray(targets).argmax(axis=1)
    rng = np.random.default_rng(seed)
    per_class = max(1, int(size) // n_classes)
    indices = []
    for class_id in range(n_classes):
        class_indices = np.flatnonzero(labels == class_id)
        take = min(per_class, len(class_indices))
        indices.extend(rng.choice(class_indices, size=take, replace=False).tolist())
    return sorted(int(i) for i in indices)

def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True