import numpy as np
import torch
import torch.nn as nn


def mixup(batch, ab):
    raw_x, targets, d_targets = batch
    x = raw_x.clone()

    indices = torch.randperm(x.size(0))
    shuffled_x = x[indices] + torch.normal(mean=0, std=0.01, size=raw_x.shape).to(raw_x.device)

    lam = np.random.beta(ab[0], ab[1])

    mixed_x = lam * x + (1 - lam) * shuffled_x
    mixed_targets = (targets, targets[indices], lam)
    mixed_d_targets = (d_targets, d_targets[indices], lam)

    return mixed_x, mixed_targets, mixed_d_targets


def mixup_group(batch, ab):
    x, targets = batch
    shuffled_x = x.clone()

    unique_groups = torch.unique(targets)
    for group in unique_groups:
        group_indices = torch.nonzero(targets == group, as_tuple=True)[0]
        group_x = x[group_indices]
        shuffled_indices = torch.randperm(len(group_indices))
        shuffled_x[group_indices] = group_x[shuffled_indices]
    shuffled_x = shuffled_x + torch.normal(mean=0, std=0.01, size=shuffled_x.shape).to(shuffled_x.device)

    lam = np.random.beta(ab[0], ab[1])

    mixed_x = lam * x + (1 - lam) * shuffled_x

    return mixed_x


class MixUpCriterion:
    """
    Used to calculate the loss of mixed samples
    """
    def __init__(self, reduction='mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


