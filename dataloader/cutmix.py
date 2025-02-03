import numpy as np
import torch
import torch.nn as nn


def cutmix(batch, ab):
    """
    CutMix: used to mix samples across classes and domains
    Arguments:
        batch: (input data, class labels, domain labels)
        ab: parameters for beta distribution
    """
    raw_data, targets, d_targets = batch
    data = raw_data.clone()

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices] + torch.normal(mean=0, std=0.01, size=raw_data.shape).to(raw_data.device)

    lam = np.random.beta(ab[0], ab[1])

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    mixed_targets = (targets, targets[indices], lam)
    mixed_d_targets = (d_targets, d_targets[indices], lam)

    return data, mixed_targets, mixed_d_targets


def cutmix_group(batch, ab):
    """
    Grouping CutMix: used to mix samples in classes and across domains
    Arguments:
        batch: (input data, class labels)
        ab: parameters for beta distribution
    """
    raw_data, targets = batch
    data = raw_data.clone()
    shuffled_data = raw_data.clone()

    unique_groups = torch.unique(targets)
    for group in unique_groups:
        group_indices = torch.nonzero(targets == group, as_tuple=True)[0]
        group_data = data[group_indices]
        shuffled_indices = torch.randperm(len(group_indices))
        shuffled_data[group_indices] = group_data[shuffled_indices]
    shuffled_data = shuffled_data + torch.normal(mean=0, std=0.01, size=raw_data.shape).to(shuffled_data.device)

    lam = np.random.beta(ab[0], ab[1])

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    return data


def cutmix2(raw_data, aug_data, ab):
    """
    Mixing two images (raw_data and aug_data) using CutMix
    """
    data = raw_data.clone()
    aug_data = aug_data + torch.normal(mean=0, std=0.01, size=aug_data.shape).to(aug_data.device)

    lam = np.random.beta(ab[0], ab[1])

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = aug_data[:, :, y0:y1, x0:x1]

    return data


def fine_tune_cutmix(batch, aug_data, ab, nums):
    """
    Given a batch of samples, generate nums times mixed ones
    """
    data, targets = batch
    data_outputs = data.clone()
    for i in range(nums):
        data_aug = data.clone()
        for j in range(len(data_aug)):
            data_j = data_aug[j]
            label_j = targets[j]
            idx = np.random.randint(len(aug_data[int(label_j)][0]))
            data_aug_j = torch.load(aug_data[int(label_j)][0][idx])
            data_aug[j] = cutmix2(data_j, data_aug_j, ab)
        data_outputs = torch.cat([data_outputs, data_aug], dim=0)
    targets = torch.tensor(list(targets)*(nums+1)).to(targets.device)
    return data_outputs, targets


class CutMixCriterion:
    """
    Used to calculate the loss of mixed samples
    """
    def __init__(self, reduction='mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)

