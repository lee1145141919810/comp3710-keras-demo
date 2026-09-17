"""Dice similarity coefficient (DSC) - loss and evaluation metric for segmentation.

DSC(A, B) = 2 |A ∩ B| / (|A| + |B|)  -> 1 for a perfect overlap, 0 for none.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target_one_hot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable 1 - mean-class soft Dice between softmax probabilities and one-hot targets.

    Computed per class over the whole batch, then averaged over classes, so that small classes
    (CSF) weigh as much as large ones (background) - this directly counters class imbalance.
    """
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)  # batch + spatial
    intersection = (probs * target_one_hot).sum(dims)
    denominator = probs.sum(dims) + target_one_hot.sum(dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class DiceAccumulator:
    """Accumulates per-class overlap statistics over many batches to report dataset-level DSC.

    Aggregating |P ∩ G|, |P| and |G| first (instead of averaging per-image scores) avoids the
    undefined 0/0 case for slices in which a tissue class is absent.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.intersection = torch.zeros(n_classes, dtype=torch.float64)
        self.pred_sum = torch.zeros(n_classes, dtype=torch.float64)
        self.target_sum = torch.zeros(n_classes, dtype=torch.float64)
        self.per_image: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, pred_labels: torch.Tensor, target_labels: torch.Tensor) -> None:
        """``pred_labels`` / ``target_labels``: integer label maps ``[B, H, W]``."""
        # Per-image pixel counts fit exactly in float32 (< 2^24); float64 is only used for the
        # running totals on the CPU because Apple MPS has no float64 support.
        pred = F.one_hot(pred_labels, self.n_classes).permute(0, 3, 1, 2).float()
        tgt = F.one_hot(target_labels, self.n_classes).permute(0, 3, 1, 2).float()
        inter = (pred * tgt).sum((2, 3)).cpu().double()  # [B, C]
        p_sum, t_sum = pred.sum((2, 3)).cpu().double(), tgt.sum((2, 3)).cpu().double()
        self.intersection += inter.sum(0)
        self.pred_sum += p_sum.sum(0)
        self.target_sum += t_sum.sum(0)
        # per-image DSC, only counting classes present in either prediction or ground truth
        denom = p_sum + t_sum
        self.per_image.append(torch.where(denom > 0, 2 * inter / denom.clamp_min(1e-12), torch.nan))

    def per_class(self) -> torch.Tensor:
        """Dataset-level DSC for each class ``[C]``."""
        return (2 * self.intersection / (self.pred_sum + self.target_sum).clamp_min(1e-12)).float()

    def per_image_mean(self) -> torch.Tensor:
        """Mean over images of the per-image DSC, for each class ``[C]`` (ignoring absent classes)."""
        return torch.nanmean(torch.cat(self.per_image), dim=0).float()

    def pixel_accuracy(self) -> float:
        return float(self.intersection.sum() / self.target_sum.sum())
