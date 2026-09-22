"""
losses.py
=========
Section 4 (loss half): Class-Weighted Focal Loss with Label Smoothing.

Focal Loss down-weights easy, well-classified examples so training spends
more gradient signal on hard/ambiguous fundus images (e.g. borderline
early-stage DR vs. Normal). Label smoothing softens the one-hot targets so
the model doesn't become overconfident on inherently ambiguous edges
(vessel occlusion, poor illumination, borderline cup-to-disc ratio).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLabelSmoothing(nn.Module):
    r"""
    Multi-class focal loss combined with label smoothing.

    For target class :math:`y` and predicted probability :math:`p_y`:

    .. math::
        \text{FL}(p_y) = -\alpha_y (1 - p_y)^{\gamma} \log(p_y)

    Label smoothing replaces the one-hot target distribution
    :math:`q(y)` with:

    .. math::
        q'(k) = (1 - \epsilon) \cdot q(k) + \epsilon / K

    where :math:`K` is the number of classes, before computing the
    per-class cross-entropy term that focal weighting is applied to.

    Args:
        class_weights: Tensor of shape (K,) -- per-class alpha weights,
            typically inverse class frequency (see
            `dataset.compute_class_weights`).
        gamma: focusing parameter; higher values down-weight easy examples
            more aggressively.
        label_smoothing: epsilon in [0, 1).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Registered as a buffer so it moves with .to(device) automatically.
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Smoothed one-hot target distribution.
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Per-sample, per-class cross-entropy contribution, then apply the
        # focal modulating factor (1 - p_t)^gamma using the *true* class
        # probability p_t (standard focal-loss formulation).
        ce_per_class = -true_dist * log_probs  # (B, K)

        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        focal_weight = (1.0 - p_t).clamp(min=1e-6) ** self.gamma  # (B,)

        loss_per_sample = ce_per_class.sum(dim=1) * focal_weight  # (B,)

        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device)[targets]  # (B,)
            loss_per_sample = loss_per_sample * alpha_t

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


def build_criterion(cfg, class_weights: torch.Tensor) -> FocalLossWithLabelSmoothing:
    """Factory reading loss hyperparameters from Config."""
    return FocalLossWithLabelSmoothing(
        class_weights=class_weights,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
        reduction="mean",
    )
