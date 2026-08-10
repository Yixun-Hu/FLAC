"""exp_12 arm B -- DINO / iBOT / Gram-anchoring / KoLeo objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Teacher-centred cross-entropy over prototype assignments, student vs teacher."""

    def __init__(self, out_dim: int, center_momentum: float = 0.9, student_temp: float = 0.1):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_out: list[torch.Tensor],
        teacher_out: list[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        t = [F.softmax((x - self.center) / teacher_temp, dim=-1).detach() for x in teacher_out]
        s = [F.log_softmax(x / self.student_temp, dim=-1) for x in student_out]
        total, n = student_out[0].new_zeros(()), 0
        for i, ti in enumerate(t):
            for j, sj in enumerate(s):
                if i == j:
                    continue                      # never predict a view from itself
                total = total + torch.sum(-ti * sj, dim=-1).mean()
                n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def update_center(self, teacher_out: list[torch.Tensor]) -> None:
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class IBOTLoss(nn.Module):
    """Masked-patch prediction: the student sees mask tokens where the teacher saw geometry."""

    def __init__(self, out_dim: int, center_momentum: float = 0.9, student_temp: float = 0.1):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_patch: torch.Tensor,      # [M, K] student outputs at masked positions
        teacher_patch: torch.Tensor,      # [M, K] teacher outputs at the same positions
        teacher_temp: float,
    ) -> torch.Tensor:
        if student_patch.numel() == 0:
            return student_patch.new_zeros(())
        t = F.softmax((teacher_patch - self.center) / teacher_temp, dim=-1).detach()
        s = F.log_softmax(student_patch / self.student_temp, dim=-1)
        return torch.sum(-t * s, dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_patch: torch.Tensor) -> None:
        if teacher_patch.numel() == 0:
            return
        self.center.mul_(self.center_momentum).add_(
            teacher_patch.mean(dim=0, keepdim=True), alpha=1 - self.center_momentum
        )


def gram_loss(student_patch: torch.Tensor, target_patch: torch.Tensor) -> torch.Tensor:
    """Mean squared difference of per-image patch Gram matrices (l2-normalised features).

    Zero iff the two fields agree up to a per-image orthogonal transform, so it constrains
    the geometry of the patch field without freezing the features themselves."""
    s = F.normalize(student_patch.float(), dim=-1, eps=1e-6)
    t = F.normalize(target_patch.float(), dim=-1, eps=1e-6)
    gs = s @ s.transpose(1, 2)
    gt = t @ t.transpose(1, 2)
    return (gs - gt).pow(2).mean()


def koleo_loss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differential-entropy regulariser: push each embedding away from its nearest neighbour."""
    if x.shape[0] < 2:
        return x.new_zeros(())
    z = F.normalize(x.float(), dim=-1, eps=1e-6)
    d = torch.cdist(z, z)
    # Mask the diagonal OUT OF PLACE: fill_diagonal_ mutates cdist's saved output and
    # autograd refuses it ("modified by an inplace operation"). Normalised distances are
    # <= 2, so a large finite constant excludes self-pairs without inf gradients.
    d = d + torch.eye(z.shape[0], device=z.device, dtype=d.dtype) * 1e6
    return -torch.log(d.min(dim=1).values + eps).mean()


def cosine_schedule(step: int, total: int, start: float, end: float, warmup: int = 0) -> float:
    """Linear warmup from 0 to `start`, then cosine from `start` to `end`."""
    if warmup and step < warmup:
        return start * step / max(warmup, 1)
    p = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    import math

    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * p))
