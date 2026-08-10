"""exp_12 arm B -- heads, EMA teacher, Gram teacher.

The global representation fed to the DINO head is `pooler_output`, the patch mean -- NOT a
CLS token. Two reasons: (1) it is exactly what the downstream FLAC conditioner consumes, so
the pretext objective shapes the vector the task actually uses; (2) it is exactly
roll-invariant on this backbone, so the global objective is well defined on a cylinder.
The CLS + registers still exist under `prefix_mode='m0_registers'` and still do work as
attention sinks; they are simply not the readout.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    """DINOv2-style projector: MLP -> l2-normalised bottleneck -> weight-normed prototypes."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers.append(nn.Linear(d, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # Fix the prototype magnitudes (DINO): direction learns, norm stays 1.
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad_(False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, eps=1e-6)
        return self.last_layer(x)


class SSLModel(nn.Module):
    """Backbone + DINO head (global) + iBOT head (patch)."""

    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int = 8192,
        ibot_out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.hidden_size
        self.dino_head = DINOHead(dim, out_dim, hidden_dim, bottleneck_dim)
        self.ibot_head = DINOHead(dim, ibot_out_dim, hidden_dim, bottleneck_dim)

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(pixel_values, bool_masked_pos=bool_masked_pos)
        return {"patch": out.last_hidden_state, "pooled": out.pooler_output}


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, m: float) -> None:
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


def make_teacher(student: SSLModel) -> SSLModel:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


class GramTeacher:
    """Frozen snapshot of the teacher backbone used as the Gram-anchoring target.

    DINOv3's fix for dense-feature drift in long runs: the *relative* geometry of the patch
    field (its Gram matrix) is held to an earlier, healthier snapshot while the features
    themselves keep improving. That dense field is exactly what FLAC pools, so this is the
    load-bearing regulariser here, not a nicety.
    """

    def __init__(self) -> None:
        self.backbone: nn.Module | None = None

    def refresh(self, teacher: SSLModel) -> None:
        self.backbone = copy.deepcopy(teacher.backbone).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @property
    def ready(self) -> bool:
        return self.backbone is not None

    @torch.no_grad()
    def patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        return self.backbone(pixel_values).last_hidden_state
