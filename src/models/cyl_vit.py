"""Cylindrical azimuth-equivariant ViT geometry encoder.

The geometry input is a source-relative 3D-coordinate panorama ``[B, 3, 256, 512]``
whose azimuth origin is an arbitrary gauge: rolling the columns by ``k`` and rotating
the horizontal channels by ``Rz(2*pi*k/W)`` is a label-preserving symmetry of the mono
RIR (see ``src.data.yaw_rotation``). :class:`CylindricalViT` is
equivariant to patch-level azimuth rolls by construction, so its mean-pooled descriptor
is invariant to that symmetry group (``C16`` at 512/32 = 16 azimuth patches).

Three pieces implement the equivariance:

(a) Per-pixel gauge alignment. For the pixel at azimuth ``theta_c = (c+0.5)*2*pi/W - pi``
    the horizontal vector ``(x, y)`` is rotated by ``Rz(-theta_c)`` into that column's own
    azimuth frame: ``(x*cos(theta_c) + y*sin(theta_c), -x*sin(theta_c) + y*cos(theta_c), z)``.
    Under a ``k``-roll a pixel moves to ``theta_c + alpha`` and its vector becomes
    ``Rz(alpha) v``, so the aligned value ``Rz(-(theta_c+alpha)) Rz(alpha) v = Rz(-theta_c) v``
    is unchanged: the channel content becomes roll-invariant and only the spatial position
    carries the group action. ``cos(theta_c)`` / ``sin(theta_c)`` are fixed per-column buffers.

(b) Roll-equivariant body. The selected patch embedding maps the panorama to a ``16x16``
    token grid (16 elevation rows, 16 azimuth columns): either the original non-overlapping
    ``16x32`` linear projection or a shared hierarchical CNN applied independently to each
    non-overlapping ``[3, 16, 32]`` patch. No patch-embedding operation mixes neighboring
    tokens.
    Absolute position encoding is kept for elevation ONLY (the azimuth component is removed).
    A circular relative position bias is added to the attention logits: a per-head table indexed by
    ``(delta_el, delta_az_circ)`` with ``delta_az_circ = ((ai - aj + 8) mod 16) - 8``
    (signed circular azimuth distance, Swin-style). Attention with only relative-circular
    azimuth information is exactly roll-equivariant at patch granularity.

(c) Readout. Mean-pool over all tokens is performed by the caller
    (:class:`src.models.conditioners.GeometryConditioner` with ``token_pool="mean"``);
    the mean of an equivariant token grid is invariant, so the ``[B, 512]`` descriptor
    is unchanged under patch-level rolls.

Sub-patch rolls (``k`` not a multiple of 32) are not exactly invariant; the encoder-only
probe in ``worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/encoder_yaw_invariance.py``
shows the residual deviation is much smaller than the vanilla ViT's.
"""
import math

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_1d(n: int, dim: int, temperature: int = 10000,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Sinusoidal 1D position embedding for elevation rows.

    Parameters
    ----------
    n : int
        Number of positions (elevation token rows).
    dim : int
        Embedding width (must be even).
    temperature : int, optional
        Frequency temperature, defaults to 10000.
    dtype : torch.dtype, optional
        Output dtype, defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Position embedding of shape ``[n, dim]``.
    """
    assert (dim % 2) == 0, "feature dimension must be even for sincos emb"
    pos = torch.arange(n)
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)
    pos = pos[:, None] * omega[None, :]
    pe = torch.cat((pos.sin(), pos.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    """Pre-norm MLP block (matches AGREE's SimpleViT feed-forward)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Parameters
        ----------
        dim : int
            Token width.
        hidden_dim : int
            Hidden width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block to ``x`` of shape ``[B, N, dim]``."""
        return self.net(x)


class RelPosAttention(nn.Module):
    """
    Multi-head self-attention with a circular relative position bias.

    The bias table is per-head and per-layer (Swin-style). It is indexed by a fixed
    ``[N, N]`` relative-position index (``rel_index``) that maps each token pair to a
    ``(delta_el, delta_az_circ)`` bucket; the azimuth component is a signed circular
    distance, making the bias invariant to a common azimuth shift of all tokens.

    Attributes:
        rel_bias (nn.Parameter): Per-head bias table of shape ``[heads, table_size]``.
    """

    def __init__(self, dim: int, table_size: int, heads: int = 8, dim_head: int = 64) -> None:
        """
        Parameters
        ----------
        dim : int
            Token width.
        table_size : int
            Number of relative-position buckets ``(2*H_tok-1) * W_tok``.
        heads : int, optional
            Number of attention heads, defaults to 8.
        dim_head : int, optional
            Head width, defaults to 64.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(heads, table_size))

    def forward(self, x: torch.Tensor, rel_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape ``[B, N, dim]``.
        rel_index : torch.Tensor
            Long tensor of shape ``[N, N]`` indexing ``rel_bias`` along its last dim.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``[B, N, dim]``.
        """
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        bias = self.rel_bias[:, rel_index]                 # [heads, N, N]
        dots = dots + bias[None]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Pre-norm transformer with circular-relative-bias attention."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 table_size: int) -> None:
        """
        Parameters
        ----------
        dim : int
            Token width.
        depth : int
            Number of layers.
        heads : int
            Number of attention heads.
        dim_head : int
            Head width.
        mlp_dim : int
            Feed-forward hidden width.
        table_size : int
            Relative-position table size passed to each attention layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RelPosAttention(dim, table_size, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ]))

    def forward(self, x: torch.Tensor, rel_index: torch.Tensor) -> torch.Tensor:
        """Run the transformer over ``x`` of shape ``[B, N, dim]``."""
        for attn, ff in self.layers:
            x = attn(x, rel_index) + x
            x = ff(x) + x
        return self.norm(x)


class LayerNorm2d(nn.LayerNorm):
    """Apply channel-wise LayerNorm independently at every spatial position."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``[B, C, H, W]`` over ``C`` without mixing spatial positions."""
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class CylindricalConvStage(nn.Module):
    """One zero-padded, strided stage of the shared patch-local CNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        activate: bool,
    ) -> None:
        super().__init__()
        kernel_height, kernel_width = kernel_size
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("CylindricalConvStage requires odd convolution kernels")
        self.height_padding = kernel_height // 2
        self.width_padding = kernel_width // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.height_padding, self.width_padding),
            bias=False,
        )
        self.norm = LayerNorm2d(out_channels)
        self.activation = nn.GELU() if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample one batch of independent ``[C, 16, 32]`` patches."""
        return self.activation(self.norm(self.conv(x)))


class CylindricalCNNPatchEmbedding(nn.Module):
    """Shared patch-local, parameter-matched CNN embedding for the C16 experiment."""

    patch_height = 16
    patch_width = 32

    def __init__(self, in_channels: int = 3, dim: int = 512) -> None:
        super().__init__()
        if in_channels != 3 or dim != 512:
            raise ValueError(
                "The C16 CNN patch embedding is fixed to in_channels=3 and dim=512"
            )
        self.stages = nn.ModuleList([
            CylindricalConvStage(in_channels, 34, (3, 3), (2, 2), activate=True),
            CylindricalConvStage(34, 44, (3, 3), (2, 2), activate=True),
            CylindricalConvStage(44, 96, (3, 3), (2, 2), activate=True),
            CylindricalConvStage(96, dim, (3, 5), (2, 4), activate=False),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed every non-overlapping ``[3, 16, 32]`` patch independently."""
        if x.ndim != 4:
            raise ValueError(f"CNN patch embedding expects [B, C, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"CNN patch embedding expects 3 input channels, got {channels}")
        if height % self.patch_height or width % self.patch_width:
            raise ValueError(
                "CNN patch embedding requires H and W divisible by "
                f"({self.patch_height}, {self.patch_width}), got ({height}, {width})"
            )

        h_tok = height // self.patch_height
        w_tok = width // self.patch_width
        x = rearrange(
            x,
            "b c (h ph) (w pw) -> (b h w) c ph pw",
            ph=self.patch_height,
            pw=self.patch_width,
        )
        for stage in self.stages:
            x = stage(x)
        if x.shape[-2:] != (1, 1):
            raise RuntimeError(
                "Patch-local CNN must reduce each [3, 16, 32] patch to [dim, 1, 1], "
                f"got spatial shape {tuple(x.shape[-2:])}"
            )
        return rearrange(
            x,
            "(b h w) c 1 1 -> b (h w) c",
            b=batch,
            h=h_tok,
            w=w_tok,
        )


class CylindricalViT(nn.Module):
    """
    Azimuth-equivariant ViT over a source-relative geometry panorama.

    Exposes the same constructor surface as the repo's ``SimpleViT`` where it matters
    (``image_size``, ``patch_size``, ``dim``, ``depth``, ``heads``) and returns the token
    sequence ``[B, N, dim]`` so :class:`src.models.conditioners.GeometryConditioner` can
    mean-pool it, matching the plain-ViT interface. The gauge alignment happens inside
    :meth:`forward`, so datasets and eval keep feeding the raw ``[B, 3, H, W]`` geometry.

    Attributes:
        dim (int): Token feature width.
        num_tokens (int): Number of patch tokens ``H_tok * W_tok``.
        to_patch_embedding (nn.Module): Linear or hierarchical CNN patch embed.
        transformer (Transformer): Circular-relative-bias transformer body.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: tuple[int, int] = (256, 512),
        patch_size: tuple[int, int] = (16, 32),
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int | None = None,
        patch_embed_type: str = "linear",
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int, optional
            Geometry channels, defaults to 3 (must be 3: channels are ``(x, y, z)``).
        image_size : tuple[int, int], optional
            Input spatial size ``(H, W)``, defaults to ``(256, 512)``.
        patch_size : tuple[int, int], optional
            Patch size ``(patch_h, patch_w)``, defaults to ``(16, 32)``.
        dim : int, optional
            Token width, defaults to 512.
        depth : int, optional
            Number of transformer layers, defaults to 12.
        heads : int, optional
            Number of attention heads, defaults to 8.
        dim_head : int, optional
            Head width, defaults to 64.
        mlp_dim : int, optional
            Feed-forward hidden width, defaults to ``dim``.
        patch_embed_type : {"linear", "cnn"}, optional
            Patch embedding variant. ``"linear"`` preserves the original flattened-patch
            projection; ``"cnn"`` selects a four-stage CNN shared by independent patches.
        """
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"CylindricalViT expects 3 (x, y, z) channels, got {in_channels}")
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."
        mlp_dim = dim if mlp_dim is None else mlp_dim

        self.dim = dim
        self.h_tok = image_height // patch_height
        self.w_tok = image_width // patch_width
        self.num_tokens = self.h_tok * self.w_tok
        self.patch_embed_type = patch_embed_type

        # (a) Per-pixel gauge alignment buffers: theta_c = (c + 0.5) * 2*pi/W - pi.
        cols = torch.arange(image_width, dtype=torch.float32)
        theta = (cols + 0.5) * (2.0 * math.pi / image_width) - math.pi
        self.register_buffer("cos_theta", torch.cos(theta), persistent=False)
        self.register_buffer("sin_theta", torch.sin(theta), persistent=False)

        if patch_embed_type == "linear":
            patch_dim = in_channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        elif patch_embed_type == "cnn":
            if (patch_height, patch_width) != (16, 32):
                raise ValueError(
                    "The C16 CNN patch embedding requires patch_size=(16, 32)"
                )
            self.to_patch_embedding = CylindricalCNNPatchEmbedding(in_channels, dim)
        else:
            raise ValueError(
                f"Unknown patch_embed_type={patch_embed_type!r}; expected 'linear' or 'cnn'"
            )

        # (b) Elevation-only absolute position encoding, broadcast across azimuth.
        el_pe = posemb_sincos_1d(self.h_tok, dim)                     # [H_tok, dim]
        pos_embedding = el_pe[:, None, :].expand(self.h_tok, self.w_tok, dim).reshape(self.num_tokens, dim)
        self.register_buffer("pos_embedding", pos_embedding.contiguous(), persistent=False)

        # (b) Circular relative position index into the per-head bias table.
        rel_index, table_size = self._build_rel_index(self.h_tok, self.w_tok)
        self.register_buffer("rel_index", rel_index, persistent=False)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, table_size)

    @staticmethod
    def _build_rel_index(h_tok: int, w_tok: int) -> tuple[torch.Tensor, int]:
        """
        Build the ``[N, N]`` relative-position index and the flat table size.

        Parameters
        ----------
        h_tok : int
            Elevation token rows.
        w_tok : int
            Azimuth token columns.

        Returns
        -------
        tuple[torch.Tensor, int]
            The long index tensor of shape ``[N, N]`` and the bias table size
            ``(2*h_tok - 1) * w_tok``.
        """
        el = torch.arange(h_tok).repeat_interleave(w_tok)            # token elevation, [N] (h-major)
        az = torch.arange(w_tok).repeat(h_tok)                       # token azimuth, [N]
        d_el = el[:, None] - el[None, :] + (h_tok - 1)               # [N, N] in [0, 2*h_tok-2]
        d_az = (az[:, None] - az[None, :] + w_tok // 2) % w_tok      # signed circular, in [0, w_tok-1]
        rel_index = (d_el * w_tok + d_az).long()
        table_size = (2 * h_tok - 1) * w_tok
        return rel_index, table_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode a source-relative geometry panorama into azimuth-equivariant tokens.

        Parameters
        ----------
        img : torch.Tensor
            Geometry image of shape ``[B, 3, H, W]`` (channels ``x, y, z``).

        Returns
        -------
        torch.Tensor
            Patch tokens of shape ``[B, num_tokens, dim]``.
        """
        cos = self.cos_theta.to(img.dtype)
        sin = self.sin_theta.to(img.dtype)
        x, y, z = img[:, 0], img[:, 1], img[:, 2]                    # [B, H, W] each
        x_a = x * cos + y * sin
        y_a = -x * sin + y * cos
        aligned = torch.stack([x_a, y_a, z], dim=1)                  # [B, 3, H, W]

        tokens = self.to_patch_embedding(aligned)                   # [B, N, dim]
        tokens = tokens + self.pos_embedding.to(tokens.dtype)
        return self.transformer(tokens, self.rel_index)
