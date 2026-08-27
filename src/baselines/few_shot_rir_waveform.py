"""Material-blind Few-ShotRIR adaptation with a direct waveform decoder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .protocol import TARGET_SAMPLES


class FLACCoordinateEmbedding(nn.Module):
    """FLAC-compatible Fourier features followed by a trainable projection."""

    def __init__(
        self,
        output_dim: int | None = None,
        *,
        num_frequencies: int = 20,
        max_frequency: float = 10.0,
        max_value: float = 5.0,
    ) -> None:
        super().__init__()
        if num_frequencies <= 0 or max_value <= 0:
            raise ValueError("coordinate frequency count and max_value must be positive")
        frequencies = 2.0 ** torch.linspace(0.0, float(max_frequency), num_frequencies)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.max_value = float(max_value)
        self.feature_dim = 3 * (1 + 2 * num_frequencies)
        self.projection = (
            nn.Linear(self.feature_dim, int(output_dim))
            if output_dim is not None
            else nn.Identity()
        )

    def features(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim < 2 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must end in an xyz axis")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        normalized = coordinates / self.max_value
        parts = [normalized]
        parts.extend(torch.sin(normalized * frequency) for frequency in self.frequencies)
        parts.extend(torch.cos(normalized * frequency) for frequency in self.frequencies)
        return torch.cat(parts, dim=-1)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(coordinates))


class _AudioContextEncoder(nn.Module):
    def __init__(self, output_dim: int, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(channels * 2, output_dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network(waveforms).squeeze(-1))


class _GeometryEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(channels * 2, output_dim)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network(geometry).flatten(1))


class _DirectWaveformDecoder(nn.Module):
    UPSAMPLING_FACTOR = 32

    def __init__(self, embedding_dim: int, channels: int, output_samples: int) -> None:
        super().__init__()
        if output_samples <= 0 or output_samples % self.UPSAMPLING_FACTOR:
            raise ValueError("output_samples must be positive and divisible by 32")
        self.output_samples = int(output_samples)
        self.coarse_samples = self.output_samples // self.UPSAMPLING_FACTOR
        self.input = nn.Linear(embedding_dim, channels * self.coarse_samples)
        blocks: list[nn.Module] = []
        for _ in range(5):
            blocks.extend(
                [
                    nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
        self.upsample = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.Conv1d(channels, 1, kernel_size=7, padding=3), nn.Tanh())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        coarse = self.input(embedding).reshape(
            embedding.shape[0], -1, self.coarse_samples
        )
        waveform = self.output(self.upsample(coarse))
        if waveform.shape[-1] != self.output_samples:
            raise RuntimeError("waveform decoder produced an unexpected length")
        return waveform


class FewShotRIRWaveform(nn.Module):
    """Context-memory RIR predictor using no RGB or pretrained model weights."""

    def __init__(
        self,
        *,
        geometry_channels: int = 3,
        embedding_dim: int = 128,
        num_heads: int = 4,
        context_layers: int = 2,
        decoder_layers: int = 2,
        audio_channels: int = 32,
        geometry_channels_hidden: int = 32,
        waveform_channels: int = 32,
        output_samples: int = TARGET_SAMPLES,
        coordinate_num_frequencies: int = 20,
        coordinate_max_frequency: float = 10.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.geometry_channels = int(geometry_channels)
        self.audio_encoder = _AudioContextEncoder(embedding_dim, audio_channels)
        self.geometry_encoder = _GeometryEncoder(
            self.geometry_channels, embedding_dim, geometry_channels_hidden
        )
        coordinate_args = dict(
            num_frequencies=coordinate_num_frequencies,
            max_frequency=coordinate_max_frequency,
            max_value=5.0,
        )
        self.context_coordinate_encoder = FLACCoordinateEmbedding(
            embedding_dim, **coordinate_args
        )
        self.query_coordinate_encoder = FLACCoordinateEmbedding(
            embedding_dim, **coordinate_args
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU()
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU()
        )
        self.geometry_modality = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.audio_modality = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=context_layers, norm=nn.LayerNorm(embedding_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(embedding_dim)
        )
        self.waveform_decoder = _DirectWaveformDecoder(
            embedding_dim, waveform_channels, output_samples
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.geometry_modality, std=0.02)
        nn.init.normal_(self.audio_modality, std=0.02)

    def _validate_inputs(
        self,
        geometry: torch.Tensor,
        context_audio: torch.Tensor,
        context_coordinates: torch.Tensor,
        query_source: torch.Tensor,
        query_receiver: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if geometry.ndim != 4 or geometry.shape[1] != self.geometry_channels:
            raise ValueError("geometry must have shape [B, geometry_channels, H, W]")
        if context_audio.ndim != 4 or context_audio.shape[2] != 1:
            raise ValueError("context audio must have shape [B, K, 1, T]")
        if context_coordinates.ndim != 3 or context_coordinates.shape[-1] != 3:
            raise ValueError("context coordinates must have shape [B, K, 3]")
        batch, count = context_audio.shape[:2]
        if geometry.shape[0] != batch or context_coordinates.shape[:2] != (batch, count):
            raise ValueError("geometry, audio, and coordinate batch/context axes must agree")
        if query_source.shape != (batch, 3) or query_receiver.shape != (batch, 3):
            raise ValueError("query source and receiver must have shape [B, 3]")
        tensors = (geometry, context_audio, context_coordinates, query_source, query_receiver)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("model inputs must be finite")
        if context_mask is None:
            context_mask = torch.ones(batch, count, dtype=torch.bool, device=geometry.device)
        if context_mask.shape != (batch, count) or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be boolean with shape [B, K]")
        if not context_mask.any(dim=1).all():
            raise ValueError("every item must contain at least one valid context")
        return context_mask

    def forward(
        self,
        geometry: torch.Tensor,
        context_audio: torch.Tensor,
        context_coordinates: torch.Tensor,
        query_source: torch.Tensor,
        query_receiver: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_mask = self._validate_inputs(
            geometry,
            context_audio,
            context_coordinates,
            query_source,
            query_receiver,
            context_mask,
        )
        batch, count = context_audio.shape[:2]
        audio = self.audio_encoder(context_audio.reshape(batch * count, 1, -1)).reshape(
            batch, count, -1
        )
        coordinates = self.context_coordinate_encoder(context_coordinates)
        context_tokens = self.context_fusion(torch.cat((audio, coordinates), dim=-1))
        context_tokens = context_tokens + self.audio_modality
        geometry_token = self.geometry_encoder(geometry).unsqueeze(1) + self.geometry_modality
        memory_input = torch.cat((geometry_token, context_tokens), dim=1)
        memory_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=context_mask.device),
                ~context_mask,
            ),
            dim=1,
        )
        memory = self.context_transformer(
            memory_input, src_key_padding_mask=memory_padding
        )
        source = self.query_coordinate_encoder(query_source)
        receiver = self.query_coordinate_encoder(query_receiver)
        query = self.query_fusion(torch.cat((source, receiver), dim=-1)).unsqueeze(1)
        decoded = self.query_decoder(
            query, memory, memory_key_padding_mask=memory_padding
        ).squeeze(1)
        return self.waveform_decoder(decoded).float()


def _masked_waveforms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, C, T]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if padding_mask is None:
        return prediction, target, None
    if padding_mask.ndim == 2:
        padding_mask = padding_mask.unsqueeze(1)
    if padding_mask.shape not in (prediction.shape, (prediction.shape[0], 1, prediction.shape[-1])):
        raise ValueError("padding_mask must have shape [B, T] or [B, 1/C, T]")
    mask = padding_mask.to(device=prediction.device, dtype=prediction.dtype)
    return prediction * mask, target * mask, mask.bool()


def multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    padding_mask: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    prediction, target, _ = _masked_waveforms(prediction, target, padding_mask)
    flattened_prediction = prediction.flatten(0, 1)
    flattened_target = target.flatten(0, 1)
    losses = []
    for fft_size in tuple(int(value) for value in fft_sizes):
        if fft_size <= 1 or fft_size > prediction.shape[-1]:
            raise ValueError("every FFT size must be in [2, waveform length]")
        window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
        kwargs = dict(
            n_fft=fft_size,
            hop_length=max(1, fft_size // 4),
            win_length=fft_size,
            window=window,
            return_complex=True,
        )
        predicted_stft = torch.stft(flattened_prediction, **kwargs)
        target_stft = torch.stft(flattened_target, **kwargs)
        predicted_magnitude = predicted_stft.abs()
        target_magnitude = target_stft.abs()
        convergence = torch.linalg.vector_norm(
            predicted_magnitude - target_magnitude
        ) / torch.linalg.vector_norm(target_magnitude).clamp_min(epsilon)
        log_magnitude = F.l1_loss(
            torch.log1p(predicted_magnitude), torch.log1p(target_magnitude)
        )
        losses.append(convergence + log_magnitude)
    if not losses:
        raise ValueError("at least one FFT size is required")
    return torch.stack(losses).mean()


def energy_decay_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    prediction, target, valid_mask = _masked_waveforms(prediction, target, padding_mask)

    def curve(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = torch.flip(
            torch.cumsum(torch.flip(waveform.square(), dims=(-1,)), dim=-1), dims=(-1,)
        )
        energy = energy / energy[..., :1].clamp_min(epsilon)
        return 10.0 * torch.log10(energy.clamp_min(epsilon)), energy

    predicted_curve, _predicted_energy = curve(prediction)
    target_curve, target_energy = curve(target)
    mask = target_energy.sum(dim=1, keepdim=True) > epsilon
    if valid_mask is not None:
        if valid_mask.shape[1] != 1:
            valid_mask = valid_mask.any(dim=1, keepdim=True)
        mask = mask & valid_mask
    if not mask.any():
        raise ValueError("target contains no valid RIR energy")
    return (predicted_curve - target_curve).abs().masked_select(mask).mean()


class FewShotRIRWaveformLoss(nn.Module):
    def __init__(
        self,
        *,
        waveform_weight: float = 1.0,
        mrstft_weight: float = 1.0,
        edc_weight: float = 1e-2,
        fft_sizes: Sequence[int] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        weights = (waveform_weight, mrstft_weight, edc_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("loss weights must be finite and non-negative")
        self.waveform_weight, self.mrstft_weight, self.edc_weight = map(float, weights)
        self.fft_sizes = tuple(int(value) for value in fft_sizes)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        masked_prediction, masked_target, valid_mask = _masked_waveforms(
            prediction, target, padding_mask
        )
        absolute_error = (masked_prediction - masked_target).abs()
        if valid_mask is None:
            waveform = absolute_error.mean()
        else:
            expanded_mask = valid_mask.expand_as(absolute_error)
            valid_count = expanded_mask.sum()
            if valid_count == 0:
                raise ValueError("padding_mask contains no valid waveform sample")
            waveform = absolute_error.sum() / valid_count
        zero = prediction.new_zeros(())
        mrstft = (
            multi_resolution_stft_loss(
                prediction, target, fft_sizes=self.fft_sizes, padding_mask=padding_mask
            )
            if self.mrstft_weight
            else zero
        )
        edc = (
            energy_decay_loss(prediction, target, padding_mask=padding_mask)
            if self.edc_weight
            else zero
        )
        total = (
            self.waveform_weight * waveform
            + self.mrstft_weight * mrstft
            + self.edc_weight * edc
        )
        return {"loss": total, "waveform": waveform, "mrstft": mrstft, "edc": edc}
