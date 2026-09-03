"""Official-architecture FewshotRiR adapted to monaural AcousticRooms data.

The module retains the released Few-ShotRIR computation graph: independent
visual and acoustic context entries, a shared sinusoidal pose encoder, modality
tags, a full encoder/decoder Transformer, and a 2-D transposed-convolution
magnitude decoder.  Dataset-specific changes are limited to depth-only views,
monaural STFTs, 3-D source/receiver poses, and the AcousticRooms sample grid.

Upstream reference: SAGNIKMJR/few-shot-rir at commit
16c0edf1cd677d61682dca52f16a3bafc60c2b3b (MIT), especially policy.py and the
visual/audio/positional/memory/fusion model modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


FEWSHOT_RIR_NAME = "FewshotRiR"


def _resnet18(input_channels: int) -> nn.Module:
    """Build the same unpretrained ResNet-18 family used by the release."""

    try:
        from torchvision.models import resnet18
    except ImportError as error:  # pragma: no cover - exercised by environment setup
        raise ImportError(
            "FewshotRiR requires torchvision, which is declared in pyproject.toml"
        ) from error
    try:
        network = resnet18(weights=None)
    except TypeError:  # torchvision before the weights API
        network = resnet18(pretrained=False)
    network.conv1 = nn.Conv2d(
        int(input_channels),
        network.conv1.out_channels,
        kernel_size=network.conv1.kernel_size,
        stride=network.conv1.stride,
        padding=network.conv1.padding,
        bias=False,
    )
    nn.init.kaiming_normal_(network.conv1.weight, mode="fan_out", nonlinearity="relu")
    network.fc = nn.Identity()
    return network


class _TinyImageEncoder(nn.Module):
    """Small contract-compatible encoder used only by unit-test configurations."""

    def __init__(self, input_channels: int, output_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_features),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DepthContextEncoder(nn.Module):
    """Depth-only replacement for the official RGB-D ResNet-18."""

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        output_features: int = 512,
    ) -> None:
        super().__init__()
        if architecture == "resnet18":
            if output_features != 512:
                raise ValueError("official ResNet-18 context encoders output 512 features")
            self.network = _resnet18(1)
        elif architecture == "tiny_test":
            self.network = _TinyImageEncoder(1, output_features)
        else:
            raise ValueError(f"unsupported context encoder architecture: {architecture}")
        self.output_features = int(output_features)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [N, 1, H, W]")
        return self.network(depth.float())


class AcousticContextEncoder(nn.Module):
    """Monaural magnitude-spectrogram ResNet-18 from the official AudioEnc."""

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        output_features: int = 512,
        log_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if architecture == "resnet18":
            if output_features != 512:
                raise ValueError("official ResNet-18 context encoders output 512 features")
            self.network = _resnet18(1)
        elif architecture == "tiny_test":
            self.network = _TinyImageEncoder(1, output_features)
        else:
            raise ValueError(f"unsupported context encoder architecture: {architecture}")
        if not math.isfinite(log_epsilon) or log_epsilon <= 0:
            raise ValueError("log_epsilon must be finite and positive")
        self.output_features = int(output_features)
        self.log_epsilon = float(log_epsilon)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if spectrogram.ndim != 4 or spectrogram.shape[-1] != 1:
            raise ValueError("context spectrogram must have shape [N, F, W, 1]")
        if not torch.isfinite(spectrogram).all() or torch.any(spectrogram < 0):
            raise ValueError("context magnitude spectrograms must be finite and nonnegative")
        values = torch.log(spectrogram + self.log_epsilon).permute(0, 3, 1, 2)
        return self.network(values)


class SharedPoseEncoder(nn.Module):
    """Official alternating sinusoidal encoder over AR receiver/source xyz poses."""

    MIN_FREQUENCY = 1e-4

    def __init__(self, pose_dimensions: int = 6, num_frequencies: int = 8) -> None:
        super().__init__()
        if pose_dimensions <= 0 or num_frequencies <= 0:
            raise ValueError("pose dimensions and frequency count must be positive")
        frequencies = self.MIN_FREQUENCY ** (
            2
            * (torch.arange(num_frequencies, dtype=torch.float32) // 2)
            / float(num_frequencies)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.pose_dimensions = int(pose_dimensions)
        self.output_features = self.pose_dimensions * int(num_frequencies)

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim != 2 or poses.shape[-1] != self.pose_dimensions:
            raise ValueError(
                f"poses must have shape [N, {self.pose_dimensions}]"
            )
        encoded = poses.unsqueeze(-1) * self.frequencies
        output = encoded.clone()
        output[..., ::2] = torch.cos(encoded[..., ::2])
        output[..., 1::2] = torch.sin(encoded[..., 1::2])
        return output.flatten(1)


class _FusionNet(nn.Module):
    """Concatenate modality, pose and type features then project to d_model."""

    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.network = (
            nn.Identity()
            if input_features == output_features
            else nn.Linear(input_features, output_features, bias=False)
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if not features or any(value.ndim != 2 for value in features):
            raise ValueError("fusion inputs must be nonempty matrices")
        return self.network(torch.cat(features, dim=-1))


def _upconv(input_channels: int, output_channels: int, *, output_padding=(0, 0)):
    return nn.Sequential(
        nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=tuple(int(value) for value in output_padding),
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class MagnitudeSpectrogramDecoder(nn.Module):
    """Official AudioDec topology with a monaural AcousticRooms output plane."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        input_channels: int = 64,
        initial_size: tuple[int, int] = (4, 4),
        channels: Sequence[int] = (512, 256, 128, 64, 32, 16),
        final_output_padding: tuple[int, int] = (0, 1),
        output_frequency_bins: int = 256,
        output_frames: int = 257,
    ) -> None:
        super().__init__()
        initial_height, initial_width = map(int, initial_size)
        if embedding_dim != input_channels * initial_height * initial_width:
            raise ValueError(
                "embedding_dim must equal decoder input_channels * initial height * initial width"
            )
        decoder_channels = tuple(int(value) for value in channels)
        if not decoder_channels or any(value <= 0 for value in decoder_channels):
            raise ValueError("decoder channels must be a nonempty positive sequence")
        blocks = []
        previous = int(input_channels)
        for index, output in enumerate(decoder_channels):
            output_padding = final_output_padding if index == len(decoder_channels) - 1 else (0, 0)
            blocks.append(_upconv(previous, output, output_padding=output_padding))
            previous = output
        self.upsample = nn.Sequential(*blocks)
        self.output = nn.Conv2d(previous, 1, kernel_size=3, padding=1, bias=False)
        self.input_channels = int(input_channels)
        self.initial_height = initial_height
        self.initial_width = initial_width
        self.output_frequency_bins = int(output_frequency_bins)
        self.output_frames = int(output_frames)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # Preserve the released call literally.  Its second positional
                # argument is ``a`` (not gain), so changing this to
                # nonlinearity="relu" changes the initialization std by sqrt(3).
                nn.init.kaiming_normal_(
                    module.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(module, nn.BatchNorm2d) and module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("decoder embeddings must have shape [N, D]")
        values = embeddings.reshape(
            embeddings.shape[0],
            self.input_channels,
            self.initial_height,
            self.initial_width,
        )
        output = self.output(self.upsample(values))
        expected = (self.output_frequency_bins, self.output_frames)
        if output.shape[-2:] != expected:
            raise RuntimeError(
                f"magnitude decoder produced {tuple(output.shape[-2:])}, expected {expected}"
            )
        return output.permute(0, 2, 3, 1)


class FewshotRiR(nn.Module):
    """Depth-only, monaural AcousticRooms adaptation of official Few-ShotRIR."""

    def __init__(
        self,
        *,
        sample_rate: int = 22050,
        sample_size: int = 10240,
        n_fft: int = 511,
        hop_length: int = 40,
        win_length: int = 248,
        embedding_dim: int = 1024,
        hidden_dim: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        pose_dimensions: int = 6,
        pose_num_frequencies: int = 8,
        modality_tag_features: int = 8,
        context_encoder_architecture: str = "resnet18",
        context_encoder_features: int = 512,
        decoder_input_channels: int = 64,
        decoder_initial_size: Sequence[int] = (4, 4),
        decoder_channels: Sequence[int] = (512, 256, 128, 64, 32, 16),
        decoder_final_output_padding: Sequence[int] = (0, 1),
        log_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        integer_values = (sample_rate, sample_size, n_fft, hop_length, win_length)
        if any(int(value) <= 0 for value in integer_values):
            raise ValueError("sample and STFT dimensions must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if n_fft % 2 != 1:
            raise ValueError("FewshotRiR retains the official odd-sized STFT")
        self.sample_rate = int(sample_rate)
        self.sample_size = int(sample_size)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.log_epsilon = float(log_epsilon)
        self.output_frequency_bins = self.n_fft // 2 + 1
        self.output_frames = self.sample_size // self.hop_length + 1

        self.depth_encoder = DepthContextEncoder(
            architecture=context_encoder_architecture,
            output_features=context_encoder_features,
        )
        self.acoustic_encoder = AcousticContextEncoder(
            architecture=context_encoder_architecture,
            output_features=context_encoder_features,
            log_epsilon=log_epsilon,
        )
        self.pose_encoder = SharedPoseEncoder(
            pose_dimensions=pose_dimensions,
            num_frequencies=pose_num_frequencies,
        )
        self.modality_tags = nn.Embedding(2, modality_tag_features)
        fusion_features = (
            context_encoder_features
            + self.pose_encoder.output_features
            + modality_tag_features
        )
        self.depth_fusion = _FusionNet(fusion_features, embedding_dim)
        self.acoustic_fusion = _FusionNet(fusion_features, embedding_dim)
        self.query_fusion = _FusionNet(self.pose_encoder.output_features, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        self.audio_decoder = MagnitudeSpectrogramDecoder(
            embedding_dim,
            input_channels=decoder_input_channels,
            initial_size=tuple(int(value) for value in decoder_initial_size),
            channels=decoder_channels,
            final_output_padding=tuple(int(value) for value in decoder_final_output_padding),
            output_frequency_bins=self.output_frequency_bins,
            output_frames=self.output_frames,
        )
        self.register_buffer(
            "stft_window", torch.hann_window(self.win_length), persistent=False
        )

    def waveform_to_magnitude(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Convert tensors ending in T to the librosa-compatible F,W,1 grid."""

        if waveforms.shape[-1] != self.sample_size:
            raise ValueError(
                f"waveforms must contain exactly {self.sample_size} samples"
            )
        original_shape = waveforms.shape[:-1]
        flattened = waveforms.reshape(-1, self.sample_size).float()
        # The release uses librosa with an odd n_fft.  Its centered padding has
        # one extra sample on the right, unlike torch.stft(center=True).
        left = self.n_fft // 2
        right = self.n_fft - left
        padded = F.pad(flattened.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(device=flattened.device, dtype=flattened.dtype),
            center=False,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        return magnitude.reshape(
            *original_shape, self.output_frequency_bins, self.output_frames, 1
        )

    def output_to_magnitude(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Map the official log-domain decoder output back to linear magnitude."""

        return (torch.exp(raw_output) - self.log_epsilon).clamp_min(0.0)

    def magnitude_to_waveform(
        self,
        magnitude: torch.Tensor,
        *,
        iterations: int = 32,
        momentum: float = 0.99,
    ) -> torch.Tensor:
        """Deterministic Griffin-Lim with no target-phase access."""

        if magnitude.shape[-3:] != (
            self.output_frequency_bins,
            self.output_frames,
            1,
        ):
            raise ValueError("magnitude has an incompatible FewshotRiR STFT shape")
        if iterations <= 0 or not 0.0 <= momentum < 1.0:
            raise ValueError("Griffin-Lim iterations/momentum are invalid")
        leading_shape = magnitude.shape[:-3]
        flattened = magnitude[..., 0].reshape(
            -1, self.output_frequency_bins, self.output_frames
        )
        window = self.stft_window.to(device=flattened.device, dtype=flattened.dtype)
        window_left = (self.n_fft - self.win_length) // 2
        window_right = self.n_fft - self.win_length - window_left
        full_window = F.pad(window, (window_left, window_right))

        def inverse(spectrum: torch.Tensor) -> torch.Tensor:
            frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-2)
            frames = frames * full_window[None, :, None]
            output_length = self.n_fft + self.hop_length * (self.output_frames - 1)
            waveform_padded = F.fold(
                frames,
                output_size=(1, output_length),
                kernel_size=(1, self.n_fft),
                stride=(1, self.hop_length),
            )[:, 0, 0]
            normalizer = F.fold(
                full_window.square()[None, :, None].expand(frames.shape[0], -1, self.output_frames),
                output_size=(1, output_length),
                kernel_size=(1, self.n_fft),
                stride=(1, self.hop_length),
            )[:, 0, 0]
            waveform_padded = waveform_padded / normalizer.clamp_min(1e-11)
            left = self.n_fft // 2
            return waveform_padded[:, left : left + self.sample_size]

        def analysis(waveform: torch.Tensor) -> torch.Tensor:
            left = self.n_fft // 2
            right = self.n_fft - left
            padded = F.pad(waveform.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
            return torch.stft(
                padded,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=False,
                return_complex=True,
            )

        angles = torch.ones_like(flattened, dtype=torch.complex64)
        previous: torch.Tensor | float = 0.0
        fast_momentum = float(momentum) / (1.0 + float(momentum))
        for _ in range(int(iterations)):
            rebuilt = analysis(inverse(flattened * angles))
            update = rebuilt - previous * fast_momentum if fast_momentum else rebuilt
            angles = update / update.abs().clamp_min(1e-16)
            previous = rebuilt
        waveform = inverse(flattened * angles)
        return waveform.reshape(*leading_shape, 1, self.sample_size).float()

    @staticmethod
    def _validate_mask(mask: torch.Tensor | None, batch: int, count: int, name: str):
        if mask is None:
            return torch.ones(batch, count, dtype=torch.bool)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean with shape [B, {count}]")
        return mask

    def forward(
        self,
        *,
        context_depth: torch.Tensor,
        context_spectrograms: torch.Tensor,
        context_poses: torch.Tensor,
        query_poses: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_depth.ndim != 5 or context_depth.shape[2] != 1:
            raise ValueError("context_depth must have shape [B, K, 1, H, W]")
        if context_spectrograms.ndim != 5 or context_spectrograms.shape[-3:] != (
            self.output_frequency_bins,
            self.output_frames,
            1,
        ):
            raise ValueError("context_spectrograms has an incompatible STFT shape")
        batch, contexts = context_depth.shape[:2]
        queries = query_poses.shape[1] if query_poses.ndim == 3 else -1
        expected_pose = self.pose_encoder.pose_dimensions
        if context_poses.shape != (batch, contexts, expected_pose):
            raise ValueError("context_poses has an incompatible shape")
        if query_poses.shape != (batch, queries, expected_pose) or queries <= 0:
            raise ValueError("query_poses has an incompatible shape")
        if context_spectrograms.shape[:2] != (batch, contexts):
            raise ValueError("context modality axes must agree")
        context_mask = self._validate_mask(
            context_mask, batch, contexts, "context_mask"
        ).to(context_depth.device)
        query_mask = self._validate_mask(query_mask, batch, queries, "query_mask").to(
            context_depth.device
        )
        if not context_mask.any(dim=1).all() or not query_mask.any(dim=1).all():
            raise ValueError("every episode needs a valid context and query")

        flat_depth = context_depth.reshape(
            batch * contexts, *context_depth.shape[2:]
        )
        flat_audio = context_spectrograms.reshape(
            batch * contexts, *context_spectrograms.shape[2:]
        )
        flat_context_pose = context_poses.reshape(batch * contexts, expected_pose)
        pose_features = self.pose_encoder(flat_context_pose)
        tags = self.modality_tags(
            torch.arange(2, device=context_depth.device)
        )
        depth_tag = tags[0].expand(batch * contexts, -1)
        acoustic_tag = tags[1].expand(batch * contexts, -1)
        depth_tokens = self.depth_fusion(
            self.depth_encoder(flat_depth), pose_features, depth_tag
        ).reshape(batch, contexts, -1)
        acoustic_tokens = self.acoustic_fusion(
            self.acoustic_encoder(flat_audio), pose_features, acoustic_tag
        ).reshape(batch, contexts, -1)
        memory_input = torch.cat((depth_tokens, acoustic_tokens), dim=1).permute(1, 0, 2)
        memory_padding = torch.cat((~context_mask, ~context_mask), dim=1)

        flat_query_pose = query_poses.reshape(batch * queries, expected_pose)
        query_tokens = self.query_fusion(
            self.pose_encoder(flat_query_pose)
        ).reshape(batch, queries, -1).permute(1, 0, 2)
        query_attention_mask = torch.ones(
            (queries, queries), device=query_tokens.device, dtype=torch.bool
        )
        query_attention_mask.fill_diagonal_(False)
        decoded = self.transformer(
            memory_input,
            query_tokens,
            tgt_mask=query_attention_mask,
            src_key_padding_mask=memory_padding,
            memory_key_padding_mask=memory_padding,
        ).permute(1, 0, 2)
        raw = self.audio_decoder(decoded.reshape(batch * queries, -1))
        return raw.reshape(
            batch,
            queries,
            self.output_frequency_bins,
            self.output_frames,
            1,
        )


class FewshotRiRLoss(nn.Module):
    """Official magnitude L1 plus frequency-collapsed spectral decay loss."""

    def __init__(
        self,
        *,
        spectral_weight: float = 1.0,
        energy_decay_weight: float = 1e-2,
        log_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        weights = (float(spectral_weight), float(energy_decay_weight))
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        self.spectral_weight, self.energy_decay_weight = weights
        self.log_epsilon = float(log_epsilon)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand_as(values)
        count = expanded.sum()
        if count == 0:
            raise ValueError("query mask contains no valid entries")
        return values.masked_select(expanded).sum() / count

    def forward(
        self,
        raw_prediction: torch.Tensor,
        target_magnitude: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if raw_prediction.shape != target_magnitude.shape or raw_prediction.ndim != 5:
            raise ValueError("prediction and target must share shape [B, Q, F, W, C]")
        if query_mask.shape != raw_prediction.shape[:2] or query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be boolean with shape [B, Q]")
        if not torch.all(query_mask):
            raise ValueError(
                "upstream Few-ShotRIR energy-decay loss requires every query slot to be valid"
            )
        prediction = torch.exp(raw_prediction) - self.log_epsilon
        spectral = self._masked_mean((prediction - target_magnitude).abs(), query_mask)

        def decay_curve(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            full_band = values.sum(dim=-3)
            energy = torch.flip(
                torch.cumsum(torch.flip(full_band.square(), dims=(-2,)), dim=-2),
                dims=(-2,),
            )
            curve = 10.0 * torch.log10(energy + 1e-13)
            curve = curve - curve[..., :1, :]
            return curve[..., 1:, :], energy[..., 1:, :]

        predicted_decay, _ = decay_curve(prediction)
        target_decay, target_energy = decay_curve(target_magnitude)
        valid = (target_energy != 0).to(target_energy.dtype)
        energy_decay = F.l1_loss(
            predicted_decay * valid,
            target_decay * valid,
        )
        total = (
            self.spectral_weight * spectral
            + self.energy_decay_weight * energy_decay
        )
        return {
            "loss": total,
            "spectral_l1": spectral,
            "spectral_energy_decay": energy_decay,
        }
