import os
import pytorch_lightning as pl
import gc
import random
import torch
import torchaudio
import typing as tp

from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..data.are_anchor import (
    apply_anchor_addback,
    build_anchor_bank,
    compute_are_anchors,
)
from ..data.yaw_rotation import (
    invariant_conditioning,
    DEFAULT_FRAME_ANGLES,
    POSE_KEYS,
    draw_yaw_offsets,
    offsets_to_radians,
    rotate_scene_metadata,
)
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, sample_flow_pingpong, truncated_logistic_normal_rescaled, DistributionShift, sample_timesteps_logsnr
from ..models.diffusion import ConditionedDiffusionModelWrapper
from .losses import MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_metric

from time import time

_MASK64 = (1 << 64) - 1
# SplitMix64's golden-ratio increment and its two finalising multipliers
# (Steele/Lea/Flood, "Fast splittable pseudorandom number generators").
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_MIX1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MIX2 = 0x94D049BB133111EB


def _splitmix64(x: int) -> int:
    """One SplitMix64 round: avalanche a 64-bit word into a 64-bit word."""
    z = (x + _SPLITMIX64_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX64_MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX64_MIX2) & _MASK64
    return z ^ (z >> 31)


_MASK32 = (1 << 32) - 1
# Domain of the (step, rank) packing below. 2**20 steps is 26x the pre-registered
# 40,000-step budget and 2**12 ranks is 512x the 8-rank rung; both are asserted
# rather than wrapped, because a wrap would silently alias one cell's yaw stream
# onto another's.
_YAW_AUG_RANK_BITS = 12
_YAW_AUG_MAX_RANK = 1 << _YAW_AUG_RANK_BITS
_YAW_AUG_MAX_STEP = 1 << (32 - _YAW_AUG_RANK_BITS)
# Murmur3 fmix32 constants; both multipliers are odd, hence invertible mod 2**32.
_FMIX32_MUL1 = 0x85EBCA6B
_FMIX32_MUL2 = 0xC2B2AE35


def _fmix32(x: int) -> int:
    """Murmur3's 32-bit finaliser — a *bijection* of ``[0, 2**32)`` onto itself.

    Each step is individually invertible (``x ^= x >> k`` for k >= 16 on 32 bits,
    and multiplication by an odd constant mod 2**32), so the composition is a
    permutation: distinct inputs can never collide. That property, not the
    avalanche quality, is what the seed derivation needs.
    """
    x &= _MASK32
    x ^= x >> 16
    x = (x * _FMIX32_MUL1) & _MASK32
    x ^= x >> 13
    x = (x * _FMIX32_MUL2) & _MASK32
    x ^= x >> 16
    return x


def _yaw_aug_step_seed(seed: int, step: int, rank: int) -> int:
    """Deterministic generator seed for one (run seed, training step, rank).

    The yaw offsets of exp_15's augmentation are a pure function of
    ``(yaw_aug.seed, global_step, global_rank, within-batch index)`` — a
    *counter-based* scheme with no stream state (plan §3.1, review F5). Nothing
    is carried between steps, so a run killed at step N and resumed there
    reproduces exactly the draws the uninterrupted run would have made; a
    sequential ``torch.Generator`` would instead have to be checkpointed, and
    PL checkpoints carry no RNG state.

    **The result is 32 bits wide on purpose** (round-1 code review, finding 1).
    The pinned torch CPU generator seeds MT19937 from the low 32 bits only, so
    ``s`` and ``s + 2**32`` are the same stream: a nominally 63-bit hash bought
    no extra entropy and *did* collide — 10 collisions over the armed 40,000-step
    x 8-rank domain, e.g. (526, 2) and (10156, 7) drew identical yaws. Instead of
    hashing and hoping, the counters are packed into one 32-bit word and mapped
    through a **keyed bijection** of that word:

        v = (step << 12) | rank                  (injective on the asserted domain)
        out = fmix32(v ^ k_lo) ^ k_hi            (bijections: xor-by-key, fmix32)

    ``k_lo``/``k_hi`` are the two halves of ``splitmix64(seed)``, so different run
    seeds give unrelated assignments while, *for a fixed seed*, distinct
    ``(step, rank)`` cells provably get distinct effective seeds and therefore
    distinct MT19937 streams. Zero collisions by construction, not by luck.

    Parameters
    ----------
    seed : int
        The run's ``training.yaw_aug.seed``.
    step : int
        ``global_step`` at this optimisation step, in ``[0, 2**20)``.
    rank : int
        ``global_rank`` of this process, in ``[0, 2**12)``.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    seed, step, rank = int(seed), int(step), int(rank)
    if not 0 <= step < _YAW_AUG_MAX_STEP:
        raise ValueError(
            f"yaw_aug step must be in [0, {_YAW_AUG_MAX_STEP}), got {step}: outside "
            "that domain the (step, rank) packing is no longer injective"
        )
    if not 0 <= rank < _YAW_AUG_MAX_RANK:
        raise ValueError(
            f"yaw_aug rank must be in [0, {_YAW_AUG_MAX_RANK}), got {rank}: outside "
            "that domain the (step, rank) packing is no longer injective"
        )

    key = _splitmix64(seed & _MASK64)
    k_lo = key & _MASK32
    k_hi = (key >> 32) & _MASK32

    v = ((step << _YAW_AUG_RANK_BITS) | rank) & _MASK32
    return _fmix32(v ^ k_lo) ^ k_hi


class Profiler:
    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal", "log_snr"] = "log_snr",
            timestep_sampler_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            p_one_shot: float = 0.0,
            test_param: tp.Optional[tp.Dict[str, tp.Any]] = None,
            cond_method: str = "vanilla",
            frame_avg_angles: tp.Optional[tp.List[float]] = None,
            frame_avg_max_fwd_samples: tp.Optional[int] = None,
            yaw_aug_enabled: bool = False,
            yaw_aug_img_w: int = 512,
            yaw_aug_seed: int = 0,
            are_lambda: tp.Optional[float] = None,
            are_anchor: tp.Optional[dict] = None
    ):
        super().__init__()

        self.diffusion = model

        # Conditioning symmetrization method (exp_03 Route 1). Validated at
        # construction so an unknown value fails fast rather than at first step;
        # _compute_conditioning keeps the same raise as a backstop.
        if cond_method not in ("vanilla", "fa_invariant"):
            raise ValueError(
                f"Unknown cond_method: {cond_method!r}. "
                "Valid options: 'vanilla', 'fa_invariant'."
            )
        self.cond_method = cond_method
        self.frame_avg_angles = (
            tuple(frame_avg_angles) if frame_avg_angles is not None else DEFAULT_FRAME_ANGLES
        )
        # exp_14: the frame-average chunk plan as a declared arm property
        # (plan §3.1). None means "the yaw_rotation module default (64)", which is
        # what every recipe already in the record ran under; the value is PASSED to
        # invariant_conditioning, never assigned to the module global. Validated
        # raw here so direct construction is as fail-closed as the config path
        # (the exp_15 yaw_aug precedent).
        if frame_avg_max_fwd_samples is not None:
            if (isinstance(frame_avg_max_fwd_samples, bool)
                    or not isinstance(frame_avg_max_fwd_samples, int)):
                raise ValueError(
                    "frame_avg_max_fwd_samples must be an int or None, got "
                    f"{frame_avg_max_fwd_samples!r}"
                )
            if frame_avg_max_fwd_samples < 1:
                raise ValueError(
                    f"frame_avg_max_fwd_samples must be >= 1, got {frame_avg_max_fwd_samples}"
                )
        self.frame_avg_max_fwd_samples = frame_avg_max_fwd_samples

        # exp_15: random-yaw TRAINING augmentation (plan §§3.1, 6.3). Off by
        # default and read only by training_step; validation/test are untouched.
        # The RAW arguments are validated — never coerced — with the same rules
        # the factory applies, so direct construction is as fail-closed as the
        # config path (round-1 review, finding 4): bool("false") is True and
        # int("512") is 512, and either would have armed or disarmed the
        # treatment on a typo without a word of complaint.
        if not isinstance(yaw_aug_enabled, bool):
            raise ValueError(
                f"yaw_aug_enabled must be a bool, got {yaw_aug_enabled!r}"
            )
        if isinstance(yaw_aug_img_w, bool) or not isinstance(yaw_aug_img_w, int):
            raise ValueError(f"yaw_aug_img_w must be an int, got {yaw_aug_img_w!r}")
        if yaw_aug_img_w <= 0:
            raise ValueError(f"yaw_aug_img_w must be > 0, got {yaw_aug_img_w}")
        if isinstance(yaw_aug_seed, bool) or not isinstance(yaw_aug_seed, int):
            raise ValueError(f"yaw_aug_seed must be an int, got {yaw_aug_seed!r}")
        if yaw_aug_enabled and cond_method == "fa_invariant":
            raise ValueError(
                "yaw_aug_enabled=True with cond_method='fa_invariant' is an "
                "untested combination and out of scope for exp_15."
            )

        self.yaw_aug_enabled = yaw_aug_enabled
        self.yaw_aug_img_w = yaw_aug_img_w
        self.yaw_aug_seed = yaw_aug_seed

        # exp_16 (ARE): the target reparameterisation z -> z - lambda*A(p).
        # ``None`` means "not an ARE arm" and is NOT the same as 0.0: at None the
        # anchor code is never entered at all, which is what keeps the absent-key
        # path identical to the one P1 (the lambda=0 control) was trained through.
        # Validated on the RAW arguments, as fail-closed as the config path.
        if are_lambda is None:
            if are_anchor is not None:
                raise ValueError(
                    "are_anchor was given without are_lambda: an anchor block that "
                    "nothing weights would be silently ignored")
        else:
            if isinstance(are_lambda, bool) or not isinstance(are_lambda, (int, float)):
                raise ValueError(
                    f"are_lambda must be a number in [0, 1], got {are_lambda!r}")
            if not 0.0 <= float(are_lambda) <= 1.0:
                raise ValueError(f"are_lambda must be in [0, 1], got {are_lambda!r}")
            if not isinstance(are_anchor, dict):
                raise ValueError(
                    f"are_lambda={are_lambda!r} requires an are_anchor dict carrying "
                    "the calibrated constants and the model's time base, got "
                    f"{are_anchor!r}")
            are_lambda = float(are_lambda)
        self.are_lambda = are_lambda
        self.are_anchor = dict(are_anchor) if isinstance(are_anchor, dict) else None
        self._are_bank = None

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.timestep_sampler_options = {} if timestep_sampler_options is None else timestep_sampler_options

        if self.timestep_sampler == "log_snr":
            self.mean_logsnr = self.timestep_sampler_options.get("mean_logsnr", -1.2)
            self.std_logsnr = self.timestep_sampler_options.get("std_logsnr", 2.0)

        self.p_one_shot = p_one_shot

        self.diffusion_objective = model.diffusion_objective

        self.loss_modules = [
            MSELoss("output",
                   "targets",
                   weight=1.0,
                   mask_key="padding_mask" if self.mask_padding else None,
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        # Validation
        self.validation_timesteps = validation_timesteps
        self.validation_step_outputs = {}
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []
        
        # Test
        if test_param is not None:
            self.set_test_config(
                samples=test_param.get("samples", 10240),
                cfg_scale=test_param.get("cfg_scale", 1.0),
                steps=int(test_param.get("steps", 1)),
                sample_rate=test_param.get("sample_rate", 22050),
                audio_channels=test_param.get("audio_channels", 1),
                metrics=test_param.get("metrics", {}), 
                store_predictions = test_param.get("store_predictions", False),
            )

    def set_test_config(self, samples, cfg_scale, steps, sample_rate, audio_channels, metrics, store_predictions=False):
        self.samples = samples
        self.cfg_scale = cfg_scale
        self.steps = steps
        self.store_predictions = store_predictions
        self.preds = []

        from ..metrics.metric_callback import AcousticMetricsCallback
        self.metric_callback = AcousticMetricsCallback(
            sample_rate=sample_rate,
            sample_size=self.samples,
            audio_channels=audio_channels,
            dataset_name= metrics.get("dataset_name", "AcousticRooms"),

            eval_T60=metrics.get("eval_T60", False),
            eval_C50=metrics.get("eval_C50", False),
            eval_EDT=metrics.get("eval_EDT", False),
            eval_l1_distance=metrics.get("eval_l1_distance", False),
            eval_l1_distance_multires=metrics.get("eval_l1_distance_multires", False),
            eval_FD=metrics.get("eval_FD", False),
            eval_retrieval = metrics.get("eval_retrieval", False),
            eval_env = metrics.get("eval_env", False),

            AGREE_ckpt=metrics.get("AGREE_ckpt", None), 
            dump_dir=metrics.get("dump_dir", None),
            eval_per_scene=True if metrics.get("dataset_name") == "HAA" else False,
        )

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    @rank_zero_only
    def _print_yaw_aug_banner(self):
        # flush: the launcher tees torchrun's stdout through a FIFO without
        # PYTHONUNBUFFERED, so a buffered banner could reach the log after the
        # post-launch gate has already looked for it.
        print(
            f"yaw_aug ENABLED img_w={self.yaw_aug_img_w} seed={self.yaw_aug_seed}",
            flush=True,
        )

    def on_fit_start(self):
        """Announce the treatment once, on rank 0, before step 0.

        The launch kit greps the job log for this line and kills a run that does
        not print it: exp_15's whole contrast is "this arm was augmented, the
        control was not", and a silently-disabled treatment would look like a
        perfectly healthy run.
        """
        if self.yaw_aug_enabled:
            self._print_yaw_aug_banner()

    def _check_yaw_aug_metadata(self, metadata):
        """Fail-closed schema validation of one training batch (plan §3.1, F7).

        Every check is a hard error rather than a skip: a batch that quietly
        declined to rotate would train the treatment arm as a partial control and
        nothing downstream could detect it. In particular the panorama width is
        validated against the ACTUAL depth tensor, never trusted from the config
        — a config/data mismatch would otherwise roll by the wrong number of
        columns and desynchronise depth from the poses.
        """
        if not isinstance(metadata, (list, tuple)):
            raise ValueError(
                f"yaw_aug: metadata must be a list of per-sample dicts, got "
                f"{type(metadata).__name__}"
            )
        if len(metadata) == 0:
            raise ValueError("yaw_aug: metadata is empty; nothing to augment")

        for i, md in enumerate(metadata):
            if not isinstance(md, dict):
                raise ValueError(
                    f"yaw_aug: metadata[{i}] must be a dict, got {type(md).__name__}"
                )
            if "depth" not in md:
                raise ValueError(f"yaw_aug: metadata[{i}] has no 'depth' field")
            depth = md["depth"]
            if not torch.is_tensor(depth):
                raise ValueError(
                    f"yaw_aug: metadata[{i}]['depth'] must be a tensor, got "
                    f"{type(depth).__name__}"
                )
            if depth.ndim != 3 or depth.shape[0] != 3:
                raise ValueError(
                    f"yaw_aug: metadata[{i}]['depth'] must have shape [3, H, W], got "
                    f"{list(depth.shape)}"
                )
            if depth.shape[2] != self.yaw_aug_img_w:
                raise ValueError(
                    f"yaw_aug: metadata[{i}]['depth'] is {depth.shape[2]} columns "
                    f"wide but yaw_aug_img_w is {self.yaw_aug_img_w}"
                )
            for key in POSE_KEYS:
                # REQUIRED, not optional: rotate_scene_metadata skips absent keys,
                # so a sample missing one would be rotated only partially (depth
                # and three poses moved, the fourth left behind) — geometric
                # nonsense that nothing downstream could detect.
                if key not in md:
                    raise ValueError(f"yaw_aug: metadata[{i}] has no {key!r} pose field")
                pose = md[key]
                if not torch.is_tensor(pose):
                    raise ValueError(
                        f"yaw_aug: metadata[{i}][{key!r}] must be a tensor, got "
                        f"{type(pose).__name__}"
                    )
                # ndim first: a 0-d pose would make shape[-1] raise IndexError.
                if pose.ndim < 1 or pose.shape[-1] != 3:
                    raise ValueError(
                        f"yaw_aug: metadata[{i}][{key!r}] must have trailing "
                        f"dimension 3, got shape {list(pose.shape)}"
                    )

    def _apply_yaw_aug(self, metadata):
        """Rotate each sample's conditioning by its own freshly drawn yaw.

        The offsets are a pure function of ``(yaw_aug_seed, global_step,
        global_rank, index)`` — see :func:`_yaw_aug_step_seed`. The generator is
        private to this call, so the global python/NumPy/torch streams are left
        untouched and the augmented arm differs from the control only through the
        treatment, never through RNG displacement.

        ``reals`` (the target RIR), ``context_audio``, ``padding_mask`` and
        ``scene`` are not rotated: a rigid yaw of the whole scene leaves the
        impulse response unchanged, which is exactly what makes the rotated
        conditioning + unchanged target a valid training pair.
        """
        self._check_yaw_aug_metadata(metadata)

        generator = torch.Generator()
        generator.manual_seed(
            _yaw_aug_step_seed(self.yaw_aug_seed, int(self.global_step), int(self.global_rank))
        )
        offsets = draw_yaw_offsets(len(metadata), self.yaw_aug_img_w, generator)
        angles = offsets_to_radians(offsets, self.yaw_aug_img_w)
        return [
            rotate_scene_metadata(md, alpha, self.yaw_aug_img_w)
            for md, alpha in zip(metadata, angles)
        ]

    def _compute_conditioning(self, metadata):
        """Dispatch conditioning by cond_method (exp_03 plan §2c).

        ``fa_invariant`` routes through the Route-1 yaw-symmetrized path
        (cylindrical pose features + C4 frame average of the ViT conditioners);
        ``vanilla`` is the unchanged single conditioner pass. Called from
        training_step, validation_step AND test_step so all inference/training
        sites share one dispatch point (the repo's flow_source precedent). The
        raise is a backstop; the constructor already rejects unknown values.

        ``max_fwd_samples`` is passed as a KEYWORD: the parameter after ``angles``
        is ``vit_ids``, so a positional slip would disable the frame average
        instead of changing the chunk plan.

        The no-cap case takes a SEPARATE branch that issues the original
        four-argument call verbatim (r1 review finding 1). ``max_fwd_samples=None``
        is numerically identical, but "identical behaviour" and "the literal
        pre-change call" are different guarantees, and every arm already in the
        record — including the cluster's in-flight exp_12C run — was trained
        through the four-argument form. Only an arm that DECLARES a cap gets a
        different call shape.
        """
        if self.cond_method == "fa_invariant":
            if self.frame_avg_max_fwd_samples is None:
                return invariant_conditioning(
                    self.diffusion.conditioner, metadata, self.device, self.frame_avg_angles
                )
            return invariant_conditioning(
                self.diffusion.conditioner, metadata, self.device, self.frame_avg_angles,
                max_fwd_samples=self.frame_avg_max_fwd_samples,
            )
        if self.cond_method == "vanilla":
            return self.diffusion.conditioner(metadata, self.device)
        raise ValueError(f"Unknown cond_method: {self.cond_method}")

    def _are_anchor_bank(self):
        """The frozen-VAE anchor bank for this arm, built once, lazily.

        Lazy because the pretransform's weights are loaded by ``train.py`` after
        the wrapper is constructed, and because a non-ARE arm must never pay for
        (or fail on) an anchor bank it does not use.
        """
        if self.are_lambda is None:
            raise ValueError(
                "this arm declares no are_lambda, so it has no anchor bank")
        if self._are_bank is None:
            anchor_cfg = {k: v for k, v in self.are_anchor.items()
                          if k not in ("sample_rate", "sample_size")}
            self._are_bank = build_anchor_bank(
                self.diffusion.pretransform,
                self.are_anchor["sample_rate"], self.are_anchor["sample_size"],
                anchor_cfg)
        return self._are_bank

    def _are_anchors(self, metadata, like):
        """Per-sample anchors for one batch, shape-checked against the latent.

        ``compute_are_anchors`` is looked up as a module global on purpose: it is
        the seam the tests spy on to prove that a non-ARE arm never enters this
        code at all.
        """
        anchors = compute_are_anchors(self._are_anchor_bank(), metadata, self.device)
        if tuple(anchors.shape) != tuple(like.shape):
            raise ValueError(
                f"ARE anchor shape {tuple(anchors.shape)} does not match the latent "
                f"shape {tuple(like.shape)}: the anchor's sample_size/hop disagree "
                "with the pretransform actually in use")
        return anchors

    def _are_residual_target(self, diffusion_input, metadata):
        """``z -> z - lambda*A(p)`` — the TRAINING half of the reparameterisation.

        Called from ``training_step`` AND ``validation_step``; ``test_step`` adds
        the anchor back instead (CLAUDE.md's all-dispatch-sites rule). Consumes no
        RNG, so an ARE arm sees exactly the noise sequence its lambda=0 control saw.
        """
        anchors = self._are_anchors(metadata, diffusion_input)
        return diffusion_input - self.are_lambda * anchors.to(diffusion_input.dtype)

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        # exp_15: TRAINING-ONLY random yaw of the conditioning (plan §6.3).
        if self.yaw_aug_enabled:
            metadata = self._apply_yaw_aug(metadata)

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")
        conditioning = self._compute_conditioning(metadata)

        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # Check for wrapped padding masks to avoid interpolation error
        first_padding_mask = metadata[0]["padding_mask"]
        if isinstance(first_padding_mask, list) and len(first_padding_mask) == 1:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)
        else:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        p.tick("conditioning")
        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast('cuda') and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        # exp_16 (ARE), dispatch site 1 of 3. Placed AFTER the encode and BEFORE
        # the timestep/noise draws, so the treatment changes the target and
        # nothing else: the anchor path touches no generator, hence this arm
        # consumes the RNG stream identically to its lambda=0 control.
        if self.are_lambda is not None:
            diffusion_input = self._are_residual_target(diffusion_input, metadata)

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)
            # Flip the distribution
            t = 1 - t
        elif self.timestep_sampler == "log_snr":
            t = sample_timesteps_logsnr(reals.shape[0], mean_logsnr=self.mean_logsnr, std_logsnr=self.std_logsnr).to(self.device)
        elif self.timestep_sampler == "ones":
            t = torch.ones(reals.shape[0], device=self.device)
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            print('Applying distribution shift to timesteps')
            t = self.diffusion.dist_shift.time_shift(t, reals.shape[2])

        if self.p_one_shot > 0:
            print('Applying one-shot timesteps')
            # Set t to 1 with probability p_one_shot
            t = torch.where(torch.rand_like(t) < self.p_one_shot, torch.ones_like(t), t)

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            alphas, sigmas = 1-t, t
        
        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)

        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            targets = noise - diffusion_input

        p.tick("noise")
        extra_args = {}
        if use_padding_mask:
            extra_args["mask"] = padding_masks

        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
        
        p.tick("diffusion")
        loss_info.update({
            "output": output,
            "targets": targets,
            "padding_mask": padding_masks if use_padding_mask else None,
        })

        loss, losses = self.losses(loss_info)

        p.tick("loss")
        if self.log_loss_info:
            # Loss debugging logs
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none")

            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

            # gather loss_all across all GPUs
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

            # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def validation_step(self, batch, batch_idx):
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        with torch.amp.autocast('cuda') and torch.no_grad():
            conditioning = self._compute_conditioning(metadata)

        # TODO: decide what to do with padding masks during validation
        # # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        # use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        # # Create batch tensor of attention masks from the "mask" field of the metadata array
        # if use_padding_mask:
        #     padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast('cuda') and torch.no_grad():
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    # if use_padding_mask:
                    #     padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        # exp_16 (ARE), dispatch site 2 of 3: validation measures the objective the
        # arm is actually training, so it uses the same residual target.
        if self.are_lambda is not None:
            diffusion_input = self._are_residual_target(diffusion_input, metadata)

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                alphas, sigmas = 1-t, t

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                targets = noise - diffusion_input
            
            extra_args = {}

            # if use_padding_mask:
            #     extra_args["mask"] = padding_masks

            with torch.amp.autocast('cuda') and torch.no_grad():
                output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0, **extra_args)
                val_loss = F.mse_loss(output, targets)
                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())

    def on_validation_epoch_end(self):
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])

            # Gather losses across all GPUs
            val_loss = self.all_gather(val_loss).mean().item()

            log_metric(self.logger, outputs_key, val_loss, step=self.global_step)
            self.log(outputs_key, val_loss, sync_dist=False)

        # Get average over all timesteps
        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        # Gather losses across all GPUs
        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)
        self.log('val/avg_loss', val_loss, prog_bar=True, sync_dist=False)

        # Reset validation losses
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []
        
    def test_step(self, batch, batch_idx):
        reals, metadata = batch

        B = reals.shape[0]

        if self.diffusion.pretransform is not None:
            samples = self.samples // self.diffusion.pretransform.downsampling_ratio
        else:
            samples = self.samples

        noise = torch.randn([B, self.diffusion.io_channels, samples]).to(self.device)

        with torch.amp.autocast('cuda') and torch.no_grad():
            conditioning = self._compute_conditioning(metadata)

        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)

        with torch.amp.autocast('cuda'):
            model = self.diffusion_ema.ema_model if self.diffusion_ema is not None else self.diffusion.model

            if self.diffusion_objective == "v":
                fakes = sample(model, noise, self.steps, 0, **cond_inputs, cfg_scale=self.cfg_scale, dist_shift=self.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            elif self.diffusion_objective == "rectified_flow":
                fakes = sample_discrete_euler(model, noise, self.steps, **cond_inputs, cfg_scale=self.cfg_scale, dist_shift=self.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            elif self.diffusion_objective == "rf_denoiser":
                logsnr = torch.linspace(-6, 2, self.steps+1).to(self.device)
                sigmas = torch.sigmoid(-logsnr)
                sigmas[0] = 1.0
                sigmas[-1] = 0.0
                fakes = sample_flow_pingpong(model, noise, sigmas=sigmas, **cond_inputs, cfg_scale=self.cfg_scale, dist_shift=self.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)

            # exp_16 (ARE), dispatch site 3 of 3 — the INFERENCE half. The sampler
            # produced the residual, so the query anchor is added back to the
            # LATENT, before the single decode.
            if self.are_lambda is not None:
                fakes = apply_anchor_addback(
                    fakes, self._are_anchors(metadata, fakes), self.are_lambda)

            if self.diffusion.pretransform is not None:
                fakes = self.diffusion.pretransform.decode(fakes)

            if self.store_predictions:
                self.preds.append(fakes.cpu())

             # Clamp and pad if necessary
            fakes = fakes.clamp(-1.0, 1.0) 
            if fakes.shape != reals.shape:
                if fakes.shape[-1] < reals.shape[-1]:
                    fakes = torch.nn.functional.pad(fakes, (0, reals.shape[-1] - fakes.shape[-1]))
                else:
                    reals = torch.nn.functional.pad(reals, (0, fakes.shape[-1] - reals.shape[-1]))
    
            scene_list = [md["scene"] for md in metadata]
            depth_list = [md["depth"] if 'depth' in md else None for md in metadata]
            query_list = [md["source"] if 'source' in md else None for md in metadata]
            depthMinusSource_list = [(d[:3, :, :] - source_pose[:, None, None]).unsqueeze(0).float()for d, source_pose in zip(depth_list, query_list)]
            self.metric_callback.update_metrics("test", fakes, reals, scene_list, depth=depthMinusSource_list)
        
    def on_test_epoch_end(self):    
        metrics_dict = self.metric_callback.compute_metrics("test")
        for metric_name, metric_value in metrics_dict.items():
            if metric_name == 'T60' or 'to' in metric_name:
                metric_name += ' (%)'
            elif metric_name == 'EDT':
                metric_name += ' (ms)'
            elif metric_name == 'C50':
                metric_name += ' (dB)'
            print('Test/' + metric_name, metric_value)
        self.metrics_dict = metrics_dict


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
            print("Exporting EMA model weights")

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)
