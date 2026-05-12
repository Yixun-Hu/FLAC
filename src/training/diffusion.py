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

from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, sample_flow_pingpong, truncated_logistic_normal_rescaled, DistributionShift, sample_timesteps_logsnr
from ..models.diffusion import ConditionedDiffusionModelWrapper
from .losses import MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_metric

from time import time

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


def _pick_nearest_reference(
    metadata: tp.Sequence[dict],
    device: tp.Union[torch.device, str],
    drop: bool,
) -> tp.Tuple[torch.Tensor, tp.Sequence[dict]]:
    """
    Per-sample: pick the reference whose source is closest to the query source.

    For each sample in the batch, find the index ``k_star`` such that
    ``||context_poses[k_star] - source||_2`` is minimized, then return that
    reference's RIR waveform. When ``drop`` is True, also remove the picked
    entry from ``context_audio`` / ``context_poses`` / ``context_poses_vit``
    of that sample's metadata dict (this is the training / validation path,
    where we want cross-attention to *not* re-see the reference that is
    already being used as the flow's starting point). When ``drop`` is False,
    leave the metadata untouched (this is the test / eval path, so that
    cross-attention sees the full set of K references the user provided,
    enabling generalization to small K such as K=1).

    Parameters
    ----------
    metadata : tp.Sequence[dict]
        Per-sample info dicts produced by ``SampleDataset.__getitem__`` plus
        ``AR_md.get_custom_metadata``, then passed through ``collation_fn``
        (which leaves dict-typed entries untouched). Length equals batch
        size B. Each dict in the sequence must contain at least:

          - 'source'            : torch.Tensor [3]      query source position
          - 'context_poses'     : torch.Tensor [N, 3]   N reference source positions
          - 'context_poses_vit' : torch.Tensor [N, 3]
          - 'context_audio'     : torch.Tensor [N, 1, T_ctx]

        If ``drop`` is True, ``N`` must be >= 2 (so cross-attention has at
        least one reference left after dropping). If ``drop`` is False,
        ``N`` must be >= 1 (need at least one reference to pick from).
    device : torch.device | str
        Device to move the picked reference waveforms onto.
    drop : bool
        If True, mutate ``metadata`` in place: remove the picked entry from
        each sample's ``context_audio`` / ``context_poses`` /
        ``context_poses_vit``, leaving N-1 entries.
        If False, leave ``metadata`` unchanged.

    Returns
    -------
    ref_audio : torch.Tensor
        Shape ``[B, 1, T_ctx]``. Stacked nearest-reference waveforms,
        already moved to ``device``.
    metadata : tp.Sequence[dict]
        The same object as the input. Mutated in place when ``drop=True``,
        unchanged when ``drop=False``. Returned for explicit reassignment so
        call sites read symmetrically across the two modes.

    Raises
    ------
    AssertionError
        If any sample has fewer references than required by ``drop``, or if
        ``context_poses`` has an unexpected shape.
    KeyError
        If any of the required keys is missing from a sample's metadata
        dict (we deliberately do not use ``.get`` with defaults here, so
        upstream contract violations fail loudly).

    Examples
    --------
    >>> # Training path: drop the chosen reference from cross-attn context.
    >>> ref_audio, metadata = _pick_nearest_reference(metadata, device, drop=True)
    >>> # Inference path: keep the full context for cross-attn.
    >>> ref_audio, metadata = _pick_nearest_reference(metadata, device, drop=False)
    """
    ref_audio_list: tp.List[torch.Tensor] = []
    n_min = 2 if drop else 1
    for md in metadata:
        s_q = md['source'].to(device).float()                  # [3]
        s_k = md['context_poses'].to(device).float()           # [N, 3]
        assert s_k.dim() == 2 and s_k.shape[-1] == 3, (
            f"context_poses must be [N, 3], got {tuple(s_k.shape)}"
        )
        assert s_k.shape[0] >= n_min, (
            f"need >={n_min} context references, got N={s_k.shape[0]} (drop={drop})"
        )

        d = (s_k - s_q.unsqueeze(0)).norm(dim=-1)              # [N]
        k_star = int(d.argmin().item())

        ref_audio_list.append(md['context_audio'][k_star])     # [1, T_ctx]
        if drop:
            keep = [i for i in range(s_k.shape[0]) if i != k_star]
            md['context_audio']     = md['context_audio'][keep]
            md['context_poses']     = md['context_poses'][keep]
            md['context_poses_vit'] = md['context_poses_vit'][keep]
    return torch.stack(ref_audio_list, dim=0).to(device), metadata


class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            flow_source: str,
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
            test_param: tp.Optional[tp.Dict[str, tp.Any]] = None
    ):
        super().__init__()

        self.diffusion = model

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

        # Flow-matching starting distribution selector. Validated at the
        # dispatch sites in training_step / validation_step / test_step (and
        # in eval_FLAC.py) via exhaustive if/elif/else: raise. Keeping the
        # raw value here without a whitelist keeps the dispatch sites the
        # single source of truth, so adding a new mode means updating exactly
        # those sites and not also a stale whitelist here.
        self.flow_source = flow_source

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

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        # When flow_source == "nearest_ref", pick the reference RIR whose
        # source is closest to the query source (one per sample), AND drop
        # that reference from each sample's cross-attention context (drop=True
        # for the training path; see _pick_nearest_reference docstring and
        # plan §3.3.1 for why we drop on training but not on inference).
        # ref_audio is the picked waveform, used below to build z_ref.
        ref_audio: tp.Optional[torch.Tensor] = None
        if self.flow_source == "nearest_ref":
            ref_audio, metadata = _pick_nearest_reference(metadata, self.device, drop=True)

        conditioning = self.diffusion.conditioner(metadata, self.device)

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

        # Encode the picked reference RIR through the same VAE pretransform
        # to obtain z_ref, the starting point of the rectified-flow trajectory
        # (used in place of Gaussian noise). pretransform.encode applies the
        # same /scale normalization as the diffusion_input path, so z_ref and
        # diffusion_input live on the same scale by construction. We wrap with
        # no_grad so this branch does not produce extra gradients through the
        # encoder (mirrors how torch.randn never backprops in the gaussian path).
        if self.flow_source == "nearest_ref":
            assert ref_audio is not None, "ref_audio must be set when flow_source=nearest_ref"
            T_target = reals.shape[-1]   # waveform length = sample_size
            if ref_audio.shape[-1] < T_target:
                ref_audio = F.pad(ref_audio, (0, T_target - ref_audio.shape[-1]))
            elif ref_audio.shape[-1] > T_target:
                ref_audio = ref_audio[..., :T_target]
            with torch.amp.autocast('cuda'), torch.no_grad():
                z_ref = self.diffusion.pretransform.encode(ref_audio)
            assert z_ref.shape == diffusion_input.shape, (
                f"z_ref shape {tuple(z_ref.shape)} != "
                f"diffusion_input shape {tuple(diffusion_input.shape)}"
            )

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

        # Pick the rectified-flow starting point ("noise" keeps its name only
        # for backward compatibility with the formula below). Exhaustive
        # if/elif/else: any new flow_source mode added later must extend this
        # branch explicitly, otherwise the dispatch raises rather than
        # silently falling back to a Gaussian source.
        if self.flow_source == "gaussian":
            noise = torch.randn_like(diffusion_input)
        elif self.flow_source == "nearest_ref":
            noise = z_ref
        else:
            raise ValueError(f"Unknown flow_source: {self.flow_source!r}")

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

        # Mirror training_step: pick nearest reference and drop it from each
        # sample's cross-attention context (drop=True for the validation path,
        # so val_loss can be compared against train_loss on equal footing).
        ref_audio: tp.Optional[torch.Tensor] = None
        if self.flow_source == "nearest_ref":
            ref_audio, metadata = _pick_nearest_reference(metadata, self.device, drop=True)

        with torch.amp.autocast('cuda') and torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)

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

        # Encode z_ref once per batch (the loop below evaluates the velocity
        # objective at multiple validation timesteps, but z_ref is independent
        # of t). Reusing the same z_ref across timesteps reduces variance in
        # per-t logged val losses, which is what we want for monitoring.
        if self.flow_source == "nearest_ref":
            assert ref_audio is not None, "ref_audio must be set when flow_source=nearest_ref"
            T_target = reals.shape[-1]
            if ref_audio.shape[-1] < T_target:
                ref_audio = F.pad(ref_audio, (0, T_target - ref_audio.shape[-1]))
            elif ref_audio.shape[-1] > T_target:
                ref_audio = ref_audio[..., :T_target]
            with torch.amp.autocast('cuda'), torch.no_grad():
                z_ref = self.diffusion.pretransform.encode(ref_audio)
            assert z_ref.shape == diffusion_input.shape, (
                f"z_ref shape {tuple(z_ref.shape)} != "
                f"diffusion_input shape {tuple(diffusion_input.shape)}"
            )

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

            # Pick the rectified-flow starting point. Same exhaustive
            # dispatch as in training_step; see comment there.
            if self.flow_source == "gaussian":
                noise = torch.randn_like(diffusion_input)
            elif self.flow_source == "nearest_ref":
                noise = z_ref
            else:
                raise ValueError(f"Unknown flow_source: {self.flow_source!r}")

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

        # Get average over all timesteps
        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        # Gather losses across all GPUs
        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)

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

        # Build the rectified-flow integration starting point. Exhaustive
        # dispatch on flow_source: any new mode must extend this branch
        # explicitly, otherwise raise rather than silently falling back.
        # The inference path uses drop=False so cross-attention sees the full
        # set of K references the user provided (supports K=1..N including
        # K=1 deployment, where dropping would leave cross-attn empty).
        # See plan §3.3.1 / §5.3 for why train/inference asymmetry is
        # deliberate (cross-attn is set-style; what must match across
        # train/inference is `flow_source`, not the cross-attn token count).
        if self.flow_source == "gaussian":
            noise = torch.randn([B, self.diffusion.io_channels, samples]).to(self.device)
        elif self.flow_source == "nearest_ref":
            ref_audio, metadata = _pick_nearest_reference(metadata, self.device, drop=False)
            T_target = self.samples   # waveform length, NOT the local latent-length `samples`
            if ref_audio.shape[-1] < T_target:
                ref_audio = F.pad(ref_audio, (0, T_target - ref_audio.shape[-1]))
            elif ref_audio.shape[-1] > T_target:
                ref_audio = ref_audio[..., :T_target]
            with torch.amp.autocast('cuda'), torch.no_grad():
                noise = self.diffusion.pretransform.encode(ref_audio)
            assert noise.shape[-1] == samples, (
                f"z_ref latent length {noise.shape[-1]} != expected {samples} "
                f"(self.samples={self.samples}, "
                f"downsampling_ratio={self.diffusion.pretransform.downsampling_ratio})"
            )
        else:
            raise ValueError(f"Unknown flow_source: {self.flow_source!r}")

        with torch.amp.autocast('cuda') and torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)

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
