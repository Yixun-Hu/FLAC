import torch
from torch import nn
from torch.nn import functional as F
import typing as tp
from time import time

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..inference.generation import generate_diffusion_cond
from ..inference.sampling import DistributionShift

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

class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()

class DiffusionModelWrapper(nn.Module):
    def __init__(
                self,
                model: DiffusionModel,
                io_channels,
                sample_size,
                sample_rate,
                min_input_length,
                pretransform: tp.Optional[Pretransform] = None,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.model = model

        if pretransform is not None:
            self.pretransform = pretransform
        else:
            self.pretransform = None

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
            distribution_shift_options = None,
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            query_phase_cond_id: tp.Optional[str] = None,
            phase_aliases: tp.Optional[tp.Dict[str, str]] = None,
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.query_phase_cond_id = query_phase_cond_id
        self.phase_aliases = dict(phase_aliases or {})
        self.min_input_length = min_input_length

        # Stream type embeddings are part of the opt-in relative-phase model,
        # not the legacy wrapper.  Keeping the inactive attribute as ``None``
        # avoids adding any parameters/state-dict keys to old checkpoints.
        self.cross_attn_type_embeddings = None
        if self.query_phase_cond_id is not None:
            expected_cross_ids = (
                "source_vit",
                "context_poses_vit",
                "context_poses",
                "context_audio",
            )
            if tuple(self.cross_attn_cond_ids) != expected_cross_ids:
                raise ValueError(
                    "Yaw-phase DiT V0 requires cross-attention IDs in fixed "
                    f"order {expected_cross_ids}; got {tuple(self.cross_attn_cond_ids)}"
                )
            if self.query_phase_cond_id != "source":
                raise ValueError(
                    "Yaw-phase DiT V0 requires query_phase_cond_id='source'"
                )
            cond_token_dim = getattr(getattr(model, "model", None), "cond_token_dim", None)
            if not isinstance(cond_token_dim, int) or cond_token_dim <= 0:
                raise ValueError(
                    "Phase-aware conditioning requires a positive DiT cond_token_dim"
                )
            if len(self.cross_attn_cond_ids) == 0:
                raise ValueError(
                    "Phase-aware conditioning requires cross-attention condition IDs"
                )
            self.cross_attn_type_embeddings = nn.ParameterDict({
                key: nn.Parameter(torch.zeros(cond_token_dim))
                for key in self.cross_attn_cond_ids
            })
        elif self.phase_aliases:
            raise ValueError(
                "phase_aliases require query_phase_cond_id to enable phase-aware conditioning"
            )

        self.dist_shift = None
        if distribution_shift_options is not None:
            self.dist_shift = DistributionShift(**distribution_shift_options)     

    @property
    def phase_aware(self) -> bool:
        return self.query_phase_cond_id is not None

    @staticmethod
    def _conditioning_entry(
            conditioning_tensors: tp.Dict[str, tp.Any],
            key: str,
    ) -> tp.Sequence[torch.Tensor]:
        if key not in conditioning_tensors:
            raise ValueError(f"Conditioning entry '{key}' is missing")
        entry = conditioning_tensors[key]
        if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
            raise ValueError(
                f"Conditioning entry '{key}' must contain [content, mask] or "
                "[content, mask, phase]"
            )
        return entry

    @staticmethod
    def _phase_content_and_mask(
            entry: tp.Sequence[torch.Tensor],
            key: str,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        content, mask = entry[0], entry[1]
        if not isinstance(content, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError(f"Conditioning entry '{key}' content and mask must be tensors")

        if content.ndim == 2:
            content = content.unsqueeze(1)
        if content.ndim != 3:
            raise ValueError(
                f"Conditioning entry '{key}' content must have shape [B, N, D]"
            )

        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        if mask.ndim != 2:
            raise ValueError(
                f"Conditioning entry '{key}' mask must have shape [B, N]"
            )
        if mask.shape != content.shape[:2]:
            raise ValueError(
                f"Conditioning entry '{key}' mask length {tuple(mask.shape)} does not "
                f"match token shape {tuple(content.shape[:2])}"
            )
        return content, mask

    @staticmethod
    def _phase_from_entry(
            entry: tp.Sequence[torch.Tensor],
            key: str,
            content: torch.Tensor,
    ) -> tp.Optional[torch.Tensor]:
        if len(entry) == 2:
            return None
        phase = entry[2]
        if not isinstance(phase, torch.Tensor):
            raise ValueError(f"Conditioning entry '{key}' phase must be a tensor")
        if phase.ndim == 1:
            phase = phase.unsqueeze(1)
        if phase.ndim != 2:
            raise ValueError(
                f"Conditioning entry '{key}' phase must have shape [B, N]"
            )
        if phase.shape != content.shape[:2]:
            raise ValueError(
                f"Conditioning entry '{key}' phase length {tuple(phase.shape)} does not "
                f"match token shape {tuple(content.shape[:2])}"
            )
        return phase.float()

    def _get_phase_aware_cross_inputs(
            self,
            conditioning_tensors: tp.Dict[str, tp.Any],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = {}
        batch_size = None
        total_tokens = 0

        # First validate every content/mask pair in the configured order.  Do
        # not concatenate yet: K>1 and malformed phases must fail before a
        # non-V0 sequence can be materialized.
        for key in self.cross_attn_cond_ids:
            entry = self._conditioning_entry(conditioning_tensors, key)
            content, mask = self._phase_content_and_mask(entry, key)
            expected_token_count = {
                "source_vit": 16,
                "context_poses_vit": 16,
                "context_poses": 1,
                "context_audio": 1,
            }[key]
            if content.shape[1] != expected_token_count:
                raise ValueError(
                    f"Yaw-phase DiT V0 stream '{key}' requires exactly "
                    f"{expected_token_count} tokens; got {content.shape[1]}"
                )
            if batch_size is None:
                batch_size = content.shape[0]
            elif content.shape[0] != batch_size:
                raise ValueError(
                    f"Conditioning entry '{key}' batch size does not match other streams"
                )

            type_embedding = self.cross_attn_type_embeddings[key]
            if content.shape[-1] != type_embedding.numel():
                raise ValueError(
                    f"Conditioning entry '{key}' channel dimension {content.shape[-1]} "
                    f"does not match cond_token_dim {type_embedding.numel()}"
                )
            phase = self._phase_from_entry(entry, key, content)
            if phase is not None and key in self.phase_aliases:
                raise ValueError(
                    f"Conditioning entry '{key}' has both a direct phase and a phase alias"
                )
            streams[key] = {
                "content": content,
                "mask": mask,
                "phase": phase,
            }
            total_tokens += content.shape[1]

        # Resolve phase-free streams only through explicit aliases.  V0 aliases
        # a single context audio item to a single context pose, so an alias with
        # any other token count is an unsupported K>1 batch.
        for key in self.cross_attn_cond_ids:
            if streams[key]["phase"] is not None:
                continue
            alias_key = self.phase_aliases.get(key)
            if alias_key is None:
                raise ValueError(
                    f"Conditioning entry '{key}' has no phase and no phase alias"
                )

            if alias_key in streams:
                alias_phase = streams[alias_key]["phase"]
            else:
                alias_entry = self._conditioning_entry(conditioning_tensors, alias_key)
                alias_content, _ = self._phase_content_and_mask(alias_entry, alias_key)
                alias_phase = self._phase_from_entry(
                    alias_entry, alias_key, alias_content
                )
            if alias_phase is None:
                raise ValueError(
                    f"Phase alias '{alias_key}' for '{key}' does not provide a phase"
                )
            if alias_phase.shape != streams[key]["content"].shape[:2]:
                raise ValueError(
                    f"Phase alias '{alias_key}' shape {tuple(alias_phase.shape)} does not "
                    f"match '{key}' token shape {tuple(streams[key]['content'].shape[:2])}"
                )
            if alias_phase.shape[1] != 1:
                raise ValueError(
                    "Yaw-phase DiT V0 requires K=1 for aliased context pose/audio streams"
                )
            streams[key]["phase"] = alias_phase.float()

        if total_tokens != 34:
            raise ValueError(
                f"Yaw-phase DiT V0 requires exactly 34 cross-attention tokens; got {total_tokens}"
            )

        query_entry = self._conditioning_entry(
            conditioning_tensors, self.query_phase_cond_id
        )
        query_content, _ = self._phase_content_and_mask(
            query_entry, self.query_phase_cond_id
        )
        query_phase = self._phase_from_entry(
            query_entry, self.query_phase_cond_id, query_content
        )
        if query_phase is None:
            raise ValueError(
                f"Query conditioning entry '{self.query_phase_cond_id}' has no phase"
            )
        if query_phase.shape[0] != batch_size or query_phase.shape[1] != 1:
            raise ValueError(
                "Yaw-phase DiT V0 query phase must have shape [B, 1] for K=1"
            )

        contents = []
        masks = []
        phases = []
        for key in self.cross_attn_cond_ids:
            content = streams[key]["content"]
            type_embedding = self.cross_attn_type_embeddings[key].to(
                device=content.device, dtype=content.dtype
            )
            contents.append(content + type_embedding.view(1, 1, -1))
            masks.append(streams[key]["mask"])
            phases.append(streams[key]["phase"].float())

        return (
            torch.cat(contents, dim=1),
            torch.cat(masks, dim=1),
            torch.cat(phases, dim=1).float(),
            query_phase[:, 0].float(),
        )

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        cross_attention_phases = None
        query_phase = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if len(self.cross_attn_cond_ids) > 0:
            if self.phase_aware:
                (
                    cross_attention_input,
                    cross_attention_masks,
                    cross_attention_phases,
                    query_phase,
                ) = self._get_phase_aware_cross_inputs(conditioning_tensors)
            else:
                # Keep the legacy path unchanged: old two-item entries, shape
                # normalization and concatenation retain their exact behavior.
                cross_attention_input = []
                cross_attention_masks = []

                for key in self.cross_attn_cond_ids:
                    cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                    # Add sequence dimension if it's not there
                    if len(cross_attn_in.shape) == 2:
                        cross_attn_in = cross_attn_in.unsqueeze(1)
                        cross_attn_mask = cross_attn_mask.unsqueeze(1)

                    cross_attention_input.append(cross_attn_in)
                    cross_attention_masks.append(cross_attn_mask)

                cross_attention_input = torch.cat(cross_attention_input, dim=1)
                cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if negative:
            result = {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
            if self.phase_aware:
                result["negative_cross_attn_phases"] = cross_attention_phases
                result["negative_query_phase"] = query_phase
            return result
        else:
            result = {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask
            }
            if self.phase_aware:
                result["cross_attn_phases"] = cross_attention_phases
                result["query_phase"] = query_phase
            return result

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)

class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        diffusion_objective: str,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.diffusion_objective = diffusion_objective

        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                cross_attn_phases=None,
                query_phase=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_cross_attn_phases=None,
                negative_query_phase=None,
                input_concat_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        phase_api_used = any(
            value is not None
            for value in (
                cross_attn_phases,
                query_phase,
                negative_cross_attn_phases,
                negative_query_phase,
            )
        )
        if phase_api_used and any(
            value is not None
            for value in (
                negative_cross_attn_cond,
                negative_cross_attn_mask,
                negative_cross_attn_phases,
                negative_query_phase,
                negative_input_concat_cond,
                negative_global_cond,
            )
        ):
            raise ValueError(
                "independent negative conditioning is unsupported in the "
                "V0 phase-aware path; omit every negative input for null CFG"
            )

        # Do not inject new kwargs into the legacy DiffusionTransformer path.
        # Round-4's phase-aware DiT accepts these keys; conditional forwarding
        # keeps old configs numerically and structurally unchanged meanwhile.
        phase_kwargs = {}
        if cross_attn_phases is not None:
            phase_kwargs["cross_attn_phases"] = cross_attn_phases
        if query_phase is not None:
            phase_kwargs["query_phase"] = query_phase
        if negative_cross_attn_phases is not None:
            phase_kwargs["negative_cross_attn_phases"] = negative_cross_attn_phases
        if negative_query_phase is not None:
            phase_kwargs["negative_query_phase"] = negative_query_phase

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            **phase_kwargs,
            **kwargs)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == 'dit':
        diffusion_model = DiTWrapper(diffusion_objective=diffusion_objective, **diffusion_model_config)
    else:
        raise NotImplementedError(f"Unknown diffusion model type: {diffusion_model_type}")

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])
    query_phase_cond_id = diffusion_config.get('query_phase_cond_id', None)
    phase_aliases = diffusion_config.get('phase_aliases', None)

    pretransform = model_config.get("pretransform", None)

    distribution_shift_options = diffusion_config.get("distribution_shift_options", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config, pretransform=pretransform)

    if diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size
    else: 
        raise NotImplementedError(f"Unknown diffusion model type: {diffusion_model_type}")

    # Get the proper wrapper class
    extra_kwargs = {}

    if model_type == "diffusion_cond":
        wrapper_fn = ConditionedDiffusionModelWrapper

        extra_kwargs["diffusion_objective"] = diffusion_objective
        
    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        query_phase_cond_id=query_phase_cond_id,
        phase_aliases=phase_aliases,
        pretransform=pretransform,
        io_channels=io_channels,
        distribution_shift_options=distribution_shift_options,
        **extra_kwargs
    )
