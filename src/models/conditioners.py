
import functools

import torch
import torch.utils.checkpoint
import typing as tp

from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .utils import load_ckpt_state_dict

from torch import nn
import torchaudio
import torchvision.models as models
import numpy as np

from .simplevit import SimpleViT
from .cyl_vit import CylindricalViT
from transformers import AutoModel, AutoConfig

# exp_03: the PINNED seed of the isolated CPU RNG stream the max_mlp head's hidden layer is
# drawn from (plan §2.1). It is deliberately NOT the training seed: the hidden layer must be
# reproducible AND must not consume a single draw from the global stream, so that every other
# module of the model initialises byte-identically to the legacy (mean+Linear) arm.
_COND_MLP_HIDDEN_SEED = 4242

# exp_04: the REGISTERED `cond_pool` values -> the `dino_pool` selector each serves. The MLP
# head construction below is SHARED by every entry (one code path, one RNG contract), so a
# same-seed build of any two arms is bitwise identical and the pooling SELECTOR is their only
# difference — which is exactly what the exp_04 (mean) vs exp_03 (max) factor-isolation
# comparison rests on. Adding a value here is the ONLY way to register a new arm; anything
# absent from this mapping fails closed.
_COND_POOL_TO_DINO_POOL = {
    'max_mlp': 'max',    # exp_03: token-axis amax
    'mean_mlp': 'mean',  # exp_04: the LEGACY patch mean (pooler_output)
}


class AudioResNet18(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 log_instead_of_log1p_in_logspace=True,
                 log_eps=1.0e-8):
        """
        ResNet-18.
        Takes in observations (binaural IR magnitude spectrograms) and produces an acoustic embedding
        :param log_instead_of_log1p_in_logspace: compute log of magnitude spect. instead of log(1 + ...)
        :param log_eps: epsilon to be used to compute log for numerical stability
        """
        super().__init__()

        self._log_instead_of_log1p_in_logspace = log_instead_of_log1p_in_logspace
        self._log_eps = log_eps

        self._n_input = in_channels

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()

        self.cnn.conv1 = nn.Conv2d(self._n_input,
                                   self.cnn.conv1.out_channels,
                                   kernel_size=self.cnn.conv1.kernel_size,
                                   stride=self.cnn.conv1.stride,
                                   padding=self.cnn.conv1.padding,
                                   bias=False)

        nn.init.kaiming_normal_(
            self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
        )

    @property
    def n_out_feats(self):
        return 512

    def forward(self, audio_spect):
        cnn_input = []
        if self._log_instead_of_log1p_in_logspace:
            audio_spect_observations = torch.log(audio_spect + self._log_eps)
        else:
            audio_spect_observations = torch.log1p(audio_spect) 
        cnn_input.append(audio_spect_observations)
        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)
    
class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False, sample_size: int = 10240, name: str = "PretransformConditioner"):
        in_dim = pretransform.encoded_channels * (sample_size // pretransform.downsampling_ratio)
        super().__init__(in_dim, output_dim)
        self.name = name

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.pretransform.eval()
        
    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        context = False 
        
        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        if audio.dim() == 4: # Context audio [B, N, C, T]
            context = True
            B, N, C, T = audio.shape
            audio = audio.view(-1, audio.shape[-2], audio.shape[-1]) # [B*N, C, T]

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
            B, C, T = audio.shape

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)
        
        latents = self.pretransform.encode(audio)

        if context:
            latents = latents.view(B, N, -1)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]
    
class RIRConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 in_channels: int = 2,
                 n_fft: int = 511,
                 win_length: int = 248,
                 hop_length: int = 62,
                 project_out: bool = False, 
                 name: str = "RIRConditioner"):
        input_dim = 512 
        super().__init__(input_dim, output_dim, project_out=project_out)
        self.name = name
        self.net = AudioResNet18(in_channels)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            win_length=win_length, 
            hop_length=hop_length, 
            power=None, 
        )

    def forward(self, audios: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.net = self.net.to(device)
        audios = torch.stack(audios, dim=0) # [B, N, C, T]
        audios = audios.to(device)
        if audios.dim() == 3:
            audios = audios.unsqueeze(1)
        B, N, C, T = audios.shape
        audios = audios.view(B * N, C, T)  # [B*N, C, T]
        audios = self.stft(audios)  # [B*N, C, F, T]

        audios = torch.sqrt(
            torch.clamp((audios.real**2) + (audios.imag**2), min=1e-8)
        )

        encoded = self.net(audios) # [batch, 512]
        encoded = encoded.view(B, N, -1)
        out = self.proj_out(encoded)  # [B, N, D]

        return [out, torch.ones(encoded.shape[0], 1).to(device)]
    
def _make_vit_ckpt_forward(layer: nn.Module, orig_forward):
    """Instance-level forward wrapper applying non-reentrant activation
    checkpointing to one ViT encoder layer. Checkpoints ONLY when the layer is in
    training mode AND grad is enabled; eval/no_grad calls pass through unchanged
    (zero overhead). ``torch.utils.checkpoint.checkpoint`` is resolved by
    attribute at call time so its invocation is observable (and shimmable) in
    tests. ``use_reentrant=False`` is REQUIRED for DDP find_unused_parameters
    compatibility; the non-reentrant path also supports the kwargs DINOv3 layers
    receive (attention_mask, position_embeddings)."""
    @functools.wraps(orig_forward)
    def ckpt_forward(*args, **kwargs):
        if layer.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                orig_forward, *args, use_reentrant=False, **kwargs
            )
        return orig_forward(*args, **kwargs)
    return ckpt_forward


def _checkpoint_vit_layers(vit: nn.Module) -> int:
    """Wrap every encoder layer of ``vit`` (its ``.layer`` nn.ModuleList) with
    explicit non-reentrant activation checkpointing; returns the number of
    checkpointed layers.

    Codex review of the pinned transformers==4.57.0 rejected relying on HF's
    ``gradient_checkpointing_enable()`` here: DINOv3 delegates checkpointing to
    the ``GradientCheckpointingLayer.__call__`` base-class hook
    (transformers/modeling_layers.py), whose enable-time ``functools.partial``
    binds ``torch.utils.checkpoint.checkpoint`` opaquely and gates only on
    ``.training`` — invisible to instrumentation and easy to mistake for a no-op
    (``modeling_dinov3_vit.py`` itself never references
    ``_gradient_checkpointing_func``). Checkpointing each layer's forward at OUR
    call site keeps the semantics pinned, additionally gates on
    ``torch.is_grad_enabled()``, and is directly testable.

    Instance-level forward replacement only: registers no modules/params/buffers,
    so the state_dict stays byte-identical. Idempotent across the two
    ViTCoordinates conditioners that share one backbone (already-wrapped layers
    are skipped — never double-checkpointed). Fail-closed: a backbone without a
    non-empty ``nn.ModuleList`` at ``.layer`` (e.g. the in-repo
    SimpleViT/CylindricalViT) raises ValueError rather than silently ignoring
    the request."""
    layers = getattr(vit, "layer", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise ValueError(
            "gradient_checkpointing=True was requested for GeometryConditioner, "
            f"but the wrapped ViT backbone ({type(vit).__name__}) has no non-empty "
            "nn.ModuleList at .layer to checkpoint (expected an HF DINOv3-style "
            "encoder). Remove the flag or use a compatible backbone."
        )
    for layer in layers:
        if getattr(layer, "_flac_ckpt_wrapped", False):
            continue
        layer.forward = _make_vit_ckpt_forward(layer, layer.forward)
        layer._flac_ckpt_wrapped = True
    return len(layers)


class GeometryConditioner(Conditioner):
    def __init__(self, 
                 vit_model, 
                 vit_proj,
                 lin_proj,
                 output_dim: int,
                 max_value: float = 5.0,
                 dim: int = 512,
                 model_type: str = "vit",
                 token_pool: str = "linear",
                 dino_pool: str = "mean",
                 gradient_checkpointing: bool = False,
                 name="GeometryConditioner"):
        super().__init__(dim, output_dim, project_out=False)
        self.name = name
        self.vit = vit_model
        self.proj_out = vit_proj
        self.lin_proj = lin_proj
        self.max_value = max_value
        self.model_type = model_type
        self.token_pool = token_pool
        # exp_03: how the model_type='dino' path pools the backbone's token field before the
        # projection head. "mean" == the legacy `pooler_output` (patch mean) and is the DEFAULT,
        # so every existing caller is byte-identical; "max" reads `last_hidden_state.amax(dim=1)`
        # (the exp_03 max-pool ablation). Fail-closed on anything else (never a silent fallback).
        if dino_pool not in ("mean", "max"):
            raise ValueError(
                f"Unknown dino_pool: {dino_pool!r}. Supported: 'mean' (legacy patch mean via "
                "pooler_output) or 'max' (token-axis amax)."
            )
        self.dino_pool = dino_pool

        # Opt-in activation checkpointing for the ViT backbone: trades backward-time
        # recompute for a large activation-memory saving with numerically identical
        # weights/gradients (lets the C4 frame-averaging arm fit micro-batch 32 on a
        # 48GB card). Absent/False -> byte-identical to before. Wrapped explicitly at
        # our own call site rather than via HF's gradient_checkpointing_enable() —
        # see _checkpoint_vit_layers for the transformers==4.57.0 review finding.
        if gradient_checkpointing:
            self._vit_ckpt_layers = _checkpoint_vit_layers(self.vit)

    def forward(self, coord, device: tp.Union[torch.device, str] = "cuda") -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.vit.to(device)
        self.proj_out.to(device)

        depth_coords, coords = [], []
        for c in coord:
            coords.append(c['coord'].float().to(device))
            depth_coords.append(c['depth'].float().to(device))

        coord = torch.stack(coords, dim=0)  # [B, 3] or [B, N, 3]
        if coord.ndim == 2:
            coord = coord.unsqueeze(1) # [B, 1, 3]
        depth_coord = torch.stack(depth_coords, dim=0)

        encoded_coords = []
        for i in range(coord.shape[1]):
            c = (coord[:, i, :, None, None] - depth_coord) / self.max_value # [B, 3, H, W]
            if self.model_type == 'dino':
                outputs = self.vit(c)
                if self.dino_pool == "max":
                    # exp_03 max pool: [B, N_tok, H] -> [B, H]. Azimuth roll acts on the token
                    # axis as a permutation and amax over that axis is permutation-invariant,
                    # so the pooled vector keeps the backbone's yaw invariance.
                    pooled_output = outputs.last_hidden_state.amax(dim=1)
                else:
                    pooled_output = outputs.pooler_output
                c = self.lin_proj(pooled_output).unsqueeze(1)  # [B, 1, D]
            elif self.model_type == 'vit':
                c = self.vit(c) 
                c = self.proj_out(c) 
                if self.token_pool == "mean":
                    c = c.mean(dim=1, keepdim=True)  # [B, 1, D]
                elif self.token_pool == "linear":
                    c = self.lin_proj(c.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)  # [B, 1, D]
                else:
                    raise ValueError(f"Unknown token_pool: {self.token_pool}")
            else: 
                raise NotImplementedError('model_type must be either "dino" or "vit"')
            encoded_coords.append(c)
        out = torch.cat(encoded_coords, dim=1)  # [B, N, D]

        return [out, torch.ones(out.shape[0], 1).to(device)]

class DistEmbedderConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int,
                 project_out: bool = False,
                 max_val: float = 5.0,
                 funcs=[torch.sin, torch.cos], 
                 num_freqs=20, 
                 max_freq=10, 
                 ch_dim=1, 
                 include_in=True, 
                 name: str = "DistEmbedderConditioner", 
                 dist_embedder_proj: tp.Optional[torch.nn.Module] = None):
        
        in_dim = (len(funcs) * num_freqs + (1 if include_in else 0)) * 3
        super().__init__(in_dim, output_dim, project_out=False)
        self.funcs = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.include_in = include_in
        self.max_val = max_val
        self.name = name
        self.dist_embedder_proj = dist_embedder_proj

    def forward(self, x_input, device: tp.Union[torch.device, str] = "cuda") -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x_input = torch.stack(x_input, dim=0).to(device)

        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        outs = []
        for i in range(x_input.shape[1]):
            x = (x_input[:, i:(i+1)]) / self.max_val 
            if self.include_in:
                out_list = [x]
            else:
                out_list = []
            for func in self.funcs:
                for freq in self.freqs:
                    out_list.append(func(x*freq))
            out = torch.cat(out_list, dim=self.ch_dim).view(x_input.shape[0], -1)
            out = self.dist_embedder_proj(out)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        return [out, torch.ones(out.shape[0], 1).to(out.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str], only_ids: tp.Optional[tp.Iterable[str]] = None) -> tp.Dict[str, tp.Any]:
        # only_ids: when given, run (and return) exactly the conditioners whose id
        # is in this collection; default None keeps the full behaviour unchanged.
        # Used by yaw_rotation.invariant_conditioning to re-run only the ViT
        # conditioners per rotation frame without re-running stateful non-ViT ones.
        output = {}

        for key, conditioner in self.conditioners.items():
            if only_ids is not None and key not in only_ids:
                continue
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")
                
                if conditioner.name == 'GeometryConditioner':
                    add_input = 'depth'
                    if add_input not in x:
                        raise ValueError(f"Conditioner {key} requires depth input, but it is not present in the batch metadata")
                    else:
                        if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                            coord = x[condition_key][0]
                        else: 
                            coord = x[condition_key]
                    if isinstance(x[add_input], list) or isinstance(x[add_input], tuple) and len(x[add_input]) == 1:
                        conditioner_input = {'coord': coord, 'depth': x[add_input][0]}
                    else:
                        conditioner_input = {'coord': x[condition_key], 'depth': x[add_input]}

                else:
                    #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                    if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                        conditioner_input = x[condition_key][0]
                    else:
                        conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device=device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    vit_model = None
    dist_embedder_proj = None
    # The FIRST ViT block IFF the shared backbone was built via the cylindrical_dinov3
    # branch. Used only there for a defense-in-depth equality check on a second
    # ViTCoordinates conditioner (below); stays None for every legacy path, so the
    # reuse branch is byte-identical for legacy configs.
    _cyl_first_vit_block = None
    # exp_03/exp_04: how the dino path pools tokens. "mean" (the legacy patch mean) for EVERY
    # path; only the cylindrical branch's `cond_pool` knob can change it, via
    # _COND_POOL_TO_DINO_POOL ("max_mlp" -> "max"; "mean_mlp" keeps "mean"). Initialised here
    # so the (unchanged) shared-backbone reuse branch can pass it on for the second conditioner.
    dino_pool = 'mean'

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]
        conditioner_type = conditioner_info["type"]
        conditioner_config = {"output_dim": cond_dim}
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "rir":
            conditioners[id] = RIRConditioner(**conditioner_config)

        elif conditioner_type == "ViTCoordinates":
            if vit_model is None:
                vit_config = conditioner_config.pop("ViT", {})

                # exp-09 (Stage B): an explicit `implementation` field routes into the
                # cylindrical DINOv3 backbone (official weights + XYZ gauge). The field
                # being ABSENT preserves EXACT legacy behavior — every existing config
                # takes a byte-identical code path below. Any OTHER (non-absent) value
                # fails closed with a ValueError rather than silently falling through to
                # AutoModel (plan §2 / review r1 #4).
                implementation = vit_config.get("implementation", None)

                if implementation == "cylindrical_dinov3":
                    from cylindrical_dinov3 import CylindricalDINOv3ViTModel

                    model_name_or_path = vit_config.get('hf_model_name_or_path', None)
                    if model_name_or_path is None:
                        raise ValueError(
                            "implementation='cylindrical_dinov3' requires 'hf_model_name_or_path' "
                            "(the official DINOv3 weights to load)."
                        )
                    # no-SSL == official weights: from_scratch is unsupported in exp-09.
                    if vit_config.get('from_scratch', False):
                        raise ValueError(
                            "implementation='cylindrical_dinov3' does not support "
                            "from_scratch=True: the no-SSL exp-09 baseline loads the "
                            "official DINOv3 weights."
                        )
                    channels = vit_config.get('ch_dim', 3)
                    assert channels == 3, "Only 3 channels are supported"

                    # Audit-blessed gauge-ON default (Stage A: valid_pass). An absent
                    # `gauge` key defaults to "cylindrical_xyz" here (NOT the package's
                    # own "none" default), so exp-09 configs cannot silently run gauge-off.
                    gauge = vit_config.get('gauge', 'cylindrical_xyz')
                    print(f"Loading cylindrical_dinov3 ViT from {model_name_or_path} "
                          f"(gauge={gauge}, attn=eager)...")
                    vit_model = CylindricalDINOv3ViTModel.from_pretrained(
                        model_name_or_path, gauge=gauge, attn_implementation="eager",
                    )

                    if vit_config.get('freeze', False):
                        print('Freezing ViT model parameters...')
                        for param in vit_model.parameters():
                            param.requires_grad = False

                    hidden_size = vit_model.config.hidden_size

                    n_trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
                    n_total_params = sum(p.numel() for p in vit_model.parameters())
                    print(f"{n_trainable_params / 1e6:.2f}M/{n_total_params / 1e6:.2f}M parameters are trainable")

                    # exp_03/exp_04 (MLP conditioning head): two ADDITIVE, fail-closed knobs,
                    # parsed ONLY inside this cylindrical branch so no legacy config can reach
                    # them.
                    #   * `cond_pool` ABSENT  -> the legacy mean-pool + bare Linear head, on a
                    #     byte-identical code path (same modules, same RNG draws, same forward);
                    #   * `cond_pool: "max_mlp"`  -> token-axis amax + Linear->GELU->Linear;
                    #   * `cond_pool: "mean_mlp"` -> the LEGACY mean pool (pooler_output) + the
                    #     IDENTICAL Linear->GELU->Linear head, built by the SAME code below;
                    #   * any other value, an ORPHAN `cond_mlp_hidden` (no `cond_pool`), or a
                    #     non-int/bool/<=0 width -> ValueError (never a silent fallback).
                    # Both ViT blocks must carry equal values; the shared-backbone block-equality
                    # guard below is what enforces that.
                    cond_pool = vit_config.get('cond_pool', None)
                    cond_mlp_hidden = vit_config.get('cond_mlp_hidden', None)
                    # The isinstance guard keeps the membership test fail-closed for unhashable
                    # (list/dict) and non-string values, which `in` alone would raise TypeError on.
                    if cond_pool is not None and (not isinstance(cond_pool, str)
                                                  or cond_pool not in _COND_POOL_TO_DINO_POOL):
                        raise ValueError(
                            f"Unknown cond_pool: {cond_pool!r}. Supported: "
                            f"{sorted(_COND_POOL_TO_DINO_POOL)} ('max_mlp' = exp_03 max-pool + "
                            "MLP head, 'mean_mlp' = exp_04 mean-pool + the same MLP head), or "
                            "omit the field for the legacy mean-pool + Linear head (fail-closed: "
                            "an unrecognised cond_pool never falls through to the legacy head)."
                        )
                    if cond_pool is None and cond_mlp_hidden is not None:
                        raise ValueError(
                            "cond_mlp_hidden is set without cond_pool: the width would be "
                            "silently ignored by the legacy head. Set a registered cond_pool "
                            f"({sorted(_COND_POOL_TO_DINO_POOL)}) or remove cond_mlp_hidden."
                        )
                    if cond_pool is not None:
                        if cond_mlp_hidden is None:
                            cond_mlp_hidden = hidden_size   # declared default (384 at ViT-S/16)
                        if (isinstance(cond_mlp_hidden, bool)
                                or not isinstance(cond_mlp_hidden, int)
                                or cond_mlp_hidden <= 0):
                            raise ValueError(
                                f"cond_mlp_hidden must be a positive int, got "
                                f"{cond_mlp_hidden!r} ({type(cond_mlp_hidden).__name__})."
                            )
                        if cond_mlp_hidden != hidden_size:
                            raise ValueError(
                                f"cond_mlp_hidden must equal the backbone hidden_size "
                                f"({hidden_size}), got {cond_mlp_hidden}: the OUTPUT layer of the "
                                "MLP head is the legacy Linear(hidden_size, cond_dim) drawn at "
                                "the legacy code point (that is what keeps it bitwise-equal to the "
                                "legacy projection and leaves the downstream RNG stream intact), so "
                                "the hidden layer must map hidden_size -> hidden_size."
                            )

                    # model_type='dino' -> forward reads outputs.pooler_output (our patch
                    # mean, [B, hidden]) and projects it with lin_proj. vit_proj is unused
                    # on the dino path (Identity), matching the legacy DINO branch.
                    lin_proj = nn.Linear(hidden_size, cond_dim) if cond_dim != hidden_size else nn.Identity()
                    if cond_pool is not None:
                        # ONE construction for EVERY registered cond_pool value (exp_04 delta 1b):
                        # the arms must be the same network up to `dino_pool`, so this block is
                        # deliberately shared rather than copied per arm — a per-arm copy could
                        # drift in draw order and silently break the cross-arm identity oracle.
                        # Construction order is load-bearing (plan §2.1 / review r2-F1):
                        #   1. the OUTPUT layer is the line above -- the IDENTICAL draws the legacy
                        #      lin_proj makes, at the identical code point, so the post-Linear
                        #      global RNG state (hence every downstream module's init) is
                        #      byte-identical to legacy AND this layer is bitwise-equal to the
                        #      legacy projection;
                        #   2. the HIDDEN layer is drawn inside fork_rng(devices=[]) from a pinned
                        #      CPU-generator seed -- isolated draws, no device state touched.
                        #      `torch.random.default_generator.manual_seed` (NOT torch.manual_seed,
                        #      which also seeds CUDA/MPS/XPU that devices=[] does not restore).
                        out_layer = lin_proj
                        with torch.random.fork_rng(devices=[]):
                            torch.random.default_generator.manual_seed(_COND_MLP_HIDDEN_SEED)
                            hidden_layer = nn.Linear(hidden_size, cond_mlp_hidden)
                        lin_proj = nn.Sequential(hidden_layer, nn.GELU(), out_layer)
                        dino_pool = _COND_POOL_TO_DINO_POOL[cond_pool]
                    vit_proj = nn.Identity()
                    model_type = 'dino'
                    _cyl_first_vit_block = vit_config  # for the shared-backbone equality guard

                elif implementation is not None:
                    raise ValueError(
                        f"Unknown ViT implementation: {implementation!r}. Supported: "
                        "'cylindrical_dinov3', or omit the field for the legacy "
                        "AutoModel / cyl_vit / SimpleViT paths (fail-closed: an "
                        "unrecognised implementation never falls through to AutoModel)."
                    )

                # DINO Encoder
                elif vit_config.get('hf_model_name_or_path', None) is not None:
                    model_name_or_path = vit_config.get('hf_model_name_or_path', None)

                    if vit_config.get('from_scratch', False):
                        print(f"Loading ViT model from scratch: {model_name_or_path}...")
                        vit_model = AutoModel.from_config(AutoConfig.from_pretrained(model_name_or_path))
                    else:
                        print(f"Loading ViT model from {model_name_or_path}...")
                        vit_model = AutoModel.from_pretrained(model_name_or_path)

                    if vit_config.get('freeze', False):
                        print('Freezing ViT model parameters...')
                        for param in vit_model.parameters():
                            param.requires_grad = False
                    
                    if 'convnext' in model_name_or_path:
                        hidden_size = vit_model.config.hidden_sizes[-1]  
                        raise NotImplementedError("ConvNeXt-based conditioners are not currently tested and may require changes")  
                    else:
                        hidden_size = vit_model.config.hidden_size

                    channels=vit_config.get('ch_dim', 3)
                    assert channels == 3, "Only 3 channels are supported"
                    
                    n_trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
                    n_total_params = sum(p.numel() for p in vit_model.parameters())
                    print(f"{n_trainable_params / 1e6:.2f}M/{n_total_params / 1e6:.2f}M parameters are trainable")

                    lin_proj = nn.Linear(hidden_size, cond_dim) if cond_dim != hidden_size else nn.Identity()
                    vit_proj = nn.Identity()
                    model_type = 'dino'

                elif vit_config.get('arch') == 'cyl_vit':
                    vit_model = CylindricalViT(
                        in_channels=vit_config.get('ch_dim', 3),
                        image_size=(vit_config['img_h'], vit_config['img_w']),
                        patch_size=(vit_config['patch_h'], vit_config['patch_w']),
                        dim=vit_config.get('dim', 512),
                        depth=vit_config.get('depth', 12),
                        heads=vit_config.get('heads', 8),
                        dim_head=vit_config.get('dim_head', 64),
                        mlp_dim=vit_config.get('mlp_dim', vit_config.get('dim', 512)),
                        patch_embed_type=vit_config.get('patch_embed_type', 'linear'),
                    )
                    vit_proj = nn.Linear(vit_config.get('dim', 512), cond_dim) if cond_dim != vit_config.get('dim', 512) else nn.Identity()
                    lin_proj = nn.Linear(vit_model.num_tokens, 1)
                    model_type = 'vit'

                else: # Simple ViT Encoder (from xRIR)
                    vit_dim = vit_config.get('dim', 512)
                    vit_model = SimpleViT(
                        image_size=(vit_config['img_h'], vit_config['img_w']),
                        patch_size=(vit_config['patch_h'], vit_config['patch_w']),
                        dim=vit_dim,
                        depth=vit_config.get('depth', 12),
                        heads=vit_config.get('heads', 8),
                        mlp_dim=vit_config.get('mlp_dim', vit_dim),
                        channels=vit_config.get('ch_dim', 3),
                    )
                    vit_proj = nn.Linear(vit_dim, cond_dim) if cond_dim != vit_dim else nn.Identity()
                    num_tokens = (vit_config['img_h'] // vit_config['patch_h']) * (vit_config['img_w'] // vit_config['patch_w'])
                    lin_proj = nn.Linear(num_tokens, 1)
                    model_type = 'vit'
            else:
                second_vit_block = conditioner_config.pop("ViT", None)
                # Defense-in-depth (exp-09, Codex blocker 2c): when the shared backbone
                # was built by the cylindrical_dinov3 branch, a second ViTCoordinates
                # conditioner carrying a DIFFERENT ViT block is a silent-mismatch trap —
                # the factory reuses the first backbone and would otherwise discard the
                # divergent block unnoticed (e.g. a swapped implementation/gauge). Gated
                # strictly on the cylindrical territory (_cyl_first_vit_block is None for
                # every legacy config), so legacy behavior stays byte-identical.
                if (_cyl_first_vit_block is not None and second_vit_block is not None
                        and second_vit_block != _cyl_first_vit_block):
                    raise ValueError(
                        "cylindrical_dinov3: a second ViTCoordinates conditioner's ViT "
                        "block differs from the source_vit block that built the shared "
                        "backbone; the divergent block would be silently ignored. Make "
                        f"the blocks equal or remove the second one.\n  source_vit={_cyl_first_vit_block}"
                        f"\n  second={second_vit_block}"
                    )
            conditioners[id] = GeometryConditioner(**conditioner_config, vit_model=vit_model, vit_proj=vit_proj, lin_proj=lin_proj, model_type=model_type, dino_pool=dino_pool)

        elif conditioner_type == "dist_embedder":
            if dist_embedder_proj is None and not conditioner_config.get("init_cond", False): # share the same projection for all DistEmbedderConditioners
                in_channels = conditioner_config.get('in_channels', 3)
                dist_embedder_proj = nn.Linear((2 * conditioner_config['num_freqs'] + (1 if conditioner_config['include_in'] else 0)) * in_channels, cond_dim)
            conditioner_config.pop('in_channels', None)
            conditioners[id] = DistEmbedderConditioner(**conditioner_config, dist_embedder_proj=dist_embedder_proj)
        
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")), strict=True)

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)

        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)
