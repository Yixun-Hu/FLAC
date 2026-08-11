
import functools
import hashlib
import os

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


def _default_local_root():
    """``<repo_root>/models``, derived from THIS FILE's location, never the CWD.

    ``src/models/conditioners.py`` -> ``src/models`` -> ``src`` -> repo root. Kept a
    named function so the lowest-priority root is one nameable thing (and so tests
    can pin the priority order against a synthetic root); behavior is identical to
    inlining the derivation.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "models")


def resolve_vit_model_path(name_or_path, local_root=None):
    """Prefer a local ViT snapshot directory over a bare hub id.

    WHY: the FLAC release itself trained its ViT from a *local directory* — the
    released checkpoint's embedded ``model_config`` carries
    ``./Models/dinov3-vits16-pretrain-lvd1689m`` — while this repo's JSON configs
    carry the portable hub id ``facebook/dinov3-vits16-pretrain-lvd1689m``. On an
    offline cluster node (della compute nodes have no network) a hub id cannot be
    fetched, so this maps the id onto the repo-level ``models/`` snapshot (a
    symlink into scratch there) *at load time*. The configs therefore stay
    byte-unchanged and checkpoints keep embedding the portable hub id.

    Root resolution order: an input that is already an existing directory wins
    outright; otherwise ``local_root`` argument -> ``$FLAC_LOCAL_MODEL_ROOT`` ->
    ``<repo_root>/models``, where ``repo_root`` is derived from THIS FILE's
    location (``src/models/`` -> ``src/`` -> repo root), never the CWD, so a job's
    working directory can never redirect the backbone. An id with no snapshot
    under any root — or one whose basename is unresolvable by construction
    (empty, ``.``, ``..``) — passes through unchanged (normal hub/cache behavior,
    and its offline error, then applies).

    Args:
        name_or_path: hub id or path, as written in the model config.
        local_root: optional highest-priority root to search.

    Returns:
        ``(resolved_path, source_tag)`` with ``source_tag`` one of
        ``explicit-dir``, ``local-root-arg``, ``env-root``, ``repo-root``,
        ``passthrough`` — so the call site can log which rule fired.
    """
    if os.path.isdir(name_or_path):
        return name_or_path, "explicit-dir"

    basename = os.path.basename(str(name_or_path).rstrip("/"))
    # "" / "." / ".." would join to the root itself or its PARENT: unresolvable by
    # construction, never a snapshot. Fail to passthrough rather than escape a root.
    if not basename or basename in (os.curdir, os.pardir):
        return name_or_path, "passthrough"

    roots = []
    if local_root is not None:
        roots.append((local_root, "local-root-arg"))
    env_root = os.environ.get("FLAC_LOCAL_MODEL_ROOT")
    if env_root:
        roots.append((env_root, "env-root"))
    roots.append((_default_local_root(), "repo-root"))

    for root, source_tag in roots:
        candidate = os.path.join(root, basename)
        if os.path.isdir(candidate):
            return candidate, source_tag

    return name_or_path, "passthrough"


def _vit_weights_provenance(resolved_path):
    """Size + sha256 prefix of ``<resolved_path>/model.safetensors``, or None.

    The run log's only record of WHICH ViT weights a job actually loaded. Returns
    None (never raises) when the file is absent or unreadable — config-only
    snapshot dirs and the ``from_scratch`` path must not be broken by logging.
    """
    weights = os.path.join(resolved_path, "model.safetensors")
    try:
        size = os.path.getsize(weights)
        h = hashlib.sha256()
        with open(weights, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return f"model.safetensors: {size} bytes, sha256:{h.hexdigest()[:16]}"


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

                # DINO Encoder
                if vit_config.get('hf_model_name_or_path', None) is not None:
                    model_name_or_path = vit_config.get('hf_model_name_or_path', None)

                    # Map a hub id onto a local snapshot when one exists (offline
                    # cluster nodes); pass-through otherwise. Resolve ONCE so both
                    # load branches see the same path, and log which rule fired.
                    resolved_path, resolve_source = resolve_vit_model_path(model_name_or_path)

                    if vit_config.get('from_scratch', False):
                        print(f"Loading ViT model from scratch: {model_name_or_path} -> {resolved_path} [{resolve_source}]...")
                        vit_model = AutoModel.from_config(AutoConfig.from_pretrained(resolved_path))
                    else:
                        print(f"Loading ViT model from {model_name_or_path} -> {resolved_path} [{resolve_source}]...")
                        vit_model = AutoModel.from_pretrained(resolved_path)

                    if resolved_path != model_name_or_path:
                        provenance = _vit_weights_provenance(resolved_path)
                        if provenance is not None:
                            print(f"Local ViT snapshot {resolved_path} {provenance}")

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
                conditioner_config.pop("ViT", None)
            conditioners[id] = GeometryConditioner(**conditioner_config, vit_model=vit_model, vit_proj=vit_proj, lin_proj=lin_proj, model_type=model_type)

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
