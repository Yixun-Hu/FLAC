#!/usr/bin/env python3
"""Build paired exp06 CylViT initializations and audit their equality contract.

Only the geometry conditioner is initialized from scratch.  Every non-geometry
tensor is loaded exactly from the released FLAC EMA checkpoint.  The linear and
CNN variants then share identical random geometry-body/projection tensors; only
``vit.to_patch_embedding`` is allowed to differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
FLAC_ROOT = SCRIPT_DIR.parents[2]
if str(FLAC_ROOT) not in sys.path:
    sys.path.insert(0, str(FLAC_ROOT))

from src.models import create_model_from_config  # noqa: E402


DEFAULT_LINEAR_CONFIG = "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json"
DEFAULT_CNN_CONFIG = "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json"
DEFAULT_RELEASED_CKPT = "weights/FLAC/FLAC_EMA.ckpt"

GEOMETRY_PREFIXES = (
    "conditioner.conditioners.source_vit.",
    "conditioner.conditioners.context_poses_vit.",
)
STEM_MARKERS = (
    ".vit.to_patch_embedding.",
    ".vit.patch_embedding.",
    ".vit.patch_embed.",
)


def resolve_from_flac(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (FLAC_ROOT / candidate).resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_digest(state: Mapping[str, torch.Tensor], keys: list[str]) -> str:
    """Hash names, metadata and raw tensor bytes in a stable order."""
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        # ``view(dtype)`` rejects 0-D tensors (for example scalar Long
        # buffers).  Flattening first keeps the exact underlying bytes while
        # making the operation valid for tensors of every rank.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Checkpoint is missing a state_dict: {path}")
    raw = payload["state_dict"]
    if not isinstance(raw, dict):
        raise RuntimeError(f"Checkpoint state_dict is not a mapping: {path}")

    # Released FLAC_EMA.ckpt is already a bare model state dict.  The remapping
    # also accepts a Lightning wrapper so this script fails safely if the source
    # checkpoint is changed later.
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("diffusion_ema.") or key.startswith("losses."):
            continue
        new_key = key[len("diffusion.") :] if key.startswith("diffusion.") else key
        normalized[new_key] = value.detach().cpu()

    ema_prefix = "diffusion_ema.ema_model."
    ema_items = [(key, value) for key, value in raw.items() if key.startswith(ema_prefix)]
    for key, value in ema_items:
        if not isinstance(value, torch.Tensor):
            continue
        normalized["model." + key[len(ema_prefix) :]] = value.detach().cpu()
    return normalized


def is_geometry_key(key: str) -> bool:
    return key.startswith(GEOMETRY_PREFIXES)


def is_stem_key(key: str) -> bool:
    return is_geometry_key(key) and any(marker in key for marker in STEM_MARKERS)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def serializable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU tensors without cloning shared source/context parameter storage."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def assert_shared_vit(model: torch.nn.Module, variant: str) -> None:
    conditioners = model.conditioner.conditioners
    source = conditioners["source_vit"]
    context = conditioners["context_poses_vit"]
    if source.vit is not context.vit:
        raise AssertionError(f"{variant}: source_vit and context_poses_vit do not share one ViT")


def build_model(config_path: Path, seed: int, variant: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_everything(seed)
    config = load_json(config_path)
    model = create_model_from_config(config)
    assert_shared_vit(model, variant)

    vit = model.conditioner.conditioners["source_vit"].vit
    configured_type = getattr(vit, "patch_embed_type", None)
    if configured_type is not None and configured_type != variant:
        raise AssertionError(
            f"{variant}: config constructed patch_embed_type={configured_type!r}, expected {variant!r}"
        )
    return model, config


def load_released_non_geometry(
    model: torch.nn.Module,
    released_state: Mapping[str, torch.Tensor],
    variant: str,
) -> dict[str, int]:
    model_state = cpu_state_dict(model)
    model_non_geometry = {key for key in model_state if not is_geometry_key(key)}
    released_non_geometry = {key for key in released_state if not is_geometry_key(key)}

    missing = sorted(model_non_geometry - released_non_geometry)
    unexpected = sorted(released_non_geometry - model_non_geometry)
    mismatched = sorted(
        key
        for key in model_non_geometry & released_non_geometry
        if model_state[key].shape != released_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{variant}: released non-geometry load is not exact: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatch={mismatched[:8]}"
        )

    for key in model_non_geometry:
        model_state[key] = released_state[key].clone()
    model.load_state_dict(model_state, strict=True)

    return {
        "loaded_tensor_count": len(model_non_geometry),
        # state_dict contains both source/context aliases of the shared ViT.
        # This is therefore an audited state-element total, not a unique
        # trainable-parameter count.
        "loaded_state_element_count_with_aliases": sum(
            model_state[key].numel() for key in model_non_geometry
        ),
        "excluded_released_geometry_tensor_count": sum(
            1 for key in released_state if is_geometry_key(key)
        ),
    }


def copy_common_geometry(
    linear_model: torch.nn.Module,
    cnn_model: torch.nn.Module,
) -> dict[str, Any]:
    linear_state = cpu_state_dict(linear_model)
    cnn_state = cpu_state_dict(cnn_model)

    linear_common = {key for key in linear_state if is_geometry_key(key) and not is_stem_key(key)}
    cnn_common = {key for key in cnn_state if is_geometry_key(key) and not is_stem_key(key)}
    linear_stem = sorted(key for key in linear_state if is_stem_key(key))
    cnn_stem = sorted(key for key in cnn_state if is_stem_key(key))

    if not linear_stem or not cnn_stem:
        raise RuntimeError(
            "Could not identify both patch stems under vit.to_patch_embedding; "
            f"linear={linear_stem[:5]}, cnn={cnn_stem[:5]}"
        )
    if linear_common != cnn_common:
        raise RuntimeError(
            "The two variants differ outside the patch stem: "
            f"linear_only={sorted(linear_common - cnn_common)[:8]}, "
            f"cnn_only={sorted(cnn_common - linear_common)[:8]}"
        )

    mismatched = sorted(key for key in linear_common if linear_state[key].shape != cnn_state[key].shape)
    if mismatched:
        raise RuntimeError(f"Common geometry tensor shapes differ: {mismatched[:8]}")

    for key in linear_common:
        cnn_state[key] = linear_state[key].clone()
    cnn_model.load_state_dict(cnn_state, strict=True)

    # Re-read after loading and assert bitwise equality, rather than trusting the
    # assignment above (shared-module aliases appear twice in state_dict()).
    linear_state = cpu_state_dict(linear_model)
    cnn_state = cpu_state_dict(cnn_model)
    unequal = [key for key in sorted(linear_common) if not torch.equal(linear_state[key], cnn_state[key])]
    if unequal:
        raise AssertionError(f"Common geometry initialization is not bitwise equal: {unequal[:8]}")

    return {
        "common_geometry_keys": sorted(linear_common),
        "common_geometry_tensor_count": len(linear_common),
        "common_geometry_state_element_count_with_aliases": sum(
            linear_state[key].numel() for key in linear_common
        ),
        "common_geometry_sha256": tensor_mapping_digest(linear_state, sorted(linear_common)),
        "linear_stem": {
            key: {"shape": list(linear_state[key].shape), "dtype": str(linear_state[key].dtype)}
            for key in linear_stem
        },
        "cnn_stem": {
            key: {"shape": list(cnn_state[key].shape), "dtype": str(cnn_state[key].dtype)}
            for key in cnn_stem
        },
    }


def assert_non_geometry_equal(linear_model: torch.nn.Module, cnn_model: torch.nn.Module) -> dict[str, Any]:
    linear_state = cpu_state_dict(linear_model)
    cnn_state = cpu_state_dict(cnn_model)
    linear_keys = {key for key in linear_state if not is_geometry_key(key)}
    cnn_keys = {key for key in cnn_state if not is_geometry_key(key)}
    if linear_keys != cnn_keys:
        raise AssertionError("Non-geometry key sets differ between variants")
    unequal = [key for key in sorted(linear_keys) if not torch.equal(linear_state[key], cnn_state[key])]
    if unequal:
        raise AssertionError(f"Non-geometry tensors are not bitwise equal: {unequal[:8]}")
    return {
        "tensor_count": len(linear_keys),
        "state_element_count_with_aliases": sum(
            linear_state[key].numel() for key in linear_keys
        ),
        "sha256": tensor_mapping_digest(linear_state, sorted(linear_keys)),
    }


def checkpoint_paths(output_dir: Path, seed: int) -> dict[str, Path]:
    return {
        "linear": output_dir / f"cylvit_pe_linear_trainS{seed}_init.ckpt",
        "cnn": output_dir / f"cylvit_pe_cnn_trainS{seed}_init.ckpt",
        "audit": output_dir / f"matched_initialization_trainS{seed}_audit.json",
        "sha256sums": output_dir / f"matched_initialization_trainS{seed}_sha256sums.txt",
    }


def verify_existing(paths: Mapping[str, Path], expected: Mapping[str, Any]) -> bool:
    existing = {name: path.exists() for name, path in paths.items()}
    if not any(existing.values()):
        return False
    if not all(existing.values()):
        raise RuntimeError(f"Partial initialization artifact set exists; refusing to overwrite: {existing}")

    audit = load_json(paths["audit"])
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(
                f"Existing initialization audit has {key}={audit.get(key)!r}, expected {value!r}; "
                "use --force only after intentionally choosing to replace it"
            )
    for variant in ("linear", "cnn"):
        actual = sha256_file(paths[variant])
        recorded = audit["outputs"][variant]["sha256"]
        if actual != recorded:
            raise RuntimeError(f"Existing {variant} init SHA mismatch: {actual} != {recorded}")
    print(f"[exp06:init] verified and reused existing matched initializations in {paths['audit'].parent}")
    return True


def save_initialization(
    model: torch.nn.Module,
    path: Path,
    variant: str,
    seed: int,
    source_sha256: str,
    config_path: Path,
    config_sha256: str,
    common_geometry_sha256: str,
    non_geometry_sha256: str,
) -> None:
    metadata = {
        "format_version": 1,
        "experiment": "exp06_cylvit_patch_embed_ablation",
        "variant": variant,
        "training_seed": seed,
        "source_checkpoint_sha256": source_sha256,
        "model_config": str(config_path),
        "model_config_sha256": config_sha256,
        "excluded_released_prefixes": list(GEOMETRY_PREFIXES),
        "common_geometry_sha256": common_geometry_sha256,
        "non_geometry_sha256": non_geometry_sha256,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": serializable_state_dict(model), "exp06_init": metadata}, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-config", default=DEFAULT_LINEAR_CONFIG)
    parser.add_argument("--cnn-config", default=DEFAULT_CNN_CONFIG)
    parser.add_argument("--released-ckpt", default=DEFAULT_RELEASED_CKPT)
    parser.add_argument(
        "--output-dir",
        default=str(FLAC_ROOT / "outputs_FLAC" / "exp06_cylvit_pe_matched_initializations"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a complete existing initialization set. Partial sets always fail closed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linear_config = resolve_from_flac(args.linear_config)
    cnn_config = resolve_from_flac(args.cnn_config)
    released_ckpt = resolve_from_flac(args.released_ckpt)
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in (linear_config, cnn_config, released_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_sha256 = sha256_file(released_ckpt)
    config_sha256 = {
        "linear": sha256_file(linear_config),
        "cnn": sha256_file(cnn_config),
    }
    paths = checkpoint_paths(output_dir, args.seed)
    existing = {name: path.exists() for name, path in paths.items()}
    if any(existing.values()) and not all(existing.values()):
        raise RuntimeError(
            f"Partial initialization artifact set exists; refusing to overwrite even with --force: {existing}"
        )
    expected_existing = {
        "format_version": 1,
        "training_seed": args.seed,
        "source_checkpoint_sha256": source_sha256,
        "model_config_sha256": config_sha256,
    }
    if not args.force and verify_existing(paths, expected_existing):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        # Overwrites are intentional here; no unrelated path is ever removed.
        for path in paths.values():
            if path.exists():
                path.unlink()

    print("[exp06:init] constructing random linear geometry branch")
    linear_model, _ = build_model(linear_config, args.seed, "linear")
    print("[exp06:init] constructing random CNN geometry branch")
    cnn_model, _ = build_model(cnn_config, args.seed, "cnn")
    released_state = load_state_dict(released_ckpt)

    linear_load = load_released_non_geometry(linear_model, released_state, "linear")
    cnn_load = load_released_non_geometry(cnn_model, released_state, "cnn")
    if linear_load != cnn_load:
        raise AssertionError(f"Released-load reports differ: {linear_load} != {cnn_load}")

    common = copy_common_geometry(linear_model, cnn_model)
    non_geometry = assert_non_geometry_equal(linear_model, cnn_model)
    assert_shared_vit(linear_model, "linear")
    assert_shared_vit(cnn_model, "cnn")

    save_initialization(
        linear_model,
        paths["linear"],
        "linear",
        args.seed,
        source_sha256,
        linear_config,
        config_sha256["linear"],
        common["common_geometry_sha256"],
        non_geometry["sha256"],
    )
    save_initialization(
        cnn_model,
        paths["cnn"],
        "cnn",
        args.seed,
        source_sha256,
        cnn_config,
        config_sha256["cnn"],
        common["common_geometry_sha256"],
        non_geometry["sha256"],
    )

    output_info = {
        variant: {"path": str(paths[variant]), "sha256": sha256_file(paths[variant])}
        for variant in ("linear", "cnn")
    }
    audit = {
        "format_version": 1,
        "experiment": "exp06_cylvit_patch_embed_ablation",
        "training_seed": args.seed,
        "source_checkpoint": str(released_ckpt),
        "source_checkpoint_sha256": source_sha256,
        "model_configs": {"linear": str(linear_config), "cnn": str(cnn_config)},
        "model_config_sha256": config_sha256,
        "excluded_released_prefixes": list(GEOMETRY_PREFIXES),
        "released_load": linear_load,
        "shared_vit_asserted": {"linear": True, "cnn": True},
        "non_geometry_equality": non_geometry,
        "common_geometry_equality": common,
        "outputs": output_info,
    }
    with paths["audit"].open("w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with paths["sha256sums"].open("w") as handle:
        for variant in ("linear", "cnn"):
            handle.write(f"{output_info[variant]['sha256']}  {paths[variant].name}\n")
        handle.write(f"{sha256_file(paths['audit'])}  {paths['audit'].name}\n")

    print(f"[exp06:init] linear={paths['linear']} sha256={output_info['linear']['sha256']}")
    print(f"[exp06:init] cnn={paths['cnn']} sha256={output_info['cnn']['sha256']}")
    print(
        "[exp06:init] audit passed: "
        f"non_geometry={non_geometry['tensor_count']} tensors, "
        f"common_geometry={common['common_geometry_tensor_count']} tensors"
    )


if __name__ == "__main__":
    main()
