#!/usr/bin/env python3
"""Fail-closed audit for the FLAC -> CylindricalViT config substitution."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "src/configs/model_configs/FLAC/AR/FLAC_AR.json"
CYL_CONFIG = ROOT / "worklog/worklog_zhixuan/cyl_vit_test/FLAC_AR_CylViT.json"

CYL_VIT_BLOCK = {
    "arch": "cyl_vit",
    "ch_dim": 3,
    "freeze": False,
    "from_scratch": True,
    "img_h": 256,
    "img_w": 512,
    "patch_h": 16,
    "patch_w": 32,
    "dim": 512,
    "depth": 12,
    "heads": 8,
    "mlp_dim": 512,
    "patch_embed_type": "linear",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def expected_cyl_config(base: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(base)
    configs = expected["model"]["conditioning"]["configs"]
    by_id = {entry["id"]: entry for entry in configs}
    for conditioner_id in ("source_vit", "context_poses_vit"):
        cfg = by_id[conditioner_id]["config"]
        cfg["ViT"] = copy.deepcopy(CYL_VIT_BLOCK)
        cfg["token_pool"] = "mean"
    return expected


def first_difference(left: Any, right: Any, path: str = "root") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            difference = first_difference(lhs, rhs, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def validate_configs(base_path: Path = BASE_CONFIG, cyl_path: Path = CYL_CONFIG) -> dict[str, Any]:
    base = load_json(base_path)
    actual = load_json(cyl_path)
    expected = expected_cyl_config(base)
    difference = first_difference(actual, expected)
    if difference:
        raise ValueError(
            "CylindricalViT config differs from FLAC outside the registered ViT "
            f"substitution, or the substitution drifted: {difference}"
        )

    configs = actual["model"]["conditioning"]["configs"]
    by_id = {entry["id"]: entry for entry in configs}
    source = by_id["source_vit"]["config"]
    context = by_id["context_poses_vit"]["config"]
    if source != context:
        raise ValueError("source_vit and context_poses_vit must have identical configs")
    return actual


def instantiate_and_validate(config: dict[str, Any]) -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from src.models import create_model_from_config
    from src.models.cyl_vit import CylindricalViT

    model = create_model_from_config(config)
    source = model.conditioner.conditioners["source_vit"]
    context = model.conditioner.conditioners["context_poses_vit"]
    if not isinstance(source.vit, CylindricalViT):
        raise TypeError(f"source_vit built {type(source.vit)!r}, expected CylindricalViT")
    if source.vit is not context.vit:
        raise ValueError("source/context conditioners must share the same CylindricalViT")
    if source.lin_proj is not context.lin_proj:
        raise ValueError("source/context conditioners must share the same token pool")
    if source.token_pool != "mean" or context.token_pool != "mean":
        raise ValueError("registered CylindricalViT recipe requires mean token pooling")
    if source.vit.num_tokens != 256:
        raise ValueError(f"expected a 16x16 token grid (256 tokens), got {source.vit.num_tokens}")


def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def validate_assets(mode: str, vae_ckpt: Path) -> None:
    required: list[Path] = []
    if mode in {"train", "all"}:
        required.append(resolve_from_root(vae_ckpt))
    if mode in {"eval", "all"}:
        required.append(ROOT / "weights/AGREE/AGREE_fullAR.pt")
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required assets: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--assets", choices=("none", "train", "eval", "all"), default="none")
    parser.add_argument("--vae-ckpt", type=Path, default=Path("weights/FLAC/VAE.safetensors"))
    args = parser.parse_args()

    config = validate_configs()
    if args.instantiate:
        instantiate_and_validate(config)
    validate_assets(args.assets, args.vae_ckpt)
    print("CylindricalViT config audit: PASS")


if __name__ == "__main__":
    main()
