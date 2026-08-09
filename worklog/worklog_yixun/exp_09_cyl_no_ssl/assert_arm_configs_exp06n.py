#!/usr/bin/env python3
"""exp_06 (MAX-pool + the LEGACY bare-Linear conditioning head) pre-launch pin + ARM-WIRING gate.

This is ``assert_arm_configs_exp03n.py`` adapted to the exp_06 delta, with the exp_04
gate's pin-alias resolver (B4). It keeps ALL of the exp-09 provenance/clean-tree/HEAD/
official-weight machinery (imported, never re-implemented) and enforces the arm-wiring
list of plan Rev 2 B2:

  - DUAL PER-VARIANT config contract (B1): the bound config is (i) its exp03n sibling with
    EXACTLY ``cond_pool`` flipped ("max_linear" vs "max_mlp") and EXACTLY
    ``cond_mlp_hidden`` REMOVED, and (ii) its exp-09 counterpart plus EXACTLY ``cond_pool``
    in BOTH ViT blocks — deleting only that key must reconstruct the reference as a parsed
    object. Applied base<->base and online<->online, never across (the online-eval variant
    intentionally differs in ``use_ema`` and omits the gradient_checkpointing keys).
  - INSTANTIATED-MODEL arm wiring: both GeometryConditioners report ``dino_pool == 'max'``;
    the head is ONE SHARED bare ``Linear(384,256)`` — an MLP head (the exp_03 arm) is
    REFUSED, as is an unshared head or backbone.
  - PARAMETER-COUNT oracle: trainable-parameter delta vs legacy EXACTLY ZERO (pooling has
    no parameters; the head IS the legacy projection).
  - LIVE CPU FORWARD: the served condition equals ``Linear(amax(last_hidden_state, dim=1))``
    computed OUTSIDE the conditioner (bitwise), AND explicitly does NOT equal
    ``Linear(pooler_output)`` on the same input — the sabotage guard that makes the first
    assertion mean something (a silent revert to mean pooling leaves every shape, count AND
    the full state dict intact; only the live forward can catch it).
  - PER-USE GRADIENT FLOW: one backward per conditioner USE (grads zeroed in between) must
    deliver finite nonzero grads to the shared Linear THROUGH the max-pool path and to a
    backbone parameter.
  - BITWISE FULL-INIT IDENTITY (the exp_06 oracle, STRONGER than exp_03/04's): a seed-42
    full-model build and a seed-42 LEGACY build have IDENTICAL state dicts — exact key-set
    equality BOTH ways + ``torch.equal`` per tensor — and bitwise-equal post-construction
    global RNG states (the max_linear branch may not consume one extra draw).

Both config variants are accepted (``--config``); the variant is auto-detected from the
filename and can be forced with ``--config-variant``. ``EXPECT_PACKAGE_SHA`` and the
worktree pin (``EXPECT_EXP06_SHA``, with ``EXPECT_EXP09_SHA`` still accepted as the
inherited alias) are REQUIRED on the blessed path: no defaults, no registered fallback set.
The pin spellings are ALIASES OF ONE VALUE: if two of the four registered sources (either
CLI flag, either environment variable) are supplied with different SHAs, the gate refuses
and names them — it never resolves the ambiguity by precedence (B4, the exp_04 lesson).

CPU-only, offline. Run from anywhere:
    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_09_cyl_no_ssl/assert_arm_configs_exp06n.py
"""
import argparse
import copy
import json
import os
import subprocess
import sys

# CPU-only (mandate) + dodge the deepspeed op-probe, BEFORE torch is imported anywhere.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import assert_arm_configs_exp09 as exp09_gate  # noqa: E402  (sets up REPO + sys.path too)
import torch  # noqa: E402

from src.models import create_model_from_config  # noqa: E402
from src.models.conditioners import (  # noqa: E402
    create_multi_conditioner_from_conditioning_config,
)
from src.training import create_training_wrapper_from_config  # noqa: E402

REPO = exp09_gate.REPO

# ---- the exp_06 delta --------------------------------------------------------------- #
NEW_KEYS = {"cond_pool": "max_linear"}
# The exp_03 sibling's knob (what B1(i) flips/removes against).
SIBLING_KEYS = {"cond_pool": "max_mlp", "cond_mlp_hidden": 384}
# (exp06n config -> exp-09 reference it must reconstruct to, exp03n sibling per B1(i))
VARIANT_REFERENCE = {
    "base": ("FLAC_AR_exp06n.json", "FLAC_AR_exp09.json", "FLAC_AR_exp03n.json"),
    "online": ("FLAC_AR_exp06n_online_eval.json", "FLAC_AR_exp09_online_eval.json",
               "FLAC_AR_exp03n_online_eval.json"),
}
EXPECTED_PARAM_DELTA = 0   # pooling has no parameters; the head IS the legacy projection
OFFICIAL_HIDDEN = 384
COND_DIM = 256
CONDITIONER_ALIASES = ("source_vit", "context_poses_vit")

# ---- cylindrical-dinov3 package pin (THIS cluster) ---------------------------------- #
CYL_VERSION = "0.2.0"
# exp-09 pinned a /home/yixunhu/... prefix; the neuronic checkout lives here. Overridable
# via CYL_SRC_PREFIX for a relocated checkout (the HEAD pin below is the real protection).
CYL_PATH_PREFIX = os.environ.get(
    "CYL_SRC_PREFIX", "/n/fs/gatrdp/codespace/cylindrical-dinov3/src/cylindrical_dinov3/"
)


# ------------------------------------------------------------------------------------ #
# config contract (B1, BOTH bindings)
# ------------------------------------------------------------------------------------ #
def variant_for_config(config_path):
    """The registered variant of a config FILE: ``online`` iff the name says so."""
    return "online" if "online_eval" in os.path.basename(config_path) else "base"


def _vit_blocks(cfg):
    return [c["config"]["ViT"] for c in cfg["model"]["conditioning"]["configs"]
            if c["type"] == "ViTCoordinates"]


def assert_config_contract(config_path, variant=None):
    """B1: the bound config is (ii) its exp-09 counterpart + EXACTLY ``cond_pool`` in BOTH
    ViT blocks with ``cond_mlp_hidden`` ABSENT, and (i) its exp03n sibling up to EXACTLY
    {cond_pool value flipped, cond_mlp_hidden removed}. Returns ``(cfg, variant)``; raises
    RuntimeError on any deviation."""
    variant = variant or variant_for_config(config_path)
    if variant not in VARIANT_REFERENCE:
        raise RuntimeError(f"config-contract: unknown variant {variant!r} "
                           f"(expected one of {sorted(VARIANT_REFERENCE)})")
    _, reference_name, sibling_name = VARIANT_REFERENCE[variant]
    with open(config_path) as f:
        cfg = json.load(f)
    with open(os.path.join(HERE, reference_name)) as f:
        reference = json.load(f)
    with open(os.path.join(HERE, sibling_name)) as f:
        sibling = json.load(f)

    blocks = _vit_blocks(cfg)
    if len(blocks) != 2:
        raise RuntimeError(f"config-contract: expected 2 ViT blocks, got {len(blocks)}")
    for i, block in enumerate(blocks):
        for key, value in NEW_KEYS.items():
            if block.get(key, "<absent>") != value:
                raise RuntimeError(
                    f"config-contract[{variant}]: ViT block {i} has {key}="
                    f"{block.get(key, '<absent>')!r}, expected {value!r}"
                )
        if "cond_mlp_hidden" in block:
            raise RuntimeError(
                f"config-contract[{variant}]: ViT block {i} carries cond_mlp_hidden="
                f"{block['cond_mlp_hidden']!r} — the exp_06 bare-Linear head has no hidden "
                "layer; the key must be REMOVED (B1)"
            )
    if blocks[0] != blocks[1]:
        raise RuntimeError(
            "config-contract: the two ViT blocks are NOT deep-equal (the factory reuses the "
            f"first backbone and would discard the second):\n  {blocks[0]}\n  {blocks[1]}"
        )

    # (ii) minus cond_pool == the exp-09 reference, parsed-object-exactly
    stripped = copy.deepcopy(cfg)
    for c in stripped["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            for key in NEW_KEYS:
                del c["config"]["ViT"][key]
    if stripped != reference:
        changed, added, removed = exp09_gate._flatten_diff(reference, stripped)
        raise RuntimeError(
            f"config-contract[{variant}]: {os.path.basename(config_path)} minus cond_pool "
            f"does not reconstruct {reference_name} — the arm differs from exp-09 by MORE "
            f"than the pooling.\n  changed={changed}\n  added={added}\n  removed={removed}"
        )

    # (i) as-sibling transform == the exp03n counterpart, parsed-object-exactly
    as_sibling = copy.deepcopy(cfg)
    for c in as_sibling["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["ViT"].update(SIBLING_KEYS)
    if as_sibling != sibling:
        changed, added, removed = exp09_gate._flatten_diff(sibling, as_sibling)
        raise RuntimeError(
            f"config-contract[{variant}]: {os.path.basename(config_path)} differs from "
            f"{sibling_name} by MORE than {{cond_pool value, cond_mlp_hidden removal}}.\n"
            f"  changed={changed}\n  added={added}\n  removed={removed}"
        )
    print(f"[cfg] contract[{variant}] OK: {os.path.basename(config_path)} == {reference_name} "
          f"+ {NEW_KEYS} in BOTH ViT blocks (cond_mlp_hidden absent), and == {sibling_name} "
          "up to the registered sibling delta")
    return cfg, variant


# ------------------------------------------------------------------------------------ #
# package pin (this cluster)
# ------------------------------------------------------------------------------------ #
def assert_cyl_pin(expect_package_sha):
    import cylindrical_dinov3 as cyl

    if not expect_package_sha:
        raise RuntimeError(
            "EXPECT_PACKAGE_SHA absent — the blessed path requires the cylindrical-dinov3 HEAD "
            "pinned to an exact SHA (no registered fallback set on this cluster). Refuse."
        )
    if cyl.__version__ != CYL_VERSION:
        raise RuntimeError(f"cylindrical_dinov3 version {cyl.__version__!r} != pinned {CYL_VERSION!r}")
    if not cyl.__file__.startswith(CYL_PATH_PREFIX):
        raise RuntimeError(
            f"cylindrical_dinov3 imported from {cyl.__file__!r}, not the pinned prefix "
            f"{CYL_PATH_PREFIX!r}"
        )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cyl.__file__))))
    try:
        sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:  # noqa: BLE001 - fail closed on any git error
        raise RuntimeError(f"cylindrical-dinov3 HEAD unreadable at {repo}: {exc}")
    if sha != expect_package_sha:
        raise RuntimeError(
            f"cylindrical-dinov3 HEAD {sha!r} != EXPECT_PACKAGE_SHA {expect_package_sha!r} — refuse"
        )
    clean = exp09_gate.package_src_clean(repo)
    if clean is not True:
        raise RuntimeError(
            f"cylindrical-dinov3 {exp09_gate.PACKAGE_SRC_PATHSPEC} is not byte-clean "
            f"(clean={clean!r}) — dirty package source could contaminate the run; refuse"
        )
    print(f"[pin] cylindrical_dinov3 {CYL_VERSION} @ {sha[:12]}… (path OK, src clean) OK")


# ------------------------------------------------------------------------------------ #
# instantiated-model arm wiring (B2)
# ------------------------------------------------------------------------------------ #
def _geoms(mc):
    return [c for c in mc.conditioners.values()
            if getattr(c, "name", None) == "GeometryConditioner"]


def assert_head_wiring(mc, hidden_size=OFFICIAL_HIDDEN, cond_dim=COND_DIM):
    """Both conditioners MAX-pool and SHARE one bare ``Linear(hidden, cond_dim)`` head.
    Returns the head. Raises RuntimeError on any deviation (never an assert: survives -O)."""
    from torch import nn

    geoms = _geoms(mc)
    if len(geoms) != 2:
        raise RuntimeError(f"arm-wiring: expected 2 GeometryConditioners, got {len(geoms)}")
    for geom in geoms:
        pool = getattr(geom, "dino_pool", None)
        if pool != "max":
            raise RuntimeError(f"arm-wiring: conditioner dino_pool={pool!r}, expected 'max'")
        if geom.model_type != "dino":
            raise RuntimeError(f"arm-wiring: model_type={geom.model_type!r}, expected 'dino'")
    head = geoms[0].lin_proj
    if geoms[1].lin_proj is not head:
        raise RuntimeError(
            "arm-wiring: the two conditioners do not reference ONE shared head object "
            "(a duplicated head would diverge under training)"
        )
    if geoms[0].vit is not geoms[1].vit:
        raise RuntimeError("arm-wiring: the two conditioners do not share ONE backbone object")
    if type(head) is not nn.Linear:
        raise RuntimeError(
            f"arm-wiring: head is {head!r}, expected the LEGACY bare nn.Linear — an "
            "nn.Sequential MLP head would be the exp_03 arm (max pooling alone does not "
            "distinguish them); refuse"
        )
    if (head.in_features, head.out_features) != (hidden_size, cond_dim):
        raise RuntimeError(
            f"arm-wiring: head is Linear({head.in_features},{head.out_features}), expected "
            f"Linear({hidden_size},{cond_dim})"
        )
    print(f"[wire] dino_pool='max' on both conditioners; ONE shared bare "
          f"Linear({hidden_size},{cond_dim}) head OK")
    return head


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_param_delta(legacy_mc, new_mc, expected=EXPECTED_PARAM_DELTA):
    delta = n_trainable(new_mc) - n_trainable(legacy_mc)
    if delta != expected:
        raise RuntimeError(
            f"arm-wiring: trainable-parameter delta {delta} != {expected} (the exp_06 arm "
            "adds NO parameters: pooling is parameter-free and the head IS the legacy "
            "projection)"
        )
    print(f"[wire] trainable-parameter delta == legacy + {expected} exactly OK")
    return delta


def _synthetic_coord(height, width, batch=1, seed=0):
    """A deterministic, non-axisymmetric XYZ depth field + pose, in the shape
    GeometryConditioner.forward consumes."""
    import math

    j = torch.arange(width, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / width
    i = torch.arange(height, dtype=torch.float32)
    el = (i + 0.5) * math.pi / height - math.pi / 2.0
    theta_g = theta.view(1, width).expand(height, width)
    el_g = el.view(height, 1).expand(height, width)
    d = 3.0 + torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
    depth = torch.stack([d * torch.cos(el_g) * torch.cos(theta_g),
                         d * torch.cos(el_g) * torch.sin(theta_g),
                         d * torch.sin(el_g)], dim=0).contiguous()
    g = torch.Generator().manual_seed(seed)
    return [{"coord": torch.randn(1, 3, generator=g), "depth": depth} for _ in range(batch)]


def assert_live_forward(mc, height=256, width=512, seed=0):
    """B2(8): the SERVED condition equals ``Linear(amax(last_hidden_state, dim=1))`` computed
    outside the conditioner (bitwise: both sides run the same modules on the same input),
    and does NOT equal ``Linear(pooler_output)`` on that same input. The second half is the
    SABOTAGE guard: state-dict identity and equivariance both survive a silent revert to
    mean pooling, so only this live oracle can catch it."""
    geoms = _geoms(mc)
    coord = _synthetic_coord(height, width, batch=1, seed=seed)
    was_training = geoms[0].training
    mc.eval()
    try:
        with torch.no_grad():
            served = geoms[0](coord, device="cpu")[0]
            field = (coord[0]["coord"].float().unsqueeze(0)[:, 0, :, None, None]
                     - coord[0]["depth"].float().unsqueeze(0)) / geoms[0].max_value
            outputs = geoms[0].vit(field)
            oracle = geoms[0].lin_proj(outputs.last_hidden_state.amax(dim=1)).unsqueeze(1)
            sabotage = geoms[0].lin_proj(outputs.pooler_output).unsqueeze(1)
    finally:
        if was_training:
            mc.train()
    if served.shape != oracle.shape or not torch.equal(served, oracle):
        raise RuntimeError(
            "arm-wiring: the live forward does NOT equal Linear(amax(last_hidden_state, dim=1)) "
            f"(shapes {tuple(served.shape)} vs {tuple(oracle.shape)}, max abs diff "
            f"{(served - oracle).abs().max().item() if served.shape == oracle.shape else float('nan'):.3e}) "
            "— the served pooling is not max"
        )
    if torch.allclose(served, sabotage):
        raise RuntimeError(
            "arm-wiring: Linear(pooler_output) is INDISTINGUISHABLE from the served condition — "
            "the mean/max distinction is not being measured (the oracle above would pass for "
            "the legacy arm too); refuse"
        )
    print(f"[wire] live CPU forward == Linear(amax(tokens)) bitwise at {height}x{width}; "
          f"Linear(pooler_output) differs (sabotage guard) OK")


def assert_grad_flow(mc, height=256, width=512, seed=0):
    """B2(7): ONE backward per conditioner USE, grads zeroed in between — a summed backward
    would let a detached conditioner hide behind the other use of the SHARED head/backbone.
    Each use must deliver finite nonzero grads to the shared Linear (through the max-pool
    path) and to a backbone parameter."""
    geoms = _geoms(mc)
    if len(geoms) != 2:
        raise RuntimeError(f"grad-flow: expected 2 GeometryConditioners, got {len(geoms)}")
    head = geoms[0].lin_proj
    backbone_param = geoms[0].vit.layer[0].mlp.up_proj.weight
    if not backbone_param.requires_grad:
        raise RuntimeError("grad-flow: the probed backbone parameter is frozen — vacuous check")
    was_training = geoms[0].training
    mc.train()
    try:
        for use in (0, 1):
            mc.zero_grad(set_to_none=True)
            coord = _synthetic_coord(height, width, batch=1, seed=seed)
            geoms[use](coord, device="cpu")[0].float().pow(2).mean().backward()
            for name, param in (("head.weight", head.weight), ("head.bias", head.bias),
                                ("backbone", backbone_param)):
                grad = param.grad
                if grad is None:
                    raise RuntimeError(f"grad-flow: conditioner {use}: {name} got no grad")
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"grad-flow: conditioner {use}: {name} grad not finite")
                if grad.abs().max().item() <= 0.0:
                    raise RuntimeError(f"grad-flow: conditioner {use}: {name} grad all zeros")
    finally:
        mc.zero_grad(set_to_none=True)
        if not was_training:
            mc.eval()
    print("[wire] per-use gradient flow through the max-pool path (both conditioners, "
          "shared Linear + backbone) OK")


def assert_full_init_identity(new_cfg, legacy_cfg, seed=42):
    """B2(1-3), the exp_06 bitwise oracle: full-model construction under one seed — the
    state dicts are IDENTICAL (exact key-set equality both ways, ``torch.equal`` per
    tensor) and the post-construction global RNG states are bitwise equal."""
    torch.manual_seed(seed)
    legacy = create_model_from_config(copy.deepcopy(legacy_cfg)).state_dict()
    legacy_rng = torch.random.get_rng_state()
    torch.manual_seed(seed)
    new = create_model_from_config(copy.deepcopy(new_cfg)).state_dict()
    new_rng = torch.random.get_rng_state()

    only_new, only_legacy = set(new) - set(legacy), set(legacy) - set(new)
    if only_new:
        raise RuntimeError(f"full-init: keys only in the max_linear build: {sorted(only_new)} "
                           "— the arm added tensors it must not add")
    if only_legacy:
        raise RuntimeError(f"full-init: keys only in the legacy build: {sorted(only_legacy)} "
                           "— the arm dropped tensors it must not drop")
    for key in sorted(legacy):
        if not torch.equal(legacy[key], new[key]):
            raise RuntimeError(
                f"full-init: tensor {key} is NOT bitwise identical to the legacy seed-{seed} "
                "reference — the ablation would not be pooling-only"
            )
    if not torch.equal(legacy_rng, new_rng):
        raise RuntimeError(
            "full-init: the max_linear build left a DIFFERENT global RNG state than the "
            "legacy build — the branch consumed RNG it must not touch"
        )
    print(f"[init] {len(legacy)} tensors bitwise identical @ seed {seed} (key sets equal both "
          "ways); post-build global RNG states equal OK")


# ------------------------------------------------------------------------------------ #
def main(argv=None):
    parser = argparse.ArgumentParser(description="exp_06 (exp06n) pre-launch pin + arm-wiring gate")
    parser.add_argument("--config", default=os.path.join(HERE, VARIANT_REFERENCE["base"][0]),
                        help="the model-config the run will actually load (train/EMA or online-eval)")
    parser.add_argument("--config-variant", default=None, choices=("base", "online"),
                        help="force the registered reference (default: auto-detect from the filename)")
    parser.add_argument("--expect-package-sha", default=None,
                        help="REQUIRED pin for the cylindrical-dinov3 HEAD (env: EXPECT_PACKAGE_SHA)")
    parser.add_argument("--expect-exp06-sha", default=None,
                        help="REQUIRED pin for the FLAC worktree HEAD (env: EXPECT_EXP06_SHA)")
    # Inherited alias (B4): exp_03's tooling names the worktree pin EXPECT_EXP09_SHA;
    # both spellings are accepted so a copied command line cannot silently run unpinned.
    parser.add_argument("--expect-exp09-sha", default=None,
                        help="alias of --expect-exp06-sha (env: EXPECT_EXP09_SHA)")
    parser.add_argument("--allow-unpinned-exp09-sha", action="store_true",
                        help="non-blessed only: downgrade an absent worktree pin to a SKIP")
    parser.add_argument("--skip-full-model", action="store_true",
                        help="non-blessed only: skip the two full-model builds (full-init identity)")
    args = parser.parse_args(argv)

    expect_package_sha = args.expect_package_sha or os.environ.get("EXPECT_PACKAGE_SHA") or None

    # The worktree pin can arrive by FOUR registered spellings (B4: EXPECT_EXP06_SHA is this
    # arm's name, EXPECT_EXP09_SHA the inherited alias, each with a CLI flag). ANY two of
    # them carrying DIFFERENT values is a submission-environment mix-up -- typically a stale
    # export sitting beside a fresh one -- so the resolver refuses instead of silently
    # picking a precedence winner, and names every source it saw.
    pin_sources = (
        ("--expect-exp06-sha", args.expect_exp06_sha),
        ("--expect-exp09-sha", args.expect_exp09_sha),
        ("EXPECT_EXP06_SHA", os.environ.get("EXPECT_EXP06_SHA")),
        ("EXPECT_EXP09_SHA", os.environ.get("EXPECT_EXP09_SHA")),
    )
    supplied = [(name, value) for name, value in pin_sources if value]
    if len({value for _, value in supplied}) > 1:
        raise RuntimeError(
            "worktree pin: the registered spellings disagree — "
            + "; ".join(f"{name}={value!r}" for name, value in supplied)
            + ". They are aliases of ONE pin, so every supplied spelling must carry the SAME "
            "SHA; refuse rather than pick one."
        )
    expect_worktree_sha = supplied[0][1] if supplied else None

    # 1. provenance: worktree executable trees clean + HEAD pinned (exp-09 machinery)
    exp09_gate.assert_exp09_provenance(REPO, expect_worktree_sha,
                                       strict=not args.allow_unpinned_exp09_sha)

    # 2. official weights + package pin
    snap = exp09_gate.assert_vit_pin()
    assert_cyl_pin(expect_package_sha)

    # 3. per-variant DUAL config contract (binds the config the run will LOAD; B1)
    cfg, variant = assert_config_contract(args.config, args.config_variant)
    reference_name = VARIANT_REFERENCE[variant][1]

    from cylindrical_dinov3 import CylindricalDINOv3ViTModel, CylindricalXYZGauge

    # 4. strict-load provenance: HF load-info must list ALL empty
    _, info = CylindricalDINOv3ViTModel.from_pretrained(
        snap, gauge="cylindrical_xyz", attn_implementation="eager", output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise RuntimeError(f"HF load-info {key} is non-empty: {info[key][:5]}")
    print("[load] HF load-info lists all empty (missing/unexpected/mismatched/error_msgs) OK")

    # 5. arm wiring on the REAL instantiated conditioners
    mc = create_multi_conditioner_from_conditioning_config(copy.deepcopy(cfg["model"]["conditioning"]))
    geoms = _geoms(mc)
    vit = geoms[0].vit
    if type(vit) is not CylindricalDINOv3ViTModel:
        raise RuntimeError(f"ViT backbone is {type(vit).__name__}, not CylindricalDINOv3ViTModel")
    if not isinstance(getattr(vit, "gauge", None), CylindricalXYZGauge):
        raise RuntimeError("gauge module absent or wrong type")
    if vit.config.gauge != "cylindrical_xyz":
        raise RuntimeError(f"gauge is {vit.config.gauge!r}, expected 'cylindrical_xyz'")
    if vit.config._attn_implementation != "eager":
        raise RuntimeError(f"attn_implementation is {vit.config._attn_implementation!r}, expected 'eager'")
    if vit.config.hidden_size != OFFICIAL_HIDDEN:
        raise RuntimeError(f"hidden_size {vit.config.hidden_size} != {OFFICIAL_HIDDEN}")
    assert_head_wiring(mc, hidden_size=OFFICIAL_HIDDEN, cond_dim=COND_DIM)

    # 6. parameter-count oracle vs the LEGACY (exp-09 reference) build: delta EXACTLY zero
    with open(os.path.join(HERE, reference_name)) as f:
        legacy_cfg = json.load(f)
    legacy_mc = create_multi_conditioner_from_conditioning_config(
        copy.deepcopy(legacy_cfg["model"]["conditioning"])
    )
    assert_param_delta(legacy_mc, mc)
    del legacy_mc

    # 7. live forward == Linear(amax(tokens)), and != Linear(pooler_output) (sabotage guard)
    assert_live_forward(mc, height=256, width=512)

    # 8. per-use gradient flow through the max-pool path (B2 7)
    assert_grad_flow(mc, height=256, width=512)
    del mc

    # 9. bitwise FULL-init identity vs legacy (the exp_06 oracle), full models @ seed 42
    if args.skip_full_model:
        print("[init] full-init identity SKIPPED (--skip-full-model; non-blessed)")
    else:
        assert_full_init_identity(cfg, legacy_cfg, seed=42)

    # 10. the fa_invariant[0.0] training pin (unchanged recipe)
    torch.manual_seed(42)
    wrapper = create_training_wrapper_from_config(cfg, create_model_from_config(copy.deepcopy(cfg)))
    if wrapper.cond_method != "fa_invariant":
        raise RuntimeError(f"cond_method {wrapper.cond_method!r} != 'fa_invariant'")
    if wrapper.frame_avg_angles != (0.0,):
        raise RuntimeError(f"frame_avg_angles {wrapper.frame_avg_angles} != (0.0,)")
    print(f"[wrap] cond_method={wrapper.cond_method!r}  frame_avg_angles={wrapper.frame_avg_angles} OK")

    print("\nALL exp06n PRELAUNCH PINS PASSED — the max-pool + bare-Linear arm is wired, "
          "pinned, one-factor vs exp-09 and BITWISE identical to the legacy init.")


if __name__ == "__main__":
    main()
