#!/usr/bin/env python3
"""exp_04 (MEAN-pool + the SAME MLP conditioning head) pre-launch pin + ARM-WIRING gate.

This is ``assert_arm_configs_exp03n.py`` adapted to the exp_04 delta. It keeps ALL of the
exp-09 provenance/clean-tree/HEAD/official-weight machinery (imported, never
re-implemented) and all of exp_03's fail-closed checks, with the pooling selector flipped
and ONE new oracle:

  - PER-VARIANT config contract: the bound config is its exp-09 counterpart plus EXACTLY
    the two conditioning keys in BOTH ViT blocks. Deleting ONLY those keys must
    reconstruct the reference as a parsed object -- ``FLAC_AR_exp04n.json`` ->
    ``FLAC_AR_exp09.json`` (training/EMA) and ``FLAC_AR_exp04n_online_eval.json`` ->
    ``FLAC_AR_exp09_online_eval.json`` (which intentionally differs in ``use_ema`` and
    omits the gradient_checkpointing keys). A single shared reference would hide a swap.
  - INSTANTIATED-MODEL arm wiring: both GeometryConditioners report ``dino_pool ==
    'mean'`` (the LEGACY pooling path -- exp_04 changes the head, not the pooling); the
    head is ONE SHARED object of exactly ``Linear(384,384) -> GELU -> Linear(384,256)``.
  - PARAMETER-COUNT oracle: trainable params == legacy + 147,840 EXACTLY (the hidden
    layer alone -- the output layer replaces, and is bitwise-equal to, the legacy
    98,560-param projection).
  - LIVE CPU FORWARD: the served condition equals ``head(pooler_output)`` computed OUTSIDE
    the conditioner (bitwise), AND explicitly does NOT equal ``head(amax(tokens))`` on the
    same input -- the sabotage guard that makes the first assertion mean something (a
    silent revert to max pooling leaves every shape and count intact).
  - BITWISE COMMON-INIT identity vs a legacy build under seed 42, INCLUDING the mapped
    comparison ``legacy .lin_proj.{weight,bias} == new .lin_proj.2.{weight,bias}`` for
    BOTH conditioner aliases.
  - CROSS-ARM IDENTITY (delta 1b, the factor-isolation proof): a same-seed ``mean_mlp``
    build and ``max_mlp`` build have BITWISE-EQUAL FULL state dicts (hidden AND output
    layers included) and bitwise-equal post-construction global RNG states. Together with
    ``dino_pool == 'mean'`` and the live-forward oracle this proves the pooling SELECTOR
    is the sole difference between exp_04 and exp_03.

Both config variants are accepted (``--config``); the variant is auto-detected from the
filename and can be forced with ``--config-variant``. ``EXPECT_PACKAGE_SHA`` and the
worktree pin (``EXPECT_EXP04_SHA``, with ``EXPECT_EXP09_SHA`` still accepted as the
inherited alias) are REQUIRED on the blessed path: no defaults, no registered fallback set.

CPU-only, offline. Run from anywhere:
    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_09_cyl_no_ssl/assert_arm_configs_exp04n.py
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

# ---- the exp_04 delta --------------------------------------------------------------- #
NEW_KEYS = {"cond_pool": "mean_mlp", "cond_mlp_hidden": 384}
# The exp_03 arm's knob value, used ONLY by the cross-arm identity oracle below.
SIBLING_COND_POOL = "max_mlp"
# (exp04n config -> the exp-09 reference it must reconstruct to)
VARIANT_REFERENCE = {
    "base": ("FLAC_AR_exp04n.json", "FLAC_AR_exp09.json"),
    "online": ("FLAC_AR_exp04n_online_eval.json", "FLAC_AR_exp09_online_eval.json"),
}
EXPECTED_PARAM_DELTA = 147840   # 384*384 + 384: the hidden layer ALONE (single oracle)
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
# config contract
# ------------------------------------------------------------------------------------ #
def variant_for_config(config_path):
    """The registered variant of a config FILE: ``online`` iff the name says so."""
    return "online" if "online_eval" in os.path.basename(config_path) else "base"


def _vit_blocks(cfg):
    return [c["config"]["ViT"] for c in cfg["model"]["conditioning"]["configs"]
            if c["type"] == "ViTCoordinates"]


def assert_config_contract(config_path, variant=None):
    """The bound config is its exp-09 counterpart + EXACTLY the two conditioning keys in
    BOTH ViT blocks. Returns ``(cfg, variant)``; raises RuntimeError on any deviation."""
    variant = variant or variant_for_config(config_path)
    if variant not in VARIANT_REFERENCE:
        raise RuntimeError(f"config-contract: unknown variant {variant!r} "
                           f"(expected one of {sorted(VARIANT_REFERENCE)})")
    reference_name = VARIANT_REFERENCE[variant][1]
    with open(config_path) as f:
        cfg = json.load(f)
    with open(os.path.join(HERE, reference_name)) as f:
        reference = json.load(f)

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
    if blocks[0] != blocks[1]:
        raise RuntimeError(
            "config-contract: the two ViT blocks are NOT deep-equal (the factory reuses the "
            f"first backbone and would discard the second):\n  {blocks[0]}\n  {blocks[1]}"
        )

    stripped = copy.deepcopy(cfg)
    for c in stripped["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            for key in NEW_KEYS:
                del c["config"]["ViT"][key]
    if stripped != reference:
        changed, added, removed = exp09_gate._flatten_diff(reference, stripped)
        raise RuntimeError(
            f"config-contract[{variant}]: {os.path.basename(config_path)} minus the two "
            f"conditioning keys does not reconstruct {reference_name} — the arm differs from "
            f"exp-09 by MORE than the head.\n  changed={changed}\n  added={added}\n  "
            f"removed={removed}"
        )
    print(f"[cfg] contract[{variant}] OK: {os.path.basename(config_path)} == {reference_name} "
          f"+ {NEW_KEYS} in BOTH ViT blocks")
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
# instantiated-model arm wiring
# ------------------------------------------------------------------------------------ #
def _geoms(mc):
    return [c for c in mc.conditioners.values()
            if getattr(c, "name", None) == "GeometryConditioner"]


def assert_head_wiring(mc, hidden_size=OFFICIAL_HIDDEN, cond_dim=COND_DIM):
    """Both conditioners MEAN-pool (the legacy pooling path) and SHARE one head of exactly
    Linear->GELU->Linear. Returns the head. Raises RuntimeError on any deviation (never an
    assert: survives -O)."""
    from torch import nn

    geoms = _geoms(mc)
    if len(geoms) != 2:
        raise RuntimeError(f"arm-wiring: expected 2 GeometryConditioners, got {len(geoms)}")
    for geom in geoms:
        pool = getattr(geom, "dino_pool", None)
        if pool != "mean":
            raise RuntimeError(f"arm-wiring: conditioner dino_pool={pool!r}, expected 'mean'")
        if geom.model_type != "dino":
            raise RuntimeError(f"arm-wiring: model_type={geom.model_type!r}, expected 'dino'")
    head = geoms[0].lin_proj
    if geoms[1].lin_proj is not head:
        raise RuntimeError(
            "arm-wiring: the two conditioners do not reference ONE shared head object "
            "(a duplicated head would double the new parameters and diverge under training)"
        )
    if geoms[0].vit is not geoms[1].vit:
        raise RuntimeError("arm-wiring: the two conditioners do not share ONE backbone object")
    if not isinstance(head, nn.Sequential) or len(head) != 3:
        raise RuntimeError(
            f"arm-wiring: head is {head!r}, expected a 3-module nn.Sequential (a bare Linear "
            "would be the LEGACY head: mean pooling alone does not distinguish the arms)"
        )
    if type(head[0]) is not nn.Linear or (head[0].in_features, head[0].out_features) != (hidden_size, hidden_size):
        raise RuntimeError(f"arm-wiring: head[0] is {head[0]!r}, expected Linear({hidden_size},{hidden_size})")
    if type(head[1]) is not nn.GELU:
        raise RuntimeError(f"arm-wiring: head[1] is {head[1]!r}, expected nn.GELU")
    if type(head[2]) is not nn.Linear or (head[2].in_features, head[2].out_features) != (hidden_size, cond_dim):
        raise RuntimeError(f"arm-wiring: head[2] is {head[2]!r}, expected Linear({hidden_size},{cond_dim})")
    print(f"[wire] dino_pool='mean' on both conditioners; ONE shared head "
          f"Linear({hidden_size},{hidden_size}) -> GELU -> Linear({hidden_size},{cond_dim}) OK")
    return head


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_param_delta(legacy_mc, new_mc, expected=EXPECTED_PARAM_DELTA):
    delta = n_trainable(new_mc) - n_trainable(legacy_mc)
    if delta != expected:
        raise RuntimeError(
            f"arm-wiring: trainable-parameter delta {delta} != {expected} (the hidden layer "
            "ALONE; the output layer replaces the legacy projection of identical shape)"
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
    """The SERVED condition equals ``head(pooler_output)`` computed outside the conditioner
    (bitwise: both sides run the same modules on the same input), and does NOT equal
    ``head(amax(last_hidden_state, dim=1))`` on that same input. The second half is the
    SABOTAGE guard: without it the assertion would also pass for the exp_03 max-pool arm,
    and a silent swap of the pooling selector leaves every shape and count intact."""
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
            oracle = geoms[0].lin_proj(outputs.pooler_output).unsqueeze(1)
            sabotage = geoms[0].lin_proj(outputs.last_hidden_state.amax(dim=1)).unsqueeze(1)
    finally:
        if was_training:
            mc.train()
    if served.shape != oracle.shape or not torch.equal(served, oracle):
        raise RuntimeError(
            "arm-wiring: the live forward does NOT equal head(pooler_output) "
            f"(shapes {tuple(served.shape)} vs {tuple(oracle.shape)}, max abs diff "
            f"{(served - oracle).abs().max().item() if served.shape == oracle.shape else float('nan'):.3e}) "
            "— the served pooling is not the mean"
        )
    if torch.allclose(served, sabotage):
        raise RuntimeError(
            "arm-wiring: head(amax(tokens)) is INDISTINGUISHABLE from the served condition — "
            "the mean/max distinction is not being measured (the oracle above would pass for "
            "the exp_03 arm too); refuse"
        )
    print(f"[wire] live CPU forward == head(pooler_output) bitwise at {height}x{width}; "
          f"head(amax(tokens)) differs (sabotage guard) OK")


def assert_common_init_identity(new_cfg, legacy_cfg, seed=42):
    """Full-model construction under one seed: every tensor the two arms SHARE is bitwise
    equal, only ``lin_proj.0.*`` (hidden) is new, and the renamed output layer
    ``lin_proj.2.*`` equals the legacy ``lin_proj.*`` for BOTH conditioner aliases."""
    torch.manual_seed(seed)
    legacy = create_model_from_config(copy.deepcopy(legacy_cfg)).state_dict()
    torch.manual_seed(seed)
    new = create_model_from_config(copy.deepcopy(new_cfg)).state_dict()

    aliases = [f"conditioner.conditioners.{a}" for a in CONDITIONER_ALIASES]
    expect_new = {f"{a}.lin_proj.{i}.{t}" for a in aliases for i in (0, 2) for t in ("weight", "bias")}
    expect_gone = {f"{a}.lin_proj.{t}" for a in aliases for t in ("weight", "bias")}
    only_new, only_legacy = set(new) - set(legacy), set(legacy) - set(new)
    if only_new != expect_new:
        raise RuntimeError(f"common-init: new-only keys {sorted(only_new)} != {sorted(expect_new)}")
    if only_legacy != expect_gone:
        raise RuntimeError(f"common-init: legacy-only keys {sorted(only_legacy)} != {sorted(expect_gone)}")

    for key in sorted(set(legacy) & set(new)):
        if not torch.equal(legacy[key], new[key]):
            raise RuntimeError(
                f"common-init: common tensor {key} is NOT bitwise identical to the legacy "
                f"seed-{seed} reference — the ablation would not be one-factor"
            )
    for alias in aliases:
        for tensor in ("weight", "bias"):
            if not torch.equal(legacy[f"{alias}.lin_proj.{tensor}"], new[f"{alias}.lin_proj.2.{tensor}"]):
                raise RuntimeError(
                    f"common-init: {alias} output layer lin_proj.2.{tensor} != the legacy "
                    f"lin_proj.{tensor} (the mapped comparison a key intersection would skip)"
                )
    print(f"[init] {len(set(legacy) & set(new))} common tensors bitwise identical @ seed {seed}; "
          f"only the hidden layer is new; mapped output layer equal for both aliases OK")


def assert_cross_arm_identity(cfg, seed=42):
    """CROSS-ARM IDENTITY ORACLE (delta 1b). Build the bound conditioning config, and the
    SAME config with ``cond_pool`` switched to the exp_03 value, under one seed: the FULL
    state dicts (hidden AND output layers included) must be bitwise equal and the
    post-construction GLOBAL RNG states must be bitwise equal. That is the proof that
    exp_04 differs from exp_03 by the pooling SELECTOR alone — if the head construction
    were copy-pasted per arm (a second fork_rng, a reordered draw) this fails."""
    mean_cond = copy.deepcopy(cfg["model"]["conditioning"])
    sibling_cond = copy.deepcopy(mean_cond)
    n = 0
    for c in sibling_cond["configs"]:
        if c["type"] == "ViTCoordinates":
            block = c["config"]["ViT"]
            if block.get("cond_pool") != NEW_KEYS["cond_pool"]:
                raise RuntimeError(
                    f"cross-arm: ViT block cond_pool={block.get('cond_pool')!r}, expected "
                    f"{NEW_KEYS['cond_pool']!r} before the sibling switch"
                )
            block["cond_pool"] = SIBLING_COND_POOL
            n += 1
    if n != 2:
        raise RuntimeError(f"cross-arm: switched {n} ViT blocks, expected 2")

    torch.manual_seed(seed)
    mean_mc = create_multi_conditioner_from_conditioning_config(copy.deepcopy(mean_cond))
    mean_rng = torch.random.get_rng_state()
    torch.manual_seed(seed)
    sibling_mc = create_multi_conditioner_from_conditioning_config(copy.deepcopy(sibling_cond))
    sibling_rng = torch.random.get_rng_state()

    mean_sd, sibling_sd = mean_mc.state_dict(), sibling_mc.state_dict()
    if set(mean_sd) != set(sibling_sd):
        raise RuntimeError(
            f"cross-arm: state-dict key sets differ: {sorted(set(mean_sd) ^ set(sibling_sd))}"
        )
    head_keys = [k for k in mean_sd if ".lin_proj.0." in k or ".lin_proj.2." in k]
    if not head_keys:
        raise RuntimeError("cross-arm: no MLP head tensors in the state dict — oracle vacuous")
    for key in sorted(mean_sd):
        if not torch.equal(mean_sd[key], sibling_sd[key]):
            raise RuntimeError(
                f"cross-arm: tensor {key} differs between the {NEW_KEYS['cond_pool']} and "
                f"{SIBLING_COND_POOL} builds at seed {seed} — the two arms are NOT the same "
                "network up to the pooling selector"
            )
    if not torch.equal(mean_rng, sibling_rng):
        raise RuntimeError(
            "cross-arm: the two arms left DIFFERENT global RNG states — the head construction "
            "is not the shared code path"
        )
    # non-vacuity: the two builds really are wired to different poolings
    pools = (getattr(_geoms(mean_mc)[0], "dino_pool", None),
             getattr(_geoms(sibling_mc)[0], "dino_pool", None))
    if pools != ("mean", "max"):
        raise RuntimeError(f"cross-arm: dino_pool pair {pools} != ('mean', 'max') — oracle vacuous")
    print(f"[cross] {len(mean_sd)} tensors ({len(head_keys)} head) bitwise identical between "
          f"{NEW_KEYS['cond_pool']} and {SIBLING_COND_POOL} @ seed {seed}; global RNG states "
          f"equal; dino_pool {pools[0]!r} vs {pools[1]!r} OK")


# ------------------------------------------------------------------------------------ #
def main(argv=None):
    parser = argparse.ArgumentParser(description="exp_04 (exp04n) pre-launch pin + arm-wiring gate")
    parser.add_argument("--config", default=os.path.join(HERE, VARIANT_REFERENCE["base"][0]),
                        help="the model-config the run will actually load (train/EMA or online-eval)")
    parser.add_argument("--config-variant", default=None, choices=("base", "online"),
                        help="force the registered reference (default: auto-detect from the filename)")
    parser.add_argument("--expect-package-sha", default=None,
                        help="REQUIRED pin for the cylindrical-dinov3 HEAD (env: EXPECT_PACKAGE_SHA)")
    parser.add_argument("--expect-exp04-sha", default=None,
                        help="REQUIRED pin for the FLAC worktree HEAD (env: EXPECT_EXP04_SHA)")
    # Inherited alias (delta 0): exp_03's tooling names the worktree pin EXPECT_EXP09_SHA;
    # both spellings are accepted so a copied command line cannot silently run unpinned.
    parser.add_argument("--expect-exp09-sha", default=None,
                        help="alias of --expect-exp04-sha (env: EXPECT_EXP09_SHA)")
    parser.add_argument("--allow-unpinned-exp09-sha", action="store_true",
                        help="non-blessed only: downgrade an absent worktree pin to a SKIP")
    parser.add_argument("--skip-full-model", action="store_true",
                        help="non-blessed only: skip the two full-model builds (common-init identity)")
    args = parser.parse_args(argv)

    expect_package_sha = args.expect_package_sha or os.environ.get("EXPECT_PACKAGE_SHA") or None
    expect_worktree_sha = (args.expect_exp04_sha or args.expect_exp09_sha
                           or os.environ.get("EXPECT_EXP04_SHA")
                           or os.environ.get("EXPECT_EXP09_SHA") or None)
    if (args.expect_exp04_sha and args.expect_exp09_sha
            and args.expect_exp04_sha != args.expect_exp09_sha):
        raise RuntimeError("--expect-exp04-sha and --expect-exp09-sha disagree — refuse")

    # 1. provenance: worktree executable trees clean + HEAD pinned (exp-09 machinery)
    exp09_gate.assert_exp09_provenance(REPO, expect_worktree_sha,
                                       strict=not args.allow_unpinned_exp09_sha)

    # 2. official weights + package pin
    snap = exp09_gate.assert_vit_pin()
    assert_cyl_pin(expect_package_sha)

    # 3. per-variant config contract (binds the config the run will LOAD)
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

    # 6. parameter-count oracle vs the LEGACY (exp-09 reference) build
    with open(os.path.join(HERE, reference_name)) as f:
        legacy_cfg = json.load(f)
    legacy_mc = create_multi_conditioner_from_conditioning_config(
        copy.deepcopy(legacy_cfg["model"]["conditioning"])
    )
    assert_param_delta(legacy_mc, mc)
    del legacy_mc

    # 7. live forward == head(pooler_output), and != head(amax(tokens)) (sabotage guard)
    assert_live_forward(mc, height=256, width=512)
    del mc

    # 8. bitwise common-init identity (one-factor discipline), full models @ seed 42
    if args.skip_full_model:
        print("[init] common-init identity SKIPPED (--skip-full-model; non-blessed)")
    else:
        assert_common_init_identity(cfg, legacy_cfg, seed=42)

    # 9. cross-arm identity vs the exp_03 arm (delta 1b — the factor-isolation proof)
    assert_cross_arm_identity(cfg, seed=42)

    # 10. the fa_invariant[0.0] training pin (unchanged recipe)
    torch.manual_seed(42)
    wrapper = create_training_wrapper_from_config(cfg, create_model_from_config(copy.deepcopy(cfg)))
    if wrapper.cond_method != "fa_invariant":
        raise RuntimeError(f"cond_method {wrapper.cond_method!r} != 'fa_invariant'")
    if wrapper.frame_avg_angles != (0.0,):
        raise RuntimeError(f"frame_avg_angles {wrapper.frame_avg_angles} != (0.0,)")
    print(f"[wrap] cond_method={wrapper.cond_method!r}  frame_avg_angles={wrapper.frame_avg_angles} OK")

    print("\nALL exp04n PRELAUNCH PINS PASSED — the mean-pool + MLP arm is wired, pinned, "
          "one-factor vs exp-09 and identical to exp_03 up to the pooling selector.")


if __name__ == "__main__":
    main()
