#!/usr/bin/env python3
"""exp-09 pre-launch pin gate (plan §2; extends exp_07's assert_arm_configs.py).

Fail-closed prelaunch check for the cylindrical no-SSL arm. Instantiates the exp-09
model + training wrapper from FLAC_AR_exp09.json (CPU, official DINOv3 weights from
the local HF cache, offline) and REFUSES to launch unless ALL of the following hold:

  - custom class identity: the ViT conditioner backbone is EXACTLY
    ``CylindricalDINOv3ViTModel`` (not the plain AutoModel DINOv3);
  - package provenance: ``cylindrical_dinov3`` imports from the pinned source path
    prefix, at the pinned version, and the cylindrical-dinov3 repo HEAD is one of the
    accepted (byte-identical-src) SHAs;
  - official-weight provenance: the HF cache holds EXACTLY the pinned DINOv3 snapshot
    (single revision) and its ``model.safetensors`` sha256 matches the pin;
  - eager attention (``config._attn_implementation == 'eager'``);
  - gauge == 'cylindrical_xyz' with the gauge module actually constructed;
  - HF load-info lists ALL empty (missing / unexpected / mismatched / error_msgs) —
    100% of the official weights load with strict compatibility;
  - shared backbone: source_vit and context_poses_vit reference the SAME object;
  - config delta vs FLAC_AR_BF.json is EXACTLY the registered set;
  - cond_method == 'fa_invariant' and frame_avg_angles == (0.0,);
  - scoped CLEAN-TREE for BOTH repos (integrative-review finding 3 / r2 blocker 1): the
    package's ``src/cylindrical_dinov3`` subtree AND the exp-09 worktree's executable trees
    (``src/`` + the exp09 dir + the executed ROOT files ``train.py``/``eval_FLAC.py``/
    ``defaults.ini``, untracked run logs/JSON excluded) are byte-clean;
  - exp-09 worktree HEAD PINNED to an exact SHA (r2 blocker 1): a different clean exp-09
    commit is refused. The pin is supplied by ``--expect-exp09-sha``/``EXPECT_EXP09_SHA``;
    unlike the package seam (which has a registered fallback set), an ABSENT exp-09 pin is a
    REFUSAL on the blessed (strict-default) path.

Explicit raises everywhere (survive ``python -O``). Run offline with HF_HUB_OFFLINE=1.
``--expect-package-sha``/``EXPECT_PACKAGE_SHA`` pins the accepted package HEAD to exactly one
SHA at records freeze (no code edit); otherwise the registered byte-identical-src set is
accepted. ``--allow-unpinned-exp09-sha`` downgrades an absent exp-09 pin to a non-blessed SKIP.

Run from repo root:  python worklog/worklog_yixun/exp_09_cyl_no_ssl/assert_arm_configs_exp09.py
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

# CPU-only (mandate) and dodge the deepspeed op-probe in this env when CUDA is
# visible-but-unusable. Set BEFORE torch is imported by the model factory.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402


def _repo_root(p):  # marker-walk (layout-proof; ``.git`` is a FILE in a worktree)
    p = os.path.abspath(p)
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # guard against a stale pip-installed src copy

from src.models import create_model_from_config  # noqa: E402
from src.models.conditioners import (  # noqa: E402
    create_multi_conditioner_from_conditioning_config,
)
from src.training import create_training_wrapper_from_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- official DINOv3 ViT-S/16 pin (identical to exp_07/Stage A) ----
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"

# ---- cylindrical-dinov3 package pin ----
CYL_VERSION = "0.2.0"
CYL_PATH_PREFIX = "/home/yixunhu/codespace/cylindrical-dinov3/src/cylindrical_dinov3/"
# Accepted repo HEADs: the Stage-A frozen SHA and the Stage-B-dispatch SHA. The
# packaged src/cylindrical_dinov3 is byte-identical across them (verified: the
# intervening commits are worklog/records only — `git diff --stat <a> <b> --
# src/cylindrical_dinov3/` is empty). Any OTHER HEAD is refused.
CYL_ACCEPTED_SHAS = {
    "1f2c015905980a070c01a9aebce68bdebe00dbd2",  # Stage A blessed-audit freeze
    "977c58439a581d497c78b286a71dceaa86085ded",  # Stage B dispatch
}

# ---- scoped clean-tree checks (integrative-review finding 3) ----
# The gate imports EXECUTABLE code from BOTH working trees, so it must refuse when that code is
# dirty (a dirty tree could retain an accepted HEAD yet contaminate C2). Reuses the audit gate's
# approach (audit_convention.py: git status --porcelain=v1 -z --untracked-files=all, faithfully
# reimplemented here so this launch gate stays dependency-light).
PACKAGE_SRC_PATHSPEC = "src/cylindrical_dinov3"                       # scope in the package repo
# Executed ROOT files (integrative-review r2 blocker 1): C1/C2 run `python train.py` (c1_fit,
# c1_smoke, exp09_launch); the screen runs `python eval_FLAC.py` and c1_smoke imports
# `eval_FLAC.check_load_integrity`; train.py sources its defaults from `defaults.ini` (prefigure
# get_all_args). These live at the repo ROOT (outside src/ and the exp09 dir) yet are executed, so
# a dirty one must block. finetune_cond.py is EXCLUDED: it is only named in eval_FLAC comments/help,
# never imported or invoked by the C1/C2/smoke path.
EXP09_EXECUTED_ROOT_FILES = ("train.py", "eval_FLAC.py", "defaults.ini")
EXP09_TREE_PATHSPECS = ("src", "worklog/worklog_yixun/exp_09_cyl_no_ssl") + EXP09_EXECUTED_ROOT_FILES
# NARROW exclusion for the worktree scope: an entry is ignorable ONLY when it is UNTRACKED ('??')
# AND its basename ends in .log or .json (a fresh run output). A TRACKED modification to ANY file
# - code OR a source .json config - always blocks; a brand-new untracked non-output file (e.g. a
# .py) also blocks; a rename is a tracked change and never excluded. Fail-closed by construction.
_OUTPUT_ARTIFACT_RE = re.compile(r".+\.(?:log|json)$", re.IGNORECASE)


def _parse_porcelain_z(out):
    """Parse ``git status --porcelain=v1 -z`` into entries (faithful to
    audit_convention._parse_porcelain_z): NUL-separated, unquoted; a rename/copy emits
    ``<dst>\\0<src>`` (both sides recorded); ``is_untracked`` iff the status is exactly ``??``."""
    tokens = out.split("\0")
    entries, i, n = [], 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "":
            i += 1
            continue
        status = tok[:2]
        dst = tok[3:]
        is_rename = status[0] in ("R", "C") or status[1] in ("R", "C")
        if is_rename:
            src = tokens[i + 1] if i + 1 < n else ""
            paths = [src, dst]
            i += 2
        else:
            src, paths = None, [dst]
            i += 1
        entries.append({"status": status, "src": src, "dst": dst, "paths": paths,
                        "is_untracked": status == "??", "is_rename": is_rename})
    return entries


def _scoped_status_entries(repo, pathspecs):
    """Scoped ``git status --porcelain=v1 -z --untracked-files=all -- <pathspecs...>`` entries, or
    None on any git failure (the caller then fails closed). ``--untracked-files=all`` defeats a
    repo-local ``status.showUntrackedFiles=no``."""
    try:
        out = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
             "--", *pathspecs],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:  # noqa: BLE001
        return None
    return _parse_porcelain_z(out)


def _entry_is_output_artifact(entry):
    """True iff an entry may be dropped from the WORKTREE executable-tree cleanliness set: an
    UNTRACKED ('??') file whose basename is a run output (.log/.json). Nothing else is excluded."""
    if entry.get("is_rename") or not entry.get("is_untracked"):
        return False
    base = entry["dst"].rsplit("/", 1)[-1]
    return _OUTPUT_ARTIFACT_RE.fullmatch(base) is not None


def package_src_clean(cyl_repo):
    """True iff the package's ``src/cylindrical_dinov3`` subtree is byte-clean (index OR work
    tree; both sides of a rename); None on git failure. ANY dirty path under the scope => not
    clean (the package source has no run outputs, so there is nothing to exclude)."""
    entries = _scoped_status_entries(cyl_repo, [PACKAGE_SRC_PATHSPEC])
    if entries is None:
        return None
    return len(entries) == 0


def exp09_tree_clean(repo):
    """True iff the exp-09 EXECUTABLE trees (``src/`` + the exp09 dir) are clean AFTER dropping
    ONLY untracked run-output artifacts; None on git failure. A tracked modification to any file
    (code OR a source config) and a new untracked non-output file both make it dirty."""
    entries = _scoped_status_entries(repo, list(EXP09_TREE_PATHSPECS))
    if entries is None:
        return None
    for e in entries:
        if _entry_is_output_artifact(e):
            continue
        return False
    return True


def accepted_shas(expect_package_sha=None):
    """The accepted package-HEAD set. ``--expect-package-sha`` (records freeze) pins it to EXACTLY
    that one SHA WITHOUT editing the code; otherwise the registered byte-identical-src set is used."""
    if expect_package_sha:
        return {expect_package_sha}
    return set(CYL_ACCEPTED_SHAS)


def exp09_head_sha(repo):
    """The exp-09 worktree HEAD SHA, or None on git failure (caller then fails closed)."""
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def assert_exp09_provenance(repo, expect_exp09_sha=None, strict=True):
    """exp-09 provenance gate (r2 blocker 1): the worktree executable trees (src/ + the exp09 dir
    + the executed ROOT files) must be CLEAN, and the worktree HEAD must be PINNED to an exact
    expected SHA. Unlike the package seam (which has a registered byte-identical-src fallback set),
    the exp-09 HEAD has NO fallback — so an ABSENT ``expect_exp09_sha`` is reported SKIPPED and, in
    strict mode (the blessed launch path), is treated as a REFUSAL. Raises on any failure."""
    clean = exp09_tree_clean(repo)
    if clean is not True:
        raise RuntimeError(
            f"exp-09 worktree executable trees {EXP09_TREE_PATHSPECS} not clean (clean={clean!r}) "
            "— dirty executable code (incl. executed root files) could contaminate C2; refuse"
        )
    head = exp09_head_sha(repo)
    if head is None:
        raise RuntimeError("exp-09 worktree HEAD unreadable (git failure) — refuse")
    if expect_exp09_sha:
        if head != expect_exp09_sha:
            raise RuntimeError(
                f"exp-09 worktree HEAD {head!r} != EXPECT_EXP09_SHA {expect_exp09_sha!r} — refuse"
            )
        print(f"[pin] exp-09 worktree clean + HEAD pinned to {head[:12]}… OK")
        return
    # absent expected SHA: SKIPPED, and in strict mode that SKIP is a refusal.
    print("[pin] exp-09 HEAD pin SKIPPED (EXPECT_EXP09_SHA absent)")
    if strict:
        raise RuntimeError(
            "EXPECT_EXP09_SHA absent — the blessed path requires the exp-09 worktree HEAD pinned to "
            "an exact SHA (records freeze sets it via env or --expect-exp09-sha). Refuse. "
            "(--allow-unpinned-exp09-sha downgrades this to a non-blessed SKIP.)"
        )

# Registered config delta vs FLAC_AR_BF.json (configs[1]=source_vit, [2]=context_poses_vit).
# EXACT path->value (not membership in a value-set: a swapped implementation<->gauge on
# the second block keeps both values 'cylindrical_*' yet is a real bug — Codex blocker 2).
REGISTERED_ADDED_VALUES = {
    "model.conditioning.configs[1].config.ViT.implementation": "cylindrical_dinov3",
    "model.conditioning.configs[1].config.ViT.gauge": "cylindrical_xyz",
    "model.conditioning.configs[2].config.ViT.implementation": "cylindrical_dinov3",
    "model.conditioning.configs[2].config.ViT.gauge": "cylindrical_xyz",
}
REGISTERED_ADDED = set(REGISTERED_ADDED_VALUES)
REGISTERED_CHANGED = {"training.frame_avg_angles"}
REGISTERED_FRAME_ANGLES = ([0.0, 90.0, 180.0, 270.0], [0.0])

# Online eval-variant ADDITIONAL delta vs FLAC_AR_BF.json (D-stage, Codex D-tool F2). The
# eval config (FLAC_AR_exp09_online_eval.json) drops gradient_checkpointing on BOTH ViT
# conditioners and flips use_ema false; the BASE (C2 train) config has NEITHER of these (it
# matches B-F's grad-ckpt=true / use_ema=true on those fields). So the delta set the pin gate
# must expect DIFFERS by variant — the gate binds the config the eval will actually load.
REGISTERED_ONLINE_REMOVED = {
    "model.conditioning.configs[1].config.gradient_checkpointing",
    "model.conditioning.configs[2].config.gradient_checkpointing",
}
REGISTERED_ONLINE_USE_EMA = (True, False)  # (B-F value, online-eval value)


def expected_config_delta(variant):
    """The (added_values, changed_values, removed_keys) delta vs FLAC_AR_BF.json expected for a
    config VARIANT. ``base`` = the C2 train config (FLAC_AR_exp09.json); ``online`` = the
    eval-time variant. Raises on an unknown variant (fail-closed)."""
    added = dict(REGISTERED_ADDED_VALUES)
    changed = {"training.frame_avg_angles": REGISTERED_FRAME_ANGLES}
    removed = set()
    if variant == "online":
        changed["training.use_ema"] = REGISTERED_ONLINE_USE_EMA
        removed = set(REGISTERED_ONLINE_REMOVED)
    elif variant != "base":
        raise RuntimeError(f"unknown config-variant {variant!r} (expected 'base' or 'online')")
    return added, changed, removed


def assert_vit_pin():
    from huggingface_hub.constants import HF_HUB_CACHE
    hub = os.path.join(HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
    snap_dir = os.path.join(hub, "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir} — refuse to launch")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(f"DINOv3 cache snapshots {snaps} != pinned [{VIT_REV!r}] — refuse")
    st = os.path.join(snap_dir, VIT_REV, "model.safetensors")
    h = hashlib.sha256()
    with open(st, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != VIT_SHA256:
        raise RuntimeError(f"DINOv3 model.safetensors sha256 {h.hexdigest()} != pinned {VIT_SHA256}")
    print(f"[pin] official DINOv3: single snapshot {VIT_REV[:12]}…, sha256 {VIT_SHA256[:12]}… OK")
    return os.path.join(snap_dir, VIT_REV)


def assert_cyl_pin(expect_package_sha=None):
    import cylindrical_dinov3 as cyl
    if cyl.__version__ != CYL_VERSION:
        raise RuntimeError(f"cylindrical_dinov3 version {cyl.__version__!r} != pinned {CYL_VERSION!r}")
    if not cyl.__file__.startswith(CYL_PATH_PREFIX):
        raise RuntimeError(
            f"cylindrical_dinov3 imported from {cyl.__file__!r}, not the pinned prefix {CYL_PATH_PREFIX!r}"
        )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cyl.__file__))))
    sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    accepted = accepted_shas(expect_package_sha)
    if sha not in accepted:
        raise RuntimeError(
            f"cylindrical-dinov3 HEAD {sha!r} not in accepted set {sorted(accepted)} — refuse"
        )
    # Scoped package-source cleanliness (finding 3): a dirty src/cylindrical_dinov3 could keep
    # the accepted HEAD yet ship different executable code into C2 — refuse (fail closed).
    clean = package_src_clean(repo)
    if clean is not True:
        raise RuntimeError(
            f"cylindrical-dinov3 {PACKAGE_SRC_PATHSPEC} is not byte-clean (clean={clean!r}) — "
            "dirty package source could contaminate C2; refuse"
        )
    print(f"[pin] cylindrical_dinov3 {CYL_VERSION} @ {sha[:12]}… (path OK, src clean) OK")


def _flatten_diff(a, b, prefix=""):
    changed, added, removed = {}, {}, {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = f"{prefix}.{k}" if prefix else k
            if k not in a:
                added[p] = b[k]
            elif k not in b:
                removed[p] = a[k]
            else:
                c, ad, rm = _flatten_diff(a[k], b[k], p)
                changed.update(c); added.update(ad); removed.update(rm)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            c, ad, rm = _flatten_diff(x, y, f"{prefix}[{i}]")
            changed.update(c); added.update(ad); removed.update(rm)
    else:
        if a != b:
            changed[prefix] = (a, b)
    return changed, added, removed


def assert_config_delta(exp09_cfg, variant="base"):
    exp_added, exp_changed, exp_removed = expected_config_delta(variant)
    with open(os.path.join(REPO, "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json")) as f:
        bf = json.load(f)
    changed, added, removed = _flatten_diff(bf, exp09_cfg)
    if set(added) != set(exp_added):
        raise RuntimeError(f"config-delta[{variant}] ADDED keys {set(added)} != registered {set(exp_added)}")
    if set(changed) != set(exp_changed):
        raise RuntimeError(f"config-delta[{variant}] CHANGED keys {set(changed)} != registered {set(exp_changed)}")
    if set(removed) != set(exp_removed):
        raise RuntimeError(f"config-delta[{variant}] REMOVED keys {set(removed)} != registered {set(exp_removed)}")
    # EXACT path->value (Codex blocker 2a): pin each added value precisely, not just its path.
    for path, value in exp_added.items():
        if added[path] != value:
            raise RuntimeError(f"config-delta[{variant}] {path} = {added[path]!r}, expected {value!r}")
    for path, value in exp_changed.items():
        if changed[path] != value:
            raise RuntimeError(f"config-delta[{variant}] {path} {changed[path]} != {value}")
    # Real-JSON block equality (Codex blocker 2b): the two ViT blocks must be deep-equal to
    # EACH OTHER, so a swapped/drifted second block (silently discarded by the factory) is caught.
    vit_blocks = [c["config"]["ViT"] for c in exp09_cfg["model"]["conditioning"]["configs"]
                  if c["type"] == "ViTCoordinates"]
    if len(vit_blocks) != 2:
        raise RuntimeError(f"expected 2 ViT blocks in the config, got {len(vit_blocks)}")
    if vit_blocks[0] != vit_blocks[1]:
        raise RuntimeError(
            "the two ViT blocks in the config are NOT deep-equal:\n  "
            f"source_vit={vit_blocks[0]}\n  context_poses_vit={vit_blocks[1]}"
        )
    print(f"[cfg] delta[{variant}] vs B-F is exactly the registered path->value set; both ViT "
          f"blocks deep-equal ({len(added)} added, {len(changed)} changed, {len(removed)} removed) OK")


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp-09 pre-launch pin gate")
    parser.add_argument(
        "--expect-package-sha", default=None,
        help="pin the accepted cylindrical-dinov3 HEAD to EXACTLY this one SHA (records freeze), "
             "instead of the registered byte-identical-src set — no code edit needed",
    )
    parser.add_argument(
        "--expect-exp09-sha", default=None,
        help="pin the exp-09 worktree HEAD to EXACTLY this SHA (records freeze). Mirrors the package "
             "seam's plumbing; unlike it, ABSENCE fails in strict mode (no fallback set exists).",
    )
    parser.add_argument(
        "--allow-unpinned-exp09-sha", action="store_true",
        help="non-blessed only: downgrade an absent EXPECT_EXP09_SHA from a refusal to a SKIP",
    )
    parser.add_argument(
        "--config", default=os.path.join(HERE, "FLAC_AR_exp09.json"),
        help="the model-config to pin/build (default the base C2 train config). D-stage evals pass "
             "the ACTUAL config the eval will load (base or the online-eval variant), so the delta "
             "check binds it (Codex D-tool F2).",
    )
    parser.add_argument(
        "--config-variant", default="base", choices=("base", "online"),
        help="which registered delta set applies to --config: 'base' (C2 train: grad-ckpt on, "
             "use_ema true) or 'online' (eval variant: grad-ckpt dropped, use_ema false).",
    )
    args = parser.parse_args(argv)

    # Env seam mirrors the package-SHA plumbing: c1_fit/exp09_launch/exp09_screen inherit these
    # (they cannot pass a new flag), while c1_smoke/records may pass the flag; flag overrides env.
    expect_package_sha = args.expect_package_sha or os.environ.get("EXPECT_PACKAGE_SHA") or None
    expect_exp09_sha = args.expect_exp09_sha or os.environ.get("EXPECT_EXP09_SHA") or None

    # exp-09 provenance (r2 blocker 1): scoped cleanliness over src/ + the exp09 dir + the executed
    # ROOT files (train.py, eval_FLAC.py, defaults.ini), AND an exact HEAD pin. Fail-closed BEFORE
    # building anything; absent HEAD pin is a refusal on the blessed (strict-default) path.
    assert_exp09_provenance(REPO, expect_exp09_sha, strict=not args.allow_unpinned_exp09_sha)

    snap = assert_vit_pin()
    assert_cyl_pin(expect_package_sha)

    from cylindrical_dinov3 import CylindricalDINOv3ViTModel, CylindricalXYZGauge

    cfg = json.load(open(args.config))
    print(f"[cfg] binding config {os.path.relpath(args.config, REPO)} as variant '{args.config_variant}'")
    assert_config_delta(cfg, args.config_variant)

    # --- direct strict-load provenance: HF load-info must list ALL empty ---
    _, info = CylindricalDINOv3ViTModel.from_pretrained(
        snap, gauge="cylindrical_xyz", attn_implementation="eager", output_loading_info=True,
    )
    for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(k):
            raise RuntimeError(f"HF load-info {k} is non-empty: {info[k][:5]}")
    print("[load] HF load-info lists all empty (missing/unexpected/mismatched/error_msgs) OK")

    # --- build the conditioner from the config's conditioning subtree; assert wiring ---
    mc = create_multi_conditioner_from_conditioning_config(cfg["model"]["conditioning"])
    geoms = [c for c in mc.conditioners.values() if getattr(c, "name", None) == "GeometryConditioner"]
    if len(geoms) != 2:
        raise RuntimeError(f"expected 2 GeometryConditioners, got {len(geoms)}")
    vit = geoms[0].vit
    if type(vit) is not CylindricalDINOv3ViTModel:
        raise RuntimeError(f"ViT backbone is {type(vit).__name__}, not CylindricalDINOv3ViTModel")
    if not isinstance(getattr(vit, "gauge", None), CylindricalXYZGauge):
        raise RuntimeError("gauge module absent or wrong type")
    if vit.config.gauge != "cylindrical_xyz":
        raise RuntimeError(f"gauge is {vit.config.gauge!r}, expected 'cylindrical_xyz'")
    if vit.config._attn_implementation != "eager":
        raise RuntimeError(f"attn_implementation is {vit.config._attn_implementation!r}, expected 'eager'")
    if vit.config.hidden_size != 384:
        raise RuntimeError(f"hidden_size {vit.config.hidden_size} != 384")
    if geoms[0].vit is not geoms[1].vit:
        raise RuntimeError("source_vit and context_poses_vit do not share ONE backbone object")
    print("[wire] custom class + gauge cylindrical_xyz + eager + hidden 384 + shared backbone OK")

    # --- full model + wrapper: cond_method / frame_avg_angles (the fa_invariant[0.0] pin) ---
    torch.manual_seed(42)
    model = create_model_from_config(cfg)
    wrapper = create_training_wrapper_from_config(cfg, model)
    if wrapper.cond_method != "fa_invariant":
        raise RuntimeError(f"cond_method {wrapper.cond_method!r} != 'fa_invariant'")
    if wrapper.frame_avg_angles != (0.0,):
        raise RuntimeError(f"frame_avg_angles {wrapper.frame_avg_angles} != (0.0,)")
    print(f"[wrap] cond_method={wrapper.cond_method!r}  frame_avg_angles={wrapper.frame_avg_angles} OK")

    print("\nALL exp-09 PRELAUNCH PINS PASSED — cylindrical no-SSL arm is wired and pinned.")


if __name__ == "__main__":
    main()
