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
  - cond_method == 'fa_invariant' and frame_avg_angles == (0.0,).

Explicit raises everywhere (survive ``python -O``). Run offline with HF_HUB_OFFLINE=1.

Run from repo root:  python worklog/worklog_yixun/exp_09_cyl_no_ssl/assert_arm_configs_exp09.py
"""
import hashlib
import json
import os
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

# Registered config delta vs FLAC_AR_BF.json (configs[1]=source_vit, [2]=context_poses_vit).
REGISTERED_ADDED = {
    "model.conditioning.configs[1].config.ViT.implementation",
    "model.conditioning.configs[1].config.ViT.gauge",
    "model.conditioning.configs[2].config.ViT.implementation",
    "model.conditioning.configs[2].config.ViT.gauge",
}
REGISTERED_CHANGED = {"training.frame_avg_angles"}


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


def assert_cyl_pin():
    import cylindrical_dinov3 as cyl
    if cyl.__version__ != CYL_VERSION:
        raise RuntimeError(f"cylindrical_dinov3 version {cyl.__version__!r} != pinned {CYL_VERSION!r}")
    if not cyl.__file__.startswith(CYL_PATH_PREFIX):
        raise RuntimeError(
            f"cylindrical_dinov3 imported from {cyl.__file__!r}, not the pinned prefix {CYL_PATH_PREFIX!r}"
        )
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(cyl.__file__))))
    sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    if sha not in CYL_ACCEPTED_SHAS:
        raise RuntimeError(
            f"cylindrical-dinov3 HEAD {sha!r} not in accepted set {sorted(CYL_ACCEPTED_SHAS)} — refuse"
        )
    print(f"[pin] cylindrical_dinov3 {CYL_VERSION} @ {sha[:12]}… (path OK) OK")


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


def assert_config_delta(exp09_cfg):
    with open(os.path.join(REPO, "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json")) as f:
        bf = json.load(f)
    changed, added, removed = _flatten_diff(bf, exp09_cfg)
    if set(added) != REGISTERED_ADDED:
        raise RuntimeError(f"config-delta ADDED keys {set(added)} != registered {REGISTERED_ADDED}")
    if set(changed) != REGISTERED_CHANGED:
        raise RuntimeError(f"config-delta CHANGED keys {set(changed)} != registered {REGISTERED_CHANGED}")
    if removed:
        raise RuntimeError(f"config-delta unexpected REMOVED keys {set(removed)}")
    print(f"[cfg] delta vs B-F is exactly the registered set ({len(added)} added, {len(changed)} changed) OK")


def main():
    snap = assert_vit_pin()
    assert_cyl_pin()

    from cylindrical_dinov3 import CylindricalDINOv3ViTModel, CylindricalXYZGauge

    cfg = json.load(open(os.path.join(HERE, "FLAC_AR_exp09.json")))
    assert_config_delta(cfg)

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
