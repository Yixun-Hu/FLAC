#!/usr/bin/env python3
"""exp_10 S1: extract the ONE canonical teacher ViT from the pinned P1@55,000 ckpt.

Plan Rev 2 R2-3 / Rev 3 pins: the checkpoint's two ViT branches
(diffusion.conditioner.conditioners.{source_vit,context_poses_vit}.vit.*) are the SAME
shared backbone serialized twice (plan-r1 reviewer-verified). Fail-closed contract:
- ckpt file sha256 == the pinned teacher hash (Rev 3) before ANY load;
- exactly 211 keys under EACH prefix; identical suffix sets;
- pairwise torch.equal for every suffix (any mismatch => refuse: the shared-backbone
  premise would be false and the design needs re-review);
- output: outputs_FLAC/exp10_teacher/teacher_vit_p1s55000.pt ({suffix: tensor}, HF
  Dinov3ViTModel-loadable) + teacher_manifest.json (source sha, key/param counts,
  output sha256). Atomic, refuse-overwrite.
"""
import hashlib
import json
import os
import sys

import torch

PINNED_CKPT = "/home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl/outputs_FLAC/p1_sweep_import/epoch=12-step=55000.ckpt"
PINNED_SHA = "4b802899aa49783bb876252d252ccc2225b4cf57beb9fb3c53aecad37656c6a2"
P_SRC = "diffusion.conditioner.conditioners.source_vit.vit."
P_CTX = "diffusion.conditioner.conditioners.context_poses_vit.vit."
N_KEYS = 211


def die(msg):
    sys.exit(f"REFUSE: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(sd, n_keys=N_KEYS):
    """Pure extraction logic (unit-tested): fail-closed dual-prefix equality."""
    src = {k[len(P_SRC):]: v for k, v in sd.items() if k.startswith(P_SRC)}
    ctx = {k[len(P_CTX):]: v for k, v in sd.items() if k.startswith(P_CTX)}
    if len(src) != n_keys or len(ctx) != n_keys:
        die(f"prefix key counts {len(src)}/{len(ctx)} != {n_keys}")
    if set(src) != set(ctx):
        die("suffix sets differ between source_vit and context_poses_vit")
    for suf in src:
        if not torch.equal(src[suf], ctx[suf]):
            die(f"shared-backbone premise FALSE: tensors differ at {suf} — design re-review required")
    return {k: v.clone() for k, v in src.items()}


def main():
    ckpt = os.environ.get("EXP10_TEACHER_CKPT", PINNED_CKPT)   # env override is TEST-ONLY
    if ckpt != PINNED_CKPT and os.environ.get("P1TEST_SENTINEL") != "ckpt-parity-tests":
        die("EXP10_TEACHER_CKPT override without the test sentinel")
    out_dir = os.environ.get("EXP10_TEACHER_OUT",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", "..", "outputs_FLAC", "exp10_teacher"))
    if ckpt == PINNED_CKPT:
        actual = sha256_file(ckpt)
        if actual != PINNED_SHA:
            die(f"teacher ckpt sha {actual[:12]} != pinned {PINNED_SHA[:12]}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    teacher = extract(sd)
    n_params = sum(v.numel() for v in teacher.values())
    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pt = os.path.join(out_dir, "teacher_vit_p1s55000.pt")
    mf = os.path.join(out_dir, "teacher_manifest.json")
    for p in (pt, mf):
        if os.path.exists(p):
            die(f"output exists: {p}")
    torch.save(teacher, pt + ".tmp")
    os.rename(pt + ".tmp", pt)
    manifest = {"source_ckpt": ckpt, "source_sha256": PINNED_SHA if ckpt == PINNED_CKPT else sha256_file(ckpt),
                "n_keys": len(teacher), "n_params": n_params, "output_sha256": sha256_file(pt)}
    with open(mf + ".tmp", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.rename(mf + ".tmp", mf)
    print(f"teacher extracted: {len(teacher)} keys, {n_params:,} params, sha {manifest['output_sha256'][:12]}")


if __name__ == "__main__":
    main()
