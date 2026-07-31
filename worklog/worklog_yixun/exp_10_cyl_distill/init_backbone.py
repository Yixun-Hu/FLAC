#!/usr/bin/env python3
"""exp_10 S3 extension: load the S2-distilled cylindrical backbone into a freshly-built
FLAC conditioning model, fail-closed (plan §5 's3').

Contract (all refusals SystemExit):
- artifact sha256 must equal the caller-pinned sha (from the S2 records);
- the model must expose conditioner.conditioners.{source_vit,context_poses_vit}.vit and
  BOTH must be the SAME object (the exp-09 shared-backbone invariant);
- that object must be a CylindricalDINOv3ViTModel;
- state-dict load is strict=True; key/shape mismatch refuses;
- non-backbone parameters are NOT touched (verified by the caller-visible return:
  n_loaded params + the backbone id, logged by train.py).
"""
import hashlib
import io
import sys

import torch


def die(msg):
    sys.exit(f"REFUSE(init-backbone): {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_distilled_backbone(model, artifact_path, pinned_sha):
    """Load the distilled state dict into the SHARED cylindrical backbone. Returns
    (n_params_loaded, backbone_object) for the caller's launch log.
    r5 #1: the file is read ONCE into memory; the hash and torch.load both consume that
    same immutable buffer — no hash-then-reopen TOCTOU window."""
    with open(artifact_path, "rb") as fh:
        blob = fh.read()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != pinned_sha:
        die(f"artifact sha {actual[:12]} != pinned {pinned_sha[:12]}")
    try:
        conds = model.conditioner.conditioners
        src_vit = conds["source_vit"].vit
        ctx_vit = conds["context_poses_vit"].vit
    except (AttributeError, KeyError) as exc:
        die(f"model does not expose the expected conditioner ViTs: {exc!r}")
    if src_vit is not ctx_vit:
        die("source_vit.vit and context_poses_vit.vit are NOT the same object — "
            "shared-backbone invariant violated; design re-review required")
    cls_name = type(src_vit).__name__
    if cls_name != "CylindricalDINOv3ViTModel":
        die(f"backbone class {cls_name!r} != CylindricalDINOv3ViTModel")
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    try:
        src_vit.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        die(f"strict load failed: {exc}")
    return sum(v.numel() for v in sd.values()), src_vit
