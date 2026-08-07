#!/usr/bin/env python3
"""exp_11 Q5 — batched-orbit qualification probe on the REAL DINOv3 stack.

What this probe can and cannot prove (review findings 1, 2, 4):

* **EVAL-MODE EQUIVALENCE (gated).** With the model in ``eval()`` the DINOv3
  random RoPE rescale is off, so the batched execution and the legacy per-angle
  loop must agree numerically. This half is the acceptance gate: fp32 must pass
  BOTH a normwise-relative bound and a scale-aware max-absolute bound; bf16 is
  measured and recorded, not gated (bf16's unit roundoff near one is ~3.9e-3, so
  a max-elementwise bf16 bound is not a defensible pass/fail criterion).
  Coverage: B=8 (the pinned per-rank training batch) for C in {4,8,16,32}, plus
  the evaluation schedules B=64 (full batch) and B=1 (the 6,337-split tail case,
  the only evaluation batch whose grouping actually changes).

* **TRAIN-MODE QUALIFICATION (recorded, NOT gated; C4 only).** In train mode the RoPE
  rescale is drawn once per forward, so chunked angles share a draw where the
  loop gave them independent ones. That is a DISCLOSED RECIPE CHANGE, applied
  identically to every arm including the contemporaneous C4L bridge; it is not a
  numerical error to be tolerated. This half therefore asserts only what must
  hold — finite outputs and finite gradients through the batched orbit, with
  gradients enabled, ``torch.set_float32_matmul_precision('medium')`` as
  ``train.py`` sets it, and bf16-mixed autocast — and RECORDS the batched-vs-loop
  divergence for the disclosure record.

VRAM requalification is NOT this probe's job: the batched peak is measured by the
8x8 P0 spot cells (C4L/C8/C16/C32), which exercise the real train path including
backward and checkpoint recomputation via ``p0_runner.py``. That is also why the
train half stops at C4 — see ``TRAIN_ORBITS``.

Fail-closed: every expected cell must produce both ViT ids with all-finite
tensors, and an empty or short result set is a FAIL. Emits exactly one
machine-parseable ``EQUIVPROBE`` line.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys

# Gates (fp32, eval mode). Both must hold; the absolute floor keeps the relative
# metric out of the ill-conditioned near-zero regime.
#
# ===================== BOUND JUSTIFICATION (rel_norm) =========================
# ADJUSTED AFTER MEASUREMENT, 2026-08-06 — subject to final reviewer sign-off.
#
# The pre-registered rel_norm bound was 1e-6. It was breached, and the two
# breaches have different causes, only one of which is a defect:
#
#   attempt 4 (job 3646626, mm='medium')   3.479e-04 .. 5.415e-04 on all 8 B=1
#                                          cells; = TF32 unit roundoff 2^-11.
#                                          A REAL DEFECT: the gate compared a
#                                          1-row GEMV against a multi-row GEMM
#                                          with TF32 enabled, i.e. two different
#                                          precisions. Root-caused and removed by
#                                          running the gate at mm='highest'.
#   attempt 5 (job 3646634, mm='highest')  0.0 .. 1.979e-06 across all 24 gated
#                                          cells (6 above 1e-6: C4/C8/C16 at B8
#                                          and C4 at B1, both conditioner ids);
#                                          max_abs peaked at 7.987e-06, inside
#                                          the 1e-5 companion bound.
#                                          NOT a defect: shape-dependent fp32
#                                          summation order inside the kernels.
#                                          Expected scale sqrt(D) * 2^-24 at
#                                          D=384 is 1.17e-06 — the measured
#                                          envelope sits exactly there.
#
# So rel_norm moves to 5e-6: 2.5x headroom over the measured 1.979e-06 envelope,
# still ~70x (1.8 orders) below the smallest failure mode this gate exists to
# catch (the TF32 band above), and ~5.3 orders below a semantic slice/mapping
# error, which is O(1) and is separately bounded by the CPU angle-identity test
# (test_invariant_conditioning.py::test_batched_orbit_maps_every_angle_to_its_slice).
# max_abs is unchanged at 1e-5: it was never breached and it is the scale-aware
# companion that keeps a small-norm tensor from hiding a large single deviation.
# =============================================================================
TOL_REL_FP32 = 5e-6          # normwise: ||a-b|| / ||b||  (see the block above)
TOL_ABS_FP32 = 1e-5          # max |a-b|, scale-aware companion (unchanged)
REL_ABS_FLOOR = 1e-8         # elementwise rel = |a-b| / max(|b|, floor)
EVAL_ORBITS = (4, 8, 16, 32)
EVAL_TRAIN_BATCH = 8
EVAL_SCHEDULE_BATCHES = (64, 1)
# Train-mode qualification is C4 ONLY. A train cell holds a full gradient graph
# for BOTH the batched and the loop orbit; at C32 that is ~42.3 GiB and it OOMed
# inside the loop reference on a 46 GB L40 (job 3646616). That memory shape only
# exists distributed in the real run (micro-8 per rank x 8 ranks), so a
# single-GPU probe cannot host it and does not need to: C32's train-path memory
# and throughput are qualified by the 8x8 P0 spot cell on the real trainer, which
# is already a launch precondition.
TRAIN_ORBITS = (4,)
N_SAMPLES = 8

# Pinned DINOv3 initialiser (same constants as assert_arm_configs_exp11.py).
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
VIT_IDS = ("source_vit", "context_poses_vit")


def _repo_root(p):
    p = os.path.abspath(p)
    # `.git` is a DIRECTORY in a normal checkout and a FILE in a linked worktree —
    # measurements run from a pinned worktree, so both must count as the root.
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# pure functions (unit-tested in src/tests/test_exp11_equiv_probe.py)
# --------------------------------------------------------------------------- #
def orbit(n):
    """The uniform Cn orbit in degrees."""
    return tuple(k * 360.0 / n for k in range(n))


def precision_for(mode):
    """Matmul precision policy per cell mode.

    Job 3646626 failed every B=1 fp32 cell at rel_norm 3.5e-4..5.4e-4 while B=8
    and B=64 were clean at ~1.3e-7. That band IS TF32's unit roundoff (2^-11 =
    4.88e-4): the probe had copied train.py's global
    ``set_float32_matmul_precision('medium')``, which sets
    ``torch.backends.cuda.matmul.allow_tf32 = True``, so the "fp32" gate was not
    fp32. A 1-row matmul is a GEMV and does not take the reduced-precision GEMM
    path, while the batched side's 3..31-row matmul does — so the two sides ran
    at DIFFERENT precisions and the gate measured cuBLAS kernel policy, not
    batching. (CPU repro at B=1 is exact; see
    test_invariant_conditioning.py::test_batched_orbit_maps_every_angle_to_its_slice.)

    So: the EQUIVALENCE GATE runs in true fp32 ('highest', TF32 off everywhere),
    and only the TRAIN qualification cell keeps train.py's 'medium', because
    there the point is to mirror the training path rather than to compare
    numbers."""
    return "medium" if mode == "train" else "highest"


@contextlib.contextmanager
def matmul_precision(mode):
    """Set the fp32 matmul precision (and the cuDNN TF32 flag, which
    ``set_float32_matmul_precision`` does NOT touch) for one cell, then restore."""
    import torch
    prev = torch.get_float32_matmul_precision()
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision(mode)
    torch.backends.cudnn.allow_tf32 = (mode != "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def expected_cells(eval_orbits=EVAL_ORBITS, train_batch=EVAL_TRAIN_BATCH,
                   schedule_batches=EVAL_SCHEDULE_BATCHES, train_orbits=TRAIN_ORBITS):
    """The exact cell set the probe must produce, as ``(mode, n, batch)`` keys."""
    cells = [("eval", n, train_batch) for n in eval_orbits]
    cells += [("eval", n, b) for b in schedule_batches for n in eval_orbits]
    cells += [("train", n, train_batch) for n in train_orbits]
    return tuple(cells)


def deviation(a, b):
    """``(max_abs, rel_norm, rel_max)`` between two tensors, NaN-proof.

    ``rel_norm`` is normwise ``||a-b|| / ||b||`` (well conditioned); ``rel_max``
    is the elementwise ratio against ``max(|b|, REL_ABS_FLOOR)`` so a near-zero
    reference cannot manufacture a huge ratio. A non-finite input yields ``inf``
    rather than a suppressed NaN."""
    import torch
    a = a.detach().float()
    b = b.detach().float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("inf"), float("inf"), float("inf")
    diff = (a - b).abs()
    max_abs = float(diff.max())
    denom_norm = float(b.norm())
    rel_norm = float(diff.norm()) / denom_norm if denom_norm > 0 else float(diff.norm())
    rel_max = float((diff / b.abs().clamp_min(REL_ABS_FLOOR)).max())
    return max_abs, rel_norm, rel_max


def verdict(results, expected, tol_rel=TOL_REL_FP32, tol_abs=TOL_ABS_FP32):
    """``(ok, reasons)`` for the collected results.

    Fail-closed: every expected cell must be present with BOTH ViT ids and finite
    metrics; every gated (fp32 eval) cell must satisfy both bounds. Train-mode
    cells are recorded, and only their finiteness is required."""
    reasons = []
    missing = [c for c in expected if c not in results]
    if missing:
        reasons.append(f"missing cells: {sorted(missing)}")
    for cell in expected:
        res = results.get(cell)
        if res is None:
            continue
        ids = sorted(res.get("ids", {}))
        if ids != sorted(VIT_IDS):
            reasons.append(f"{cell}: ViT ids {ids} != {sorted(VIT_IDS)}")
            continue
        for vit, m in res["ids"].items():
            if not all(math.isfinite(v) for v in (m["max_abs"], m["rel_norm"], m["rel_max"])):
                reasons.append(f"{cell}/{vit}: non-finite metric {m}")
                continue
            if not res.get("finite", True):
                reasons.append(f"{cell}/{vit}: non-finite tensors in the forward")
            if res["gated"]:
                if m["rel_norm"] > tol_rel:
                    reasons.append(f"{cell}/{vit}: rel_norm {m['rel_norm']:.3e} > {tol_rel:g}")
                if m["max_abs"] > tol_abs:
                    reasons.append(f"{cell}/{vit}: max_abs {m['max_abs']:.3e} > {tol_abs:g}")
    if not results:
        reasons.append("no results collected")
    return (not reasons), reasons


def summarize(results, key):
    """Worst value of ``key`` over gated cells and over recorded cells."""
    gated, recorded = [0.0], [0.0]
    for res in results.values():
        for m in res.get("ids", {}).values():
            (gated if res["gated"] else recorded).append(m[key])
    return max(gated), max(recorded)


def record_id(meta, idx):
    """A stable EXACT identifier for one dataset record: ``<idx>:<relpath>``.

    The scene label is NOT an identifier — eight different records of one room
    all carry ``scene='Cafe'``, which is what the earlier probe emitted eight
    times over. The dataset exposes ``idx``, ``path`` and ``relpath``; the
    relative path (falling back to the basename, then the raw index) is the
    per-record key, and the loader index makes it order-explicit."""
    rel = meta.get("relpath")
    if not rel:
        raw = meta.get("path")
        rel = os.path.basename(str(raw)) if raw else None
    if not rel:
        rel = f"record{meta.get('idx', idx)}"
    return f"{int(meta.get('idx', idx))}:{rel}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_vit_pin(hf_cache=None):
    """The same pinned-DINOv3 gate the arm launcher runs (offline mode alone does
    not bind external-cache identity)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    root = hf_cache or HF_HUB_CACHE
    snap_dir = os.path.join(root, "models--facebook--dinov3-vits16-pretrain-lvd1689m", "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir}")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(f"DINOv3 snapshots {snaps} != pinned [{VIT_REV}]")
    got = sha256_file(os.path.join(snap_dir, VIT_REV, "model.safetensors"))
    if got != VIT_SHA256:
        raise RuntimeError(f"DINOv3 weight sha256 {got} != pinned {VIT_SHA256}")
    return f"{VIT_REV[:12]}/{VIT_SHA256[:12]}"


# --------------------------------------------------------------------------- #
# real stack + deterministic real data
# --------------------------------------------------------------------------- #
def build_conditioner(config_path, device):
    from src.models.factory import create_model_from_config
    cfg = json.load(open(config_path))
    model = create_model_from_config(cfg)
    cond = model.conditioner.to(device)
    return cfg, cond


def real_samples(dataset_config_path, n_samples, device, seed=42):
    """The FIRST ``n_samples`` dataset items, deterministically.

    The shared dataloader factory forces ``persistent_workers=True`` (which
    rejects ``num_workers=0``) and shuffles, so the probe indexes the dataset
    object directly instead: no sampler, no workers, no augmentation ordering,
    and the exact item ids are recorded. Requires exactly ``n_samples`` records.
    """
    import numpy as np
    import torch
    from src.data.dataset import create_dataloader_from_config

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg = json.load(open(dataset_config_path))
    dl = create_dataloader_from_config(ds_cfg, batch_size=1, num_workers=1,
                                       sample_rate=22050, sample_size=10240,
                                       audio_channels=1, shuffle=False)
    dataset = dl.dataset
    md, ids = [], []
    for idx in range(n_samples):
        item = dataset[idx]
        meta = item[1] if isinstance(item, (list, tuple)) else item["metadata"]
        if isinstance(meta, (list, tuple)):
            meta = meta[0]
        md.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in meta.items()})
        ids.append(record_id(meta, idx))
    if len(md) != n_samples:
        raise RuntimeError(f"loaded {len(md)} records, expected exactly {n_samples}")
    if len(set(ids)) != n_samples:
        raise RuntimeError(f"record identifiers are not distinct: {ids}")
    return md, ids


def reference(cond, metadata, device, angles):
    """Orbit average in the legacy per-angle order, via the library's own helper."""
    from src.data import yaw_rotation as yr
    md_inv = [yr.cylindrical_pose_features(m) for m in metadata]
    base = cond(md_inv, device)
    present = [i for i in VIT_IDS if i in base]
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = yr._orbit_average_loop(cond, md_inv, base, present, angles, img_w, device)
    for i in present:
        base[i][0] = accum[i] / float(len(angles))
    return base


def run_cell(cond, md, device, angles, mode, use_bf16, seed=1234):
    """One probe cell; returns the per-id metrics plus a finiteness flag.

    The cell runs entirely inside its mode's matmul-precision policy (see
    :func:`precision_for`), so both compared paths always use the same one."""
    import torch
    from src.data import yaw_rotation as yr

    train = mode == "train"
    cond.train(train)
    if use_bf16 and device != "cuda":
        raise RuntimeError("a bf16 cell requires CUDA; refusing to run it as fp32 on CPU")
    autocast = (torch.autocast(device_type=device, dtype=torch.bfloat16) if use_bf16
                else torch.autocast(device_type=device, enabled=False))
    grad = torch.enable_grad() if train else torch.no_grad()

    # Memory hygiene (job 3646616): keep ONE path's graph resident at a time. Each
    # side is run, immediately reduced to detached snapshots (plus, for the
    # batched side, its backward), then its graph is dropped and the allocator
    # cache released before the other side starts. Peak is therefore one path,
    # not two.
    snapshots, grads_finite = {}, None
    for label in ("batched", "loop"):
        torch.manual_seed(seed)           # identical RNG state before each side
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        with matmul_precision(precision_for(mode)), grad, autocast:
            got = (yr.invariant_conditioning(cond, md, device, angles) if label == "batched"
                   else reference(cond, md, device, angles))
        side = {k: got[k][0] for k in VIT_IDS if k in got}
        if train and label == "batched":
            loss = sum(t.float().pow(2).mean() for t in side.values())
            cond.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in cond.parameters() if p.grad is not None]
            grads_finite = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
            cond.zero_grad(set_to_none=True)
            del loss, grads
        snapshots[label] = {k: v.detach().float().clone() for k, v in side.items()}
        del got, side                      # release this path's graph before the next
        if device == "cuda":
            torch.cuda.empty_cache()

    finite = all(torch.isfinite(t).all().item()
                 for side in snapshots.values() for t in side.values())
    ids = {}
    for vit in VIT_IDS:
        if vit not in snapshots["batched"] or vit not in snapshots["loop"]:
            continue
        max_abs, rel_norm, rel_max = deviation(snapshots["batched"][vit], snapshots["loop"][vit])
        ids[vit] = {"max_abs": max_abs, "rel_norm": rel_norm, "rel_max": rel_max}
    del snapshots
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"ids": ids, "finite": finite, "grads_finite": grads_finite,
            "matmul": precision_for(mode),
            "gated": mode == "eval" and not use_bf16}


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_11 batched-orbit qualification probe")
    ap.add_argument("--config", default=os.path.join(HERE, "FLAC_AR_BF_C32.json"))
    ap.add_argument("--dataset-config",
                    default=os.path.join(REPO, "src/configs/dataset_configs/AR/train/acousticroom_train.json"))
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = ap.parse_args(argv)

    import torch
    from src.data import yaw_rotation as yr

    # NEW-2: the bf16 half of the qualification is meaningless without CUDA, and a
    # CPU fallback would be reported as if it had run bf16. Require the GPU.
    if not torch.cuda.is_available():
        print("EQUIVPROBE-ABORT: CUDA is not available; the bf16 qualification cannot run "
              "and a CPU/fp32 result must never be recorded as one")
        return 3
    device = "cuda"
    try:
        torch.zeros(1, device=device)                 # force CUDA init HERE, not mid-cell
    except Exception as exc:
        print(f"EQUIVPROBE-ABORT: CUDA initialisation failed: {type(exc).__name__}: {exc}")
        return 3
    pin = assert_vit_pin()
    cfg, cond = build_conditioner(args.config, device)
    md, sample_ids = real_samples(args.dataset_config, args.n_samples, device)
    print(f"probe: device={device} samples={len(md)} ids={','.join(sample_ids)} "
          f"cap={yr.FRAME_AVG_MAX_FWD_SAMPLES} vit_pin={pin} "
          f"matmul=gate:{precision_for('eval')}/train:{precision_for('train')}")

    results, bf16_results = {}, {}
    plan = expected_cells()
    for cell in plan:
        mode, n, batch = cell
        cell_md = [md[i % len(md)] for i in range(batch)]
        # fp32 decides the cell; bf16 is measured alongside for the B8 and train
        # cells (recorded, not gated) but its NON-FINITENESS still fails the run.
        precisions = (False, True) if (mode == "train" or batch == EVAL_TRAIN_BATCH) else (False,)
        for use_bf16 in precisions:
            res = run_cell(cond, cell_md, device, orbit(n), mode, use_bf16)
            if not use_bf16:
                results[cell] = res
            gate = "GATED" if res["gated"] else "recorded"
            for vit, m in sorted(res["ids"].items()):
                print(f"  {mode:<5} C{n:<3} B{batch:<3} {'bf16' if use_bf16 else 'fp32'} "
                      f"mm={res['matmul']:<7} {vit:<18} max_abs={m['max_abs']:.3e} "
                      f"rel_norm={m['rel_norm']:.3e} rel_max={m['rel_max']:.3e} [{gate}]")
            if res["grads_finite"] is not None:
                print(f"        grads_finite={res['grads_finite']} outputs_finite={res['finite']}")
            if use_bf16:                                  # NEW-5: auditable in the result line
                bf16_results[(mode, n, batch)] = res
            unhealthy = (not res["finite"]) or (res["grads_finite"] is False)
            if unhealthy and cell in results:
                results[cell]["finite"] = False          # a bf16 NaN fails the whole cell

    ok, reasons = verdict(results, plan)
    gated_rel, rec_rel = summarize(results, "rel_norm")
    gated_abs, rec_abs = summarize(results, "max_abs")
    bf16_rel = max([0.0] + [m["rel_norm"] for r in bf16_results.values()
                            for m in r["ids"].values()])
    bf16_abs = max([0.0] + [m["max_abs"] for r in bf16_results.values()
                            for m in r["ids"].values()])
    for r in reasons:
        print(f"  !! {r}")
    print(f"EQUIVPROBE cfg={sha256_file(args.config)[:12]} vit_pin={pin} device={device} "
          f"nsamples={len(md)} sample_ids={','.join(sample_ids)} cells={len(results)}/{len(plan)} "
          f"gate_rel_norm={gated_rel:.3e} gate_max_abs={gated_abs:.3e} "
          f"rec_rel_norm={rec_rel:.3e} rec_max_abs={rec_abs:.3e} "
          f"bf16_cells={len(bf16_results)} bf16_rel_norm={bf16_rel:.3e} bf16_max_abs={bf16_abs:.3e} "
          f"gate_matmul={precision_for('eval')} train_matmul={precision_for('train')} "
          f"tol_rel={TOL_REL_FP32:g} tol_abs={TOL_ABS_FP32:g} "
          f"verdict={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
