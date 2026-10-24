#!/usr/bin/env python3
"""exp_25 depth_canon -- the numerical-composition control and eps_cond
(plan_depth_canon.md v8 §1 Tier 2 (ii) + eps_cond, §6 test 21b).

For the first ``--n-items`` items of an evaluation loader and each registered column
offset d in {64, 128, 256, 384, random in [1, W-1]}, A = canon(md) and
B = canon(rot_d(md)) are pushed through the SAME loaded checkpoint with the SAME fixed
per-item noise, and compared stage by stage: conditioning tensors, the one-step DiT
velocity, the latent after the single Euler step, the decoded waveform after
clamp(-1, 1). relL2 / relLinf use ``src/tools/depth_canon_stats`` (pinned
denominators, A is the reference). Tier 1 (canonical recovered-depth sha256 of A == B)
is recorded for the same items. eps_cond = max fp32 conditioning relL2; PASS <= 1e-5.

GPU for the registered run (the Planner launches it); CPU-testable on a toy stack.

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §1/§6/§8.
"""
import contextlib
import os
import sys

import numpy as np
import torch


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")) and not os.path.isfile(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # this checkout's src/ and eval_FLAC.py, never a stale installed copy

from src.tools import depth_canon_stats as stats  # noqa: E402

FIXED_OFFSETS = (64, 128, 256, 384)
OFFSET_KEYS = tuple(f"d{d}" for d in FIXED_OFFSETS) + ("random",)
RANDOM_OFFSET_SEED_SHIFT = 1  # the random offsets' generator is seeded with seed + 1


# --------------------------------------------------------------------------- #
# offsets and fixed noise
# --------------------------------------------------------------------------- #
def _check_int(name, v, lo):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")


def draw_offsets(n_items, img_w, seed):
    """``{key: [d_i per item]}``: the four fixed column offsets and one independent
    uniform draw in ``[1, img_w - 1]`` per item from a generator seeded with
    ``seed + RANDOM_OFFSET_SEED_SHIFT``."""
    _check_int("n_items", n_items, 1)
    _check_int("img_w", img_w, max(FIXED_OFFSETS) + 1)  # the registered offsets must not wrap
    _check_int("seed", seed, 0)
    out = {f"d{d}": [d] * n_items for d in FIXED_OFFSETS}
    g = torch.Generator().manual_seed(seed + RANDOM_OFFSET_SEED_SHIFT)
    out["random"] = [int(d) for d in torch.randint(1, img_w, (n_items,), generator=g)]
    return out


def item_noise(i, shape, seed):
    """Item ``i``'s fixed noise: its own generator, seeded from ``(seed, i)`` through
    numpy's SeedSequence, so it does not depend on how many items run."""
    s = int(np.random.SeedSequence([int(seed), int(i)]).generate_state(1, dtype=np.uint64)[0])
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(s & (2 ** 63 - 1)))


def noise_bank(n_items, shape, seed):
    """``[n_items, *shape]`` float32: one seeded noise tensor per item, reused for A and B."""
    return torch.stack([item_noise(i, shape, seed) for i in range(n_items)])


# --------------------------------------------------------------------------- #
# A = canon(md), B = canon(rot_d(md)), Tier 1
# --------------------------------------------------------------------------- #
def canonical_pairs(metadata, offsets, img_w):
    """``(A, {key: B}, tier1)`` for one batch of loader metadata dicts.

    ``A[i] = canonicalize_scene_metadata(md_i)``; ``B[key][i]`` canonicalizes
    ``rotate_scene_metadata(md_i, d_i)`` with ``d_i = offsets[key][i]`` (the reviewed
    operators, nothing re-implemented). ``tier1[i]`` records ``dj_A``, the classes, the
    canonical recovered-depth sha256 of A and, per key, whether B's equals it and
    whether ``dj_B == (dj_A - d) mod W`` (the shift co-rotates). A ``StabilizedTieError``
    propagates (fail-closed, as in training and eval)."""
    from src.data import yaw_rotation as yr
    n = len(metadata)
    for key, ds in offsets.items():
        if len(ds) != n:
            raise ValueError(f"offsets[{key!r}] must hold one offset per item ({n}), got {len(ds)}")
    a_list, b_lists, tier1 = [], {k: [] for k in offsets}, []
    for i, md in enumerate(metadata):
        a, dj_a, cls_a = yr.canonicalize_scene_metadata(md, img_w)
        sha_a = yr.canonical_depth_sha256(a["depth"])
        rec = {"dj_A": dj_a, "class_A": cls_a, "depth_sha_A": sha_a, "offsets": {}}
        for key, ds in offsets.items():
            d = int(ds[i])
            b, dj_b, cls_b = yr.canonicalize_scene_metadata(
                yr.rotate_scene_metadata(md, yr.offsets_to_radians([d], img_w)[0], img_w), img_w)
            b_lists[key].append(b)
            rec["offsets"][key] = {"d": d, "dj_B": dj_b, "class_B": cls_b,
                                   "depth_sha_equal": yr.canonical_depth_sha256(b["depth"]) == sha_a,
                                   "class_equal": cls_b == cls_a,
                                   "dj_corotates": dj_b == (dj_a - d) % img_w}
        a_list.append(a)
        tier1.append(rec)
    return a_list, b_lists, tier1


# --------------------------------------------------------------------------- #
# stagewise comparison
# --------------------------------------------------------------------------- #
def _rows(t):
    """``[B, ...]`` tensor -> float64 CPU ``[B, prod(rest)]`` numpy (bf16 included)."""
    t = t.detach().to(device="cpu", dtype=torch.float64)
    return t.reshape(t.shape[0], -1).numpy()


def stage_rel(test, ref):
    """Per-item ``{"relL2", "relLinf"}`` of ``test`` against the reference ``ref``."""
    a, b = _rows(test), _rows(ref)
    return {"relL2": stats.rel_l2(a, b), "relLinf": stats.rel_linf(a, b)}


def compare_cond(test, ref):
    """Conditioning dicts ``{id: [tensor, mask]}``: per-id relL2 / relLinf of the
    tensors, the same over all ids concatenated per item (``"all"``, ids sorted), and
    whether every mask is identical."""
    if sorted(test) != sorted(ref):
        raise ValueError(f"conditioner ids differ: {sorted(test)} vs {sorted(ref)}")
    ids = sorted(ref)
    out = {"ids": ids, "masks_equal": all(
        (test[k][1] is None and ref[k][1] is None) or
        (test[k][1] is not None and ref[k][1] is not None and torch.equal(test[k][1].cpu(), ref[k][1].cpu()))
        for k in ids)}
    for k in ids:
        out[k] = stage_rel(test[k][0], ref[k][0])
    cat = lambda c: torch.cat([c[k][0].detach().double().cpu().reshape(c[k][0].shape[0], -1)  # noqa: E731
                               for k in ids], dim=1)
    out["all"] = stage_rel(cat(test), cat(ref))
    return out


def summarize(rel):
    """``{"relL2": paired_summary, "relLinf": paired_summary}`` over the items."""
    return {k: stats.paired_summary(rel[k]) for k in ("relL2", "relLinf")}


# --------------------------------------------------------------------------- #
# one forward of the evaluation's generative path, stage by stage
# --------------------------------------------------------------------------- #
STAGES = ("cond", "velocity", "latent", "waveform")
# fp32: the eps_cond measurement (autocast off, true-fp32 matmuls); eval: the
# registered evaluation protocol (--cond-autocast bf16, eval_FLAC.py's 'medium').
PROTOCOLS = {"fp32": {"cond_autocast": "off", "matmul_precision": "highest"},
             "eval": {"cond_autocast": "bf16", "matmul_precision": "medium"}}


@contextlib.contextmanager
def matmul_precision(value):
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(value)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old)


def run_stages(module, md_list, noise, reals, protocol):
    """The four stages for one list of (already canonical) metadata dicts, mirroring
    ``eval_FLAC.evaluate_model``'s loop: conditioner under the protocol's cond autocast,
    ``get_conditioning_inputs``, ``sample_discrete_euler`` with 1 step / cfg 1.0 /
    ``batch_cfg=True`` / the wrapper's ``dist_shift`` on the GIVEN noise (the DiT's one
    output recorded as the velocity), decode, ``clamp(-1, 1)``, pad to ``reals``."""
    import eval_FLAC as ev
    from src.inference import sampling
    enabled, dtype = ev.resolve_cond_autocast(protocol["cond_autocast"])
    device = module.device
    device_type = torch.device(device).type
    model = module.diffusion.model
    objective = (model.diffusion_objective if hasattr(model, "diffusion_objective")
                 else getattr(model, "objective", "rectified_flow"))
    if objective != "rectified_flow":
        raise ValueError(f"the probe mirrors the rectified_flow 1-step path only, got {objective!r}")
    velocities = []

    def recording_model(x, t, **kw):
        v = model(x, t, **kw)
        velocities.append(v)
        return v

    with matmul_precision(protocol["matmul_precision"]), torch.no_grad():
        ctx = (contextlib.nullcontext() if not enabled else torch.amp.autocast(device_type)
               if dtype is None else torch.amp.autocast(device_type, dtype=dtype))
        with ctx:
            cond = module.diffusion.conditioner(md_list, device)
        cond_inputs = module.diffusion.get_conditioning_inputs(cond)
        latent = sampling.sample_discrete_euler(
            recording_model, noise.to(device), 1, **cond_inputs, cfg_scale=1.0,
            dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
        if len(velocities) != 1:
            raise RuntimeError(f"expected one DiT call for one Euler step, saw {len(velocities)}")
        wav = latent if module.diffusion.pretransform is None else module.diffusion.pretransform.decode(latent)
        wav = wav.clamp(-1.0, 1.0)
        if wav.shape != reals.shape and wav.shape[-1] < reals.shape[-1]:
            wav = torch.nn.functional.pad(wav, (0, reals.shape[-1] - wav.shape[-1]))
    return {"cond": cond, "velocity": velocities[0], "latent": latent, "waveform": wav}


def compare_stages(test, ref):
    """``{stage: comparison}`` of two :func:`run_stages` outputs (``ref`` = A)."""
    out = {"cond": compare_cond(test["cond"], ref["cond"])}
    out.update({s: stage_rel(test[s], ref[s]) for s in STAGES[1:]})
    return out


# --------------------------------------------------------------------------- #
# loading: eval_FLAC.evaluate_model's checkpoint path
# --------------------------------------------------------------------------- #
def load_probe_module(model_config_path, ckpt_path, device, allow_canon_mismatch=False):
    """``(module, info)`` -- the checkpoint loaded the way ``evaluate_model`` loads it.

    The helpers are eval_FLAC's own (``resolve_weights_source``,
    ``resolve_are_from_checkpoint``, ``resolve_canonicalize_protocol`` with the
    ``--canonicalize`` flag set, ``check_load_integrity``, the two factories, all looked
    up on the ``eval_FLAC`` module). The prefix strip and the EMA re-keying are inline
    statements in ``evaluate_model``, so they are repeated here verbatim; the tests pin
    the state dict and both configs this hands the factories against the ones
    ``evaluate_model`` hands them for the same checkpoint. ARE checkpoints are refused
    (no add-back bank here); a non-canonical checkpoint needs ``allow_canon_mismatch``."""
    import copy
    import json
    import eval_FLAC as ev
    with open(model_config_path) as f:
        model_config = json.load(f)
    file_model_config = copy.deepcopy(model_config)
    training_config = model_config.get("training", None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    for key in list(state_dict.keys()):                            # verbatim (evaluate_model)
        if key.startswith("diffusion."):
            new_key = key.replace("diffusion.", "")
            state_dict[new_key] = state_dict.pop(key)
    weights_source = ev.resolve_weights_source(training_config, state_dict.keys())
    if training_config.get("use_ema", False) and any(k.startswith("diffusion_ema.ema_model.")
                                                     for k in state_dict.keys()):
        print("Using EMA model")
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                new_key = key.replace("diffusion_ema.ema_model.", "model.")
                state_dict[new_key] = state_dict.pop(key)
        training_config["use_ema"] = False
    are_lambda, _, _ = ev.resolve_are_from_checkpoint(ckpt.get("model_config"), file_model_config, None)
    if are_lambda is not None:
        raise ValueError(f"an ARE checkpoint (are_lambda={are_lambda}) is out of this probe's scope: "
                         "it has no anchor add-back")
    embedded = ckpt.get("model_config")
    protocol = ev.resolve_canonicalize_protocol(
        True, file_model_config.get("training"),
        embedded.get("training") if isinstance(embedded, dict) else None,
        allow_canon_mismatch, cond_method="vanilla", file_model_cfg=file_model_config)
    model = ev.create_model_from_config(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ev.check_load_integrity(missing, unexpected, False)
    model_config["training"] = training_config
    module = ev.create_training_wrapper_from_config(model_config, model)
    module.eval().requires_grad_(False)
    module.to(device)
    info = {"weights_source": weights_source, "canonicalize_trained": protocol.trained,
            "canonicalize_applied": protocol.applied, "canonicalize_mismatch": protocol.mismatch,
            "img_w": protocol.img_w, "model_config_sha256_file": ev.model_config_sha256(file_model_config),
            "model_config_sha256_embedded": ev.model_config_sha256(embedded)}
    return module, info


# --------------------------------------------------------------------------- #
# the probe proper (no I/O)
# --------------------------------------------------------------------------- #
EPS_COND_THRESHOLD = 1e-5


def _pair_lists():
    return {"relL2": [], "relLinf": []}


def _new_acc():
    acc = {"cond": {"ids": None, "masks_equal": True, "per_id": {}, "all": _pair_lists()}}
    acc.update({s: _pair_lists() for s in STAGES[1:]})
    return acc


def _append(acc, cmp):
    c = cmp["cond"]
    if acc["cond"]["ids"] not in (None, c["ids"]):
        raise RuntimeError(f"conditioner ids changed between batches: {acc['cond']['ids']} -> {c['ids']}")
    acc["cond"]["ids"] = c["ids"]
    acc["cond"]["masks_equal"] = acc["cond"]["masks_equal"] and c["masks_equal"]
    for target, src in [(acc["cond"]["per_id"].setdefault(i, _pair_lists()), c[i]) for i in c["ids"]] + \
            [(acc["cond"]["all"], c["all"])] + [(acc[s], cmp[s]) for s in STAGES[1:]]:
        for k in ("relL2", "relLinf"):
            target[k].extend(float(v) for v in src[k])


def _finish(acc):
    for d in list(acc["cond"]["per_id"].values()) + [acc["cond"]["all"]] + [acc[s] for s in STAGES[1:]]:
        d["summary"] = summarize({k: np.array(d[k]) for k in ("relL2", "relLinf")})
    return acc


def _pooled(results_p):
    """Per stage, the summary over the five registered offsets' items together."""
    out = {}
    for s in STAGES:
        rows = [results_p[k]["cond"]["all"] if s == "cond" else results_p[k][s] for k in OFFSET_KEYS]
        out[s] = summarize({m: np.array([v for r in rows for v in r[m]]) for m in ("relL2", "relLinf")})
    return out


def _eps_cond(results):
    """max fp32 conditioning relL2 over the registered offsets, conditioner ids and items."""
    if "fp32" not in results:
        return None
    value, key, cid, pos = max((v, k, i, pos) for k in OFFSET_KEYS
                               for i, d in results["fp32"][k]["cond"]["per_id"].items()
                               for pos, v in enumerate(d["relL2"]))
    return {"value": value, "threshold": EPS_COND_THRESHOLD, "pass": value <= EPS_COND_THRESHOLD,
            "argmax": {"key": key, "id": cid, "position": pos}}


def run_probe(module, reals, metadata, img_w, seed, batch_size, latent_len, protocols=None):
    """Per protocol and offset key: stagewise B-vs-A comparisons for every item, A run
    twice for the determinism floor, the Tier-1 record, eps_cond and the control p99."""
    protocols = PROTOCOLS if protocols is None else protocols
    n = len(metadata)
    if reals.shape[0] != n:
        raise ValueError(f"{reals.shape[0]} reals for {n} metadata items")
    offsets = draw_offsets(n, img_w, seed)
    noise = noise_bank(n, (module.diffusion.io_channels, latent_len), seed)
    acc = {p: {k: _new_acc() for k in OFFSET_KEYS} for p in protocols}
    det = {p: _new_acc() for p in protocols}
    items = []
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        a_list, b_lists, tier1 = canonical_pairs(list(metadata[sl]), {k: v[sl] for k, v in offsets.items()},
                                                 img_w)
        items += tier1
        for p, proto in protocols.items():
            sa = run_stages(module, a_list, noise[sl], reals[sl], proto)
            _append(det[p], compare_stages(run_stages(module, a_list, noise[sl], reals[sl], proto), sa))
            for key in OFFSET_KEYS:
                _append(acc[p][key], compare_stages(
                    run_stages(module, b_lists[key], noise[sl], reals[sl], proto), sa))
    results = {p: {k: _finish(acc[p][k]) for k in OFFSET_KEYS} for p in protocols}
    pooled = {p: _pooled(results[p]) for p in protocols}
    eps = _eps_cond(results)
    control = {p: dict({k: results[p][k]["waveform"]["summary"]["relL2"]["p99"] for k in OFFSET_KEYS},
                       pooled=pooled[p]["waveform"]["relL2"]["p99"]) for p in protocols}
    pairs = [o for it in items for o in it["offsets"].values()]
    t1 = {"n_pairs": len(pairs)} | {f: sum(o[f] for o in pairs)
                                    for f in ("depth_sha_equal", "class_equal", "dj_corotates")}
    verdict = {"eps_cond": bool(eps and eps["pass"]),
               "tier1": all(t1[f] == t1["n_pairs"] for f in ("depth_sha_equal", "class_equal", "dj_corotates")),
               "masks": all(results[p][k]["cond"]["masks_equal"] for p in protocols for k in OFFSET_KEYS)}
    verdict = {"pass": all(verdict.values()), **verdict}
    return {"offsets": offsets, "items": items, "tier1": t1, "results": results, "pooled": pooled,
            "determinism": {p: _finish(det[p]) for p in protocols}, "eps_cond": eps,
            "control_p99": control, "verdict": verdict}
