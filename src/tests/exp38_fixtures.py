"""Shared synthetic fixtures for the exp_38 tests (not a test module): a sealed E0 receipt, a sealed
TRAINING_COMPLETE receipt with its checkpoint manifest, and complete evaluation cells (metrics,
stream, receipt, log, scored-waveform dump) -- all built with the exact schemas the exp_38 kit
admits, so each test can mutate one field and expect a refusal. Every path goes through the
loaded module's own ``_abs`` / ``REPO`` (the table-generator tests load the kit from a temp root).
"""
import hashlib
import json
import os

CAMPAIGN = {"head": "f" * 40, "closure_sha256": "c" * 64, "env_receipt_sha256": "d" * 64}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_training_receipt(ev, run, rd, closure, final_bytes=b"final checkpoint bytes", e0_sha="e" * 64):
    """(record, sha256) of a sealed receipt + a 16-line CHECKPOINTS.sha256 + the checkpoint files."""
    reg = ev.reg
    arm, tag, rung = reg.parse_run_id(run)
    os.makedirs(rd, exist_ok=True)
    final = reg.ckpt_name(reg.MAX_STEPS, tag)
    lines = []
    for s in reg.cadence_steps():
        name = reg.ckpt_name(s, tag)
        data = final_bytes if s == reg.MAX_STEPS else f"ckpt {s}".encode()
        open(os.path.join(rd, name), "wb").write(data)
        lines.append(f"{sha(data)}  {name}")
    manifest = os.path.join(rd, "CHECKPOINTS.sha256")
    open(manifest, "w").write("\n".join(lines) + "\n")
    rec = {"schema": 1, "run": run, "platform": "a6000", "backend": "flash", "head": CAMPAIGN["head"],
           "closure_sha256": closure, "snapshot_sha256": "5" * 64, "env_receipt_sha256": CAMPAIGN["env_receipt_sha256"],
           "e0_receipt_sha256": e0_sha, "smoke_receipt": "x.json", "smoke_receipt_sha256": "6" * 64,
           "model_config_sha256": reg.sha256_file(reg._abs(reg.model_config_rel(arm, rung))),
           "dataset_config_sha256": reg.FRACS[tag]["dataset_config_sha256"], "split_sha256": reg.FRACS[tag]["split_sha256"],
           "argv": reg.build_train_argv(run, "a6000"), "rc": 0, "log": "x.log", "log_sha256": "7" * 64,
           "log_gate": "PASS", "evidence_sha256": "8" * 64, "evidence_gate": "PASS",
           "checkpoints_sha256": reg.sha256_file(manifest), "final_ckpt": final, "final_ckpt_sha256": sha(final_bytes),
           "final_ckpt_bytes": len(final_bytes), "gpus": [["GPU-x", "NVIDIA RTX A6000"]] * 2,
           "launched_at": "2026-11-08T04:00:00-0500", "finished_at": "2026-11-09T04:00:00-0500"}
    digest = reg.write_sealed(os.path.join(rd, reg.TRAINING_RECEIPT), rec)
    return rec, digest


def make_e0_receipt(ev, campaign=None, evaluator_files=None):
    """A sealed, valid E0 receipt at ``campaign``; returns the campaign dict incl. its sha256."""
    reg = ev.reg
    campaign = dict(campaign or CAMPAIGN)
    cells = {c["name"]: {"reference": f"E0/{c['e0']}_K{c['K']}_s42.json", "equal": True, "difference": None,
                         "problems": [], "metrics_sha256": "a" * 64, "receipt_sha256": "b" * 64} for c in ev.e0_cells()}
    files = evaluator_files or {f: (reg.sha256_file(ev._abs(f)) if os.path.exists(ev._abs(f)) else "0" * 64)
                                for f in ev.EVALUATOR_FILES}
    rec = {"schema": 1, "verdict": "PASS", "failed": [], "cells": cells, "head": campaign["head"],
           "closure_sha256": campaign["closure_sha256"], "evaluator_files": files,
           "env_receipt_sha256": campaign["env_receipt_sha256"],
           "checkpoints": {g: {"sha256": ev.E0[g]["sha"], "bytes": ev.E0_BYTES[g]} for g in ev.E0},
           "reference_manifest_sha256": reg.sha256_file(ev._abs(ev.REF_MANIFEST)), "references": ev.check_refs(),
           "input_hash": {f"K{k}_s{s}": v for (k, s), v in ev.INPUT_HASH.items()},
           "eval_pins": {"agree_sha256": reg.AGREE_SHA256, "vae_ckpt_sha256": reg.VAE_CKPT_SHA256, "vit_rev": reg.VIT_REV},
           "written_at": "2026-11-08T05:00:00-0500"}
    path = ev._abs(ev.E0_RECEIPT)
    for p in (path, path + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    campaign["sha256"] = reg.write_sealed(path, rec)
    campaign["e0_receipt"] = path
    return campaign, rec


def write_dump(ev, cell, msha, ssha, head, shape=None):
    """A complete ScoredWaveformDump cell dir exactly as eval_FLAC.finalize writes it."""
    import numpy as np
    shape = shape or ev.DUMP_SHAPE
    d = cell["dump_dir"]
    os.makedirs(d)
    arr = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    npy = os.path.join(d, "waveforms.npy")
    np.save(npy, arr, allow_pickle=False)
    wsha = ev.reg.sha256_file(npy)
    open(os.path.join(d, "waveforms.sha256"), "w").write(f"{wsha}  waveforms.npy\n")
    man = {"schema_version": 1, "n": shape[0], "dtype": "float32", "shape": list(shape), "sha256": wsha,
           "input_hash": ev.INPUT_HASH[(cell["K"], cell["seed"])], "metrics_stem": cell["stem"], "source_sha": head,
           "metrics_sha256": msha, "stream_sidecar_sha256": ssha}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    open(os.path.join(d, "COMPLETE"), "w").write(wsha + "\n")
    return man


def make_cell(ev, cell, campaign, ckpt, log_dir, dump=True):
    """Write one complete, admissible cell. ``ckpt`` = (sha256, bytes, training-receipt sha or None)."""
    reg = ev.reg
    os.makedirs(os.path.dirname(ev._abs(cell["metrics"])), exist_ok=True)
    fa = cell["cond"] == "fa_invariant"
    metrics = {k: 1.0 + i for i, k in enumerate(ev.METRIC_KEYS)}
    rec = {"metrics": metrics, "by_scene": {s: dict(metrics) for s in ev.SCENES}, "cond_method": cell["cond"],
           "cond_autocast": "bf16", "rotate_deg": 0.0, "seed": cell["seed"], "steps": 1, "cfg_scale": 1.0,
           "batch_size": 64, "n_samples": 6337, "device": "cuda", "dataset_config": ev.EVAL_CONFIGS[cell["K"]],
           "eval_name": cell["name"], "weights_source": "ema", "per_scene_schema": 1, "scene_count": 10,
           "frame_avg_angles": [0.0, 90.0, 180.0, 270.0] if fa else None, "frame_avg_fwd_cap": 64 if fa else None,
           "orbit_execution": "batched" if fa else "n/a", "source_sha": campaign["head"], "ckpt_path": cell["ckpt_link"],
           "ckpt_sha256_loaded": ckpt[0], "ckpt_bytes_loaded": ckpt[1]}
    json.dump(rec, open(ev._abs(cell["metrics"]), "w"))
    json.dump({"stream_count": 6337, "input_hash": ev.INPUT_HASH[(cell["K"], cell["seed"])]},
              open(ev._abs(cell["stream"]), "w"))
    msha, ssha = reg.sha256_file(ev._abs(cell["metrics"])), reg.sha256_file(ev._abs(cell["stream"]))
    man_sha = None
    if cell["dump"]:
        man = write_dump(ev, cell, msha, ssha, campaign["head"]) if dump else None
        copy = ev.dump_manifest_copy(cell)
        if dump:
            open(copy, "wb").write(open(os.path.join(cell["dump_dir"], "manifest.json"), "rb").read())
        else:                                                    # offline fixture: the copy only
            shape = ev.DUMP_SHAPE
            man = {"schema_version": 1, "n": 6337, "dtype": "float32", "shape": list(shape), "sha256": "9" * 64,
                   "input_hash": ev.INPUT_HASH[(cell["K"], cell["seed"])], "metrics_stem": cell["stem"],
                   "source_sha": campaign["head"], "metrics_sha256": msha, "stream_sidecar_sha256": ssha}
            json.dump(man, open(copy, "w"), indent=2)
        man_sha = reg.sha256_file(copy)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{cell['name']}.log")
    open(log, "w").write("argv: ...\n" + ev.SENTINEL + "\n")
    receipt = {"schema": 1, "cell": cell["name"], "argv": ev.eval_argv(cell), "rc": 0, "sentinel": ev.SENTINEL,
               "log": log, "log_sha256": reg.sha256_file(log), "gpu": 1, "gpu_uuid": reg.A6000_UUIDS[1],
               "gpu_name": ev.GPU_NAME, "head": campaign["head"], "closure_sha256": campaign["closure_sha256"],
               "e0_receipt_sha256": None if cell["e0"] else campaign["sha256"],
               "env_receipt_sha256": campaign["env_receipt_sha256"],
               "model_config_sha256": reg.sha256_file(ev._abs(cell["config"])),
               "dataset_config_sha256": ev.EVAL_CONFIG_SHA256[cell["K"]], "unseen_split_sha256": ev.UNSEEN_SPLIT[1],
               "agree_sha256": reg.AGREE_SHA256, "vae_ckpt_sha256": reg.VAE_CKPT_SHA256, "vit_rev": reg.VIT_REV,
               "metrics_sha256": msha, "stream_sha256": ssha, "dump_manifest_sha256": man_sha,
               "ckpt_sha256": ckpt[0], "ckpt_bytes": ckpt[1], "training_receipt_sha256": ckpt[2],
               "started_at": "2026-11-09T05:00:00-0500", "finished_at": "2026-11-09T05:30:00-0500"}
    json.dump(receipt, open(ev._abs(cell["receipt"]), "w"))
    return cell
