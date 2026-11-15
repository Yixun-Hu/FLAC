"""Shared synthetic fixtures for the exp_38 tests (not a test module): first-step evidence, train /
smoke logs, a sealed E0 receipt whose eight cells really exist (re-admissible, equal to their
references), a sealed TRAINING_COMPLETE receipt with its checkpoint manifest, retained receipt
copies, retained log and evidence, and complete evaluation cells (metrics, stream, receipt, log,
scored-waveform dump) -- all with the exact schemas the exp_38 kit admits, so each test can mutate
one field and expect a refusal. Every path goes through the loaded module's own ``_abs`` / ``REPO``.
"""
import hashlib
import json
import os

def sha(data):
    return hashlib.sha256(data).hexdigest()


# the fixtures' snapshot manifest and ITS closure (exp38_registry.closure_of_manifest of it): a consistent
# campaign binds this closure, since the training receipt's closure is re-derived from the retained manifest
SNAP_TEXT = "".join(f"{sha(str(i).encode())}  src/f{i}.py\n" for i in range(3))
FIX_CLOSURE = sha("".join(f"src/f{i}.py\0{sha(str(i).encode())}\n" for i in range(3)).encode())
CAMPAIGN = {"head": "f" * 40, "closure_sha256": FIX_CLOSURE, "env_receipt_sha256": "d" * 64,
            "eval_snapshot_sha256": "5" * 64}
PINNED_SINGLE = {
    "accumulate_grad_batches": 1, "batchnorm_nonsync_modules": 20, "flash_attn_func_set": True, "flash_calls": 48,
    "global_step_at_fit_start": 0, "global_step_recorded": 1, "gradient_clip_val": 0.0, "local_rank": 0,
    "micro_batch": 32, "n_state_tensors": 1279, "num_nodes": 1, "precision": "bf16-mixed", "rank": 0, "sdpa_calls": 0,
    "strategy": "SingleDeviceStrategy", "syncbn_modules": 0, "vit_ckpt_wrapped_layers": 12, "world_size": 1}


def records(reg, run, ngpu=2, backend="flash", max_steps=30):
    """Per-rank first-step evidence that passes reg.evidence_problems."""
    arm, tag, rung = reg.parse_run_id(run)
    cond = reg.ARMS[arm]["cond_method"]
    micro = reg.RUNGS[rung]["micro"] if ngpu > 1 else 32
    out = {}
    for r in range(ngpu):
        e = dict(PINNED_SINGLE, rank=r, local_rank=r, world_size=ngpu, max_steps=max_steps, micro_batch=micro,
                 sobol=reg.INIT_SOBOL, state_dict_sha256=reg.INIT_STATE_SHA256,
                 strategy="DDPStrategy" if ngpu > 1 else "SingleDeviceStrategy",
                 syncbn_modules=20 if ngpu > 1 else 0, batchnorm_nonsync_modules=0 if ngpu > 1 else 20,
                 rng_at_fit_start=dict(reg.RNG_AT_FIT_START["ddp" if ngpu > 1 else "single"]),
                 checkpoint_calls=[{"use_reentrant": False}] * reg.VIT_CKPT_CALLS[cond],
                 conditioner_calls=[[micro, None]] + ([[micro, ["source_vit", "context_poses_vit"]]] * 3
                                                      if cond == "fa_invariant" else []))
        if backend == "sdpa":
            e.update(flash_attn_func_set=False, flash_calls=0, sdpa_calls=48)
        out[r] = json.loads(json.dumps(e))
    return out


def write_evidence(d, recs):
    os.makedirs(d, exist_ok=True)
    for r, e in recs.items():
        json.dump(e, open(os.path.join(d, f"run_evidence.json.rank{r}.json"), "w"))


def train_log(reg, run, ngpu=2, steps=30, losses=3, end=True, backend="flash"):
    arm, tag, _ = reg.parse_run_id(run)
    parts = ["Seed set to 42"] + [f"Found {reg.FRACS[tag]['n_items']} files in 243 subfolders"] * ngpu
    if backend == "sdpa":
        parts += [reg.FLASH_MISSING + ", disabling Flash Attention"] * ngpu
    parts += [f"Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {r + 1}/{ngpu}" for r in range(ngpu)]
    if reg.ARMS[arm]["banner"]:
        parts.append(reg.ARMS[arm]["banner"])
    parts += [f"Epoch 0:   0%|          | {i + 1}/1148 [00:05<1:00:00,  0.30it/s, train/loss=0.5{i}]" for i in range(losses)]
    text = "\n".join(parts)
    return text + ("\r" + reg.FIT_END.format(n=steps) if end else "")


def fake_snapshot(reg, d):
    """A minimal, verifiable snapshot dir; returns its digest."""
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "x.txt"), "w").write("x\n")
    open(os.path.join(d, "SNAPSHOT.sha256"), "w").write(f"{sha(b'x' + bytes([10]))}  x.txt\n")
    open(os.path.join(d, "SNAPSHOT.links"), "w").write("")
    return reg.snapshot_digest(d)


def sealed(reg, path, keys, **values):
    """A sealed receipt with exactly ``keys`` (unspecified ones filled with a placeholder string)."""
    rec = {k: values.get(k, f"<{k}>") for k in keys}
    for p in (path, path + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return rec, reg.write_sealed(path, rec)


def write_snapshot_manifests(reg, d):
    """SNAPSHOT.sha256 (= SNAP_TEXT) + an empty SNAPSHOT.links in ``d``; returns their digest."""
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "SNAPSHOT.sha256"), "w").write(SNAP_TEXT)
    open(os.path.join(d, "SNAPSHOT.links"), "w").write("")
    assert reg.closure_of_manifest(SNAP_TEXT) == FIX_CLOSURE
    return reg.snapshot_digest(d)


def make_training_receipt(ev, run, rd, closure, final_bytes=b"final checkpoint bytes", e0_sha=None, e0_path=None,
                          platform=None, backend="flash", head=None, content_sha=None):
    """(record, sha256) of a complete sealed TRAINING_COMPLETE in ``rd`` with everything the consumer
    re-opens: CHECKPOINTS.sha256 + the 16 files, receipts/ (E0 copy = ``e0_path`` or a synthetic PASS,
    environment, SMOKE_DDP with valid evidence, the train log), evidence/."""
    reg = ev.reg
    platform = platform or reg.TRAINING_PLATFORM                # D2: the campaign trains on neuronic
    arm, tag, rung = reg.parse_run_id(run)
    ngpu = reg.RUNGS[rung]["ngpu"]
    head = head or CAMPAIGN["head"]
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
    rdir = os.path.join(rd, reg.RECEIPTS_DIR)
    os.makedirs(rdir, exist_ok=True)
    if e0_path:
        for src in (e0_path, e0_path + ".sha256"):
            open(os.path.join(rdir, os.path.basename(src)), "wb").write(open(src, "rb").read())
        e0_name = os.path.basename(e0_path)
        e0_digest = reg.read_sealed(os.path.join(rdir, e0_name), ev.E0_KEYS)[1]
    else:
        e0_name = "data_efficiency_e0_receipt.json"
        e0rec = synthetic_e0(ev, closure)
        for p_ in (os.path.join(rdir, e0_name), os.path.join(rdir, e0_name) + ".sha256"):
            if os.path.exists(p_):
                os.remove(p_)
        e0_digest = reg.write_sealed(os.path.join(rdir, e0_name), e0rec)
    if e0_sha is not None and e0_sha != e0_digest:
        raise AssertionError("fixture: the retained E0 copy must be the campaign's E0 receipt (pass e0_path)")
    env_name = f"data_efficiency_env_receipt_{platform}.json"
    pk = {"flash-attn": reg.FLASH_VERSION, "torch": "2.7.0"}
    _, env_digest = sealed(reg, os.path.join(rdir, env_name), reg.ENV_RECEIPT_KEYS, platform=platform, packages=pk,
                           packages_sha256=hashlib.sha256(reg.canonical(dict(sorted(pk.items()))).encode()).hexdigest(),
                           flags={"optimize": 0, "no_user_site": 1, "ignore_environment": 0, "isolated": 0})
    argv = reg.build_train_argv(run, platform, allow_nonmatched=rung != "2x32")
    smoke_name = f"data_efficiency_smoke_{run}_{platform}_2026-11-08_04-00-00.json"
    gpus = (sorted([u, "NVIDIA RTX A6000"] for u in reg.A6000_UUIDS) if platform == "a6000"
            else [[f"GPU-{i}", "NVIDIA L40"] for i in range(ngpu)])
    smoke_log = os.path.join(rdir, "smoke_log.log")
    open(smoke_log, "w").write(train_log(reg, run, ngpu, steps=reg.SMOKE_STEPS_DDP, backend=backend))
    _, smoke_digest = sealed(
        reg, os.path.join(rdir, smoke_name), reg.SMOKE_KEYS, schema=1, mode="SMOKE_DDP", run=run, platform=platform,
        backend=backend, head=head, closure_sha256=closure, env_receipt_sha256=env_digest, rc=0, checkpoints_written=0,
        log_gate="PASS", evidence_gate="PASS", verdict="PASS", argv_projection=reg.argv_projection(argv),
        gpus=gpus, snapshot_sha256="7" * 64, log_sha256=reg.sha256_file(smoke_log),
        launched_at="2026-11-08T03:00:00-0500", finished_at="2026-11-08T03:30:00-0500",
        evidence={str(r): e for r, e in records(reg, run, ngpu, backend, reg.SMOKE_STEPS_DDP).items()})
    log = os.path.join(rdir, "train.log")
    open(log, "w").write(train_log(reg, run, ngpu, steps=reg.MAX_STEPS, losses=60, backend=backend))
    snap_sha = write_snapshot_manifests(reg, rdir)
    write_evidence(os.path.join(rd, "evidence"), records(reg, run, ngpu, backend, reg.MAX_STEPS))
    rec = {"schema": 1, "run": run, "platform": platform, "backend": backend, "head": head,
           "closure_sha256": closure, "snapshot_sha256": snap_sha, "env_receipt_sha256": env_digest,
           "e0_receipt_sha256": e0_digest, "smoke_receipt": "/campaign/" + smoke_name,
           "smoke_receipt_sha256": smoke_digest,
           "model_config_sha256": reg.sha256_file(reg._abs(reg.model_config_rel(arm, rung))),
           "dataset_config_sha256": reg.FRACS[tag]["dataset_config_sha256"], "split_sha256": reg.FRACS[tag]["split_sha256"],
           "data_content_sha256": content_sha if content_sha is not None else reg.AR_CONTENT_SHA256,
           "receipts": {"e0": e0_name, "env": env_name, "smoke": smoke_name, "log": "train.log",
                        "smoke_log": "smoke_log.log", "snapshot_manifest": "SNAPSHOT.sha256",
                        "snapshot_links": "SNAPSHOT.links"},
           "data_content_checks": ["pre-launch", "post-run"],
           "argv": argv, "rc": 0, "log": "/campaign/train.log", "log_sha256": reg.sha256_file(log),
           "log_gate": "PASS", "evidence_sha256": reg.evidence_digest(os.path.join(rd, "evidence")),
           "evidence_gate": "PASS", "checkpoints_sha256": reg.sha256_file(manifest), "final_ckpt": final,
           "final_ckpt_sha256": sha(final_bytes), "final_ckpt_bytes": len(final_bytes), "gpus": gpus,
           "launched_at": "2026-11-08T04:00:00-0500", "finished_at": "2026-11-09T04:00:00-0500"}
    for p in (os.path.join(rd, reg.TRAINING_RECEIPT), os.path.join(rd, reg.TRAINING_RECEIPT) + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    digest = reg.write_sealed(os.path.join(rd, reg.TRAINING_RECEIPT), rec)
    commit(reg, rd, run, digest)
    return rec, digest


def commit(reg, rd, run, digest, status="COMPLETE"):
    """What the supervisor does last: the terminal launch record (run-dir copy) + the commit marker."""
    lr = os.path.join(rd, f"at_launch_{run}_a6000_FULL_2026-11-08_04-00-00.json")
    json.dump({"run": run, "status": status, "training_receipt_sha256": digest}, open(lr, "w"))
    for p in (os.path.join(rd, reg.COMMIT_MARKER), os.path.join(rd, reg.COMMIT_MARKER) + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    return reg.write_commit_marker(rd, run, digest, lr)


def synthetic_e0(ev, closure):
    """An E0 receipt that passes e0_static_problems (for retained copies of a synthetic campaign)."""
    reg = ev.reg
    cells = {c["name"]: {"reference": f"E0/{c['e0']}_K{c['K']}_s42.json", "equal": True, "difference": None,
                         "problems": [], "metrics_sha256": "a" * 64, "receipt_sha256": "b" * 64,
                         "accepted_deviations": {}} for c in ev.e0_cells()}
    return {"schema": 1, "verdict": "PASS", "failed": [], "cells": cells, "head": CAMPAIGN["head"],
            "e0a_decision_sha256": None,
            "closure_sha256": closure, "evaluator_files": {f: "0" * 64 for f in ev.EVALUATOR_FILES},
            "env_receipt_sha256": CAMPAIGN["env_receipt_sha256"], "eval_snapshot_sha256": CAMPAIGN["eval_snapshot_sha256"],
            "checkpoints": {g: {"sha256": ev.E0[g]["sha"], "bytes": ev.E0_BYTES[g]} for g in ev.E0},
            "reference_manifest_sha256": "1" * 64, "references": {r: "2" * 64 for r in ev.ref_set()},
            "input_hash": {f"K{k}_s{s}": v for (k, s), v in ev.INPUT_HASH.items()},
            "eval_pins": {"agree_sha256": reg.AGREE_SHA256, "vae_ckpt_sha256": reg.VAE_CKPT_SHA256, "vit_rev": reg.VIT_REV},
            "written_at": "2026-11-08T05:00:00-0500"}


def copy_to_farm(ev, rd, farm):
    """What ``prepare`` copies beside the farm link (the offline consumer reads these)."""
    reg = ev.reg
    os.makedirs(farm, exist_ok=True)
    cm = json.load(open(os.path.join(rd, reg.COMMIT_MARKER)))
    for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256", reg.COMMIT_MARKER,
              reg.COMMIT_MARKER + ".sha256", cm["launch_record"]):
        open(os.path.join(farm, f), "wb").write(open(os.path.join(rd, f), "rb").read())
    for sub in (reg.RECEIPTS_DIR, "evidence"):
        os.makedirs(os.path.join(farm, sub), exist_ok=True)
        for f in os.listdir(os.path.join(rd, sub)):
            open(os.path.join(farm, sub, f), "wb").write(open(os.path.join(rd, sub, f), "rb").read())


def make_e0_receipt(ev, campaign=None, evaluator_files=None, log_dir=None, e0a_ours=None, decision=None):
    """A sealed, valid E0 receipt at ``campaign`` whose eight cells EXIST (in the module's farm paths),
    are admissible under that campaign and equal their references -- or, with ``e0a_ours`` = {cell:
    {key: value}}, E0a cells carrying those retrieval values, judged under ``decision`` (accepted map,
    sha256); returns (campaign incl. sha256, rec)."""
    reg = ev.reg
    campaign = dict(campaign or CAMPAIGN)
    refs = ev.check_refs()
    e0camp = dict(campaign, sha256=None)
    cells = {}
    for c in ev.e0_cells():
        ref, rel = ev.e0_reference(c, refs)
        metrics = dict(ref["metrics"], **((e0a_ours or {}).get(c["name"], {})))
        make_cell(ev, c, e0camp, (ev.E0[c["e0"]]["sha"], ev.E0_BYTES[c["e0"]], None),
                  log_dir or os.path.join(os.path.dirname(ev._abs(c["metrics"])), "logs"),
                  metrics=metrics, by_scene=ref.get("by_scene"))
        ok, equal, diff, acc = ev.judge_e0(c, json.load(open(ev._abs(c["metrics"]))), ref,
                                           decision[0] if decision else None)
        assert ok, (c["name"], diff)
        cells[c["name"]] = {"reference": rel, "equal": equal, "difference": diff, "problems": [],
                            "accepted_deviations": acc,
                            "metrics_sha256": reg.sha256_file(ev._abs(c["metrics"])),
                            "receipt_sha256": reg.sha256_file(ev._abs(c["receipt"]))}
    files = evaluator_files or {f: (reg.sha256_file(ev._abs(f)) if os.path.exists(ev._abs(f)) else "0" * 64)
                                for f in ev.EVALUATOR_FILES}
    rec = {"schema": 1, "verdict": "PASS", "failed": [], "cells": cells, "head": campaign["head"],
           "closure_sha256": campaign["closure_sha256"], "evaluator_files": files,
           "env_receipt_sha256": campaign["env_receipt_sha256"], "eval_snapshot_sha256": campaign["eval_snapshot_sha256"],
           "e0a_decision_sha256": decision[1] if decision and any(v["accepted_deviations"] for v in cells.values()) else None,
           "checkpoints": {g: {"sha256": ev.E0[g]["sha"], "bytes": ev.E0_BYTES[g]} for g in ev.E0},
           "reference_manifest_sha256": reg.sha256_file(ev._abs(ev.REF_MANIFEST)), "references": refs,
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


def make_cell(ev, cell, campaign, ckpt, log_dir, dump=True, metrics=None, by_scene=None):
    """Write one complete, admissible cell. ``ckpt`` = (sha256, bytes, training-receipt sha or None)."""
    reg = ev.reg
    os.makedirs(os.path.dirname(ev._abs(cell["metrics"])), exist_ok=True)
    fa = cell["cond"] == "fa_invariant"
    metrics = metrics or {k: 1.0 + i for i, k in enumerate(ev.METRIC_KEYS)}
    rec = {"metrics": metrics, "by_scene": by_scene or {s: dict(metrics) for s in ev.SCENES}, "cond_method": cell["cond"],
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
        copy = ev.dump_manifest_copy(cell)
        if dump:
            write_dump(ev, cell, msha, ssha, campaign["head"])
            open(copy, "wb").write(open(os.path.join(cell["dump_dir"], "manifest.json"), "rb").read())
        else:                                                    # offline fixture: the copy only
            man = {"schema_version": 1, "n": 6337, "dtype": "float32", "shape": list(ev.DUMP_SHAPE), "sha256": "9" * 64,
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
               "eval_snapshot_sha256": campaign["eval_snapshot_sha256"],
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
