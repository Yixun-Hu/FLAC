#!/usr/bin/env python3
"""exp_38 evaluation driver for THIS 2×A6000 box (plan_data_efficiency.md r2 §9, §10 B8).

    exp38_eval.py plan [--bridge]                 the 60 (+10) result cells and the 8 E0 cells, full argv
    exp38_eval.py env-receipt --write | --check   the campaign environment receipt (registry)
    exp38_eval.py prepare <run>                   admit a run's final checkpoint + create its farm link
    exp38_eval.py prepare-e0                      verify + link the four E0 reference checkpoints
    exp38_eval.py run --gpu G --cells <name>...   run cells sequentially (card lock; idle card unless --shared)
    exp38_eval.py e0                              E0 verdict -> data_efficiency_e0_receipt.json (+ .sha256)
    exp38_eval.py validate <name>...              admission check of finished cells

Every cell runs ``eval_FLAC.py`` through the reviewed exp_35 no-flash launcher (SDPA, the backend of
every reference cell) with the reference cells' exact argv shape (incl. ``--num-workers 4``).

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp38_registry as reg  # noqa: E402

EXP = reg.EXP_DIR
FARM = "outputs_FLAC/exp38_data_efficiency/eval"
DUMP_ROOT = "/media/diskstation/yixunhu/FLAC/exp38_pred_waveforms"
NAS_CKPT_ROOT = "/media/diskstation/yixunhu/FLAC/checkpoints"
LOCKDIR = "/home/yixunhu/codespace"          # .exp38_card<i>.lock, shared with exp38_train_local.sh
SEEDS = (42, 43, 44, 45, 46)
EVAL_CONFIGS = {8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
                1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"}
EVAL_CONFIG_SHA256 = {8: "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55",
                      1: "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"}
UNSEEN_SPLIT = ("data/AR/unseen_eval.json", "9a9d817abc3e19f41351e07325ffa929c1f2846d0c77e80d538d9e4a21342ba8")
AGREE = "weights/AGREE/AGREE_fullAR.pt"
NO_FLASH = "overlays/no_flash/exp35_no_flash_launch.py"
EVALUATOR_FILES = ("eval_FLAC.py", NO_FLASH, "overlays/no_flash/flash_attn.py")
SENTINEL = ("EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa "
            "overlay_sha256=0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023 "
            "launcher_sha256=bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de target=eval_FLAC.py")
# fixed-mode input hashes of the unseen split per (K, seed): exp_25 P1 / YAW and exp_35 B-F agree
INPUT_HASH = {
    (1, 42): "37d2f5693fa6c7731c49b4ff4b68571ec0c5feb8455c80b0657ce06cba6fd759",
    (1, 43): "df150ea93d68a1e97caff05ddd3b69a4fb5ac1410df32ca9b3459deb82e363f8",
    (1, 44): "5709f434b246d7d881d3add6a6e26ec0285e21bcbfcff0314f814f3a33922eb5",
    (1, 45): "fb4df53656ca75b4eb316c4a17846dceb153607e56edeca65496d431eddebc2e",
    (1, 46): "7d139a5679ed198787326a544f1cc18ea891afa7799c973272add6b2ce0fcced",
    (8, 42): "4c950c1e2ed021a352f5c454b67e3b21234cdbd52060debc1c58919c9e3645ac",
    (8, 43): "1c048da9c6afd2b240d4f5650d930fff92e455e14429d37eb869b3a7b8317e20",
    (8, 44): "8398cd8655ad8b44a09cd608035504744776d88a43335176b2ae5a5492a66f83",
    (8, 45): "ef2c937139f8ec09e4034cccf3751ce15a9ba1761bbc7a5f6ec748bb6753063e",
    (8, 46): "163f89a4b84a8071f1fa8091883cc9ab2c9bff97fccdb37b5196bbb8c1490b17",
}
METRIC_KEYS = ("T60", "Invalid T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
               "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
SCENES = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
          "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants")      # the unseen split's 10 families
# E0 replay gates (plan r2 §9.3): checkpoint, its sha, the config, the reference (imported into
# ref_cells/, sha-pinned) and whether by_scene is compared.
E0 = {
    "E0a": {"ckpt": f"{NAS_CKPT_ROOT}/exp14_data_curve/dc_van_f025/epoch=34-step=40000.ckpt",
            "sha": "aa1a2cd624333a18eb2f9f019f6b46beaf2d94c1b2d7dc455f238ad0cec9433c",
            "config": reg.ARMS["p1"]["config"], "cond": "vanilla", "by_scene": False},
    "E0b": {"ckpt": f"{NAS_CKPT_ROOT}/ar_40k_endpoints/BF/epoch=8-step=40000.ckpt",
            "sha": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
            "config": reg.BF_BASE_CONFIG, "cond": "fa_invariant", "by_scene": True},
    "E0c": {"ckpt": f"{NAS_CKPT_ROOT}/ar_40k_endpoints/YAW/epoch=8-step=40000.ckpt",
            "sha": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
            "config": reg.ARMS["yaw"]["config"], "cond": "vanilla", "by_scene": True},
    "E0d": {"ckpt": f"{NAS_CKPT_ROOT}/ar_40k_endpoints/P1/epoch=8-step=40000.ckpt",
            "sha": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
            "config": reg.ARMS["p1"]["config"], "cond": "vanilla", "by_scene": True},
}
REF_MANIFEST = f"{EXP}/ref_cells/REF_MANIFEST.sha256"


class EvalError(RuntimeError):
    pass


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(reg.REPO, rel)


# ---------------------------------------------------------------- cells
def method_suffix(cond):
    return "" if cond == "vanilla" else "_fa_invariant_a4"


def make_cell(name, run, K, seed, ckpt_link, config, cond, dump, e0=None):
    stem = f"{os.path.basename(ckpt_link)[:-5]}_metrics_1_1.0_{name}{method_suffix(cond)}"
    d = os.path.dirname(ckpt_link)
    return {"name": name, "run": run, "K": K, "seed": seed, "ckpt_link": ckpt_link, "config": config,
            "cond": cond, "dump": dump, "e0": e0, "stem": stem,
            "metrics": os.path.join(d, stem + ".json"), "stream": os.path.join(d, stem + ".stream.json"),
            "receipt": os.path.join(d, stem + ".receipt.json"),
            "dump_dir": os.path.join(DUMP_ROOT, stem) if dump else None}


def run_cells(run):
    arm, tag, rung = reg.parse_run_id(run)
    link = f"{FARM}/{run}/{reg.ckpt_name(reg.MAX_STEPS, tag)}"
    return [make_cell(f"{run}_K{K}_s{s}", run, K, s, link, reg.model_config_rel(arm, rung),
                      reg.ARMS[arm]["cond_method"], dump=True)
            for K in (8, 1) for s in SEEDS]


def e0_cells():
    out = []
    for gate, g in E0.items():
        link = f"{FARM}/{gate}/{os.path.basename(g['ckpt'])}"
        for K in (8, 1):
            out.append(make_cell(f"de_{gate}_K{K}_s42", gate, K, 42, link, g["config"], g["cond"],
                                 dump=False, e0=gate))
    return out


def all_cells(bridge=False):
    runs = reg.required_runs() + (["de_p1_f025"] if bridge else [])
    return e0_cells() + [c for r in runs for c in run_cells(r)]


def eval_argv(cell):
    argv = ["--model-config", cell["config"], "--dataset-config", EVAL_CONFIGS[cell["K"]],
            "--cond-method", cell["cond"]]
    if cell["cond"] == "fa_invariant":
        argv += ["--frame-avg-angles", "0,90,180,270", "--frame-avg-max-fwd-samples", "64"]
    argv += ["--cond-autocast", "bf16", "--cfg-scale", "1.0", "--steps", "1", "--batch-size", "64",
             "--num-workers", "4", "--device", "cuda", "--record-per-scene", "--record-stream",
             "--expected-stream-count", "6337", "--seed", str(cell["seed"]), "--rotate-mode", "fixed",
             "--rotate-deg", "0", "--eval-name", cell["name"], "--ckpt-path", cell["ckpt_link"]]
    if cell["dump"]:
        argv += ["--dump-scored-waveforms", DUMP_ROOT]
    return argv


# ---------------------------------------------------------------- validation
def _finite_tree(x):
    if isinstance(x, dict):
        return all(_finite_tree(v) for v in x.values())
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return math.isfinite(x)
    return True


def strict_equal(a, b, path="$"):
    """Recursive, type-strict equality; returns the first differing path or None."""
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in sorted(a):
            d = strict_equal(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = strict_equal(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def expected_ckpt_sha(cell):
    if cell["e0"]:
        return E0[cell["e0"]]["sha"]
    rd = os.path.join(reg.PLATFORMS["a6000"]["save_root"], cell["run"])
    name = os.path.basename(cell["ckpt_link"])
    for line in open(os.path.join(rd, "CHECKPOINTS.sha256")).read().strip().split("\n"):
        digest, n = line.split("  ", 1)
        if n == name:
            return digest
    raise EvalError(f"{name} not in {rd}/CHECKPOINTS.sha256")


def validate_cell(cell, source_sha=None):
    """Problems (empty = admitted) of one finished cell (plan r2 §9.4)."""
    p = []
    try:
        rec = json.load(open(_abs(cell["metrics"])))
        stream = json.load(open(_abs(cell["stream"])))
        receipt = json.load(open(_abs(cell["receipt"])))
    except (OSError, ValueError) as err:
        return [f"{cell['name']}: artifact missing / unreadable: {err}"]
    m = rec.get("metrics", {})
    if tuple(sorted(m)) != tuple(sorted(METRIC_KEYS)) or not _finite_tree(m):
        p.append("metrics key set / finiteness")
    want = {"cond_method": cell["cond"], "cond_autocast": "bf16", "rotate_deg": 0.0, "seed": cell["seed"],
            "steps": 1, "cfg_scale": 1.0, "batch_size": 64, "n_samples": 6337, "device": "cuda",
            "dataset_config": EVAL_CONFIGS[cell["K"]], "eval_name": cell["name"], "weights_source": "ema",
            "per_scene_schema": 1, "scene_count": 10,
            "frame_avg_angles": [0.0, 90.0, 180.0, 270.0] if cell["cond"] == "fa_invariant" else None,
            "frame_avg_fwd_cap": 64 if cell["cond"] == "fa_invariant" else None,
            "orbit_execution": "batched" if cell["cond"] == "fa_invariant" else "n/a"}
    for k, v in want.items():
        if rec.get(k) != v:
            p.append(f"{k} {rec.get(k)!r} != {v!r}")
    if source_sha and rec.get("source_sha") != source_sha:
        p.append(f"source_sha {rec.get('source_sha')} != {source_sha}")
    if tuple(sorted(rec.get("by_scene", {}))) != SCENES or not _finite_tree(rec.get("by_scene", {})):
        p.append("by_scene scene set / finiteness")
    try:
        if rec.get("ckpt_sha256_loaded") != expected_ckpt_sha(cell):
            p.append("ckpt_sha256_loaded != the admitted checkpoint")
    except (OSError, EvalError) as err:
        p.append(str(err))
    if stream.get("stream_count") != 6337 or stream.get("input_hash") != INPUT_HASH[(cell["K"], cell["seed"])]:
        p.append(f"stream count / input_hash {stream.get('input_hash')} != pinned")
    links = {"metrics_sha256": reg.sha256_file(_abs(cell["metrics"])), "stream_sha256": reg.sha256_file(_abs(cell["stream"]))}
    for k, v in links.items():
        if receipt.get(k) != v:
            p.append(f"receipt {k} != the file")
    if receipt.get("rc") != 0 or receipt.get("sentinel") != SENTINEL or receipt.get("argv") != eval_argv(cell):
        p.append("receipt rc / sentinel / argv")
    for k, v in (("model_config_sha256", reg.sha256_file(_abs(cell["config"]))),
                 ("dataset_config_sha256", EVAL_CONFIG_SHA256[cell["K"]]), ("unseen_split_sha256", UNSEEN_SPLIT[1])):
        if receipt.get(k) != v:
            p.append(f"receipt {k} != pinned")
    if cell["dump"]:
        d = cell["dump_dir"]
        try:
            man = json.load(open(os.path.join(d, "manifest.json")))
            ok = (os.path.exists(os.path.join(d, "COMPLETE")) and man.get("n") == 6337 and man.get("dtype") == "float32"
                  and man.get("metrics_sha256") == links["metrics_sha256"]
                  and man.get("stream_sidecar_sha256") == links["stream_sha256"]
                  and man.get("input_hash") == stream.get("input_hash")
                  and receipt.get("dump_manifest_sha256") == reg.sha256_file(os.path.join(d, "manifest.json")))
            if not ok:
                p.append("dump manifest cross-links")
        except (OSError, ValueError) as err:
            p.append(f"dump unreadable: {err}")
    return [f"{cell['name']}: {x}" for x in p]


# ---------------------------------------------------------------- preparation
def _link(path, target):
    os.makedirs(os.path.dirname(_abs(path)), exist_ok=True)
    if os.path.islink(_abs(path)):
        if os.readlink(_abs(path)) != target:
            raise EvalError(f"{path} already links to {os.readlink(_abs(path))}")
        return
    if os.path.exists(_abs(path)):
        raise EvalError(f"{path} exists and is not our link")
    os.symlink(target, _abs(path))


def prepare(run):
    arm, tag, rung = reg.parse_run_id(run)
    rd = os.path.join(reg.PLATFORMS["a6000"]["save_root"], run)
    name = reg.ckpt_name(reg.MAX_STEPS, tag)
    want = expected_ckpt_sha(run_cells(run)[0])
    audit = reg.audit_ckpt(os.path.join(rd, name), run, reg.MAX_STEPS, run_dir=rd)
    if audit["sha256"] != want:
        raise EvalError(f"{name} sha {audit['sha256']} != CHECKPOINTS.sha256 {want}")
    _link(f"{FARM}/{run}/{name}", os.path.join(rd, name))
    return {"run": run, "admitted": audit}


def prepare_e0():
    out = {}
    for gate, g in E0.items():
        got = reg.sha256_file(g["ckpt"])
        if got != g["sha"]:
            raise EvalError(f"{gate}: {g['ckpt']} sha {got} != {g['sha']}")
        _link(f"{FARM}/{gate}/{os.path.basename(g['ckpt'])}", g["ckpt"])
        out[gate] = got
    return out


def check_refs():
    lines = open(_abs(REF_MANIFEST)).read().strip().split("\n")
    refs = {}
    for line in lines:
        digest, rel = line.split("  ", 1)
        if reg.sha256_file(_abs(os.path.join(EXP, "ref_cells", rel))) != digest:
            raise EvalError(f"reference {rel} != its manifest sha")
        refs[rel] = digest
    return refs


def e0_reference(cell):
    rel = f"E0/{cell['e0']}_K{cell['K']}_s42.json"
    return json.load(open(_abs(os.path.join(EXP, "ref_cells", rel)))), rel


# ---------------------------------------------------------------- running
def gpu_foreign(gpu):
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip().split("\n")
    uuid = {r.split(", ")[0]: r.split(", ")[1] for r in rows}[str(gpu)]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip().split("\n")
    return [a.split(", ")[1] for a in apps if a and a.split(", ")[0] == uuid]


def run_one(cell, gpu, closure, head):
    if not validate_cell(cell, head):
        return "skipped (valid)"
    for k in ("metrics", "stream", "receipt"):
        if os.path.exists(_abs(cell[k])):
            raise EvalError(f"{cell['name']}: a partial {k} exists ({cell[k]}); archive it first")
    if cell["dump"] and os.path.exists(cell["dump_dir"]):
        raise EvalError(f"{cell['name']}: dump dir {cell['dump_dir']} exists; archive it first")
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log = _abs(f"{EXP}/eval_logs/data_efficiency_{ts}_{cell['name']}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ENABLE_TORCH_COMPILE", "PYTHONSTARTUP")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": str(gpu)})
    argv = eval_argv(cell)
    with open(log, "w") as fh:
        fh.write("argv: " + " ".join(["python", NO_FLASH, "eval_FLAC.py"] + argv) + "\n")
        fh.flush()
        rc = subprocess.run([sys.executable, NO_FLASH, "eval_FLAC.py"] + argv, cwd=reg.REPO, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    text = open(log, errors="replace").read()
    sentinels = [line for line in text.split("\n") if line.startswith("EXP35_NO_FLASH_LAUNCHER")]
    receipt = {"cell": cell["name"], "argv": argv, "rc": rc, "sentinel": sentinels[0] if len(sentinels) == 1 else None,
               "log": os.path.relpath(log, reg.REPO), "log_sha256": reg.sha256_file(log), "gpu": gpu,
               "head": head, "closure_sha256": closure,
               "model_config_sha256": reg.sha256_file(_abs(cell["config"])),
               "dataset_config_sha256": reg.sha256_file(_abs(EVAL_CONFIGS[cell["K"]])),
               "unseen_split_sha256": reg.sha256_file(_abs(UNSEEN_SPLIT[0])),
               "agree_sha256": reg.sha256_file(_abs(AGREE))}
    if os.path.exists(_abs(cell["metrics"])):
        receipt["metrics_sha256"] = reg.sha256_file(_abs(cell["metrics"]))
    if os.path.exists(_abs(cell["stream"])):
        receipt["stream_sha256"] = reg.sha256_file(_abs(cell["stream"]))
    if cell["dump"] and os.path.exists(os.path.join(cell["dump_dir"], "manifest.json")):
        receipt["dump_manifest_sha256"] = reg.sha256_file(os.path.join(cell["dump_dir"], "manifest.json"))
    with open(_abs(cell["receipt"]) + ".tmp", "w") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)
    os.replace(_abs(cell["receipt"]) + ".tmp", _abs(cell["receipt"]))
    probs = validate_cell(cell, head)
    if rc != 0 or probs:
        raise EvalError(f"{cell['name']}: rc {rc}; " + "; ".join(probs))
    return "done"


def cmd_run(gpu, names, shared=False):
    cells = {c["name"]: c for c in all_cells(bridge=True)}
    unknown = [n for n in names if n not in cells]
    if unknown:
        raise EvalError(f"unknown cells {unknown}")
    if os.path.realpath(reg.REPO) != reg.CAMPAIGN_ROOT["a6000"]:
        raise EvalError(f"evaluation runs only from the campaign tree {reg.CAMPAIGN_ROOT['a6000']}")
    head = subprocess.run(["git", "-C", reg.REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if subprocess.run(["git", "-C", reg.REPO, "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise EvalError("tracked tree is not clean")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("ENABLE_TORCH_COMPILE", None)
    shadows = reg.check_untracked("a6000")
    if shadows:
        raise EvalError("campaign tree: " + "; ".join(shadows))
    reg.env_receipt("a6000")
    closure = reg.closure_digest()["closure_sha256"]
    lockfile = os.path.join(LOCKDIR, f".exp38_card{gpu}.lock")
    lock = open(lockfile, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise EvalError(f"another exp_38 training / evaluation holds card {gpu} ({lockfile})")
    try:
        for n in names:
            foreign = gpu_foreign(gpu)
            if foreign and not shared:
                raise EvalError(f"GPU {gpu} is not idle (pids {foreign})")
            if cells[n]["dump"]:
                st = os.statvfs(DUMP_ROOT)
                if st.f_bavail * st.f_frsize < 260e6 * 1.5:
                    raise EvalError("dump root has < 390 MB free")
            print(f"{n}: {run_one(cells[n], gpu, closure, head)}", flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def cmd_e0():
    head = subprocess.run(["git", "-C", reg.REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    refs = check_refs()
    verdicts, problems = {}, []
    for cell in e0_cells():
        probs = validate_cell(cell, head)
        ref, rel = e0_reference(cell)
        rec = json.load(open(_abs(cell["metrics"]))) if not probs else {}
        diff = strict_equal(rec.get("metrics"), ref.get("metrics")) if not probs else "cell not admitted"
        if not diff and E0[cell["e0"]]["by_scene"]:
            diff = strict_equal(rec.get("by_scene"), ref.get("by_scene"))
        verdicts[cell["name"]] = {"reference": rel, "equal": diff is None, "difference": diff, "problems": probs}
        if probs or diff:
            problems.append(cell["name"])
    rec = {"verdict": "PASS" if not problems else "FAIL", "failed": problems, "cells": verdicts, "head": head,
           "closure_sha256": reg.closure_digest()["closure_sha256"],
           "evaluator_files": {f: reg.sha256_file(_abs(f)) for f in EVALUATOR_FILES},
           "env_receipt": reg.env_receipt("a6000"), "checkpoints": {g: E0[g]["sha"] for g in E0},
           "reference_manifest_sha256": reg.sha256_file(_abs(REF_MANIFEST)), "references": refs,
           "input_hash": {f"K{k}_s{s}": v for (k, s), v in INPUT_HASH.items()}}
    out = _abs(f"{EXP}/data_efficiency_e0_receipt.json")
    if os.path.exists(out):
        raise EvalError(f"{out} exists (archive it to re-run E0)")
    text = json.dumps(rec, indent=1, sort_keys=True).encode()
    with open(out + ".tmp", "wb") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    with open(out + ".sha256", "w") as fh:
        fh.write(f"{hashlib.sha256(text).hexdigest()}  {os.path.basename(out)}\n")
    os.replace(out + ".tmp", out)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("--bridge", action="store_true")
    s = sub.add_parser("env-receipt"); g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true"); g.add_argument("--check", action="store_true")
    s = sub.add_parser("prepare"); s.add_argument("run")
    sub.add_parser("prepare-e0")
    s = sub.add_parser("run"); s.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    s.add_argument("--cells", nargs="+", required=True); s.add_argument("--shared", action="store_true")
    sub.add_parser("e0")
    s = sub.add_parser("validate"); s.add_argument("names", nargs="+")
    a = p.parse_args(argv)
    os.chdir(reg.REPO)
    try:
        if a.cmd == "plan":
            for c in all_cells(a.bridge):
                print(json.dumps({"name": c["name"], "argv": ["python", NO_FLASH, "eval_FLAC.py"] + eval_argv(c),
                                  "metrics": c["metrics"], "dump": c["dump_dir"]}))
        elif a.cmd == "env-receipt":
            print(json.dumps(reg.env_receipt("a6000", write=a.write)))
        elif a.cmd == "prepare":
            print(json.dumps(prepare(a.run)))
        elif a.cmd == "prepare-e0":
            print(json.dumps(prepare_e0()))
        elif a.cmd == "run":
            cmd_run(a.gpu, a.cells, a.shared)
        elif a.cmd == "e0":
            rec = cmd_e0()
            print(f"E0 {rec['verdict']}" + (f": {rec['failed']}" if rec["failed"] else ""))
            return 0 if rec["verdict"] == "PASS" else 1
        elif a.cmd == "validate":
            cells = {c["name"]: c for c in all_cells(bridge=True)}
            probs = [x for n in a.names for x in validate_cell(cells[n])]
            print("\n".join(probs) or "VALID")
            return 0 if not probs else 1
    except (EvalError, reg.GateError, OSError, KeyError, ValueError) as err:
        print(f"EXP38 EVAL REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
