#!/usr/bin/env python3
"""exp_38 evaluation driver for THIS 2×A6000 box (plan_data_efficiency.md r2 §9, §10 B8).

    exp38_eval.py plan [--bridge]                 the 60 (+10) result cells and the 8 E0 cells, full argv
    exp38_eval.py env-receipt --write | --check   the campaign environment receipt (registry)
    exp38_eval.py prepare <run>                   admit a run's final checkpoint + create its farm link
    exp38_eval.py prepare-e0                      verify + link the four E0 reference checkpoints
    exp38_eval.py run --gpu G --cells <name>...   run cells sequentially (card lock; idle card unless --shared)
    exp38_eval.py e0                              E0 verdict -> the sealed data_efficiency_e0_receipt.json
    exp38_eval.py e0-check                        validate the sealed E0 receipt against this tree
    exp38_eval.py validate [--offline] <name>...  admission check of finished cells

Every cell runs ``eval_FLAC.py`` through the reviewed exp_35 no-flash launcher (SDPA, the backend of
every reference cell) with the reference cells' exact argv shape (incl. ``--num-workers 4``).
Result cells are bound to the sealed E0 receipt (HEAD, closure, environment, evaluator files,
references), to the run's sealed TRAINING_COMPLETE receipt (checkpoint sha256 + bytes) and to this
box's registered A6000s; AGREE / VAE.ckpt / DINOv3 are pinned before and after every cell
(Codex code r1 BLOCKING 3-7).

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
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


# ---------------------------------------------------------------- exact schemas (Codex code r1 BLOCKING 3, 5)
GPU_NAME = "NVIDIA RTX A6000"
E0_BYTES = {"E0a": 723923435, "E0b": 723922667, "E0c": 723922667, "E0d": 723927531}
DUMP_SHAPE = [6337, 10240]
DUMP_SCHEMA_VERSION = 1           # eval_FLAC.SCORED_DUMP_SCHEMA_VERSION (equivalence test-covered)
METRICS_TOP_KEYS = frozenset({
    "batch_size", "by_scene", "cfg_scale", "ckpt_bytes_loaded", "ckpt_path", "ckpt_sha256_loaded", "cond_autocast",
    "cond_method", "dataset_config", "device", "eval_name", "frame_avg_angles", "frame_avg_fwd_cap", "metrics",
    "n_samples", "orbit_execution", "per_scene_schema", "rotate_deg", "scene_count", "seed", "source_sha", "steps",
    "weights_source"})
RECEIPT_KEYS = frozenset({
    "schema", "cell", "argv", "rc", "sentinel", "log", "log_sha256", "gpu", "gpu_uuid", "gpu_name", "head",
    "closure_sha256", "e0_receipt_sha256", "env_receipt_sha256", "model_config_sha256", "dataset_config_sha256",
    "unseen_split_sha256", "agree_sha256", "vae_ckpt_sha256", "vit_rev", "metrics_sha256", "stream_sha256",
    "dump_manifest_sha256", "ckpt_sha256", "ckpt_bytes", "training_receipt_sha256", "started_at", "finished_at"})
E0_KEYS = frozenset({"schema", "verdict", "failed", "cells", "head", "closure_sha256", "evaluator_files",
                     "env_receipt_sha256", "checkpoints", "reference_manifest_sha256", "references", "input_hash",
                     "eval_pins", "written_at"})
E0_CELL_KEYS = frozenset({"reference", "equal", "difference", "problems", "metrics_sha256", "receipt_sha256"})
E0_RECEIPT = reg.E0_RECEIPT


def _num(x):
    return type(x) in (int, float) and math.isfinite(x)


def metric_block_problems(block, where):
    if not isinstance(block, dict) or set(block) != set(METRIC_KEYS):
        return [f"{where}: metric keys {sorted(block) if isinstance(block, dict) else block!r}"]
    bad = sorted(k for k, v in block.items() if not _num(v))
    return [f"{where}: non-numeric / non-finite {bad}"] if bad else []


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


# ---------------------------------------------------------------- references (Codex code r1 BLOCKING 7)
def ref_set():
    """The exact 68 reference cells: 8 E0 replays + the dc_van 25/50/75 % and P1 / YAW / B-F 100 % anchors."""
    out = [f"E0/{g}_K{k}_s42.json" for g in E0 for k in (8, 1)]
    out += [f"anchors/{m}/{m}_K{k}_s{s}.json" for m in ("P1", "YAW", "BF") for k in (8, 1) for s in SEEDS]
    out += [f"anchors/dc_van/dc_van_f{t}_K{k}_s{s}.json" for t in ("025", "050", "075") for k in (8, 1) for s in SEEDS]
    return sorted(out)


def check_refs():
    """{rel: sha256} of the reference set, exactly the registered 68 files, each matching the manifest."""
    lines = open(_abs(REF_MANIFEST)).read().strip().split("\n")
    pairs = [ln.split("  ", 1) for ln in lines]
    rels = [r for _, r in pairs]
    if len(rels) != len(set(rels)) or sorted(rels) != ref_set():
        raise EvalError(f"reference manifest set != the registered {len(ref_set())} cells "
                        f"(extra {sorted(set(rels) - set(ref_set()))[:3]}, missing {sorted(set(ref_set()) - set(rels))[:3]})")
    refs = {}
    for digest, rel in pairs:
        if reg.sha256_file(_abs(os.path.join(EXP, "ref_cells", rel))) != digest:
            raise EvalError(f"reference {rel} != its manifest sha")
        refs[rel] = digest
    on_disk = sorted(os.path.relpath(os.path.join(d, f), _abs(os.path.join(EXP, "ref_cells")))
                     for d, _, fs in os.walk(_abs(os.path.join(EXP, "ref_cells"))) for f in fs if f.endswith(".json"))
    if on_disk != ref_set():
        raise EvalError("ref_cells/ holds JSON files outside the registered set")
    return refs


def load_ref(rel, refs):
    """One reference cell, read once and checked against ``refs`` (no check / read gap)."""
    data = open(_abs(os.path.join(EXP, "ref_cells", rel)), "rb").read()
    if hashlib.sha256(data).hexdigest() != refs.get(rel):
        raise EvalError(f"reference {rel} changed since its check")
    return json.loads(data)


def e0_reference(cell, refs=None):
    rel = f"E0/{cell['e0']}_K{cell['K']}_s42.json"
    return load_ref(rel, refs or check_refs()), rel


# ---------------------------------------------------------------- campaign identity
def git_head(repo=None):
    return subprocess.run(["git", "-C", repo or reg.REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def current_campaign():
    """The identity the E0 cells are produced and judged under (before an E0 receipt exists)."""
    return {"head": git_head(), "closure_sha256": reg.closure_digest()["closure_sha256"], "sha256": None,
            "env_receipt_sha256": reg.env_receipt_sha("a6000")}


def validate_e0_receipt(repo=None, offline=False, platform="a6000"):
    """The sealed E0 receipt: exact schema, a complete PASS of the eight registered replays with the
    pinned checkpoints / inputs / weights / references, and -- online -- bound to THIS tree's closure,
    evaluator files and (on this box) environment receipt. ``offline`` (the table generator, any
    host): the receipt's own bindings only. Returns the campaign identity."""
    path = _abs(E0_RECEIPT) if repo is None else os.path.join(repo, E0_RECEIPT)
    rec, digest = reg.read_sealed(path, E0_KEYS)
    p = []
    if rec["schema"] != 1 or rec["verdict"] != "PASS" or rec["failed"] != []:
        p.append(f"verdict {rec['verdict']} failed {rec['failed']}")
    cells = rec["cells"] if isinstance(rec["cells"], dict) else {}
    if sorted(cells) != sorted(c["name"] for c in e0_cells()):
        p.append(f"E0 cells {sorted(cells)} != the eight registered replays")
    for c in e0_cells():
        v = cells.get(c["name"])
        want_ref = f"E0/{c['e0']}_K{c['K']}_s42.json"
        if not (isinstance(v, dict) and set(v) == E0_CELL_KEYS and v["reference"] == want_ref and v["equal"] is True
                and v["difference"] is None and v["problems"] == [] and reg._is_sha(v["metrics_sha256"])
                and reg._is_sha(v["receipt_sha256"])):
            p.append(f"{c['name']}: not an exact, admitted replay of {want_ref}")
    want_ck = {g: {"sha256": E0[g]["sha"], "bytes": E0_BYTES[g]} for g in E0}
    if rec["checkpoints"] != want_ck:
        p.append("E0 checkpoint pins differ")
    if rec["input_hash"] != {f"K{k}_s{s}": v for (k, s), v in INPUT_HASH.items()}:
        p.append("input-hash pins differ")
    if rec["eval_pins"] != {"agree_sha256": reg.AGREE_SHA256, "vae_ckpt_sha256": reg.VAE_CKPT_SHA256, "vit_rev": reg.VIT_REV}:
        p.append("evaluator weight pins differ")
    if not (isinstance(rec["head"], str) and re.fullmatch(r"[0-9a-f]{40}", rec["head"]) and reg._is_sha(rec["closure_sha256"])
            and reg._is_sha(rec["env_receipt_sha256"]) and isinstance(rec["written_at"], str)):
        p.append("head / closure / environment fields malformed")
    if not isinstance(rec["evaluator_files"], dict) or sorted(rec["evaluator_files"]) != sorted(EVALUATOR_FILES):
        p.append("evaluator file set differs")
    try:
        refs = check_refs()
        if rec["references"] != refs or rec["reference_manifest_sha256"] != reg.sha256_file(_abs(REF_MANIFEST)):
            p.append("reference cells / manifest differ from the E0 receipt's")
    except (EvalError, OSError) as err:
        p.append(str(err))
    if not offline:
        if reg.closure_digest(repo)["closure_sha256"] != rec["closure_sha256"]:
            p.append("E0 receipt was produced at a different campaign closure")
        for rel, d in (rec["evaluator_files"] or {}).items():
            if reg.sha256_file(_abs(rel) if repo is None else os.path.join(repo, rel)) != d:
                p.append(f"{rel} changed since E0")
        if platform == "a6000" and reg.env_receipt_sha("a6000", repo) != rec["env_receipt_sha256"]:
            p.append("environment receipt changed since E0")
    if p:
        raise reg.GateError(f"{path}: " + "; ".join(p))
    return {"e0_receipt": path, "sha256": digest, "head": rec["head"], "closure_sha256": rec["closure_sha256"],
            "env_receipt_sha256": rec["env_receipt_sha256"]}


# ---------------------------------------------------------------- admission
def expected_ckpt(cell, campaign, offline=False):
    """(sha256, bytes, training-receipt sha256 or None) the cell must have evaluated."""
    if cell["e0"]:
        return E0[cell["e0"]]["sha"], E0_BYTES[cell["e0"]], None
    rd = os.path.join(reg.PLATFORMS["a6000"]["save_root"], cell["run"])
    farm = os.path.dirname(_abs(cell["ckpt_link"]))
    rec, digest = reg.check_training_receipt(cell["run"], rd, closure=campaign["closure_sha256"],
                                             receipt_dir=farm if offline else None, verify_bytes=False)
    return rec["final_ckpt_sha256"], rec["final_ckpt_bytes"], digest


def dump_manifest_copy(cell):
    return _abs(cell["metrics"])[:-len(".json")] + ".dump_manifest.json"


def dump_problems(cell, man, stream, links):
    """The evaluator's ScoredWaveformDump.is_complete(dump_dir, expected_n=6337) predicate, inlined
    (equivalence is test-covered against eval_FLAC), plus the registered shape and every cross-link."""
    import numpy as np
    d = cell["dump_dir"]
    p = []
    try:
        marker = open(os.path.join(d, "COMPLETE")).read()
        sidecar = open(os.path.join(d, "waveforms.sha256")).read()
        sha = man.get("sha256")
        shape = man.get("shape")
        if not (type(man.get("schema_version")) is int and man["schema_version"] == DUMP_SCHEMA_VERSION
                and type(man.get("n")) is int and man["n"] == 6337 and man.get("dtype") == "float32"
                and isinstance(shape, list) and shape == DUMP_SHAPE and all(type(x) is int for x in shape)
                and reg._is_sha(sha) and reg._is_sha(man.get("input_hash")) and man.get("metrics_stem") == cell["stem"]):
            p.append("dump manifest schema / n / dtype / shape / stem")
        else:
            npy = os.path.join(d, "waveforms.npy")
            arr = np.load(npy, mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.float32 or list(arr.shape) != DUMP_SHAPE or reg.sha256_file(npy) != sha:
                p.append("waveforms.npy dtype / shape / sha256")
            if sidecar != f"{sha}  waveforms.npy\n" or marker != sha + "\n":
                p.append("waveforms.sha256 / COMPLETE contents")
    except (OSError, ValueError, TypeError) as err:
        p.append(f"dump unreadable: {err}")
    return p


def validate_cell(cell, campaign=None, offline=False):
    """Problems (empty = admitted) of one finished cell (plan r2 §9.4): exact metrics / by-scene /
    receipt schemas, protocol, campaign binding (E0 receipt -> head / closure / environment), the
    checkpoint the training receipt admitted (sha + bytes), stream pins, evaluator weight pins and
    the complete scored-waveform dump. ``offline``: no NAS read -- the training receipt, checkpoint
    manifest and dump manifest come from the copies kept beside the cell (dump content unchecked)."""
    p = []
    try:
        if campaign is None:
            campaign = current_campaign() if cell["e0"] else validate_e0_receipt(offline=offline)
        rec = json.load(open(_abs(cell["metrics"])))
        stream = json.load(open(_abs(cell["stream"])))
        receipt = json.load(open(_abs(cell["receipt"])))
    except (OSError, ValueError, reg.GateError) as err:
        return [f"{cell['name']}: artifact / campaign missing or unreadable: {err}"]
    if not isinstance(rec, dict) or set(rec) != METRICS_TOP_KEYS:
        return [f"{cell['name']}: metrics schema {sorted(set(rec) ^ METRICS_TOP_KEYS) if isinstance(rec, dict) else rec!r}"]
    p += metric_block_problems(rec["metrics"], "metrics")
    bs = rec["by_scene"]
    if not isinstance(bs, dict) or tuple(sorted(bs)) != tuple(sorted(SCENES)):
        p.append(f"by_scene scenes {sorted(bs) if isinstance(bs, dict) else bs!r}")
    else:
        for sc in SCENES:
            p += metric_block_problems(bs[sc], f"by_scene.{sc}")
    fa = cell["cond"] == "fa_invariant"
    want = {"cond_method": cell["cond"], "cond_autocast": "bf16", "rotate_deg": 0.0, "seed": cell["seed"],
            "steps": 1, "cfg_scale": 1.0, "batch_size": 64, "n_samples": 6337, "device": "cuda",
            "dataset_config": EVAL_CONFIGS[cell["K"]], "eval_name": cell["name"], "weights_source": "ema",
            "per_scene_schema": 1, "scene_count": 10, "ckpt_path": cell["ckpt_link"],
            "frame_avg_angles": [0.0, 90.0, 180.0, 270.0] if fa else None, "frame_avg_fwd_cap": 64 if fa else None,
            "orbit_execution": "batched" if fa else "n/a", "source_sha": campaign["head"]}
    for k, v in want.items():
        if strict_equal(rec.get(k), v):
            p.append(f"{k} {rec.get(k)!r} != {v!r}")
    try:
        ck_sha, ck_bytes, tr_sha = expected_ckpt(cell, campaign, offline)
        if rec["ckpt_sha256_loaded"] != ck_sha or rec["ckpt_bytes_loaded"] != ck_bytes or type(rec["ckpt_bytes_loaded"]) is not int:
            p.append("ckpt sha / bytes loaded != the admitted checkpoint")
    except (OSError, EvalError, reg.GateError) as err:
        p.append(str(err))
        ck_sha = ck_bytes = tr_sha = "unavailable"
    if not isinstance(stream, dict) or stream.get("stream_count") != 6337 \
            or stream.get("input_hash") != INPUT_HASH[(cell["K"], cell["seed"])]:
        p.append(f"stream count / input_hash {stream.get('input_hash') if isinstance(stream, dict) else stream!r} != pinned")
    links = {"metrics_sha256": reg.sha256_file(_abs(cell["metrics"])), "stream_sha256": reg.sha256_file(_abs(cell["stream"]))}
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        got = set(receipt) if isinstance(receipt, dict) else set()
        p.append(f"receipt schema (missing {sorted(RECEIPT_KEYS - got)}, extra {sorted(got - RECEIPT_KEYS)})")
        return [f"{cell['name']}: {x}" for x in p]
    gpu = receipt["gpu"]
    want_r = {"schema": 1, "cell": cell["name"], "argv": eval_argv(cell), "rc": 0, "sentinel": SENTINEL,
              "gpu_name": GPU_NAME, "head": campaign["head"], "closure_sha256": campaign["closure_sha256"],
              "e0_receipt_sha256": campaign["sha256"], "env_receipt_sha256": campaign["env_receipt_sha256"],
              "model_config_sha256": reg.sha256_file(_abs(cell["config"])),
              "dataset_config_sha256": EVAL_CONFIG_SHA256[cell["K"]], "unseen_split_sha256": UNSEEN_SPLIT[1],
              "agree_sha256": reg.AGREE_SHA256, "vae_ckpt_sha256": reg.VAE_CKPT_SHA256, "vit_rev": reg.VIT_REV,
              "ckpt_sha256": ck_sha, "ckpt_bytes": ck_bytes, "training_receipt_sha256": tr_sha, **links}
    for k, v in want_r.items():
        if strict_equal(receipt[k], v):
            p.append(f"receipt {k} {str(receipt[k])[:80]!r} != {str(v)[:80]!r}")
    if gpu not in (0, 1) or type(gpu) is not int or receipt["gpu_uuid"] != reg.A6000_UUIDS[gpu if gpu in (0, 1) else 0]:
        p.append(f"receipt GPU {gpu} / {receipt['gpu_uuid']} is not this box's registered card")
    if not (reg._is_sha(receipt["log_sha256"]) and isinstance(receipt["log"], str)
            and isinstance(receipt["started_at"], str) and isinstance(receipt["finished_at"], str)):
        p.append("receipt log / time fields malformed")
    elif not offline and (not os.path.exists(_abs(receipt["log"])) or reg.sha256_file(_abs(receipt["log"])) != receipt["log_sha256"]):
        p.append("receipt log missing / changed")
    if cell["dump"]:
        try:
            copy = dump_manifest_copy(cell)
            man_path = copy if offline else os.path.join(cell["dump_dir"], "manifest.json")
            man = json.load(open(man_path))
            if not (reg.sha256_file(man_path) == receipt["dump_manifest_sha256"] == reg.sha256_file(copy)
                    and man.get("metrics_sha256") == links["metrics_sha256"]
                    and man.get("stream_sidecar_sha256") == links["stream_sha256"]
                    and man.get("input_hash") == stream.get("input_hash") and man.get("source_sha") == campaign["head"]):
                p.append("dump manifest cross-links")
            if not offline:
                p += dump_problems(cell, man, stream, links)
        except (OSError, ValueError) as err:
            p.append(f"dump manifest unreadable: {err}")
    elif receipt["dump_manifest_sha256"] is not None:
        p.append("a no-dump cell records a dump manifest")
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


def _copy_once(src, dest):
    data = open(src, "rb").read()
    if os.path.exists(dest):
        if open(dest, "rb").read() != data:
            raise EvalError(f"{dest} differs from {src}")
        return
    reg.atomic_write_bytes(dest, data)


def prepare(run):
    """Admit a run for evaluation: its sealed TRAINING_COMPLETE receipt at the E0 campaign closure,
    the final checkpoint's current sha256 AND byte count == the receipt's, a full-state audit; then
    the farm link + offline copies of the receipt and the checkpoint manifest (Codex code r1 BLOCKING 4)."""
    arm, tag, rung = reg.parse_run_id(run)
    campaign = validate_e0_receipt()
    rd = os.path.join(reg.PLATFORMS["a6000"]["save_root"], run)
    rec, digest = reg.check_training_receipt(run, rd, closure=campaign["closure_sha256"], verify_bytes=True)
    name = reg.ckpt_name(reg.MAX_STEPS, tag)
    # the checkpoint's ModelCheckpoint dirpath is the dir it was TRAINED in (neuronic runs are published
    # to the NAS afterwards): the receipt's own argv names it
    trained_in = rec["argv"][rec["argv"].index("--save-dir") + 1]
    if trained_in != reg.save_dir(rec["platform"], run):
        raise EvalError(f"{run} was trained in {trained_in}, not its registered {rec['platform']} run dir")
    audit = reg.audit_ckpt(os.path.join(rd, name), run, reg.MAX_STEPS, run_dir=trained_in)
    if (audit["sha256"], audit["bytes"]) != (rec["final_ckpt_sha256"], rec["final_ckpt_bytes"]):
        raise EvalError(f"{name}: audit {audit['sha256'][:12]} / {audit['bytes']} != the training receipt")
    _link(f"{FARM}/{run}/{name}", os.path.join(rd, name))
    for f in ("CHECKPOINTS.sha256", reg.TRAINING_RECEIPT, reg.TRAINING_RECEIPT + ".sha256"):
        _copy_once(os.path.join(rd, f), _abs(f"{FARM}/{run}/{f}"))
    return {"run": run, "admitted": audit, "training_receipt_sha256": digest}


def prepare_e0():
    out = {}
    for gate, g in E0.items():
        got = reg.sha256_file(g["ckpt"])
        if got != g["sha"] or os.path.getsize(g["ckpt"]) != E0_BYTES[gate]:
            raise EvalError(f"{gate}: {g['ckpt']} sha / bytes != {g['sha']} / {E0_BYTES[gate]}")
        _link(f"{FARM}/{gate}/{os.path.basename(g['ckpt'])}", g["ckpt"])
        out[gate] = got
    return out


# ---------------------------------------------------------------- running
def gpu_row(gpu):
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip().split("\n")
    row = [[x.strip() for x in r.split(",")] for r in rows if r.split(",")[0].strip() == str(gpu)]
    if len(row) != 1 or row[0][2] != GPU_NAME or row[0][1] != reg.A6000_UUIDS[gpu]:
        raise EvalError(f"GPU {gpu} is {row}, not this box's registered A6000 {reg.A6000_UUIDS[gpu]}")
    return row[0][1], row[0][2]


def gpu_foreign(gpu):
    uuid = reg.A6000_UUIDS[gpu]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip().split("\n")
    return [a.split(", ")[1] for a in apps if a and a.split(", ")[0] == uuid]


def eval_env(gpu):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",) + reg.FORBIDDEN_ENV}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": str(gpu)})
    return env


def run_one(cell, gpu, campaign):
    if not validate_cell(cell, campaign):
        return "skipped (valid)"
    for k in ("metrics", "stream", "receipt"):
        if os.path.exists(_abs(cell[k])):
            raise EvalError(f"{cell['name']}: a partial {k} exists ({cell[k]}); archive it first")
    if cell["dump"] and os.path.lexists(cell["dump_dir"]):
        raise EvalError(f"{cell['name']}: dump dir {cell['dump_dir']} exists; archive it first")
    ck_sha, ck_bytes, tr_sha = expected_ckpt(cell, campaign)
    pins = reg.eval_pins()
    uuid, name = gpu_row(gpu)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log = _abs(f"{EXP}/eval_logs/data_efficiency_{ts}_{cell['name']}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    argv = eval_argv(cell)
    with open(log, "w") as fh:
        fh.write("argv: " + " ".join(["python", NO_FLASH, "eval_FLAC.py"] + argv) + "\n")
        fh.flush()
        rc = subprocess.run([sys.executable, NO_FLASH, "eval_FLAC.py"] + argv, cwd=reg.REPO, env=eval_env(gpu),
                            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    if reg.eval_pins() != pins:
        raise EvalError(f"{cell['name']}: evaluator weights changed during the cell")
    text = open(log, errors="replace").read()
    sentinels = [line for line in text.split("\n") if line.startswith("EXP35_NO_FLASH_LAUNCHER")]
    exists = os.path.exists
    receipt = {"schema": 1, "cell": cell["name"], "argv": argv, "rc": rc,
               "sentinel": sentinels[0] if len(sentinels) == 1 else None,
               "log": os.path.relpath(log, reg.REPO), "log_sha256": reg.sha256_file(log), "gpu": gpu,
               "gpu_uuid": uuid, "gpu_name": name, "head": campaign["head"], "closure_sha256": campaign["closure_sha256"],
               "e0_receipt_sha256": campaign["sha256"], "env_receipt_sha256": campaign["env_receipt_sha256"],
               "model_config_sha256": reg.sha256_file(_abs(cell["config"])),
               "dataset_config_sha256": reg.sha256_file(_abs(EVAL_CONFIGS[cell["K"]])),
               "unseen_split_sha256": reg.sha256_file(_abs(UNSEEN_SPLIT[0])),
               "agree_sha256": pins["agree_sha256"], "vae_ckpt_sha256": pins["vae_ckpt_sha256"], "vit_rev": pins["vit_rev"],
               "metrics_sha256": reg.sha256_file(_abs(cell["metrics"])) if exists(_abs(cell["metrics"])) else None,
               "stream_sha256": reg.sha256_file(_abs(cell["stream"])) if exists(_abs(cell["stream"])) else None,
               "dump_manifest_sha256": None, "ckpt_sha256": ck_sha, "ckpt_bytes": ck_bytes,
               "training_receipt_sha256": tr_sha, "started_at": started,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if cell["dump"] and exists(os.path.join(cell["dump_dir"], "manifest.json")):
        _copy_once(os.path.join(cell["dump_dir"], "manifest.json"), dump_manifest_copy(cell))
        receipt["dump_manifest_sha256"] = reg.sha256_file(dump_manifest_copy(cell))
    reg.atomic_write_bytes(_abs(cell["receipt"]), (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode())
    probs = validate_cell(cell, campaign)
    if rc != 0 or probs:
        raise EvalError(f"{cell['name']}: rc {rc}; " + "; ".join(probs))
    return "done"


def campaign_gates():
    """The campaign tree at a clean HEAD, no shadows / unexpected entries, the environment receipt."""
    if os.path.realpath(reg.REPO) != reg.CAMPAIGN_ROOT["a6000"]:
        raise EvalError(f"evaluation runs only from the campaign tree {reg.CAMPAIGN_ROOT['a6000']}")
    if subprocess.run(["git", "-C", reg.REPO, "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True, check=True).stdout.strip():
        raise EvalError("tracked tree is not clean")
    probs = reg.check_untracked("a6000")
    if probs:
        raise EvalError("campaign tree: " + "; ".join(probs))
    reg.env_receipt("a6000")


def cmd_run(gpu, names, shared=False, lockdir=LOCKDIR):
    cells = {c["name"]: c for c in all_cells(bridge=True)}
    unknown = [n for n in names if n not in cells]
    if unknown:
        raise EvalError(f"unknown cells {unknown}")
    kinds = {bool(cells[n]["e0"]) for n in names}
    if len(kinds) != 1:
        raise EvalError("run E0 replays and result cells in separate calls (results need the E0 receipt)")
    campaign_gates()
    if kinds == {True}:
        campaign = current_campaign()
    else:
        campaign = validate_e0_receipt()
        if campaign["head"] != git_head():
            raise EvalError(f"HEAD {git_head()} != the E0 campaign's {campaign['head']}")
    n_dump = sum(1 for n in names if cells[n]["dump"])
    if n_dump:
        reg.storage_gate("a6000", DUMP_ROOT, int(n_dump * reg.DUMP_CELL_BYTES * 1.25), tag="exp38_dump")
    lockfile = os.path.join(lockdir, f".exp38_card{gpu}.lock")
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
            print(f"{n}: {run_one(cells[n], gpu, campaign)}", flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def cmd_e0():
    """Judge the eight replays (exact equality with the references) and seal the E0 receipt."""
    campaign_gates()
    campaign = current_campaign()
    refs = check_refs()
    verdicts, failed = {}, []
    for cell in e0_cells():
        probs = validate_cell(cell, campaign)
        ref, rel = e0_reference(cell, refs)
        rec = json.load(open(_abs(cell["metrics"]))) if not probs else {}
        diff = strict_equal(rec.get("metrics"), ref.get("metrics")) if not probs else "cell not admitted"
        if not diff and E0[cell["e0"]]["by_scene"]:
            diff = strict_equal(rec.get("by_scene"), ref.get("by_scene"))
        verdicts[cell["name"]] = {
            "reference": rel, "equal": diff is None, "difference": diff, "problems": probs,
            "metrics_sha256": reg.sha256_file(_abs(cell["metrics"])) if os.path.exists(_abs(cell["metrics"])) else None,
            "receipt_sha256": reg.sha256_file(_abs(cell["receipt"])) if os.path.exists(_abs(cell["receipt"])) else None}
        if probs or diff:
            failed.append(cell["name"])
    pins = reg.eval_pins()
    rec = {"schema": 1, "verdict": "PASS" if not failed else "FAIL", "failed": failed, "cells": verdicts,
           "head": campaign["head"], "closure_sha256": campaign["closure_sha256"],
           "evaluator_files": {f: reg.sha256_file(_abs(f)) for f in EVALUATOR_FILES},
           "env_receipt_sha256": campaign["env_receipt_sha256"],
           "checkpoints": {g: {"sha256": E0[g]["sha"], "bytes": E0_BYTES[g]} for g in E0},
           "reference_manifest_sha256": reg.sha256_file(_abs(REF_MANIFEST)), "references": refs,
           "input_hash": {f"K{k}_s{s}": v for (k, s), v in INPUT_HASH.items()},
           "eval_pins": {k: pins[k] for k in ("agree_sha256", "vae_ckpt_sha256", "vit_rev")},
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    reg.write_sealed(_abs(E0_RECEIPT), rec)
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
    sub.add_parser("e0-check")
    s = sub.add_parser("validate"); s.add_argument("names", nargs="+"); s.add_argument("--offline", action="store_true")
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
        elif a.cmd == "e0-check":
            print(json.dumps(validate_e0_receipt()))
        elif a.cmd == "validate":
            cells = {c["name"]: c for c in all_cells(bridge=True)}
            probs = [x for n in a.names for x in validate_cell(cells[n], offline=a.offline)]
            print("\n".join(probs) or "VALID")
            return 0 if not probs else 1
    except (EvalError, reg.GateError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as err:
        print(f"EXP38 EVAL REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
