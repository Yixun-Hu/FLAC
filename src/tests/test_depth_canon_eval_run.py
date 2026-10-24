"""exp_25 depth_canon round 3c: the evaluation cell manifest and the eval driver
(plan_depth_canon.md v9 §5 "cell classes as data", "Drivers:", atomic cell completion).

Part 1 pins the manifest ``depth_canon_cells.json`` against its generator
``depth_canon_make_cells.py`` and against independently derived expectations (block
counts, flags per block class, stems through eval_FLAC's own ``build_output_paths``).

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9 §5.
"""
import collections
import hashlib
import importlib.util
import json
import os

import pytest

import eval_FLAC

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")
EXPDIR = os.path.join(REPO, EXPREL)
MAKE_CELLS = os.path.join(EXPDIR, "depth_canon_make_cells.py")
CELLS_JSON = os.path.join(EXPDIR, "depth_canon_cells.json")

BLOCKS = ("T", "T-dup", "C4+45", "R", "D-i", "D-ii", "S", "P1-pin", "YAW-pin", "TRAJ")
COUNTS = {"T": 10, "T-dup": 2, "C4+45": 8, "R": 10, "D-i": 2, "D-ii": 2, "S": 10,
          "P1-pin": 28, "YAW-pin": 38, "TRAJ": 120}
SEEDS = (42, 43, 44, 45, 46)
# plan §5 common set (+ --batch-size 64 and --device cuda written out: never defaulted)
COMMON = {"--cond-method": "vanilla", "--cond-autocast": "bf16", "--cfg-scale": "1.0",
          "--steps": "1", "--batch-size": "64", "--device": "cuda", "--record-per-scene": None,
          "--record-stream": None}
NAS = "/media/diskstation/yixunhu/FLAC/checkpoints"
# NAS ar_40k_endpoints/MANIFEST.sha256 (read 2026-10-23) == depth_canon_comparator_ledger.md
ENDPOINT_SHA = {"P1": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
                "YAW": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7"}
MODEL_CONFIG = {"CANON": f"{EXPREL}/FLAC_AR_CANON.json",
                "P1": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
                "YAW": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"}
DATASET = {("unseen", 1): "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
           ("unseen", 8): "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
           ("seen", 1): "src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json",
           ("seen", 8): "src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json"}
SPLIT_JSON = {"unseen": "data/AR/unseen_eval.json", "seen": "data/AR/seen_eval.json"}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load(MAKE_CELLS, "depth_canon_make_cells")


@pytest.fixture(scope="module")
def manifest():
    return MC.build_manifest()


def _sha(path):
    with open(os.path.join(REPO, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _flags(row):
    """``{flag: value-or-None}`` of a row's evaluator argv; every flag exactly once."""
    args, out, i = row["eval_args"], {}, 0
    while i < len(args):
        flag = args[i]
        assert flag.startswith("--") and flag not in out, (row["cell"], flag)
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[flag], i = args[i + 1], i + 2
        else:
            out[flag], i = None, i + 1
    return out


def _n_items(split):
    with open(os.path.join(REPO, SPLIT_JSON[split])) as f:
        data = json.load(f)
    return sum(len(files) for rooms in data.values() for files in rooms.values())


# --------------------------------------------------------------------------- #
# structure: 230 rows, block counts, priority order, identity
# --------------------------------------------------------------------------- #
def test_block_counts_and_priority_order(manifest):
    cells = manifest["cells"]
    assert collections.Counter(c["block"] for c in cells) == COUNTS
    assert tuple(manifest["blocks"]) == BLOCKS
    assert [c["order"] for c in cells] == list(range(230))
    assert [c["priority"] for c in cells] == sorted(c["priority"] for c in cells)
    assert all(c["priority"] == BLOCKS.index(c["block"]) for c in cells)
    assert manifest["block_counts"] == COUNTS and sum(COUNTS.values()) == 230


def test_eval_names_and_metrics_stems_are_injective(manifest):
    cells = manifest["cells"]
    names = [c["cell"] for c in cells]
    assert len(set(names)) == 230
    assert all(n.startswith("exp25_") and "exp14_" not in n and "exp11_" not in n for n in names)
    stems, suffixes = set(), set()
    for c in cells:
        base = f"epoch={c['step'] // 4551}-step={c['step']}" if c["arm"] == "CANON" \
            else "epoch=8-step=40000"
        paths = eval_FLAC.build_output_paths(
            f"/farm/{c['arm']}/{base}.ckpt", 1, 1.0, c["cell"], cond_method="vanilla",
            rotate_deg=float(c["rotate_deg"] or 0), n_angles=4, rotate_mode=c["rotate_mode"],
            rotate_seed=c["rotate_seed"], are_lambda=None, canonicalize=c["canonicalize"],
            canon_mismatch=c["canon_mismatch"])
        name = os.path.basename(paths["metrics"])
        assert name == f"{base}_metrics_{c['metrics_suffix']}.json", c["cell"]
        stems.add(name[:-len(".json")])
        suffixes.add(c["metrics_suffix"])
    assert len(stems) == len(suffixes) == 230


def test_canon_tokens_follow_the_protocol(manifest):
    for c in manifest["cells"]:
        want = ("_canon" if c["canonicalize"] else "") + ("_canonmis" if c["canon_mismatch"] else "")
        rot = ("_rotrand%d" % c["seed"] if c["rotate_mode"] == "random"
               else ("" if c["rotate_deg"] == 0 else f"_rot{c['rotate_deg']}"))
        assert c["metrics_suffix"] == f"1_1.0_{c['cell']}{want}{rot}", c["cell"]


# --------------------------------------------------------------------------- #
# flags per block class (plan §5 table; announcement 05: never defaulted)
# --------------------------------------------------------------------------- #
def test_common_flags_on_every_row(manifest):
    for c in manifest["cells"]:
        f = _flags(c)
        for flag, value in COMMON.items():
            assert f.get(flag, "MISSING") == value, (c["cell"], flag)
        assert f["--eval-name"] == c["cell"] and f["--seed"] == str(c["seed"])
        assert f["--model-config"] == MODEL_CONFIG[c["arm"]] == c["model_config"]
        assert f["--dataset-config"] == DATASET[(c["split"], c["K"])] == c["dataset_config"]
        assert f["--expected-stream-count"] == str(c["expected_stream_count"])
        assert "--ckpt-path" not in f and "--dump-scored-waveforms" not in f   # runtime
        assert "--frame-avg-angles" not in f and "--are-lambda" not in f       # plan §5: inert
        assert c["dump"] is True                                               # A3 default


def _rot(f):
    if f["--rotate-mode"] == "random":
        assert "--rotate-deg" not in f
        return ("random", int(f["--rotate-seed"]))
    assert "--rotate-seed" not in f
    return ("fixed", int(f["--rotate-deg"]))


def test_flags_per_block_class(manifest):
    by = collections.defaultdict(list)
    for c in manifest["cells"]:
        by[c["block"]].append(c)
    for block, cells in by.items():
        for c in cells:
            f = _flags(c)
            canon, allow = "--canonicalize" in f, "--allow-canon-mismatch" in f
            assert (canon, allow) == (c["canonicalize"], c["allow_canon_mismatch"])
            assert f.get("--canonicalize", None) is None and f.get("--allow-canon-mismatch") is None
            rot = _rot(f)
            if c["arm"] == "CANON" and block != "D-i":
                assert canon and not allow, c["cell"]
            if block in ("T", "T-dup", "S"):
                assert (c["arm"], rot) == ("CANON", ("fixed", 0))
            if block == "T-dup":
                assert c["seed"] == 42 and c["cell"].endswith("_dup")
            if block == "C4+45":
                assert c["arm"] == "CANON" and c["seed"] == 42 and rot[0] == "fixed"
            if block == "R":
                assert c["arm"] == "CANON" and rot == ("random", c["seed"])
            if block == "D-i":
                assert (c["arm"], canon, allow, rot, c["seed"]) == ("CANON", False, True, ("fixed", 0), 42)
            if block == "D-ii":
                assert (c["arm"], canon, allow, rot, c["seed"]) == ("P1", True, True, ("fixed", 0), 42)
            if block in ("P1-pin", "YAW-pin"):
                assert c["arm"] == block.split("-")[0] and not canon and not allow
            if block == "TRAJ":
                assert c["arm"] == "CANON" and c["seed"] == 42 and rot[0] == "fixed"
            assert c["split"] == ("seen" if block == "S" or c["subblock"] == "S" else "unseen")
    c4 = {(c["K"], _rot(_flags(c))[1]) for c in by["C4+45"]}
    assert c4 == {(k, d) for k in (1, 8) for d in (90, 180, 270, 45)}
    assert {(c["K"], c["seed"]) for c in by["T"]} == {(k, s) for k in (1, 8) for s in SEEDS}
    assert {(c["K"], c["seed"]) for c in by["R"]} == {(k, s) for k in (1, 8) for s in SEEDS}
    traj = {(c["step"], c["K"], _rot(_flags(c))[1]) for c in by["TRAJ"]}
    assert traj == {(s, k, d) for s in range(2500, 40000, 2500) for k in (1, 8)
                    for d in (0, 90, 180, 270)}                        # 40k = the T / C4 cells
    for block in ("P1-pin", "YAW-pin"):
        subs = collections.Counter(c["subblock"] for c in by[block])
        assert subs == ({"T": 10, "R": 10, "C4": 8} | ({"S": 10} if block == "YAW-pin" else {}))
    assert all(c["step"] == 40000 for c in manifest["cells"] if c["block"] != "TRAJ")


def test_canon_protocol_expectations(manifest):
    for c in manifest["cells"]:
        if c["arm"] == "CANON":
            want = (True, True, not c["canonicalize"])
        elif c["block"] == "D-ii":
            want = (True, False, True)
        else:
            want = (False, None, False)
        assert (c["canon_in_play"], c["canon_trained"], c["canon_mismatch"]) == want, c["cell"]


# --------------------------------------------------------------------------- #
# inputs: configs, split sizes, checkpoints
# --------------------------------------------------------------------------- #
def test_dataset_configs_and_expected_counts(manifest):
    assert _n_items("unseen") == 6337 and _n_items("seen") == 6217   # = the tie audit's ledger
    for c in manifest["cells"]:
        with open(os.path.join(REPO, c["dataset_config"])) as f:
            cfg = json.load(f)
        assert cfg["modalities"]["acoustic_context"]["max_context"] == c["K"]
        assert cfg["datasets"][0]["json_file_path"] == SPLIT_JSON[c["split"]]
        assert cfg["drop_last"] is False
        assert c["expected_stream_count"] == _n_items(c["split"])
        assert c["dataset_config_sha256"] == _sha(c["dataset_config"])
        assert c["model_config_sha256"] == _sha(c["model_config"])


def test_arm_checkpoints(manifest):
    arms = manifest["arms"]
    assert set(arms) == {"CANON", "P1", "YAW"}
    for arm in ("P1", "YAW"):
        ck = arms[arm]["ckpt"]
        assert ck == {"kind": "endpoint", "path": f"{NAS}/ar_40k_endpoints/{arm}/epoch=8-step=40000.ckpt",
                      "sha256": ENDPOINT_SHA[arm], "nas_manifest": f"{NAS}/ar_40k_endpoints/MANIFEST.sha256",
                      "nas_manifest_key": f"{arm}/epoch=8-step=40000.ckpt"}
    # the launcher's run namespace (depth_canon_launch.sh: <save>/<W&B project>/<run>)
    assert arms["CANON"]["ckpt"] == {"kind": "canon_run",
                                     "run_ns": f"{NAS}/exp25_depth_canon/FLAC_exp25_CANON/exp25_CANON"}
    with open(os.path.join(EXPDIR, "depth_canon_launch.sh")) as f:
        launcher = f.read()
    assert f'SAVE_ROOT_PIN="{NAS}/exp25_depth_canon"' in launcher
    assert 'NAME="FLAC_exp25_CANON"; EXPNAME="exp25_CANON"' in launcher
    for arm, cfg in MODEL_CONFIG.items():
        assert arms[arm]["model_config"] == cfg


def test_manifest_defaults(manifest):
    assert manifest["dump_root"] == "/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms"
    assert manifest["farm_root"] == "outputs_FLAC/exp25_depth_canon"
    assert manifest["img_w"] == 512 and manifest["schema_version"] == 1
