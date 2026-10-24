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
# the committed manifest IS the generator's output
# --------------------------------------------------------------------------- #
def test_committed_manifest_equals_the_generator_output(manifest):
    with open(CELLS_JSON) as f:
        committed = f.read()
    assert committed == MC.render(manifest)
    assert json.loads(committed) == manifest
    assert MC.main(["--check"]) == 0


def test_render_is_one_cell_per_line(manifest):
    lines = MC.render(manifest).splitlines()
    cell_lines = [ln for ln in lines if ln.startswith('  {"cell":')]
    assert len(cell_lines) == len(manifest["cells"]) == 230


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


# --------------------------------------------------------------------------- #
# part 2: the driver's data feed -- depth_canon_collect.py --plan / --write-receipt
# --------------------------------------------------------------------------- #
C = _load(os.path.join(EXPDIR, "depth_canon_collect.py"), "depth_canon_collect")
PIN = "a" * 40
DIGESTS = dict(campaign_digest="d" * 64, spawn_digest="d" * 64, post_digest="d" * 64)   # rB-F1 B3


def _canon_index(manifest):
    ns = manifest["arms"]["CANON"]["ckpt"]["run_ns"]
    return {s: (f"{ns}/checkpoints/epoch={s // 4551}-step={s}.ckpt",
                hashlib.sha256(str(s).encode()).hexdigest()) for s in range(2500, 40001, 2500)}


def test_selection_by_blocks_only_and_limit(manifest):
    sel = C.select_cells(manifest, blocks=["P1-pin", "YAW-pin"])
    assert [c["block"] for c in sel] == ["P1-pin"] * 28 + ["YAW-pin"] * 38
    a1_off = C.select_cells(manifest, blocks=[b for b in BLOCKS if b != "YAW-pin"])
    assert len(a1_off) == 192 and all(c["block"] != "YAW-pin" for c in a1_off)
    assert [c["cell"] for c in C.select_cells(manifest)] == [c["cell"] for c in manifest["cells"]]
    names = ["exp25_YAW_S_S40000_K8_s46", "exp25_CANON_T_S40000_K1_s42"]
    assert [c["cell"] for c in C.select_cells(manifest, only=names)] == names[::-1]  # manifest order
    assert len(C.select_cells(manifest, blocks=["TRAJ"], limit=7)) == 7
    for kw in (dict(blocks=["T", "nope"]), dict(only=["exp25_nope"]), dict(limit=0), dict(blocks=[])):
        with pytest.raises(ValueError):
            C.select_cells(manifest, **kw)


def test_plan_entries(manifest):
    sel = C.select_cells(manifest, blocks=["T", "D-ii", "TRAJ"])
    plan = C.plan_cells(manifest, sel, farm_root="farm", dump_root="/nas/dumps",
                        canon_index=_canon_index(manifest), no_dump_blocks=["TRAJ"])
    assert [e["cell"] for e in plan] == [c["cell"] for c in sel]
    by = {e["cell"]: e for e in plan}
    t = by["exp25_CANON_T_S40000_K8_s44"]
    assert t["target"] == _canon_index(manifest)[40000][0] and t["sha256"] == _canon_index(manifest)[40000][1]
    assert t["link"] == "farm/CANON/epoch=8-step=40000.ckpt" and t["dump"] is True
    assert t["metrics"] == f"farm/CANON/epoch=8-step=40000_metrics_{manifest_row(manifest, t['cell'])['metrics_suffix']}.json"
    assert t["dump_dir"] == "/nas/dumps/" + os.path.basename(t["metrics"])[:-5] == "/nas/dumps/" + t["stem"]
    assert t["argv"] == manifest_row(manifest, t["cell"])["eval_args"] + [
        "--ckpt-path", t["link"], "--dump-scored-waveforms", "/nas/dumps"]
    assert t["canon"] == t["metrics"][:-5] + ".canon.json" and t["receipt"] == t["metrics"][:-5] + ".cell.json"
    d = by["exp25_P1_Dii_S40000_K1_s42"]
    assert (d["target"], d["sha256"]) == (manifest["arms"]["P1"]["ckpt"]["path"], manifest["arms"]["P1"]["ckpt"]["sha256"])
    assert d["nas_manifest_key"] == "P1/epoch=8-step=40000.ckpt" and d["canon"].endswith(".canon.json")
    traj = [e for e in plan if e["block"] == "TRAJ"]
    assert traj and all(e["dump"] is False and "--dump-scored-waveforms" not in e["argv"] for e in traj)
    pins = C.plan_cells(manifest, C.select_cells(manifest, blocks=["P1-pin"]), farm_root="farm",
                        dump_root="/d", canon_index=None, no_dump_blocks=[])
    assert all(e["canon"] is None for e in pins)                      # CANON index not needed
    with pytest.raises(KeyError):
        C.plan_cells(manifest, sel, farm_root="farm", dump_root="/d", canon_index=None, no_dump_blocks=[])
    with pytest.raises(ValueError):
        C.plan_cells(manifest, sel, farm_root="farm", dump_root="/d", canon_index={}, no_dump_blocks=["X"])


def manifest_row(manifest, cell):
    return next(c for c in manifest["cells"] if c["cell"] == cell)


def test_write_receipt_produces_what_verify_accepts(manifest, tmp_path):
    sel = C.select_cells(manifest, only=["exp25_YAW_C4_S40000_K1_s42_th45"])
    entry = C.plan_cells(manifest, sel, farm_root="farm", dump_root=str(tmp_path / "d"),
                         canon_index=None, no_dump_blocks=[])[0]
    os.makedirs(tmp_path / "farm" / "YAW")
    path = C.write_receipt(entry, manifest_row(manifest, entry["cell"]), base_dir=str(tmp_path), lane=1,
                           log="x.log", rc=0, pin=PIN, ckpt_sha256=entry["sha256"], **DIGESTS)
    assert path == str(tmp_path / entry["receipt"]) and sorted(os.listdir(tmp_path / "farm" / "YAW")) == [
        os.path.basename(entry["receipt"])]                            # atomic: no temp left behind
    rec = json.load(open(path))
    assert C.receipt_reasons(rec, manifest_row(manifest, entry["cell"]), stem=entry["stem"],
                             link=entry["link"], target=entry["target"], ckpt_sha256=entry["sha256"],
                             dump=True, dump_root=str(tmp_path / "d"), pin=PIN, expect_digest="d" * 64) == []
    with pytest.raises(ValueError, match="sha256"):
        C.write_receipt(entry, manifest_row(manifest, entry["cell"]), base_dir=str(tmp_path), lane=1,
                        log="x.log", rc=0, pin=PIN, ckpt_sha256="0" * 64, **DIGESTS)


def test_cli_plan_and_receipt(manifest, tmp_path, capsys):
    out = str(tmp_path / "plan.json")
    assert C.main(["--plan", out, "--blocks", "P1-pin", "--limit", "3", "--no-dump-blocks", "",
                   "--farm-root", "farm", "--dump-root", "/d"]) == 2          # rB-F1 B3: the digest is required
    assert C.main(["--plan", out, "--blocks", "P1-pin", "--limit", "3", "--no-dump-blocks", "",
                   "--farm-root", "farm", "--dump-root", "/d", "--campaign-digest", "d" * 64]) == 0
    plan = json.load(open(out))
    assert plan["campaign_digest"] == "d" * 64
    assert [e["cell"] for e in plan["cells"]] == [c["cell"] for c in manifest["cells"] if c["block"] == "P1-pin"][:3]
    assert (plan["farm_root"], plan["dump_root"], plan["n_selected"]) == ("farm", "/d", 3)
    assert C.main(["--plan", out, "--blocks", "T", "--canon-ckpt-manifest",
                   str(tmp_path / "absent.sha256"), "--campaign-digest", "d" * 64]) == 2   # CANON cells need it
    assert "CANON" in capsys.readouterr().out
    assert C.main(["--plan", out, "--blocks", "T,bogus", "--campaign-digest", "d" * 64]) == 2


def test_cli_write_receipt_can_verify_in_the_same_process(manifest, tmp_path, capsys):
    """One interpreter start (~5 s of imports) per finished cell instead of two."""
    out = str(tmp_path / "plan.json")
    assert C.main(["--plan", out, "--only", "exp25_P1_T_S40000_K1_s42", "--farm-root", "farm",
                   "--dump-root", str(tmp_path / "d"), "--campaign-digest", "d" * 64]) == 0
    os.makedirs(tmp_path / "farm" / "P1")
    sha = manifest["arms"]["P1"]["ckpt"]["sha256"]
    argv = ["--write-receipt", "exp25_P1_T_S40000_K1_s42", "--plan-json", out, "--base-dir", str(tmp_path),
            "--lane", "0", "--log", "x.log", "--rc", "0", "--pin", PIN, "--ckpt-sha", sha,
            "--farm-root", "farm", "--dump-root", str(tmp_path / "d"), "--no-link-check",
            "--spawn-digest", "d" * 64, "--post-digest", "d" * 64]
    assert C.main(argv + ["--verify-after"]) == 1           # nothing but the receipt exists yet
    text = capsys.readouterr().out
    assert "receipt:" in text and "INVALID" in text and "missing or unreadable metrics" in text
    assert C.main(argv) == 0                                # without --verify-after: just the receipt


# --------------------------------------------------------------------------- #
# part 3: the driver depth_canon_eval_run.sh, run as a subprocess against a THROWAWAY
# clone (git clone --shared + sparse checkout; the kit's working-tree files overlaid and
# committed inside it) in harness mode: stub evaluator + stub findmnt / nvidia-smi on
# PATH, fake checkpoints, farm / dumps / logs / lock all under one /tmp harness dir.
# Nothing in the source repo, the NAS or a GPU is touched.
# --------------------------------------------------------------------------- #
import subprocess  # noqa: E402

DRIVER_REL = f"{EXPREL}/depth_canon_eval_run.sh"
KIT = ("depth_canon_eval_run.sh", "depth_canon_collect.py", "depth_canon_make_cells.py",
       "depth_canon_cells.json", "depth_canon_audit_gate.py")
TESTMOD = os.path.join(REPO, "src", "tests", "test_depth_canon_collect.py")
STUB_EVAL = r'''#!/usr/bin/env python3
"""Harness stub for eval_FLAC.py: records its call, then publishes exactly what the real
evaluator publishes for the cell (test_depth_canon_collect.write_cell, real eval_FLAC code)."""
import importlib.util, json, os, sys
H = os.environ["DEPTH_CANON_EVAL_HARNESS"]
argv = sys.argv[1:]
with open(os.path.join(H, "calls.log"), "a") as f:
    f.write(json.dumps({"cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": argv}) + "\n")
if os.environ.get("STUB_MODE") == "crash":
    sys.exit(3)
if os.environ.get("STUB_MODE") == "mutate":        # an edit of the evaluated tree DURING the cell
    open("eval_FLAC.py", "a").write("\n# drift during the cell\n")
opts, i = {}, 0
while i < len(argv):
    nxt = argv[i + 1] if i + 1 < len(argv) else "--"
    opts[argv[i]], i = (nxt, i + 2) if not nxt.startswith("--") else (True, i + 1)
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname("@TESTMOD@"))   # its fixtures import sibling test modules (rB-F1 B1)
spec = importlib.util.spec_from_file_location("tdc", "@TESTMOD@")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
import eval_FLAC
manifest = json.load(open(os.environ["DEPTH_CANON_CELLS"]))
row = next(c for c in manifest["cells"] if c["cell"] == opts["--eval-name"])
T.write_cell(os.getcwd(), row, pin=eval_FLAC.source_sha(), n=row["expected_stream_count"],
             link=opts["--ckpt-path"], dump="--dump-scored-waveforms" in opts,
             dump_root=opts.get("--dump-scored-waveforms"), receipt=False)
'''
STUB_FINDMNT = "#!/usr/bin/env bash\nprintf '%s\\n' \"${GUARD_FINDMNT_LINE:-cifs   //10.249.7.28/shared_data}\"\n"
STUB_NVSMI = r"""#!/usr/bin/env bash
# harness nvidia-smi: index->uuid map, per-call compute-app scenarios (apps.<n>, else
# apps.default), memory.free; GUARD_* switches inject failures. It never touches a GPU.
H="$DEPTH_CANON_EVAL_HARNESS"
echo "$*" >> "$H/nvsmi.log"
[ "${GUARD_NVSMI_FAIL:-0}" = 1 ] && exit 1
case "$*" in
  *index,uuid*)
    [ "${GUARD_UUID_FAIL:-0}" = 1 ] && { echo "0, GPU-00000000-aaaa"; exit 9; }
    if [ -f "$H/uuids" ]; then cat "$H/uuids"; else printf '0, GPU-00000000-aaaa\n1, GPU-11111111-bbbb\n'; fi ;;
  *query-compute-apps*)
    n=$(( $(cat "$H/apps.count" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$H/apps.count"
    [ "${GUARD_MUTATE_AT:-0}" = "$n" ] && printf '\n# drift between cells\n' >> eval_FLAC.py   # cwd = the clone
    f="$H/apps.$n"; [ -f "$f" ] || f="$H/apps.default"
    if [ -f "$f" ]; then cat "$f"; fi
    exit "${GUARD_APPS_RC:-0}" ;;
  *memory.free*) printf '%b\n' "${GUARD_GPU_FREE:-40000}" ;;
esac
"""
UUID = {0: "GPU-00000000-aaaa", 1: "GPU-11111111-bbbb"}


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalrun") / "repo"
    _git(REPO, "rev-parse", "HEAD")
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", REPO, str(root)], check=True)
    _git(root, "sparse-checkout", "set", "--cone", "src", "data/AR",
         "worklog/worklog_yixun/exp_07_fa_scratch_claude", "worklog/worklog_yixun/exp_15_yaw_aug_claude",
         EXPREL)
    _git(root, "checkout", "-q", "--detach", _git(REPO, "rev-parse", "HEAD"))
    for f in KIT:
        if os.path.exists(os.path.join(EXPDIR, f)):
            with open(os.path.join(EXPDIR, f), "rb") as src, open(root / EXPREL / f, "wb") as dst:
                dst.write(src.read())
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "--allow-empty",
         "-m", "kit overlay")
    return root


def _fake_ckpt(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return hashlib.sha256(content).hexdigest()


@pytest.fixture()
def harness(tmp_path, clone):
    """A harness dir + its cells manifest (N = 8, checkpoints redirected to fake files)."""
    h = tmp_path / "h"
    (h / "bin").mkdir(parents=True)
    (h / ".depth_canon_eval_harness").write_text("")
    for name, body in (("findmnt", STUB_FINDMNT), ("nvidia-smi", STUB_NVSMI)):
        (h / "bin" / name).write_text(body)
        os.chmod(h / "bin" / name, 0o755)
    (h / "stub_eval.py").write_text(STUB_EVAL.replace("@TESTMOD@", TESTMOD))
    m = json.loads(json.dumps(MC.build_manifest()))
    for split in m["splits"].values():
        split["expected_stream_count"] = 8
    lines = []
    for arm in ("P1", "YAW"):
        ck = m["arms"][arm]["ckpt"]
        ck["path"] = str(h / "nas" / arm / "epoch=8-step=40000.ckpt")
        ck["sha256"] = _fake_ckpt(ck["path"], f"fake {arm}".encode())
        ck["nas_manifest"] = str(h / "nas" / "MANIFEST.sha256")
        lines.append(f"{ck['sha256']}  {ck['nas_manifest_key']}\n")
    (h / "nas" / "MANIFEST.sha256").write_text("".join(lines))
    ns = m["arms"]["CANON"]["ckpt"]["run_ns"] = str(h / "nas" / "canon")
    canon = [f"{_fake_ckpt(f'{ns}/checkpoints/epoch={s // 4551}-step={s}.ckpt', str(s).encode())}  "
             f"{ns}/checkpoints/epoch={s // 4551}-step={s}.ckpt\n" for s in range(2500, 40001, 2500)]
    (h / "canon_ckpts.sha256").write_text("".join(canon))
    for c in m["cells"]:
        c["expected_stream_count"] = 8
        c["eval_args"][c["eval_args"].index("--expected-stream-count") + 1] = "8"
    (h / "cells.json").write_text(json.dumps(m))
    return {"dir": h, "manifest": m, "head": _git(clone, "rev-parse", "HEAD")}


def drive(clone, harness, *args, env=None, timeout=600):
    h = harness["dir"]
    e = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "EXPECT_HEAD")}
    e.update(PATH=f"{h / 'bin'}:{os.environ['PATH']}", DEPTH_CANON_EVAL_HARNESS=str(h),
             DEPTH_CANON_EVAL_PY=str(h / "stub_eval.py"), DEPTH_CANON_CELLS=str(h / "cells.json"),
             EXPECT_HEAD=harness["head"], GPU_POLL_SECONDS="0", GPU_MAX_POLLS="2",
             CUDA_VISIBLE_DEVICES="")
    e.update(env or {})
    base = ["--dump-root", str(h / "dumps"), "--canon-ckpt-manifest", str(h / "canon_ckpts.sha256")]
    p = subprocess.run(["bash", str(clone / DRIVER_REL), *base, *args], env=e, capture_output=True,
                       text=True, timeout=timeout, cwd=str(h))
    return p.returncode, p.stdout + p.stderr


ONE_PER_CLASS = ["exp25_CANON_T_S40000_K1_s42", "exp25_CANON_T_S40000_K8_s42_dup",
                 "exp25_CANON_C4_S40000_K1_s42_th45", "exp25_CANON_R_S40000_K8_s44",
                 "exp25_CANON_Di_S40000_K1_s42", "exp25_P1_Dii_S40000_K8_s42", "exp25_CANON_S_S40000_K1_s43",
                 "exp25_P1_R_S40000_K1_s45", "exp25_YAW_S_S40000_K8_s46", "exp25_CANON_TRAJ_S12500_K8_s42_th180"]


def _dry_argv(out):
    return {ln.split()[1]: json.loads(ln.split(" argv: ", 1)[1]) for ln in out.splitlines()
            if ln.startswith("DRYRUN ")}


def test_dry_run_prints_the_exact_argv_per_block_class(clone, harness):
    rc, out = drive(clone, harness, "--only", ",".join(ONE_PER_CLASS), "--no-dump-blocks", "TRAJ",
                    env={"DRY_RUN": "1"})
    assert rc == 0, out
    got, h, m = _dry_argv(out), harness["dir"], harness["manifest"]
    assert sorted(got) == sorted(ONE_PER_CLASS)
    for cell in ONE_PER_CLASS:
        row = manifest_row(m, cell)
        target = (m["arms"][row["arm"]]["ckpt"]["path"] if row["arm"] != "CANON" else
                  f"{h}/nas/canon/checkpoints/epoch={row['step'] // 4551}-step={row['step']}.ckpt")
        link = f"{h}/farm/{row['arm']}/{os.path.basename(target)}"
        dump = [] if row["block"] == "TRAJ" else ["--dump-scored-waveforms", str(h / "dumps")]
        assert got[cell] == row["eval_args"] + ["--ckpt-path", link] + dump, cell
        f = got[cell]
        assert ("--canonicalize" in f) == (row["arm"] == "CANON" and row["block"] != "D-i"
                                          or row["block"] == "D-ii")
    assert "would run" in out and not (h / "calls.log").exists()        # nothing evaluated
    assert not (h / "farm").exists() and not (h / "dumps").exists()     # side-effect free
    assert not (h / "nvsmi.log").exists()                               # GPU gate skipped


@pytest.mark.parametrize("env, needle", [
    ({"EXPECT_HEAD": "0" * 40}, "HEAD PIN"), ({"EXPECT_HEAD": ""}, "EXPECT_HEAD"),
    ({"DEPTH_CANON_EVAL_HARNESS": "/var/tmp"}, "HARNESS")])
def test_pin_and_harness_refusals(clone, harness, env, needle):
    rc, out = drive(clone, harness, "--only", ONE_PER_CLASS[0], env={"DRY_RUN": "1", **env})
    assert rc == 2 and needle in out, out


def test_an_uncommitted_evaluator_aborts(clone, harness):
    path = clone / "eval_FLAC.py"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# drift\n")
        rc, out = drive(clone, harness, "--only", ONE_PER_CLASS[0], env={"DRY_RUN": "1"})
        assert rc == 2 and "NOT COMMITTED" in out and "eval_FLAC.py" in out, out
    finally:
        path.write_bytes(original)


def test_checkpoint_sha_mismatch_and_unlisted_checkpoint_abort(clone, harness):
    p1 = harness["manifest"]["arms"]["P1"]["ckpt"]["path"]
    with open(p1, "ab") as f:
        f.write(b"tampered")
    rc, out = drive(clone, harness, "--only", "exp25_P1_T_S40000_K1_s42", env={"DRY_RUN": "1"})
    assert rc == 2 and "CHECKPOINT SHA" in out, out
    nas = harness["dir"] / "nas" / "MANIFEST.sha256"
    nas.write_text("\n".join(ln for ln in nas.read_text().splitlines() if "YAW" not in ln) + "\n")
    rc, out = drive(clone, harness, "--only", "exp25_YAW_T_S40000_K1_s42", env={"DRY_RUN": "1"})
    assert rc == 2 and "NAS MANIFEST" in out, out


def test_config_drift_and_bad_arguments_abort(clone, harness):
    m = harness["manifest"]
    m["cells"][0]["model_config_sha256"] = "0" * 64
    (harness["dir"] / "cells.json").write_text(json.dumps(m))
    rc, out = drive(clone, harness, "--only", ONE_PER_CLASS[0], env={"DRY_RUN": "1"})
    assert rc == 2 and "CONFIG" in out, out
    for args in (["--lanes", "0,x"], ["--blocks", "T,nope"], ["--bogus"]):
        rc, out = drive(clone, harness, *args, env={"DRY_RUN": "1"})
        assert rc == 2, (args, out)


def test_gpu_gate_is_skipped_under_dry_run(clone, harness):
    rc, out = drive(clone, harness, "--only", ONE_PER_CLASS[0], env={"DRY_RUN": "1", "GUARD_NVSMI_FAIL": "1"})
    assert rc == 0 and not (harness["dir"] / "nvsmi.log").exists(), out


# ---- cycles 4b/4c: the resume predicate, quarantine, the GPU lanes (stub evaluator) ---- #
TWO = ["exp25_P1_T_S40000_K1_s42", "exp25_CANON_C4_S40000_K1_s42_th90"]


def _calls(harness):
    path = harness["dir"] / "calls.log"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _entry(clone, harness, cell):
    """The cell's plan entry, as the driver computes it (farm = the harness farm)."""
    h, m = harness["dir"], harness["manifest"]
    sel = C.select_cells(m, only=[cell])
    index = C.load_canon_index(str(h / "canon_ckpts.sha256"), m["arms"]["CANON"]["ckpt"]["run_ns"])
    return C.plan_cells(m, sel, farm_root=str(h / "farm"), dump_root=str(h / "dumps"),
                        canon_index=index, no_dump_blocks=[])[0]


def _verify(clone, harness, cell, **kw):
    h, m = harness["dir"], harness["manifest"]
    index = C.load_canon_index(str(h / "canon_ckpts.sha256"), m["arms"]["CANON"]["ckpt"]["run_ns"])
    return C.verify_cell(_entry(clone, harness, cell)["stem"], manifest=m, farm_root=str(h / "farm"),
                         dump_root=str(h / "dumps"), pin=harness["head"], canon_index=index,
                         base_dir=str(clone), **kw)


def test_a_run_completes_cells_on_both_lanes_and_a_rerun_skips_them(clone, harness):
    rc, out = drive(clone, harness, "--only", ",".join(TWO))
    assert rc == 0 and "CAMPAIGN SELECTION COMPLETE" in out and "2/2" in out, out
    calls = _calls(harness)
    assert sorted(c["cvd"] for c in calls) == ["0", "1"]                       # one lane per card
    assert sorted(c["argv"][c["argv"].index("--eval-name") + 1] for c in calls) == sorted(TWO)
    for cell in TWO:
        assert _verify(clone, harness, cell) == [], cell                         # links checked too
        rec = json.load(open(_entry(clone, harness, cell)["receipt"]))
        assert rec["rc"] == 0 and rec["expect_head"] == harness["head"] and rec["lane"] in (0, 1)
    rc, out = drive(clone, harness, "--only", ",".join(TWO))
    assert rc == 0 and out.count("SKIP (complete and verified)") == 2 and len(_calls(harness)) == 2, out


def _break(state, e):
    d = e["dump_dir"]
    if state == "marker_without_manifest":
        os.unlink(os.path.join(d, "manifest.json"))
    elif state == "unmarked_final_dir":
        os.unlink(os.path.join(d, "COMPLETE"))
    elif state == "staging_leftover":
        os.mkdir(d + ".staging-999")
    elif state == "metrics_without_sidecar":
        os.unlink(e["stream"])


def test_every_incomplete_state_is_quarantined_then_rerun(clone, harness):
    cell = "exp25_CANON_R_S40000_K8_s43"
    rc, out = drive(clone, harness, "--only", cell)
    assert rc == 0, out
    e = _entry(clone, harness, cell)
    for n, state in enumerate(["marker_without_manifest", "unmarked_final_dir", "staging_leftover",
                               "metrics_without_sidecar"], start=2):
        _break(state, e)
        rc, out = drive(clone, harness, "--only", cell)
        assert rc == 0 and "QUARANTINED" in out and len(_calls(harness)) == n, (state, out)
        assert _verify(clone, harness, cell) == [], state
        parked = [p for p in os.listdir(harness["dir"] / "dumps") if ".incomplete-" in p]
        assert parked, state                                                   # moved, never deleted
    farm = os.path.dirname(e["metrics"])
    assert any(".json.incomplete-" in p for p in os.listdir(farm))
    rc, out = drive(clone, harness, "--only", cell)                            # quarantine is final
    assert rc == 0 and "SKIP (complete and verified)" in out and len(_calls(harness)) == 5, out


def test_an_evaluator_failure_is_reported_and_nothing_is_marked_complete(clone, harness):
    rc, out = drive(clone, harness, "--only", TWO[0], env={"STUB_MODE": "crash"})
    assert rc == 1 and "FAILED exp25_P1_T_S40000_K1_s42" in out and "INCOMPLETE" in out, out
    assert not os.path.exists(_entry(clone, harness, TWO[0])["receipt"])


def test_the_gpu_gate_waits_and_never_runs_on_a_busy_card(clone, harness):
    rc, out = drive(clone, harness, "--only", TWO[0], env={"GUARD_GPU_FREE": "100"})
    assert rc == 2 and "GPU GATE" in out and not _calls(harness), out
    assert (harness["dir"] / "nvsmi.log").read_text().count("memory.free") >= 2


def test_a_no_dump_block_completes_without_a_dump(clone, harness):
    cell = "exp25_CANON_TRAJ_S5000_K1_s42_th0"
    rc, out = drive(clone, harness, "--only", cell, "--no-dump-blocks", "TRAJ")
    assert rc == 0, out
    e = _entry(clone, harness, cell)
    assert not os.path.exists(e["dump_dir"]) and json.load(open(e["receipt"]))["dump"] is False
    assert _verify(clone, harness, cell, expect_dump=False) == []
    rc, out = drive(clone, harness, "--only", cell, "--no-dump-blocks", "TRAJ")
    assert rc == 0 and "SKIP (complete and verified)" in out, out


# ---- rB-F1 B2 (Codex rB HIGH; decision A6): a lane needs >= 10 GiB free AND no compute process ---- #
@pytest.fixture()
def sleepers():
    """Stand-ins for foreign compute processes: OUR OWN `sleep`s (the driver must not signal them)."""
    procs = []

    def start():
        procs.append(subprocess.Popen(["sleep", "300"]))
        return procs[-1].pid
    yield start
    for p in procs:                       # our own children only
        p.kill()
        p.wait()


def _apps(harness, rows, name="apps.default"):
    (harness["dir"] / name).write_text("".join(f"{pid}, {UUID[g]}, python, 15000\n" for pid, g in rows))


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_an_occupied_lane_is_skipped_and_the_occupant_is_logged(clone, harness, sleepers):
    pid = sleepers()
    _apps(harness, [(pid, 0)])
    rc, out = drive(clone, harness, "--only", TWO[0])
    assert rc == 0, out
    assert [c["cvd"] for c in _calls(harness)] == ["1"]                   # GPU 0 is foreign-occupied
    assert f"{pid}" in out and "occupied" in out and "cwd" in out and _alive(pid), out


def test_both_lanes_occupied_waits_then_aborts_and_signals_nothing(clone, harness, sleepers):
    a, b = sleepers(), sleepers()
    _apps(harness, [(a, 0), (b, 1)])
    rc, out = drive(clone, harness, "--only", TWO[0])
    assert rc == 2 and "GPU GATE" in out and not _calls(harness), out
    assert _alive(a) and _alive(b)                                          # nothing was signalled


def test_allow_shared_gpu_shares_but_keeps_the_memory_gate(clone, harness, sleepers):
    _apps(harness, [(sleepers(), 0), (sleepers(), 1)])
    rc, out = drive(clone, harness, "--only", TWO[0], env={"ALLOW_SHARED_GPU": "1"})
    assert rc == 0 and len(_calls(harness)) == 1 and "ALLOW_SHARED_GPU" in out, out
    rc, out = drive(clone, harness, "--only", TWO[1], env={"ALLOW_SHARED_GPU": "1", "GUARD_GPU_FREE": "100"})
    assert rc == 2 and len(_calls(harness)) == 1, out                        # memory still gates
    rc, out = drive(clone, harness, "--only", TWO[1], env={"ALLOW_SHARED_GPU": "yes"})
    assert rc == 2 and "ALLOW_SHARED_GPU" in out, out


@pytest.mark.parametrize("setup, env, needle", [
    (None, {"GUARD_UUID_FAIL": "1"}, "index,uuid"),
    (None, {"GUARD_APPS_RC": "6"}, "compute-apps"),
    ("malformed", {}, "malformed"),
    ("unknown_uuid", {}, "unknown"),
    ("dup_uuid", {}, "duplicate"),
    ("no_lane_uuid", {}, "uuid"),
    (None, {"GUARD_GPU_FREE": "40000\\n40000"}, "memory"),
], ids=["uuid_query_fails", "apps_query_fails", "malformed_app_row", "unknown_uuid", "duplicate_uuid",
        "lane_without_uuid", "two_memory_rows"])
def test_a_query_failure_fails_closed(clone, harness, sleepers, setup, env, needle):
    h = harness["dir"]
    if setup == "malformed":
        (h / "apps.default").write_text("garbage row\n")
    elif setup == "unknown_uuid":
        (h / "apps.default").write_text(f"{sleepers()}, GPU-99999999-zzzz, python, 10\n")
    elif setup == "dup_uuid":
        (h / "uuids").write_text("0, GPU-00000000-aaaa\n1, GPU-00000000-aaaa\n")
    elif setup == "no_lane_uuid":
        (h / "uuids").write_text("1, GPU-11111111-bbbb\n")
    rc, out = drive(clone, harness, "--only", TWO[0], "--lanes", "0", env=env)
    assert rc == 2 and "GPU GATE FAILED" in out and needle in out and not _calls(harness), out


def test_the_lane_is_rechecked_immediately_before_the_spawn(clone, harness, sleepers):
    """Free at selection, a newcomer at the re-check: no spawn on that card."""
    (harness["dir"] / "apps.1").write_text("")
    _apps(harness, [(sleepers(), 0)])                                       # every later poll
    rc, out = drive(clone, harness, "--only", TWO[0], "--lanes", "0")
    assert rc == 2 and not _calls(harness) and "re-check" in out, out


# ---- rB-F1 B3 (Codex rB HIGH): the campaign pin closes over the evaluated BYTES ---- #
def _ran(harness):
    return [c["argv"][c["argv"].index("--eval-name") + 1] for c in _calls(harness)]


def test_the_receipt_binds_the_campaign_tree_digest(clone, harness):
    import re
    rc, out = drive(clone, harness, "--only", TWO[0])
    assert rc == 0, out
    digest = re.search(r"evaluator tree digest ([0-9a-f]{64})", out).group(1)
    rec = json.load(open(_entry(clone, harness, TWO[0])["receipt"]))
    assert rec["campaign_digest"] == rec["spawn_digest"] == rec["post_digest"] == digest
    assert _verify(clone, harness, TWO[0], expect_digest=digest) == []


def test_an_edit_between_two_cells_stops_the_campaign(clone, harness):
    path = clone / "eval_FLAC.py"
    original = path.read_bytes()
    try:
        rc, out = drive(clone, harness, "--only", ",".join(TWO), "--lanes", "0", env={"GUARD_MUTATE_AT": "3"})
    finally:
        path.write_bytes(original)
    ran = _ran(harness)
    assert rc == 4 and "PIN DRIFT" in out and len(ran) == 1, out
    other = next(c for c in TWO if c not in ran)
    assert _verify(clone, harness, ran[0]) == []                          # the first cell is intact
    assert not os.path.exists(_entry(clone, harness, other)["receipt"])   # the second never spawned


def test_an_edit_during_a_cell_fails_that_cell(clone, harness):
    path = clone / "eval_FLAC.py"
    original = path.read_bytes()
    try:
        rc, out = drive(clone, harness, "--only", TWO[0], env={"STUB_MODE": "mutate"})
    finally:
        path.write_bytes(original)
    assert rc == 4 and "changed while the cell ran" in out, out
    assert not os.path.exists(_entry(clone, harness, TWO[0])["receipt"])
