"""exp_35 phase (ii) kit, part 1 (plan_orbitrir_yaw_evidence.md §15 items 1-4 and its r1 clarifications (b) (c);
Codex plan §15 r1 S6 "Checks and durable records"): the registered pins (bound to the H35 git objects), the exact
68-cell selection and its name digest, the CANON re-pin, the runner's plan JSON re-derived row by row (checked
against the runner's own committed plans), the lane list, the report evidence, the phase-(i) completion contract
(checked against the archived REAL supervisor state), the runner's relevant-tree digest and the durable writers.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import ast
import hashlib
import json
import os

import pytest

import exp35_phase_ii_sandbox as SB

K = SB.load_kit()
X25, X35 = SB.X25, SB.X35
SHA = lambda raw: hashlib.sha256(raw).hexdigest()  # noqa: E731
BEGIN, END = "# --- BEGIN REGISTERED CONFIG", "# --- END REGISTERED CONFIG ---"


# --------------------------------------------------------------------------- #
# the registered pins
# --------------------------------------------------------------------------- #
PRODUCTION = {
    "RUN": "/home/yixunhu/codespace/exp-35-orbitrir-yaw",
    "H35": "5b065801e9d1c1844461ec84cd0558702c714326",
    "CAMPAIGN_DIGEST": "da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1",
    "RESERVATIONS": "/home/yixunhu/codespace/.gpu_reservations",
    "EXP30": "/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn",
    "EXP30_PINS": {"exp30_attn_queue.sh": "9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa",
                   "exp30_attn_eval.sh": "94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c",
                   "exp30_attn_queue_state.py": "8572bb0305bcd1f1f245a9d2b31ca9760ba7ed036bccf5bf612d2b2208b33907"},
    "QUEUE_LOG": "logs/queue_20261101T233607_741947.log",
    "NVIDIA_SMI": "/usr/bin/nvidia-smi", "PROC": "/proc", "SCAN_PROC": "/proc",
    "PY": "/home/yixunhu/miniconda3/envs/flac/bin/python",
    "GAP_S": 60, "CONFIRM_S": 60, "IDENTIFY_S": 5.0, "DRY_TIMEOUT_S": 1800, "DRY_MAX_AGE_S": 10800}


def block_values(path):
    """The literal assignments of a kit file's registered-config block (the only part a sandbox replaces)."""
    with open(path) as f:
        text = f.read()
    i = text.index(BEGIN)
    body = text[text.index("\n", i) + 1:text.index(END, i)]
    out = {}
    for node in ast.parse(body).body:
        names = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets
        vals = node.value.elts if isinstance(node.value, ast.Tuple) and len(names) > 1 else [node.value]
        out.update({n.id: ast.literal_eval(v) for n, v in zip(names, vals)})
    return out


def test_the_registered_config_block_holds_the_production_facts():
    assert block_values(SB.KIT_PY) == PRODUCTION
    cfg = K.registered_config()
    assert (cfg.run, cfg.h35, cfg.digest, cfg.reservations, cfg.nvidia_smi, cfg.proc, cfg.scan_proc, cfg.py) == (
        PRODUCTION["RUN"], PRODUCTION["H35"], PRODUCTION["CAMPAIGN_DIGEST"], PRODUCTION["RESERVATIONS"],
        PRODUCTION["NVIDIA_SMI"], "/proc", "/proc", PRODUCTION["PY"])
    assert cfg.queue_log == PRODUCTION["EXP30"] + "/" + PRODUCTION["QUEUE_LOG"]
    assert cfg.exp30_pins == {PRODUCTION["EXP30"] + "/" + k: v for k, v in PRODUCTION["EXP30_PINS"].items()}
    assert cfg.kit_dir == os.path.dirname(SB.KIT_PY) and cfg.helper == SB.KIT_PY


def test_the_fixed_pins_are_the_registered_facts():
    assert (K.CELLS, K.CELLS_SHA256, K.N_ROWS) == (f"{X25}/depth_canon_cells.json",
                                                   "440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d", 346)
    assert K.BLOCKS == ("OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45")
    assert K.COUNTS == {"OFF-P1": 20, "OFF-BF": 20, "OFF-CANON": 20, "CANON-45": 8}
    assert K.BLOCK_ARM == {"OFF-P1": "P1", "OFF-BF": "BF", "OFF-CANON": "CANON", "CANON-45": "CANON"}
    assert K.ORDERS == tuple(range(262, 330)) and K.E2_FAIL_ORDERS == tuple(range(330, 346))
    assert K.NAMES_SHA256 == "6a88112f9cceeb9de3a8056e621b68bdc7c5d1d484aebb2e05a4624c8406d427"
    assert (K.DUMP_ROOT, K.FARM) == ("/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms", "outputs_FLAC/exp25_depth_canon")
    assert K.CANON_MANIFEST == f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256"
    assert K.CANON_MANIFEST_SHA256 == "1e4814472b72375d6f3f42b3773f7bdb5d137719f882d00aef0387b770218cba"
    assert K.CANON_CKPT == ("/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon/FLAC_exp25_CANON/exp25_CANON/"
                            "checkpoints/epoch=8-step=40000.ckpt",
                            "52c64b525c8a34f1721b73abd5c4aad67295ea2f729426e9330104b6d1441ace")
    assert K.CARDS == {0: SB.GPU0, 1: SB.GPU1}
    assert K.LAUNCH_ENV == {"HOME": "/home/yixunhu", "USER": "yixunhu", "LOGNAME": "yixunhu", "SHELL": "/bin/bash",
                            "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    assert K.SHARING == ("ALLOW_SHARED_GPU", "EXP35_APPROVED_COTENANTS", "EXP35_APPROVED_COTENANTS_SHA256")
    assert (K.RUNNER, K.BRANCH) == (f"{X25}/depth_canon_eval_run.sh", "exp-35-orbitrir-yaw")
    assert (K.SUP_STATE, K.SUP_FILE, K.SUP_LOCK) == (f"{X35}/exp35_supervisor_state.json", f"{X35}/exp35_supervisor.py",
                                                     f"{X35}/exp35_supervisor_state.json.lock")
    assert (K.RUNNER_LOCK, K.KIT_LOCK) == (f"{X25}/.depth_canon_eval.lock", f"{X35}/exp35_phase_ii.lock")
    assert K.FINAL_EVENT == SB.FINAL_EVENT and K.KIT_FILES == ("exp35_phase_ii.sh", "exp35_phase_ii.py")


def test_the_pins_hold_on_the_h35_objects():
    cells, canon = SB.real_cells(), SB.real_canon()
    assert SHA(cells) == K.CELLS_SHA256 and SHA(canon) == K.CANON_MANIFEST_SHA256
    why, m, rows = K.selection(cells)
    assert why == [] and len(rows) == 68 and [r["order"] for r in rows] == list(K.ORDERS)
    assert K.names_sha256([r["cell"] for r in rows]) == K.NAMES_SHA256
    assert {r["block"] for r in rows} == set(K.BLOCKS) and all(r["dump"] is True for r in rows)
    assert K.canon_reasons(canon) == []


def test_the_name_digest_is_the_sha256_of_the_newline_joined_names():
    assert K.names_sha256(["a", "b"]) == SHA(b"a\nb\n")


# --------------------------------------------------------------------------- #
# every wrong manifest / CANON entry is refused (S6 "every wrong pin / hash / cell count / block / dump root /
# CANON entry"); each reason is named even when the sha256 pin already refuses
# --------------------------------------------------------------------------- #
def _mutated(mutation):
    m = json.loads(SB.real_cells())
    sel = [r for r in m["cells"] if r["block"] in K.BLOCKS]
    if mutation == "rows":
        m["cells"].pop()
    elif mutation == "block":
        sel[0]["block"] = "OFF-PX"
    elif mutation == "dump":
        sel[5]["dump"] = False
    elif mutation == "dump_root":
        m["dump_root"] = "/tmp/elsewhere"
    elif mutation == "farm":
        m["farm_root"] = "outputs_FLAC/elsewhere"
    elif mutation == "order":
        sel[0]["order"], sel[1]["order"] = sel[1]["order"], sel[0]["order"]
    elif mutation == "e2fail":
        sel[-1]["block"] = "E2-FAIL"
    elif mutation == "duplicate":
        sel[1]["cell"] = sel[0]["cell"]
    elif mutation == "arm":
        sel[25]["arm"] = "P1"
    elif mutation == "name":
        sel[3]["cell"] += "_x"
    return json.dumps(m, indent=1).encode()


@pytest.mark.parametrize("mutation, reason", [
    ("rows", "345 rows"), ("block", "count"), ("dump", "dump is False"), ("dump_root", "dump root"),
    ("farm", "farm root"), ("order", "orders"), ("e2fail", "E2-FAIL"), ("duplicate", "distinct"),
    ("arm", "arm P1"), ("name", "digest")])
def test_every_wrong_manifest_is_refused(mutation, reason):
    why, _, _ = K.selection(_mutated(mutation))
    assert any("hashes to" in x for x in why) and any(reason in x for x in why), why


def test_a_manifest_that_is_not_json_is_refused():
    why, m, rows = K.selection(b"{")
    assert why and m is None and rows == []


@pytest.mark.parametrize("edit", ["extra-line", "missing", "other-sha", "duplicate"])
def test_every_wrong_canon_manifest_is_refused(edit):
    raw = SB.real_canon()
    line = f"{K.CANON_CKPT[1]}  {K.CANON_CKPT[0]}\n".encode()
    raw = {"extra-line": raw + b"0" * 64 + b"  /x/epoch=9-step=42500.ckpt\n", "missing": raw.replace(line, b""),
           "other-sha": raw.replace(line, b"1" * 64 + line[64:]), "duplicate": raw + line}[edit]
    why = K.canon_reasons(raw)
    assert any("hashes to" in x for x in why)
    assert any("step-40000" in x for x in why) == (edit != "extra-line")


# --------------------------------------------------------------------------- #
# the runner's plan JSON, re-derived row by row from the manifest (the exact planned set and its 68 argv)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("plan_rel, no_dump", [
    (f"{X35}/dry_run_artifacts_5b065801/depth_canon_2026-11-05_19-24-07_eval_plan.json", ()),   # P1 / BF at H35
    (f"{X25}/depth_canon_2026-10-29_11-18-16_eval_plan.json", ()),                             # 42 CANON rows
    (f"{X25}/depth_canon_2026-10-30_03-41-59_eval_plan.json", ("TRAJ",))])                     # 120, no dumps
def test_the_runners_own_plans_are_re_derived_exactly(plan_rel, no_dump):
    with open(os.path.join(SB.REPO, plan_rel)) as f:
        real = json.load(f)
    m = json.loads(SB.real_cells())
    rows = {r["cell"]: r for r in m["cells"]}
    run = real["cells_manifest"][:-len("/" + K.CELLS)]
    index = {int(ln.rsplit("step=", 1)[1][:-len(".ckpt")]): (ln.partition("  ")[2], ln.partition("  ")[0])
             for ln in SB.real_canon().decode().splitlines()}                 # every CANON step (TRAJ), as the runner
    want = K.expected_plan(m, [rows[e["cell"]] for e in real["cells"]], run=run, digest=real["campaign_digest"],
                           no_dump=no_dump, canon_index=index)
    assert want == real


def test_the_phase_ii_plan_is_68_dumping_no_flash_entries():
    _, m, rows = K.selection(SB.real_cells())
    p = K.expected_plan(m, rows, run="/r", digest="d" * 64)
    assert p["n_selected"] == 68 and p["cells_manifest"] == "/r/" + K.CELLS and p["dump_root"] == K.DUMP_ROOT
    assert all(e["dump"] is True and e["no_flash"] is True and e["argv"][-4:] == [
        "--ckpt-path", e["link"], "--dump-scored-waveforms", K.DUMP_ROOT] for e in p["cells"])
    assert {e["arm"]: (e["target"], e["sha256"]) for e in p["cells"]} == {
        "P1": (m["arms"]["P1"]["ckpt"]["path"], "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c"),
        "BF": (m["arms"]["BF"]["ckpt"]["path"], "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328"),
        "CANON": K.CANON_CKPT}
    assert not any("--no-dump-blocks" in e["argv"] or "E2-FAIL" in json.dumps(e) for e in p["cells"])


# --------------------------------------------------------------------------- #
# the lane list (frozen before reserve; S6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, lanes", [("0", (0,)), ("1", (1,)), ("0,1", (0, 1)), ("1,0", (1, 0))])
def test_the_lane_list_parses(text, lanes):
    assert K.parse_lanes(text) == lanes and K.lanes_text(lanes) == text


@pytest.mark.parametrize("text", ["", "2", "0,0", "0,1,1", "a", "0;1", " 0", "0,", "01", "0,1,0"])
def test_a_wrong_lane_list_is_refused(text):
    with pytest.raises(ValueError):
        K.parse_lanes(text)


# --------------------------------------------------------------------------- #
# the durable writers (S6: O_EXCL records; the atomic, file-and-directory-fsynced state)
# --------------------------------------------------------------------------- #
def test_write_exclusive_never_overwrites_and_returns_the_written_identity(tmp_path):
    p = str(tmp_path / "r.json")
    s = K.write_exclusive(p, b'{"a": 1}')
    st = os.stat(p)
    assert (s.raw, s.sha256, s.dev, s.inode) == (b'{"a": 1}', SHA(b'{"a": 1}'), st.st_dev, st.st_ino)
    with pytest.raises(FileExistsError):
        K.write_exclusive(p, b"other")
    os.symlink(str(tmp_path / "nowhere"), str(tmp_path / "dangling"))
    with pytest.raises(FileExistsError):
        K.write_exclusive(str(tmp_path / "dangling"), b"x")
    assert open(p, "rb").read() == b'{"a": 1}' and not os.path.exists(tmp_path / "nowhere")


def test_write_json_atomic_replaces_the_whole_document_and_leaves_no_temp(tmp_path):
    p = str(tmp_path / "state.json")
    K.write_json_atomic(p, {"phase": "reserved"})
    K.write_json_atomic(p, {"phase": "dry_run_passed", "x": [1, 2]})
    assert json.load(open(p)) == {"phase": "dry_run_passed", "x": [1, 2]} and os.listdir(tmp_path) == ["state.json"]
    assert open(p, "rb").read() == K.dumps({"phase": "dry_run_passed", "x": [1, 2]})


def test_snapshot_reads_regular_files_only_and_never_follows_a_link(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    os.symlink(str(tmp_path / "f"), str(tmp_path / "l"))
    s = K.snapshot(str(tmp_path / "f"))
    assert s.raw == b"abc" and s.sha256 == SHA(b"abc") and s.inode == os.stat(tmp_path / "f").st_ino
    assert K.snapshot(str(tmp_path / "l")) is None and K.snapshot(str(tmp_path)) is None
    assert K.snapshot(str(tmp_path / "absent")) is None
