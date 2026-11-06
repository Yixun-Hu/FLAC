"""exp_22 A7 (plan §9 A7 r2 (iii) (vi), r3 (c) (f), r4, r5): the pre-flight line list as data -- every line pinned
against the governing command-list entry 2026-11-03T17:17:46-05:00 (identical arguments except the fresh out-dirs,
the launcher wrapping and T's JUnit flag), T against the plan and the triage log, the exact environments, and no
reachable shard / merge / report / crossarm line."""
import copy
import hashlib
import json
import os
import re
import shlex

import pytest

from conftest import load

E = load("exp22_r2c_evidence")
X = os.path.abspath(os.path.join(os.path.dirname(E.__file__), ".."))
REPO = os.path.abspath(os.path.join(X, "..", "..", ".."))
LINES = os.path.join(os.path.dirname(E.__file__), "exp22_r2c_preflight_lines.json")
PY = "/home/yixunhu/miniconda3/envs/flac/bin/python"
PYTEST_MAIN = "/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytest/__main__.py"
CV = "CUDA_VISIBLE_DEVICES=<free card> "


def doc():
    with open(LINES) as f:
        return json.load(f)


def governing_spans():
    with open(os.path.join(X, "loc_meshgrid_worklog.md")) as f:
        text = f.read()
    start = text.index("## 2026-11-03T17:17:46-05:00 — rB-F3: THE AUTHORITATIVE GPU COMMAND LIST for R2-C")
    return re.findall(r"`([^`]*)`", text[start:text.index("\n## ", start + 10)])


def span(prefix):
    hits = [s for s in governing_spans() if s.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def line(name):
    return next(ln for ln in doc()["lines"] if ln["name"] == name)


def test_the_committed_list_is_valid_and_in_the_registered_order():
    d = doc()
    assert E.lines_reasons(d) == []
    assert [ln["name"] for ln in d["lines"]] == ["T", "GV", "V_GV", "G1", "V_G1", "G2", "V_G2", "G3", "V_G3", "G3D",
                                                 "V_G3D", "G0", "V_G0", "P", "A"]


@pytest.mark.parametrize("name, prefix, output", [
    ("GV", "--cache-parity-check --expect-arm YAW ", "vanilla_cache_parity.json"),
    ("G1", "--fa-split-parity-check ", "fa_split_parity.json"),
    ("G2", "--cache-parity-check --ckpt-path weights/exp20/BF_40k.ckpt ", "fa_cache_parity.json"),
    ("G3", "--fa-production-parity-check ", "fa_production_parity.json"),
    ("G3D", "--fa-production-drift-measurement ", "fa_production_drift.json"),
    ("G0", "--replay-check ", "vanilla_replay_regression.json"),
    ("P", "--probe 1 ", "diagnostics_BF_bathrooms14.json")])
def test_every_wrapped_line_is_the_governing_line_but_the_out_dir_and_the_launcher(name, prefix, output):
    command = span(CV + "$PY localize_meshgrid.py " + prefix)
    gov = shlex.split(command[len(CV):])
    old = "outputs_loc/exp22/r2_probe_BF" if name == "P" else "outputs_loc/exp22/r2_gates"
    new = "@PROBE@" if name == "P" else "@GATES@"
    i = gov.index("--out-dir")
    assert gov[0] == "$PY" and gov[i + 1] == old and gov.count("--out-dir") == 1
    want = ["@PY@", "@L@"] + gov[1:i + 1] + [new] + gov[i + 2:]
    assert line(name)["argv"] == want
    assert line(name)["outputs"] == [f"{new}/{output}"]
    following = governing_spans()[governing_spans().index(command) + 1]
    assert following == output                                           # the governing "-> `<evidence>`"


def test_every_reverification_is_the_governing_one_liner_on_the_fresh_path():
    gov = shlex.split(span('$PY -c "import json, sys; from src.localization import meshgrid_fa_parity').split(" || ")[0])
    assert gov[0] == "$PY" and gov[1] == "-c" and gov[3] == "outputs_loc/exp22/r2_gates/<name>.json" and len(gov) == 4
    for g in ("GV", "G1", "G2", "G3", "G3D", "G0"):
        out = line(g)["outputs"][0]
        assert line(f"V_{g}")["argv"] == ["@PY@", "-c", gov[2], out] and line(f"V_{g}")["inputs"] == [out]


def test_the_a5_arithmetic_is_the_governing_one_card_line_on_the_fresh_probe_path():
    gov = shlex.split(span("$PY -c \"import json; h = json.load(open('outputs_loc/exp22/r2_probe_BF/diag"))
    old = "'outputs_loc/exp22/r2_probe_BF/diagnostics_BF_bathrooms14.json'"
    assert gov[:2] == ["$PY", "-c"] and len(gov) == 3 and gov[2].count(old) == 1
    assert line("A")["argv"] == ["@PY@", "-c", gov[2].replace(old, "'@PROBE@/diagnostics_BF_bathrooms14.json'")]
    assert "1.5 * h * 0.5145" in gov[2] and "m <= 150" in gov[2]          # the one-card form, factor 1.5 (A7 r2 (vi))


def test_t_is_the_plan_line_on_the_triage_ids():
    with open(os.path.join(X, "loc_meshgrid_2026-11-04_12:08:44_r2_merge_cpu_failures_triage.log"), "rb") as f:
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest().startswith("08c362ed")
    failed = sorted(set(re.findall(r"^FAILED (\S+)$", raw.decode().split("## (b)")[0], re.M)))
    with open(os.path.join(X, "plan_loc_meshgrid_r2_bf_yaw.md")) as f:
        plan = f.read()
    r2 = plan[plan.index("- **A7 r2 ("):plan.index("- **A7 r3 (")]
    plan_ids = re.findall(r"`(src/tests/test_[a-z_]+\.py::[^`]+)`", r2)
    t = line("T")
    assert t["argv"][:13] == ["@PY@", "@L@", "@PYTEST_MAIN@", "-x", "-q", "--color=no", "-o", "xfail_strict=true",
                              "-rA", "-p", "no:cacheprovider", "--junitxml", "@ORCH@/T_junit.xml"]
    assert t["argv"][13:] == plan_ids == list(E.T_NODE_IDS) and sorted(plan_ids) == failed and len(failed) == 11
    assert t["junit"] == "@ORCH@/T_junit.xml" and t["outputs"] == [] and t["inputs"] == []
    assert "--color=no" in plan[plan.index("- **A7 r5 ("):] and "xfail_strict=true" in plan[plan.index("- **A7 r4 ("):]


def test_the_expansion_is_exact():
    lines = E.expand_lines(doc(), run="20261106T130501", index=1, worktree="/w", py=PY, pytest_main=PYTEST_MAIN)
    base = {"HOME": "/home/yixunhu", "USER": "yixunhu", "LOGNAME": "yixunhu", "LANG": "en_US.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
    wrapped = dict(base, CUDA_VISIBLE_DEVICES="1", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH="/w/overlays/no_flash")
    cpu = dict(base, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    orch, gates = "outputs_loc/exp22/r2_preflight_20261106T130501", "outputs_loc/exp22/r2_gates_20261106T130501"
    t, gv, v, p, a = lines[0], lines[1], lines[2], lines[13], lines[14]
    assert t["argv"][:3] == [PY, "overlays/no_flash/exp35_no_flash_launch.py", PYTEST_MAIN] and t["target"] == PYTEST_MAIN
    assert t["argv"][12] == f"{orch}/T_junit.xml" and t["junit"] == f"{orch}/T_junit.xml" and t["env"] == wrapped
    assert t["node_ids"] == list(E.T_NODE_IDS) and t["wrapped"] is True
    assert gv["target"] == "localize_meshgrid.py" and gv["argv"][-1] == gates and gv["env"] == wrapped
    assert gv["outputs"] == [f"{gates}/vanilla_cache_parity.json"] and gv["log"] == f"{orch}/line_02_GV.log"
    assert gv["record"] == f"{orch}/dispatch_02_GV.json"
    assert v["wrapped"] is False and v["target"] is None and v["env"] == cpu and v["argv"][:2] == [PY, "-c"]
    assert p["argv"][-1] == "outputs_loc/exp22/r2_probe_BF_20261106T130501" and a["env"] == cpu
    assert "'outputs_loc/exp22/r2_probe_BF_20261106T130501/diagnostics_BF_bathrooms14.json'" in a["argv"][2]
    assert not [x for ln in lines for x in ln["argv"] + ln["outputs"] + ln["inputs"] if "@" in x and
                re.search(r"@[A-Z_]+@", x)]


@pytest.mark.parametrize("bad_run", ["", "2026-11-06", "20261106T1305", "../x", "20261106T130501/..", None])
def test_a_malformed_run_id_is_refused(bad_run):
    with pytest.raises(ValueError):
        E.expand_lines(doc(), run=bad_run, index=0, worktree="/w", py=PY, pytest_main=PYTEST_MAIN)


def _mutated(fn):
    d = copy.deepcopy(doc())
    fn(d)
    return E.lines_reasons(d)


SHARD = ["@PY@", "@L@", "localize_meshgrid.py", "--emit-launch-record", "x/launch_record.json", "--expect-arm", "YAW"]


@pytest.mark.parametrize("mutate", [
    lambda d: d["lines"].append(dict(d["lines"][1], n=16, name="YAW_cafe", argv=SHARD)),
    lambda d: d["lines"][1].update(argv=d["lines"][1]["argv"][:3] + ["--launch-record", "r.json"] + d["lines"][1]["argv"][3:]),
    lambda d: d["lines"][13].update(argv=d["lines"][13]["argv"] + ["--rooms", "Cafe/Cafe_idx_1"]),
    lambda d: d["lines"][14].update(argv=["@PY@", "-m", "src.localization.meshgrid_report", "--run-dir", "x"]),
    lambda d: d["lines"][12].update(argv=["@PY@", "-c", "import src.localization.meshgrid_crossarm", "@GATES@/x.json"]),
    lambda d: d["lines"][3].update(argv=[a for a in d["lines"][3]["argv"] if a != "--fa-split-parity-check"]),  # a pass
    lambda d: d["lines"][5].update(argv=[("@ORCH@" if a == "@GATES@" else a) for a in d["lines"][5]["argv"]]),
    lambda d: d["lines"][7].update(argv=d["lines"][7]["argv"] + ["--merge-shards", "a", "b"]),
    lambda d: d["lines"][9].update(argv=d["lines"][9]["argv"] + ["--resume"]),
    lambda d: d["lines"][0].update(argv=d["lines"][0]["argv"][:-1]),                     # ten ids
    lambda d: d["lines"][0].update(argv=d["lines"][0]["argv"] + ["-k", "x"]),
    lambda d: d["lines"].reverse(),
    lambda d: d["lines"][2].update(argv=d["lines"][2]["argv"][:3] + ["@GATES@/fa_split_parity.json"]),
    lambda d: d["lines"][13].update(outputs=["@PROBE@/other.json"]),
    lambda d: d["lines"][1].update(argv=d["lines"][1]["argv"] + ["@SECRET@"]),
    lambda d: d.update(schema_version=2),
], ids=["shard-line", "launch-record", "rooms", "report", "crossarm", "scored-pass", "out-dir", "merge", "resume",
        "ten-ids", "deselect", "order", "wrong-input", "wrong-output", "unknown-placeholder", "schema"])
def test_no_shard_merge_report_or_crossarm_line_is_reachable(mutate):
    assert _mutated(mutate)


def test_no_registered_argv_carries_a_shard_token():
    lines = E.expand_lines(doc(), run="20261106T130501", index=0, worktree="/w", py=PY, pytest_main=PYTEST_MAIN)
    tokens = [a for ln in lines for a in ln["argv"]]
    assert not set(tokens) & {"--emit-launch-record", "--launch-record", "--merge-shards", "--merge-out", "--rooms",
                              "--resume", "-m"}
    assert not [a for a in tokens if "meshgrid_report" in a or "meshgrid_crossarm" in a]
    outdirs = [ln["argv"][ln["argv"].index("--out-dir") + 1] for ln in lines if "--out-dir" in ln["argv"]]
    assert sorted(set(outdirs)) == ["outputs_loc/exp22/r2_gates_20261106T130501",
                                    "outputs_loc/exp22/r2_probe_BF_20261106T130501"] and len(outdirs) == 7
