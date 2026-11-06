"""exp_35 phase (ii) kit, part 1b (plan_orbitrir_yaw_evidence.md §15 items 2 and 4, r1 clarification (b); Codex plan
§15 r1 S6): the report evidence (a committed file + its sha256, binding H35, the final phase-(i) state, the frozen
lane list, the ETA and every earlier launched orchestration), the phase-(i) completion contract (checked against the
archived REAL supervisor state) and the runner's relevant-tree digest (checked against the H35 tree itself).

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import hashlib
import json
import subprocess
import sys

import pytest

import exp35_phase_ii_sandbox as SB

K = SB.load_kit()
X25, X35 = SB.X25, SB.X35
SHA = lambda raw: hashlib.sha256(raw).hexdigest()  # noqa: E731


# --------------------------------------------------------------------------- #
# the report evidence (the brief: a committed file path + its sha256; the lane list it names)
# --------------------------------------------------------------------------- #
STATE_SHA = "a" * 64
REPORT = ("# exp_35 phase (i) report to Yixun\n\nThe measured slowdown ... the card plan ...\n\n```\n"
          f"exp35-phase-ii h35 {SB.H35_REAL}\n"
          f"exp35-phase-ii phase-i-state-sha256 {STATE_SHA}\n"
          "exp35-phase-ii lanes 0\n"
          "exp35-phase-ii second-lane-decision none\n"
          "exp35-phase-ii eta 68 cells on GPU 0, about 19 h; ETD 2026-11-07T09:30-05:00\n"
          "exp35-phase-ii prior-runs none\n```\n")
FIELDS = {"h35": SB.H35_REAL, "phase-i-state-sha256": STATE_SHA, "lanes": "0", "second-lane-decision": "none",
          "eta": "68 cells on GPU 0, about 19 h; ETD 2026-11-07T09:30-05:00", "prior-runs": "none"}


def test_the_report_fields_parse():
    assert K.report_fields(REPORT.encode()) == (FIELDS, [])


@pytest.mark.parametrize("edit, reason", [
    (lambda t: t.replace("exp35-phase-ii prior-runs none\n", ""), "prior-runs"),
    (lambda t: t + "exp35-phase-ii lanes 0\n", "lanes"),
    (lambda t: t + "exp35-phase-ii colour blue\n", "colour"),
    (lambda t: t.replace("exp35-phase-ii eta", "  exp35-phase-ii eta"), "eta"),
    (lambda t: t.replace("exp35-phase-ii eta 68", "exp35-phase-ii eta  "), "eta")])
def test_a_malformed_report_is_refused(edit, reason):
    _, why = K.report_fields(edit(REPORT).encode())
    assert why and any(reason in x for x in why), why


def test_a_report_that_is_not_utf8_is_refused():
    assert K.report_fields(b"\xff\xfe")[1]


def _rr(fields=None, **kw):
    args = dict(h35=SB.H35_REAL, lanes=(0,), state_sha256=STATE_SHA, prior_runs=())
    args.update(kw)
    return K.report_reasons(dict(FIELDS, **(fields or {})), **args)


def test_the_report_binds_h35_the_state_the_lanes_and_the_prior_runs():
    assert _rr() == []
    assert _rr({"lanes": "0,1", "second-lane-decision": "Planner 2026-11-06T11:40 (Yixun: two cards)"}, lanes=(0, 1)) == []
    assert _rr({"prior-runs": "20261106T143000"}, prior_runs=("20261106T143000",)) == []


@pytest.mark.parametrize("fields, kw, reason", [
    ({"h35": "b" * 40}, {}, "h35"),
    ({}, {"state_sha256": "c" * 64}, "phase-i-state-sha256"),
    ({"phase-i-state-sha256": "xyz"}, {}, "phase-i-state-sha256"),
    ({"lanes": "0,1"}, {}, "lanes"),
    ({}, {"lanes": (0, 1)}, "lanes"),
    ({"lanes": "0,1"}, {"lanes": (0, 1)}, "second lane"),                    # an UNRECORDED second lane
    ({"second-lane-decision": "yes"}, {}, "second lane"),
    ({}, {"prior_runs": ("20261106T143000",)}, "prior-runs"),
    ({"prior-runs": "20261106T143000"}, {}, "prior-runs")])
def test_every_wrong_report_binding_is_refused(fields, kw, reason):
    why = _rr(fields, **kw)
    assert why and any(reason in x for x in why), why


# --------------------------------------------------------------------------- #
# the phase-(i) completion contract (§15 item 4; S6 "the phase-(i) state hash / schema, 33/33, no latch, E2 PASS,
# the final release event") -- against the archived REAL supervisor state
# --------------------------------------------------------------------------- #
SUP_SHA = SHA(SB.real_supervisor())


def _pi(doc, **kw):
    args = dict(h35=SB.H35_REAL, digest=SB.DIGEST_REAL, supervisor_sha256=SUP_SHA)
    args.update(kw)
    return K.phase_i_reasons(doc, **args)


def test_the_chunk_order_is_the_h35_supervisors(tmp_path):
    p = tmp_path / "exp35_supervisor.py"
    p.write_bytes(SB.real_supervisor())
    sys.dont_write_bytecode = True
    import importlib.util
    spec = importlib.util.spec_from_file_location("h35_supervisor_copy", str(p))
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    assert [(c["n"], c["kind"], c["cell"], c["gate"]) for c in sup.chunk_plan()] == list(K.PHASE_I_CHUNKS)
    assert SUP_SHA == SB.snapshot_state()["supervisor_sha256"]


def test_the_real_mid_campaign_snapshot_is_refused_as_incomplete():
    why = _pi(SB.snapshot_state())
    assert any("chunk 14" in x and "in_flight" in x for x in why) and any("chunk 33" in x for x in why)
    assert any("reservation" in x for x in why) and any("lane" in x for x in why)
    assert any("final" in x for x in why)


def test_a_completed_phase_i_passes():
    assert _pi(SB.completed_state()) == []


def _tamper(name):
    d = SB.completed_state()
    c = d["chunks"]
    if name == "planned":
        c[32] = {k: c[32][k] for k in ("n", "kind", "cell", "gate")} | {"state": "planned"}
    elif name == "latch":
        d["stop_latch"] = {"reason": "the training guard", "time": "2026-11-06T09:00:00-05:00"}
    elif name == "e2":
        c[3]["e2"] = {"rc": 1, "verdict": "FAIL", "out": "E2 FAIL"}
    elif name == "e1":
        c[0]["e1"] = {"rc": 1, "verdict": "mismatch", "out": "CTL K1 MISMATCH"}
    elif name == "probe":
        c[4]["probe_contract"] = {"rc": 1, "out": "BF PROBE INVALID"}
    elif name == "reservation":
        d["reservation"] = {"uuid": SB.GPU1, "epoch": 1, "path": "/x", "sha256": "0" * 64}
    elif name == "lane":
        d["lane"] = {"uuid": SB.GPU1, "shared": True}
    elif name == "no-final":
        d["events"].pop()
    elif name == "final-not-last":
        d["events"].append({"t": 1.0, "time": "t", "what": "supervisor started (resumed)"})
    elif name == "h35":
        d["h35"] = "b" * 40
    elif name == "cells":
        d["cells_sha256"] = "0" * 64
    elif name == "supervisor":
        d["supervisor_sha256"] = "0" * 64
    elif name == "keys":
        del d["his_log_anchor"]
    elif name == "digest":
        c[7]["campaign_digest"] = "0" * 64
    elif name == "order":
        c[5], c[6] = c[6], c[5]
    elif name == "schema":
        d["schema_version"] = 2
    elif name == "campaign":
        d["campaign"] = "exp22"
    elif name == "rc":
        c[9]["runner_rc"] = 1
    return d


@pytest.mark.parametrize("name, reason", [
    ("planned", "chunk 33"), ("latch", "stop latch"), ("e2", "E2"), ("e1", "E1"), ("probe", "probe"),
    ("reservation", "reservation"), ("lane", "lane"), ("no-final", "final"), ("final-not-last", "final"),
    ("h35", "h35"), ("cells", "cells_sha256"), ("supervisor", "supervisor_sha256"), ("keys", "keys"),
    ("digest", "digest"), ("order", "33-chunk order"), ("schema", "schema_version"), ("campaign", "campaign"),
    ("rc", "runner_rc")])
def test_every_incomplete_or_failed_phase_i_is_refused(name, reason):
    why = _pi(_tamper(name))
    assert why and any(reason in x for x in why), why


def test_a_chunk_adopted_at_a_supervisor_restart_passes_as_the_supervisor_admits_it():
    d = SB.completed_state()
    d["chunks"][9].update(runner_rc=None, ended=None, adopted_at_restart=True)
    assert _pi(d) == []
    d["chunks"][9]["adopted_at_restart"] = False
    assert any("runner_rc" in x for x in _pi(d))


def test_a_phase_i_state_that_is_not_a_dict_is_refused():
    assert _pi([]) and _pi(None)


# --------------------------------------------------------------------------- #
# the runner's relevant-tree digest (depth_canon_eval_run.sh tree_digest; §15 r1 (b): the exact campaign digest)
# --------------------------------------------------------------------------- #
def _git(w, *args):
    return subprocess.run(["git", "-C", str(w), "-c", "user.email=p2kit@sandbox", "-c", "user.name=p2kit", *args],
                          capture_output=True, text=True, check=True).stdout


def test_the_relevant_pathspec_is_the_runners_for_an_exp35_selection():
    m = json.loads(SB.real_cells())
    assert K.relevant(m) == ["eval_FLAC.py", "src", ":(exclude)src/tests", "data/AR",
                             *[a["model_config"] for a in m["arms"].values()], f"{X25}/depth_canon_eval_run.sh",
                             f"{X25}/depth_canon_collect.py", f"{X25}/depth_canon_make_cells.py",
                             f"{X25}/depth_canon_cells.json", f"{X25}/depth_canon_audit_gate.py", "overlays/no_flash",
                             f"{X35}/exp35_gpu_identity.py"]


def test_the_digest_of_the_h35_relevant_tree_is_the_campaign_digest(tmp_path):
    m = json.loads(SB.real_cells())
    paths = K.relevant(m)
    tar = subprocess.run(["git", "-C", SB.REPO, "archive", "--format=tar", SB.H35_REAL, "--",
                          *[p for p in paths if not p.startswith(":(")]], capture_output=True, check=True).stdout
    w = tmp_path / "w"
    w.mkdir()
    subprocess.run(["tar", "-x", "-C", str(w)], input=tar, check=True)
    _git(w, "init", "-q")
    _git(w, "add", "-A")
    _git(w, "commit", "-q", "-m", "H35 relevant tree")
    assert K.tree_digest(str(w), paths) == SB.DIGEST_REAL
    assert K.tree_dirty(str(w), paths) == ""


RUNNER_DIGEST = r'''lines="$(git ls-files -z --cached --others --exclude-standard -- "$@" | LC_ALL=C sort -zu \
  | xargs -0r sha256sum --)" && [ -n "$lines" ] || exit 1
printf '%s\n' "$lines" | sha256sum | awk '{print $1}'
'''


def test_the_digest_is_the_runners_bash_formula_with_untracked_and_ignored_files(tmp_path):
    w = tmp_path / "w"
    for rel, text in {"src/a.py": "a\n", "src/sub/b.py": "b\n", "src/tests/t.py": "t\n", "data/AR/x.json": "{}\n",
                      "other.txt": "o\n", ".gitignore": "*.log\n"}.items():
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        (w / rel).write_text(text)
    _git(w, "init", "-q")
    _git(w, "add", "-A")
    _git(w, "commit", "-q", "-m", "x")
    paths = ["src", ":(exclude)src/tests", "data/AR"]
    bash = lambda: subprocess.run(["bash", "-c", RUNNER_DIGEST, "_", *paths], cwd=w, capture_output=True,  # noqa: E731
                                  text=True, check=True).stdout.strip()
    assert K.tree_digest(str(w), paths) == bash() and K.tree_dirty(str(w), paths) == ""
    (w / "src" / "new.py").write_text("untracked\n")                     # untracked, not ignored: in the digest
    (w / "src" / "noise.log").write_text("ignored\n")                    # ignored: never
    (w / "src" / "tests" / "u.py").write_text("excluded\n")              # excluded by the pathspec
    assert K.tree_digest(str(w), paths) == bash() and "src/new.py" in K.tree_dirty(str(w), paths)
    assert "noise.log" not in K.tree_dirty(str(w), paths) and "tests" not in K.tree_dirty(str(w), paths)


def test_a_path_the_runners_sha256sum_would_escape_is_refused(tmp_path):
    w = tmp_path / "w"
    (w / "src").mkdir(parents=True)
    (w / "src" / "a\\b.py").write_text("x\n")
    _git(w, "init", "-q")
    _git(w, "add", "-A")
    _git(w, "commit", "-q", "-m", "x")
    with pytest.raises(ValueError):
        K.tree_digest(str(w), ["src"])
