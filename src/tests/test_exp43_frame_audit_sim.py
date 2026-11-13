"""exp_43 std_aug -- the archived frame-audit simulation scripts (Codex code review r1, finding 2).

``worklog/worklog_yixun/exp_43_std_aug_claude/frame_audit_sim/{sim_gate,design_sources}.py`` must
run from any clone and any working directory: no absolute checkout or scratch path, the repository
root resolved from the script's own location, no work at import time, a ``main`` entry point,
``design_sources.py`` writing only to a required, new ``--out`` file. Lightweight (seconds): the heavy
simulations themselves are never run here -- their two smoke runs replace the audit's scoring by
cheap stubs and only prove that the scripts' plumbing works against the current audit API.
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SIM = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude/frame_audit_sim"
SCRIPTS = {"design_sources": _SIM / "design_sources.py", "sim_gate": _SIM / "sim_gate.py"}
AUDIT_TESTS = _REPO / "src/tests/test_exp43_frame_audit.py"
W = 512


def _import(name):
    """Import a script by path (a fresh module object each time); fails fast if it would run."""
    text = SCRIPTS[name].read_text()
    assert 'if __name__ == "__main__":' in text, f"{name}: top-level code is not guarded"
    spec = importlib.util.spec_from_file_location(f"exp43_frame_audit_sim_{name}", SCRIPTS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_script_holds_no_absolute_home_or_tmp_path(name):
    text = SCRIPTS[name].read_text()
    assert "/home/" not in text and "/tmp/" not in text
    assert "scratchpad" not in text and "codespace" not in text


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_script_imports_from_any_cwd_without_running(name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["no-seed"])          # a top-level sys.argv[1] would raise
    mod = _import(name)
    assert callable(mod.main) and callable(mod.load_audit_test_module)
    assert capsys.readouterr().out == ""                    # nothing ran at import
    t = mod.load_audit_test_module()                         # resolved from the script's own file
    assert Path(t.__file__).resolve() == AUDIT_TESTS
    assert Path(t.TOOL_PATH).resolve() == _SIM.parent / "exp43_frame_audit.py"
    assert list(tmp_path.iterdir()) == []                   # nothing written


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_help_exits_zero_from_any_cwd(name, tmp_path):
    assert 'if __name__ == "__main__":' in SCRIPTS[name].read_text()   # else --help would run it
    r = subprocess.run([sys.executable, str(SCRIPTS[name]), "--help"], cwd=tmp_path, capture_output=True,
                       text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0, r.stderr
    assert r.stdout.lower().startswith("usage:")
    assert list(tmp_path.iterdir()) == []


def _boom(*a, **k):
    raise AssertionError("refused before any computation")


def test_design_sources_requires_a_new_out_file(tmp_path, monkeypatch, capsys):
    mod = _import("design_sources")
    monkeypatch.setattr(mod, "load_audit_test_module", _boom)
    with pytest.raises(SystemExit) as exc:
        mod.main([])                                         # --out is required
    assert exc.value.code == 2 and "--out" in capsys.readouterr().err
    existing = tmp_path / "designed_sources.json"
    existing.write_text("keep")
    assert mod.main(["--out", str(existing)]) == 2
    assert existing.read_text() == "keep" and "refusing to overwrite" in capsys.readouterr().err
    assert mod.main(["--out", str(tmp_path / "no_such_dir" / "x.json")]) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["designed_sources.json"]
    # the same refusal through the real command line (from another cwd)
    r = subprocess.run([sys.executable, str(SCRIPTS["design_sources"]), "--out", str(existing)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 2 and "refusing to overwrite" in r.stderr and existing.read_text() == "keep"
    r = subprocess.run([sys.executable, str(SCRIPTS["design_sources"])], cwd=tmp_path, capture_output=True,
                       text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 2 and "--out" in r.stderr


def test_sim_gate_keeps_its_seed_argument(monkeypatch):
    mod = _import("sim_gate")
    monkeypatch.setattr(mod, "load_audit_test_module", _boom)
    for argv in ([], ["one"]):
        with pytest.raises(SystemExit) as exc:
            mod.main(argv)
        assert exc.value.code == 2


def _stub_scores(n_src):
    """Cheap stand-in for score_room: a clean peak at (I, 0) for both scores."""
    a = np.zeros((2, W), dtype=np.int64)
    a[0, 0] = n_src
    b = np.zeros((2, W), dtype=np.int64)
    b[0, 0] = 10
    return {"A_count": a, "A_n": n_src, "B_agree": b, "B_viol": np.zeros((2, W), np.int64), "B_n": 10,
            "n_degenerate_pairs": 0, "n_excluded_receiver_pairs": 0}


def _stubbed_audit_tests(mod, monkeypatch):
    """A private copy of the audit test module whose scoring is stubbed (nothing heavy runs)."""
    t = mod.load_audit_test_module()
    monkeypatch.setattr(mod, "load_audit_test_module", lambda: t)
    monkeypatch.setattr(mod.torch, "set_num_threads", lambda n: None)     # leave this process alone
    return t


def test_sim_gate_main_runs_against_the_current_audit_api(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    mod = _import("sim_gate")
    t = _stubbed_audit_tests(mod, monkeypatch)
    seen = []
    monkeypatch.setattr(t, "room_struct", lambda sp, s, k: (seen.append((s, k)), sp)[1])
    monkeypatch.setattr(t.fa, "score_room", lambda sp: _stub_scores(len(sp["sources"])))
    real_eval = t.fa.evaluate_gate
    cats_seen = []

    def spy(rooms, **kw):
        cats_seen.append(kw.get("categories"))
        return real_eval(rooms, **kw)

    monkeypatch.setattr(t.fa, "evaluate_gate", spy)
    assert mod.main(["1"]) in (None, 0)        # seed 1: the room geometry of the archived run 1
    out = json.loads(capsys.readouterr().out)
    assert out["seed"] == 1 and out["verdict"] == "PASS" and out["boot_pass"] == 1000
    assert out["full_A"] == ["I:k=0"] and out["full_B"] == ["I:k=0"] and out["rooms_B_exact"] == 33
    assert set(seen) == {(0, 0)} and len(seen) == 33
    assert cats_seen == [["Ap", "Au", "Ba", "Be", "Ca", "Li", "Lv", "Me", "Of", "Re"]]
    assert list(out["loco_A"]) == cats_seen[0]
    assert list(tmp_path.iterdir()) == []


def test_design_sources_main_writes_only_its_out_file(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    mod = _import("design_sources")
    t = _stubbed_audit_tests(mod, monkeypatch)
    monkeypatch.setattr(t, "render", lambda sp, r, s, k: np.ones((2, 2)))

    def score_a(u, dflat, valid):                             # every candidate visible at (I, 0) only
        out = np.zeros((2, W), dtype=np.int64)
        out[0, 0] = 1
        return out

    monkeypatch.setattr(t.fa, "_score_a_receiver", score_a)
    out = tmp_path / "sub" / "designed.json"
    out.parent.mkdir()
    assert mod.main(["--out", str(out)]) in (None, 0)
    res = json.loads(out.read_text())
    assert list(res) == [spec["room_id"] for spec in t.ROOMS]
    assert all(len(v) == 20 and all(len(p) == 3 for p in v) for v in res.values())
    printed = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in printed] == [spec["room_id"] for spec in t.ROOMS]
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == ["sub", "sub/designed.json"]
    assert mod.main(["--out", str(out)]) == 2                 # never overwrites its own output either


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_script_uses_only_audit_names_that_exist(name):
    """Every ``t.<x>`` / ``fa.<x>`` a script touches exists in the current audit test module / tool."""
    mod = _import(name)
    t = mod.load_audit_test_module()
    used = {"t": set(), "fa": set()}
    for node in ast.walk(ast.parse(SCRIPTS[name].read_text())):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in used:
            used[node.value.id].add(node.attr)
    assert used["t"] and used["fa"]
    assert [x for x in sorted(used["t"]) if not hasattr(t, x)] == []
    assert [x for x in sorted(used["fa"]) if not hasattr(t.fa, x)] == []
