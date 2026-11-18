"""exp_46 — the registered init-digest prediction (plan_c4flip_data_curve.md r3 §3.2 "Init digest").

The C4F arm's seed-42 initial state is predicted to be exp_38's (the source change only removes two raises; the C4F
constructor creates no state tensor and draws nothing at construction): exp46_registry's INIT_STATE_SHA256,
INIT_SOBOL_SHA256 / INIT_SOBOL, N_STATE_TENSORS and RNG_AT_FIT_START are exp_38's values -- read from exp_38's own
registry (read-only) and from exp_45's at X45f, never retyped. The prediction is verified fail-closed at L3 by
exp46_init_digest.py and on every rank by the evidence gate: a C4F rank whose state, Sobol or fit-start RNG digest
differs from the prediction refuses (both topologies).

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import ast
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
sys.path.insert(0, TESTS)
import exp46_registry as reg  # noqa: E402
import exp46_fixtures as fx  # noqa: E402

X45F = "1e37ead2a352bd78d50cdda9ed02e95738a4ad04"
NAMES = ("INIT_STATE_SHA256", "INIT_SOBOL_SHA256", "N_STATE_TENSORS", "RNG_PY", "RNG_NP", "RNG_CUDA")


def _literals(src):
    """{name: value} of the module-level literal assignments of ``src`` among NAMES + RNG_AT_FIT_START's torch_cpu."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in NAMES:
                out[name] = ast.literal_eval(node.value)
            elif name == "RNG_AT_FIT_START":
                out[name] = {topo: ast.literal_eval(v.values[[k.value for k in v.keys].index("torch_cpu")])
                             for topo, v in ((k.value, v) for k, v in zip(node.value.keys, node.value.values))}
    return out


def test_the_registered_prediction_is_exp38s_values_read_from_exp38s_registry():
    sys.path.insert(0, reg._abs(reg.EXP38_DIR))
    import exp38_registry as r38                                # exp_38's own module, read-only
    assert os.path.dirname(os.path.realpath(r38.__file__)) == os.path.realpath(reg._abs(reg.EXP38_DIR))
    for name in ("INIT_STATE_SHA256", "INIT_SOBOL_SHA256", "INIT_SOBOL", "N_STATE_TENSORS", "RNG_AT_FIT_START"):
        assert getattr(reg, name) == getattr(r38, name), name
    assert reg.INIT_SOBOL["state_sha256"] == reg.INIT_SOBOL_SHA256 and set(reg.RNG_AT_FIT_START) == {"single", "ddp"}


def test_the_prediction_equals_exp45s_at_x45f_and_exp38s_file_literally():
    src38 = open(os.path.join(reg._abs(reg.EXP38_DIR), "exp38_registry.py")).read()
    src45 = subprocess.run(["git", "-C", REPO, "show", f"{X45F}:worklog/worklog_yixun/exp_45_rf_data_curve_claude/"
                            "exp45_registry.py"], capture_output=True, text=True, check=True).stdout
    src46 = open(os.path.join(EXP, "exp46_registry.py")).read()
    lit38, lit45, lit46 = _literals(src38), _literals(src45), _literals(src46)
    assert set(lit46) == set(NAMES) | {"RNG_AT_FIT_START"} and lit46 == lit45 == lit38


@pytest.mark.parametrize("ngpu", [1, 2])
@pytest.mark.parametrize("field", ["state_dict_sha256", "sobol", "rng_at_fit_start", "n_state_tensors"])
def test_a_c4f_rank_off_the_prediction_is_refused_by_the_evidence_gate(ngpu, field):
    """Fail-closed: every C4F run's per-rank evidence must report the predicted digests (as exp_38's arms do)."""
    for run in reg.required_runs():
        recs = fx.records(reg, run, ngpu=ngpu, backend="flash", max_steps=20)
        assert reg.evidence_problems(recs, run, ngpu, "flash", 20) == [], run
        bad = {r: dict(e) for r, e in recs.items()}
        if field == "state_dict_sha256":
            bad[ngpu - 1][field] = "0" * 64
        elif field == "sobol":
            bad[ngpu - 1][field] = dict(bad[ngpu - 1][field], state_sha256="1" * 64)
        elif field == "rng_at_fit_start":
            bad[ngpu - 1][field] = dict(bad[ngpu - 1][field], torch_cpu="2" * 64)
        else:
            bad[ngpu - 1][field] = reg.N_STATE_TENSORS + 1
        probs = reg.evidence_problems(bad, run, ngpu, "flash", 20)
        assert len(probs) == 1 and probs[0].startswith(f"rank {ngpu - 1}: {field} "), (run, probs)


# ------------------------------------------------------------------ fix round 1, Codex code r1 finding 5: L3 enforces it
import exp46_init_digest as idg  # noqa: E402


def _measured(run_id, **over):
    """What init_digest() measures on the CPU when the prediction holds (the shape of exp_38's recorded L3 output:
    rng_after_build = the single-topology fit-start python / numpy / torch_cpu digests); ``over`` replaces fields."""
    out = {"run": run_id, "state_dict_sha256": reg.INIT_STATE_SHA256, "n_state_tensors": reg.N_STATE_TENSORS,
           "sobol": dict(reg.INIT_SOBOL),
           "rng_after_build": {k: reg.RNG_AT_FIT_START["single"][k] for k in ("python", "numpy", "torch_cpu")}}
    out.update(over)
    return out


def test_l3_passes_exactly_the_registered_prediction(tmp_path, monkeypatch, capsys):
    """The L3 command compares the measurement with exp_38's registered values and exits 0 only when every one
    agrees; its --json record carries the measurement and the verdict."""
    for run in reg.required_runs():
        monkeypatch.setattr(idg, "init_digest", lambda r, repo=None: _measured(r))
        out = tmp_path / f"{run}.json"
        assert idg.main(["--run", run, "--json", str(out)]) == 0
        rec = json.load(open(out))
        assert rec["prediction"] == "PASS" and rec["prediction_problems"] == [] and rec["run"] == run
        assert "INIT DIGEST PREDICTION PASS" in capsys.readouterr().out
    cuda = _measured(RUN_C4F)
    cuda["rng_after_build"]["torch_cuda"] = reg.RNG_AT_FIT_START["single"]["torch_cuda"]      # CUDA initialised
    assert idg.prediction_problems(cuda, RUN_C4F) == [] and idg.prediction_problems(_measured(RUN_C4F), RUN_C4F) == []


RUN_C4F = "exp46_C4Ff025_l40_2x32"
_RNG = {k: reg.RNG_AT_FIT_START["single"][k] for k in ("python", "numpy", "torch_cpu")}
OFF = {"state digest": ("state_dict_sha256", "0" * 64), "tensor count": ("n_state_tensors", reg.N_STATE_TENSORS + 1),
       "tensor count as a float": ("n_state_tensors", float(reg.N_STATE_TENSORS)),
       "sobol state": ("sobol", dict(reg.INIT_SOBOL, state_sha256="1" * 64)),
       "sobol drawn": ("sobol", dict(reg.INIT_SOBOL, num_generated=1)),
       "cpu rng": ("rng_after_build", dict(_RNG, torch_cpu="2" * 64)),
       "the ddp cpu rng": ("rng_after_build", dict(_RNG, torch_cpu=reg.RNG_AT_FIT_START["ddp"]["torch_cpu"])),
       "python rng": ("rng_after_build", dict(_RNG, python="3" * 64)),
       "a missing rng": ("rng_after_build", {k: v for k, v in _RNG.items() if k != "numpy"}),
       "a wrong cuda rng": ("rng_after_build", dict(_RNG, torch_cuda="4" * 64)),
       "another run": ("run", "exp46_C4Ff050_l40_2x32")}


@pytest.mark.parametrize("case", sorted(OFF))
def test_l3_exits_nonzero_on_any_disagreement_with_the_prediction(tmp_path, monkeypatch, case):
    field, value = OFF[case]
    monkeypatch.setattr(idg, "init_digest", lambda r, repo=None: _measured(r, **{field: value}))
    out = tmp_path / "init_digest.json"
    assert idg.main(["--run", RUN_C4F, "--json", str(out)]) == 1
    rec = json.load(open(out))                                     # the measurement is still recorded, with its verdict
    assert rec["prediction"] == "FAIL" and rec["prediction_problems"], case
    assert idg.prediction_problems(_measured(RUN_C4F, **{field: value}), RUN_C4F), case
    extra = _measured(RUN_C4F, extra_key=1)
    assert idg.prediction_problems(extra, RUN_C4F)                 # the measured schema is exact


# ------------------------------------------------------------------ fix round 2, Codex code r2 finding 3: one path
def test_the_json_destination_is_anchored_before_init_digest_changes_the_cwd(tmp_path, monkeypatch):
    """init_digest() changes the cwd to the repo before it builds the model: --json is made absolute from the CALLER's
    cwd first, and that one path is checked, re-checked just before the write and written -- a relative destination is
    never re-resolved against the repo (the r2 probe: from src/, ``../out/x.json`` checked ``<repo>/out`` but wrote
    ``<repo>/../out``, a protected root there)."""
    tree = tmp_path / "repo"                     # the tree init_digest() chdirs into
    (tree / "src").mkdir(parents=True)
    (tree / "out").mkdir()
    protected = tmp_path / "out"                 # what the same relative string names from the repo: protected
    protected.mkdir()
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", (str(protected),))

    def measuring(run, repo=None, during=None):
        os.chdir(str(tree))
        if during:
            during()
        return _measured(run)
    monkeypatch.setattr(idg, "init_digest", measuring)
    monkeypatch.chdir(tree / "src")                                   # a non-root cwd
    assert idg.main(["--run", RUN_C4F, "--json", "../out/init.json"]) == 0
    assert os.listdir(protected) == []                                # nothing through the changed cwd
    assert json.load(open(tree / "out" / "init.json"))["prediction"] == "PASS"      # the caller's path
    monkeypatch.chdir(tree / "src")
    monkeypatch.setattr(idg, "init_digest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model built")))
    assert idg.main(["--run", RUN_C4F, "--json", "../../out/init.json"]) == 2       # `../` into the protected root
    (tree / "out" / "theirs.json").write_text("theirs\n")
    assert idg.main(["--run", RUN_C4F, "--json", "../out/theirs.json"]) == 2        # existing: refused up front
    monkeypatch.chdir(tree / "src")                                   # created meanwhile: never overwritten
    monkeypatch.setattr(idg, "init_digest", lambda r, repo=None: measuring(
        r, during=lambda: (tree / "out" / "race.json").write_text("theirs\n")))
    assert idg.main(["--run", RUN_C4F, "--json", "../out/race.json"]) == 2
    assert (tree / "out" / "race.json").read_text() == "theirs\n"
    monkeypatch.chdir(tree / "src")                                   # a component aliased meanwhile: re-checked
    (tree / "out").rename(tree / "out_moved")
    monkeypatch.setattr(idg, "init_digest", lambda r, repo=None: measuring(
        r, during=lambda: (tree / "out").symlink_to(protected)))
    assert idg.main(["--run", RUN_C4F, "--json", "../out/late.json"]) == 2
    assert os.listdir(protected) == []
