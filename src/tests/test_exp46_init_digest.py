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
