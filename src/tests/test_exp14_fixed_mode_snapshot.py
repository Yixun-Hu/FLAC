"""exp_14 round 1, STEP 0 — the fixed-mode BYTE-COMPAT CONTRACT (plan §5.3.6,
Codex plan-review finding B4).

exp_14 adds a random-yaw evaluation mode to ``eval_FLAC``. The whole campaign
(and every earlier experiment's reproducibility) rests on the fixed path being
*untouched*: same output paths, same metrics-record keys IN THE SAME ORDER, same
serialized JSON bytes, same predictions meta. A new key silently appearing in a
fixed-mode record would re-key every historical row.

``exp14_fixed_mode_golden.json`` was captured by running the CURRENT (pre-change)
``build_output_paths`` / ``build_metrics_record`` / ``build_predictions_meta``
over the matrix ``{vanilla, fa_invariant} x {rotate_deg 0.0, 45.0}`` plus a
defaults-only case, BEFORE any edit to ``eval_FLAC.py`` (the capture SHA is
recorded in the fixture's ``_meta``). These tests are therefore GREEN at the base
commit by construction — that is the point: they are the frozen reference the
round-1 changes must still reproduce.

``source_sha()`` shells out to git and is not part of the contract, so the
capture stubbed it to a literal and these tests stub it the same way.

Two call styles must produce identical golden bytes:
  * DEFAULT — no new arguments at all (what every existing caller, script and
    screen kit does today) — pinned here, green at the base commit;
  * EXPLICIT — ``rotate_mode='fixed'`` passed deliberately — added with the
    ``rotate_mode`` parameter itself (round-1 cycle 3), since that argument does
    not exist yet.
"""
import json
from pathlib import Path

import pytest

import eval_FLAC  # noqa: E402  (heavy but side-effect-free at import)


_GOLDEN_PATH = Path(__file__).resolve().parent / "exp14_fixed_mode_golden.json"
GOLDEN = json.loads(_GOLDEN_PATH.read_text())
META = GOLDEN["_meta"]
CASES = GOLDEN["cases"]
# The 2x2 matrix cases (the defaults-only case is exercised separately).
MATRIX = [c for c in CASES if "kwargs" not in c]


@pytest.fixture
def stub_sha(monkeypatch):
    """Freeze the git-derived provenance field to the value used at capture."""
    monkeypatch.setattr(eval_FLAC, "source_sha", lambda: META["source_sha_stub"])


def _case_id(case):
    return f'{case["cond_method"]}_rot{case["rotate_deg"]}'


def _record_kwargs(case):
    """The build_metrics_record kwargs the capture used, minus the mode."""
    return dict(
        cond_autocast=META["cond_autocast"],
        batch_size=META["batch_size"],
        n_samples=META["n_samples"],
        dataset_config=META["dataset_config"],
        seed=META["seed"],
        cfg_scale=META["cfg_scale"],
        steps=META["steps"],
        eval_name=META["eval_name"],
        weights_source=META["weights_source"],
        device=META["device"],
    )


# --------------------------------------------------------------------------- #
# fixture integrity
# --------------------------------------------------------------------------- #
def test_golden_fixture_is_the_expected_matrix():
    """The contract covers {vanilla, fa} x {0, 45} plus the defaults-only case."""
    assert len(CASES) == 5
    assert {(c["cond_method"], c["rotate_deg"]) for c in MATRIX} == {
        ("vanilla", 0.0), ("vanilla", 45.0),
        ("fa_invariant", 0.0), ("fa_invariant", 45.0),
    }
    assert META["captured_from_sha"] and META["source_sha_stub"]


# --------------------------------------------------------------------------- #
# DEFAULT call style: no new arguments (every existing caller)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_default_call_output_paths_match_golden(case):
    paths = eval_FLAC.build_output_paths(
        META["ckpt_path"], META["steps"], META["cfg_scale"], META["eval_name"],
        cond_method=case["cond_method"], rotate_deg=case["rotate_deg"],
        n_angles=case["n_angles"],
    )
    assert paths == case["output_paths"]


@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_default_call_metrics_record_bytes_match_golden(case, stub_sha):
    rec = eval_FLAC.build_metrics_record(
        META["metrics"], META["ckpt_path"], case["rotate_deg"], case["cond_method"],
        case["frame_avg_angles"], **_record_kwargs(case),
    )
    assert list(rec.keys()) == case["metrics_record_keys"]
    assert json.dumps(rec, sort_keys=False, indent=4) == case["metrics_record_json_indent4"]
    assert json.dumps(rec, sort_keys=False) == case["metrics_record_json_compact"]


@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_default_call_predictions_meta_bytes_match_golden(case, stub_sha):
    meta = eval_FLAC.build_predictions_meta(
        META["dataset_config"], META["seed"], META["n_samples"], case["cond_method"],
        case["frame_avg_angles"], case["rotate_deg"], META["batch_size"],
        META["cond_autocast"],
    )
    assert list(meta.keys()) == case["predictions_meta_keys"]
    assert json.dumps(meta, sort_keys=False) == case["predictions_meta_json_compact"]


def test_defaults_only_record_matches_golden(stub_sha):
    """Every optional kwarg omitted: the None defaults and their order are pinned."""
    case = next(c for c in CASES if c.get("kwargs") == "defaults-only")
    rec = eval_FLAC.build_metrics_record(
        META["metrics"], META["ckpt_path"], 0.0, "vanilla", None,
    )
    assert list(rec.keys()) == case["metrics_record_keys"]
    assert json.dumps(rec, sort_keys=False, indent=4) == case["metrics_record_json_indent4"]
    paths = eval_FLAC.build_output_paths(
        META["ckpt_path"], META["steps"], META["cfg_scale"], META["eval_name"],
    )
    assert paths == case["output_paths"]
