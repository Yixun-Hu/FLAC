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


# --------------------------------------------------------------------------- #
# EXPLICIT call style: rotate_mode='fixed' passed deliberately
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_explicit_fixed_output_paths_match_golden(case):
    paths = eval_FLAC.build_output_paths(
        META["ckpt_path"], META["steps"], META["cfg_scale"], META["eval_name"],
        cond_method=case["cond_method"], rotate_deg=case["rotate_deg"],
        n_angles=case["n_angles"], rotate_mode="fixed", rotate_seed=None,
    )
    assert paths == case["output_paths"]


@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_explicit_fixed_metrics_record_bytes_match_golden(case, stub_sha):
    """No new key, no reordered key, not one byte of difference — a fixed-mode
    record written today must still be the record exp_11's rows were written as."""
    rec = eval_FLAC.build_metrics_record(
        META["metrics"], META["ckpt_path"], case["rotate_deg"], case["cond_method"],
        case["frame_avg_angles"], rotate_mode="fixed", rotate_seed=None,
        **_record_kwargs(case),
    )
    assert list(rec.keys()) == case["metrics_record_keys"]
    assert json.dumps(rec, sort_keys=False, indent=4) == case["metrics_record_json_indent4"]
    assert json.dumps(rec, sort_keys=False) == case["metrics_record_json_compact"]
    for key in ("rotate_mode", "rotate_seed", "input_hash", "assignment_hash",
                "stream_count", "img_w"):
        assert key not in rec


@pytest.mark.parametrize("case", MATRIX, ids=_case_id)
def test_explicit_fixed_predictions_meta_bytes_match_golden(case, stub_sha):
    meta = eval_FLAC.build_predictions_meta(
        META["dataset_config"], META["seed"], META["n_samples"], case["cond_method"],
        case["frame_avg_angles"], case["rotate_deg"], META["batch_size"],
        META["cond_autocast"], rotate_mode="fixed", rotate_seed=None,
    )
    assert list(meta.keys()) == case["predictions_meta_keys"]
    assert json.dumps(meta, sort_keys=False) == case["predictions_meta_json_compact"]


def test_record_builders_reject_an_unknown_rotate_mode():
    """A typo must not silently produce a fixed-mode record for a random run."""
    with pytest.raises(ValueError, match="rotate_mode"):
        eval_FLAC.build_metrics_record(
            META["metrics"], META["ckpt_path"], 0.0, "vanilla", None, rotate_mode="randon")
    with pytest.raises(ValueError, match="rotate_mode"):
        eval_FLAC.build_predictions_meta(
            META["dataset_config"], 42, 1, "vanilla", None, 0.0, 64, "default",
            rotate_mode="randon")


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


# --------------------------------------------------------------------------- #
# round-3 CLOSURE fix FX1 (findings A1/B1, REPRODUCED by the reviewer)
#
# The per-scene lift ran UNCONDITIONALLY, so any configuration whose callback
# already emits per-scene metrics — HAA enables them independently of exp_14's
# flag (metric_callback.py:81) — had its legacy record rewritten: the nested
# metrics["by_scene"] was REMOVED and top-level by_scene / per_scene_schema /
# scene_count appeared, in a run that never asked for any of it.
#
# The existing snapshot matrix could not catch this: its callback payloads carry
# no by_scene at all. These tests drive the payload that does.
# --------------------------------------------------------------------------- #
_CALLBACK_WITH_BY_SCENE = {
    "T60": 9.0,
    "C50": 1.0,
    "by_scene": {"room_a": {"T60": 8.0, "C50": 0.9},
                 "room_b": {"T60": 10.0, "C50": 1.1}},
}

# The legacy bytes: what an unflagged run has always serialized for this payload
# — the callback result verbatim, nested by_scene included, and no per-scene
# provenance keys anywhere.
_LEGACY_RECORD_BYTES = json.dumps({
    "metrics": _CALLBACK_WITH_BY_SCENE,
    "ckpt_path": "/o/epoch=8-step=40000.ckpt",
    "rotate_deg": 0.0,
    "cond_method": "vanilla",
    "frame_avg_angles": None,
    "cond_autocast": "default",
    "orbit_execution": "n/a",
    "frame_avg_fwd_cap": None,
    "source_sha": "stubbed",
    "batch_size": None,
    "n_samples": None,
    "dataset_config": None,
    "seed": None,
    "cfg_scale": None,
    "steps": None,
    "eval_name": None,
    "weights_source": None,
    "device": None,
}, indent=4)


def _record_for(payload, by_scene, monkeypatch):
    monkeypatch.setattr(eval_FLAC, "source_sha", lambda: "stubbed")
    return eval_FLAC.build_metrics_record(
        payload, "/o/epoch=8-step=40000.ckpt", 0.0, "vanilla", None, by_scene=by_scene)


def test_an_unflagged_callback_that_emits_by_scene_is_passed_through(monkeypatch):
    """No flag ⇒ the callback result is the metrics block, verbatim."""
    payload, by_scene = eval_FLAC.resolve_metrics_payload(
        dict(_CALLBACK_WITH_BY_SCENE), record_per_scene=False)
    assert by_scene is None
    assert payload == _CALLBACK_WITH_BY_SCENE
    assert "by_scene" in payload, "the legacy NESTED per-scene block was removed"


def test_unflagged_record_bytes_are_the_legacy_bytes(monkeypatch):
    """The whole point of the frozen surface, pinned at byte level."""
    payload, by_scene = eval_FLAC.resolve_metrics_payload(
        dict(_CALLBACK_WITH_BY_SCENE), record_per_scene=False)
    record = _record_for(payload, by_scene, monkeypatch)
    assert json.dumps(record, indent=4) == _LEGACY_RECORD_BYTES
    for key in ("by_scene", "per_scene_schema", "scene_count"):
        assert key not in record, f"{key} appeared in a run that never asked for it"


def test_the_flag_still_lifts_the_block(monkeypatch):
    """...and with the flag, the exp_14 shape: flat metrics + top-level block."""
    payload, by_scene = eval_FLAC.resolve_metrics_payload(
        dict(_CALLBACK_WITH_BY_SCENE), record_per_scene=True)
    assert "by_scene" not in payload and by_scene == _CALLBACK_WITH_BY_SCENE["by_scene"]
    record = _record_for(payload, by_scene, monkeypatch)
    assert record["by_scene"] == _CALLBACK_WITH_BY_SCENE["by_scene"]
    assert record["scene_count"] == 2 and record["per_scene_schema"] == 1


def test_the_flag_refuses_a_callback_that_produced_no_per_scene_block():
    """Asking for the estimand and not getting it is an error, not a silent pass."""
    with pytest.raises(RuntimeError):
        eval_FLAC.resolve_metrics_payload({"T60": 9.0}, record_per_scene=True)


def test_evaluate_model_routes_through_the_resolver():
    """The seam must be the one the evaluation actually uses."""
    import inspect
    src = inspect.getsource(eval_FLAC.evaluate_model)
    assert "resolve_metrics_payload(" in src
    assert "split_per_scene_metrics(" not in src, (
        "evaluate_model still splits unconditionally")
