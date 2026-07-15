"""Tests for eval output-path construction, the metrics record, and the exp_02
comparator's meta guard (exp_03 TDD cycle 5, plan §2e / §3 rows
``test_build_output_paths`` and ``test_comparator_meta_guard``).

RED first (commit #10): ``eval_FLAC.build_output_paths`` and
``eval_FLAC.build_metrics_record`` do not exist yet (AttributeError), and
``compare_predictions`` rejects the new dict-format prediction files / has no
meta guard (TypeError on load, AttributeError on the missing guard helpers).
Green after commits #11a (eval_FLAC) and #11b (comparator).

CRITICAL REGRESSION PIN: the vanilla filenames these tests assert are copied
byte-for-byte from the real exp_01 / exp_02 artifacts in ``weights/FLAC/`` (and
``worklog/exp_02_.../metrics_json/``). They must never drift, or those runs stop
being reproducible:

  exp_01 vanilla K1 seed42 metrics : FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json
  exp_02 rot180 metrics            : FLAC_EMA_metrics_1_1.0_yaw_rot180_rot180.json
  exp_02 rot180 predictions (old)  : FLAC_EMA_predictions_1_1.0_yaw_rot180.pt   (buggy: no rot suffix)

``build_output_paths`` is a pure string function, so ``CKPT`` below need not
exist on disk; the leading directory is preserved by ``os.path.join``.
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import eval_FLAC  # noqa: E402  (heavy but side-effect-free at import; argparse is under main())

# The exp_02 comparator lives in a worklog dir that is not a package; add it to
# sys.path so ``import compare_predictions`` resolves to the real file.
_EXP02_DIR = (Path(__file__).resolve().parents[2]
              / "worklog" / "worklog_yixun" / "exp_02_yaw_noninvariance_claude")
if str(_EXP02_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP02_DIR))
import compare_predictions  # noqa: E402


CKPT = "weights/FLAC/FLAC_EMA.ckpt"


# --------------------------------------------------------------------------- #
# build_output_paths
# --------------------------------------------------------------------------- #
def test_legacy_vanilla_paths():
    """Vanilla + rot0 reproduces the exp_01 filenames exactly (both artifacts)."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="exp01_unseen_K1_seed42",
        cond_method="vanilla", rotate_deg=0.0,
    )
    assert paths["metrics"] == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json"
    )
    assert paths["predictions"] == (
        "weights/FLAC/FLAC_EMA_predictions_1_1.0_exp01_unseen_K1_seed42.pt"
    )


def test_legacy_rot_metrics_path():
    """exp_02 rot180 (--eval-name yaw_rot180 --rotate-deg 180, run_exp02.sh:20).

    Metrics stay byte-identical to the committed exp_02 artifact (the doubled
    ``_rot180`` is the historical name and MUST be preserved). Predictions gain
    the rot suffix now -- the bug fix -- so the two no longer collide across
    rotate_deg values that share an eval_name.
    """
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="yaw_rot180",
        cond_method="vanilla", rotate_deg=180.0,
    )
    assert paths["metrics"] == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_yaw_rot180_rot180.json"
    )
    # New behavior (fix): predictions carry _rot180 too (was ..._yaw_rot180.pt).
    assert paths["predictions"] == (
        "weights/FLAC/FLAC_EMA_predictions_1_1.0_yaw_rot180_rot180.pt"
    )


def test_fa_invariant_paths():
    """fa_invariant tags both filenames with method + angle count, before rot."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="unseen_K1",
        cond_method="fa_invariant", rotate_deg=90.0, n_angles=4,
    )
    assert paths["metrics"] == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_unseen_K1_fa_invariant_a4_rot90.json"
    )
    assert paths["predictions"] == (
        "weights/FLAC/FLAC_EMA_predictions_1_1.0_unseen_K1_fa_invariant_a4_rot90.pt"
    )
    # Method tag precedes the rot suffix.
    assert paths["metrics"].index("_fa_invariant_a4") < paths["metrics"].index("_rot90")


def test_fa_invariant_rot0_has_method_no_rot_suffix():
    """fa_invariant at rot0: method tag present, no rot suffix."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="unseen_K1",
        cond_method="fa_invariant", rotate_deg=0.0, n_angles=4,
    )
    assert paths["metrics"] == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_unseen_K1_fa_invariant_a4.json"
    )
    assert paths["predictions"] == (
        "weights/FLAC/FLAC_EMA_predictions_1_1.0_unseen_K1_fa_invariant_a4.pt"
    )


def test_metrics_and_predictions_share_suffix():
    """The metrics/predictions names differ ONLY by kind + extension (no drift)."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="e",
        cond_method="fa_invariant", rotate_deg=270.0, n_angles=4,
    )
    m = paths["metrics"].replace("_metrics_", "_KIND_").removesuffix(".json")
    p = paths["predictions"].replace("_predictions_", "_KIND_").removesuffix(".pt")
    assert m == p


def test_build_output_paths_n_angles_reflected():
    """Angle count is encoded in the method tag (a2 vs default a4)."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="e",
        cond_method="fa_invariant", rotate_deg=0.0, n_angles=2,
    )
    assert "_fa_invariant_a2" in paths["metrics"]
    assert "_fa_invariant_a2" in paths["predictions"]


# --------------------------------------------------------------------------- #
# build_metrics_record
# --------------------------------------------------------------------------- #
def test_metrics_json_records_method_vanilla():
    """Record keeps the legacy keys and adds cond_method + frame_avg_angles."""
    rec = eval_FLAC.build_metrics_record(
        {"T60": 1.23}, CKPT, rotate_deg=0.0, cond_method="vanilla",
        frame_avg_angles=None,
    )
    # legacy keys preserved (exp_01/02 JSONs had exactly these three)
    assert rec["metrics"] == {"T60": 1.23}
    assert rec["ckpt_path"] == CKPT
    assert rec["rotate_deg"] == 0.0
    # new keys
    assert rec["cond_method"] == "vanilla"
    assert rec["frame_avg_angles"] is None
    assert rec["cond_autocast"] == "default"  # omitted kwarg -> the legacy protocol
    json.loads(json.dumps(rec))  # must be JSON-dumpable like the real call


def test_metrics_json_records_method_fa_invariant():
    rec = eval_FLAC.build_metrics_record(
        {"C50": 4.5}, CKPT, rotate_deg=90.0, cond_method="fa_invariant",
        frame_avg_angles=[0.0, 90.0, 180.0, 270.0], cond_autocast="bf16",
    )
    assert rec["cond_method"] == "fa_invariant"
    assert rec["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
    assert rec["rotate_deg"] == 90.0
    assert rec["cond_autocast"] == "bf16"
    json.loads(json.dumps(rec))


# --------------------------------------------------------------------------- #
# comparator meta guard (worklog/exp_02_.../compare_predictions.py)
# --------------------------------------------------------------------------- #
def _save_bare(path, n=3, t=8):
    torch.save(torch.randn(n, 1, t), str(path))


def _save_dict(path, meta, n=3, t=8):
    torch.save({"predictions": torch.randn(n, 1, t), "meta": meta}, str(path))


def _meta(seed=42, dataset_config="cfgA.json", batch_size=32):
    return {
        "dataset_config": dataset_config, "seed": seed, "batch_size": batch_size,
        "cond_method": "vanilla", "frame_avg_angles": None, "rotate_deg": 0.0,
        "n_samples": 3, "cond_autocast": "default",
    }


def test_comparator_loads_legacy_bare(tmp_path):
    """(a) legacy bare tensor file still loads to a [N,1,T] tensor."""
    p = tmp_path / "bare.pt"
    _save_bare(p)
    tensor = compare_predictions.load_predictions(str(p))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 3 and tensor.shape == (3, 1, 8)


def test_comparator_loads_dict_returns_tensor(tmp_path):
    """(b) new dict format loads and returns the inner prediction tensor."""
    p = tmp_path / "dict.pt"
    _save_dict(p, _meta())
    tensor = compare_predictions.load_predictions(str(p))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 1, 8)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("seed", 999),
        ("dataset_config", "other.json"),
        ("batch_size", 1),
        ("cond_method", "fa_invariant"),
        ("frame_avg_angles", [0.0, 90.0, 180.0, 270.0]),
        ("cond_autocast", "bf16"),
    ],
)
def test_comparator_meta_mismatch_raises(tmp_path, field, bad):
    """(c) meta disagreeing on any guarded key (sample correspondence:
    seed/dataset/batch; method comparability: cond_method/frame_avg_angles)
    -> ValueError."""
    ref, alt = tmp_path / "ref.pt", tmp_path / "alt.pt"
    _save_dict(ref, _meta())
    bad_meta = _meta()
    bad_meta[field] = bad
    _save_dict(alt, bad_meta)
    m_ref = compare_predictions.load_prediction_meta(str(ref))
    m_alt = compare_predictions.load_prediction_meta(str(alt))
    with pytest.raises(ValueError):
        compare_predictions.guard_meta(m_ref, m_alt)


def test_comparator_meta_match_proceeds(tmp_path):
    """(d) matching meta proceeds (no raise)."""
    ref, alt = tmp_path / "ref.pt", tmp_path / "alt.pt"
    _save_dict(ref, _meta())
    _save_dict(alt, _meta())
    m_ref = compare_predictions.load_prediction_meta(str(ref))
    m_alt = compare_predictions.load_prediction_meta(str(alt))
    compare_predictions.guard_meta(m_ref, m_alt)  # must not raise


def test_comparator_rotate_deg_mismatch_allowed(tmp_path):
    """rotate_deg is intentionally EXEMPT from the guard: comparing a rotated
    run against an unrotated baseline is exactly this tool's purpose."""
    ref, alt = tmp_path / "ref.pt", tmp_path / "alt.pt"
    _save_dict(ref, _meta())
    rot_meta = _meta()
    rot_meta["rotate_deg"] = 180.0
    _save_dict(alt, rot_meta)
    m_ref = compare_predictions.load_prediction_meta(str(ref))
    m_alt = compare_predictions.load_prediction_meta(str(alt))
    compare_predictions.guard_meta(m_ref, m_alt)  # must not raise


def test_comparator_single_sided_meta_warns_not_raises(tmp_path):
    """Legacy interop: one bare (meta=None) + one dict -> warn only, no raise."""
    bare, rich = tmp_path / "bare.pt", tmp_path / "rich.pt"
    _save_bare(bare)
    _save_dict(rich, _meta())
    m_bare = compare_predictions.load_prediction_meta(str(bare))
    m_rich = compare_predictions.load_prediction_meta(str(rich))
    assert m_bare is None
    compare_predictions.guard_meta(m_bare, m_rich)  # must not raise


# --------------------------------------------------------------------------- #
# evaluate_model: cond_method validation + save-path wiring
# --------------------------------------------------------------------------- #
def test_evaluate_model_unknown_cond_method_raises_fast(tmp_path, monkeypatch):
    """An unknown programmatic cond_method raises ValueError BEFORE any file,
    model or dataloader work: every path argument points at a nonexistent file
    (late validation would surface FileNotFoundError instead), and reaching
    model construction trips a loud sentinel rather than the expected error."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"),
    )
    with pytest.raises(ValueError, match="cond_method"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "missing_model.json"),
            str(tmp_path / "missing_dataset.json"),
            str(tmp_path / "missing.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", cond_method="canon",
        )


class _FakeEvalModule:
    """Minimal stand-in for the PL wrapper evaluate_model drives: chainable
    eval()/requires_grad_()/to(), a .diffusion with no pretransform, and a
    .device. With an empty dataloader the eval loop body never runs, so the
    conditioner/sampler are never touched."""

    def __init__(self):
        self.diffusion = types.SimpleNamespace(
            model=object(), pretransform=None, conditioner=None
        )
        self.device = "cpu"

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def to(self, device):
        return self


def test_evaluate_model_save_path_flows_through_build_output_paths(tmp_path, monkeypatch):
    """Wiring proof (review finding 3): evaluate_model's save stage goes through
    build_output_paths -- a revert to inline path construction fails this test.

    Approach: spy wrapper delegating to the REAL build_output_paths while
    driving evaluate_model end-to-end on CPU with the factories stubbed at the
    eval_FLAC namespace (as test_cond_dispatch does) and an EMPTY dataloader,
    so the sampling loop is skipped without any GPU/data/model. Chosen over the
    sentinel-exception variant because it also proves the metrics JSON lands at
    exactly the path the real function returned (path *used*, not just fetched).
    """
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1, "training": {"use_ema": False},
    }))
    dataset_cfg = tmp_path / "dataset.json"
    dataset_cfg.write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {}}, str(ckpt))

    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda cfg: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], [])),
    )
    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config", lambda cfg, model: _FakeEvalModule()
    )
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: [])
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config",
        lambda *a, **k: types.SimpleNamespace(
            update_metrics=lambda *a, **k: None,
            compute_metrics=lambda split: {"T60": 1.0},
        ),
    )

    real_build = eval_FLAC.build_output_paths
    calls = []

    def spy(*args, **kwargs):
        result = real_build(*args, **kwargs)
        calls.append({"args": args, "kwargs": kwargs, "result": result})
        return result

    monkeypatch.setattr(eval_FLAC, "build_output_paths", spy)

    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt),
        steps=1, cfg_scale=1.0, device="cpu", eval_name="wiring",
        cond_method="fa_invariant", rotate_deg=90.0,
    )

    assert len(calls) == 1, "save stage did not go through build_output_paths"
    kwargs = calls[0]["kwargs"]
    assert kwargs["cond_method"] == "fa_invariant"
    assert kwargs["rotate_deg"] == 90.0
    assert kwargs["n_angles"] == 4  # len(DEFAULT_FRAME_ANGLES)
    # The metrics JSON was written at exactly the returned path, with the
    # build_metrics_record fields intact.
    metrics_path = Path(calls[0]["result"]["metrics"])
    assert metrics_path == tmp_path / "toy_metrics_1_1.0_wiring_fa_invariant_a4_rot90.json"
    saved = json.loads(metrics_path.read_text())
    assert saved["cond_method"] == "fa_invariant"
    assert saved["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]


# --------------------------------------------------------------------------- #
# launch conditions (full review C1/C2): cond-autocast control + load integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode,expected",
    [
        ("default", (True, None)),          # torch per-device default = exp_01/02 protocol
        ("bf16", (True, torch.bfloat16)),   # matches finetune_cond bf16-mixed training
        ("off", (False, None)),             # fp32, for exactness measurements
    ],
)
def test_resolve_cond_autocast_modes(mode, expected):
    assert eval_FLAC.resolve_cond_autocast(mode) == expected


def test_resolve_cond_autocast_unknown_raises():
    with pytest.raises(ValueError, match="cond_autocast"):
        eval_FLAC.resolve_cond_autocast("fp8")


def test_evaluate_model_unknown_cond_autocast_raises_fast(tmp_path, monkeypatch):
    """Programmatic callers fail fast, before any file/model work (paths are
    nonexistent: late validation would surface FileNotFoundError instead)."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"),
    )
    with pytest.raises(ValueError, match="cond_autocast"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", cond_autocast="fp8",
        )


def test_parser_cond_autocast_choices():
    """argparse enforces the --cond-autocast choices (bad value -> exit 2 with
    'invalid choice'; a valid value gets past choices to the required-args error)."""
    root = str(Path(__file__).resolve().parents[2])
    bad = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--cond-autocast", "fp8"],
        capture_output=True, cwd=root,
    )
    assert bad.returncode == 2
    assert b"invalid choice" in bad.stderr
    good = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--cond-autocast", "bf16"],
        capture_output=True, cwd=root,
    )
    assert good.returncode == 2  # still fails on the missing required args...
    assert b"invalid choice" not in good.stderr  # ...but not on choices


def test_predictions_meta_includes_cond_autocast():
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", seed=42, n_samples=7, cond_method="fa_invariant",
        frame_avg_angles=[0.0, 90.0], rotate_deg=0.0, batch_size=32,
        cond_autocast="bf16",
    )
    assert meta["cond_autocast"] == "bf16"
    assert meta["n_samples"] == 7
    for key in ("dataset_config", "seed", "cond_method", "frame_avg_angles",
                "rotate_deg", "batch_size"):
        assert key in meta


def test_comparator_missing_cond_autocast_treated_as_default():
    """Sidecars written before the cond_autocast field factually ran the default
    autocast: a missing field compares equal to an explicit 'default' (no false
    hard-error on legacy sidecars) but still mismatches an explicit 'bf16'."""
    legacy = _meta()
    legacy.pop("cond_autocast")
    compare_predictions.guard_meta(legacy, _meta())  # must not raise
    bf16 = _meta()
    bf16["cond_autocast"] = "bf16"
    with pytest.raises(ValueError):
        compare_predictions.guard_meta(legacy, bf16)


def test_load_integrity_real_state_dict():
    """Through a real load_state_dict (tiny-config wrapper pattern): clean load
    passes; PL-wrapper whitelisted leftovers (diffusion_ema./losses., the exact
    pattern of outputs_FLAC/ft_vanilla/epoch=0-step=2000.ckpt) pass; a dropped
    model key raises RuntimeError naming the key."""
    from test_cond_dispatch import _base_config
    from src.models.factory import create_model_from_config
    model = create_model_from_config(_base_config())

    missing, unexpected = model.load_state_dict(model.state_dict(), strict=False)
    eval_FLAC.check_load_integrity(missing, unexpected)  # clean: must not raise

    sd = dict(model.state_dict())
    sd["diffusion_ema.initted"] = torch.tensor(True)
    sd["diffusion_ema.step"] = torch.tensor(0)
    sd["losses.some_loss_buffer"] = torch.zeros(1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(unexpected) == 3
    eval_FLAC.check_load_integrity(missing, unexpected)  # whitelisted: no raise

    sd = dict(model.state_dict())
    dropped = sorted(sd)[0]
    del sd[dropped]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.check_load_integrity(missing, unexpected)
    assert dropped in str(exc.value)


def test_load_integrity_stray_and_escape_hatch(capsys):
    """A non-whitelisted unexpected key hard-errors naming it; --allow-partial-load
    downgrades any mismatch to a warning and continues."""
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.check_load_integrity([], ["totally.stray_key"])
    assert "totally.stray_key" in str(exc.value)

    eval_FLAC.check_load_integrity(["model.gone"], [], allow_partial_load=True)
    out = capsys.readouterr().out
    assert "model.gone" in out and "WARNING" in out
