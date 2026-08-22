"""exp_21 round 5, integrative-review BLOCKING 1 + 2: an evaluation must PROVE
what it evaluated, not claim it.

**Finding 1 (trained-as binding).** ``--cond-method fa_cartesian`` selects a
conditioning function at *evaluation* time. Nothing tied it to how the weights
were *trained*: a Vanilla or B-F checkpoint has the identical architecture, so it
loads cleanly, evaluates through the C4 Cartesian orbit, and writes a record
whose ``cond_method`` reads ``fa_cartesian``. The admission gate then verifies
that claim. That is precisely the announcement-05 mismatch (the fa-trained B-F
checkpoint reads 8.202 under fa eval and 10.652 under vanilla eval — plausible
numbers in both directions, unrecoverable after the fact).

``train.py`` embeds the whole model config in every checkpoint
(``ModelConfigEmbedderCallback``, train.py:21), so the artifact can be asked. The
binding below asks it, BEFORE any model or GPU work, and refuses the run
otherwise.

**Finding 2 (checkpoint digest).** ``ckpt_path`` names which file was *asked
for*; only a digest names which bytes were *loaded*. The validator's digest rule
could not engage because no record carried one. Every record written from here on
carries ``ckpt_sha256``.

Scope discipline (round-5 brief): the ABORT is fa_cartesian-scoped, so the
Vanilla and fa_invariant CLI paths behave exactly as before — pinned here, not
assumed. The receipt (``trained_cond_method``) and the digest are recorded for
every method that can supply them, because the D6 comparator arms are validated
by the same machinery and a comparator with no trained-as proof is the same hole
one arm over.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_FLAC  # noqa: E402

ANGLES = [0.0, 90.0, 180.0, 270.0]
TRAIN_CAP = 32          # the ARM's training cap (FLAC_AR_BFC.json)
EVAL_CAP = 64           # the common EVALUATION cap (announcement 06)


_DEFAULT = object()      # "the arm's real value", distinct from a literal None


def embedded(cond_method="fa_cartesian", angles=_DEFAULT, cap=TRAIN_CAP, drop=()):
    """A minimal embedded model_config: only the fields the binding reads.

    Deliberately minimal. A fixture that copied FLAC_AR_BFC.json wholesale would
    pass even if the binding compared nothing at all, because every field it
    could compare would already agree.
    """
    training = {"use_ema": True, "cond_method": cond_method,
                "frame_avg_angles": list(ANGLES) if angles is _DEFAULT else angles,
                "frame_avg_max_fwd_samples": cap}
    for key in drop:
        training.pop(key, None)
    return {"model_type": "diffusion_cond", "training": training}


# --------------------------------------------------------------------------- #
# 1. the binding itself — every abort case, by hand
# --------------------------------------------------------------------------- #
def test_a_conforming_checkpoint_binds_and_returns_its_trained_method():
    assert eval_FLAC.bind_fa_cartesian_checkpoint(embedded()) == "fa_cartesian"


@pytest.mark.parametrize("missing", [None, {}, [], "FLAC_AR_BFC.json", 7])
def test_a_checkpoint_with_no_embedded_config_is_refused(missing):
    """Without the embedded config there is NO evidence about training at all,
    and an assumed one is exactly what this exists to forbid."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(missing)
    msg = str(e.value)
    assert "model_config" in msg and "fa_cartesian" in msg


@pytest.mark.parametrize("trained", ["vanilla", "fa_invariant"])
def test_a_checkpoint_trained_under_another_method_is_refused(trained):
    """The catastrophic case: same architecture, loads cleanly, wrong weights."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(cond_method=trained))
    msg = str(e.value)
    assert trained in msg and "cond_method" in msg


def test_a_checkpoint_whose_config_omits_cond_method_is_refused():
    """An absent key means the factory trained it VANILLA (its default) — the
    exp_07 P1 checkpoint's embedded config is literally this shape."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(drop=("cond_method",)))
    assert "cond_method" in str(e.value)


@pytest.mark.parametrize("angles", [
    [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],   # C8, not C4
    [0.0, 90.0, 180.0],                                     # short orbit
    [0.0, 180.0, 90.0, 270.0],                              # same set, other order
    [0, 90, 180, 270],                                      # ints: a different config
    "0,90,180,270",                                         # a string, not a list
    None,
])
def test_angle_drift_is_refused(angles):
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(angles=angles))
    assert "frame_avg_angles" in str(e.value)


@pytest.mark.parametrize("cap", [64, 16, 32.0, True, None, "32"])
def test_training_cap_drift_is_refused(cap):
    """The cap is the arm's TRAINING chunk plan (D5: 32, one angle per chunk, the
    per-angle RoPE draw schedule B-F trained under). A checkpoint trained at 64
    saw a different augmentation schedule (announcement 06)."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(cap=cap))
    assert "frame_avg_max_fwd_samples" in str(e.value)


def test_an_absent_training_cap_is_refused():
    """B-F's own 40k checkpoint has exactly this shape — the knob postdates its
    training — so 'absent' must not read as 'whatever the default is'."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(drop=("frame_avg_max_fwd_samples",)))
    assert "frame_avg_max_fwd_samples" in str(e.value)


def test_every_abort_message_names_what_was_found_and_what_is_required():
    """A refusal is only actionable if it says which knob disagreed and how."""
    with pytest.raises(ValueError) as e:
        eval_FLAC.bind_fa_cartesian_checkpoint(embedded(cond_method="fa_invariant"))
    msg = str(e.value)
    assert "'fa_invariant'" in msg and "'fa_cartesian'" in msg


def test_the_required_training_contract_is_the_arms_own_config():
    """The constants are not a second opinion: they are FLAC_AR_BFC.json."""
    cfg = json.loads((Path(eval_FLAC.__file__).resolve().parent / "worklog" /
                      "worklog_yixun" / "exp_21_bf_fa_cartesian_claude" /
                      "FLAC_AR_BFC.json").read_text())
    training = cfg["training"]
    assert training["cond_method"] == eval_FLAC.FA_CARTESIAN_TRAINED_COND_METHOD
    assert training["frame_avg_angles"] == list(eval_FLAC.FA_CARTESIAN_TRAINED_ANGLES)
    assert training["frame_avg_max_fwd_samples"] == eval_FLAC.FA_CARTESIAN_TRAINED_FWD_CAP
    # ...and the arm's own config must, of course, bind.
    assert eval_FLAC.bind_fa_cartesian_checkpoint(cfg) == "fa_cartesian"


# --------------------------------------------------------------------------- #
# 2. the digest
# --------------------------------------------------------------------------- #
def test_file_sha256_matches_hashlib_and_is_streamed(tmp_path):
    blob = tmp_path / "b.bin"
    payload = bytes(range(256)) * 5000            # > one read chunk
    blob.write_bytes(payload)
    assert eval_FLAC.file_sha256(str(blob)) == hashlib.sha256(payload).hexdigest()
    assert len(eval_FLAC.file_sha256(str(blob))) == 64


def test_file_sha256_of_an_empty_file_is_the_empty_digest(tmp_path):
    blob = tmp_path / "e.bin"
    blob.write_bytes(b"")
    assert eval_FLAC.file_sha256(str(blob)) == hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------------------- #
# 3. what the record carries
# --------------------------------------------------------------------------- #
DIGEST = "a" * 64


def _record(**kw):
    base = dict(rotate_deg=0.0, cond_method="fa_cartesian",
                frame_avg_angles=list(ANGLES), cond_autocast="bf16", batch_size=64,
                n_samples=6337, dataset_config="ds.json", seed=42, cfg_scale=1.0,
                steps=1, eval_name="e", weights_source="ema", device="cpu",
                frame_avg_max_fwd_samples=EVAL_CAP)
    base.update(kw)
    return eval_FLAC.build_metrics_record({"T60": 1.0}, "c/epoch=8-step=40000.ckpt",
                                          **base)


def test_the_record_carries_the_digest_beside_the_path_it_proves():
    rec = _record(ckpt_sha256=DIGEST)
    assert rec["ckpt_sha256"] == DIGEST
    # adjacency is deliberate: a reader looking for checkpoint identity finds
    # both halves in one place, and the order is part of the file
    keys = list(rec)
    assert keys[keys.index("ckpt_path") + 1] == "ckpt_sha256"


def test_the_record_carries_the_binding_receipt():
    rec = _record(ckpt_sha256=DIGEST, trained_cond_method="fa_cartesian")
    assert rec["trained_cond_method"] == "fa_cartesian"
    keys = list(rec)
    assert keys[keys.index("cond_method") + 1] == "trained_cond_method"


@pytest.mark.parametrize("bad", ["", "zz", "A" * 64, "a" * 63, "a" * 65, 42, b"a" * 64])
def test_a_malformed_digest_is_refused_rather_than_recorded(bad):
    """A record is evidence. A digest that is not a digest would be published as
    a proven identity by a gate that only checks presence."""
    with pytest.raises(ValueError) as e:
        _record(ckpt_sha256=bad)
    assert "ckpt_sha256" in str(e.value)


def test_omitting_the_digest_leaves_the_legacy_record_byte_identical():
    """NO-CHANGE PIN. ``exp14_fixed_mode_golden.json`` freezes the exact JSON
    bytes of a fixed-mode record across {vanilla, fa_invariant} x {0, 45} and
    exists to stop history being re-keyed. Callers that supply no digest — the
    frozen goldens — must therefore be untouched; every REAL run supplies one,
    because ``evaluate_model`` computes it unconditionally (pinned below)."""
    rec = _record()
    assert "ckpt_sha256" not in rec and "trained_cond_method" not in rec


def test_predictions_meta_carries_the_same_two_fields():
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", seed=42, n_samples=7, cond_method="fa_cartesian",
        frame_avg_angles=list(ANGLES), rotate_deg=0.0, batch_size=64,
        cond_autocast="bf16", frame_avg_max_fwd_samples=EVAL_CAP,
        ckpt_sha256=DIGEST, trained_cond_method="fa_cartesian",
    )
    assert meta["ckpt_sha256"] == DIGEST
    assert meta["trained_cond_method"] == "fa_cartesian"
    with pytest.raises(ValueError):
        eval_FLAC.build_predictions_meta(
            "ds.json", seed=42, n_samples=7, cond_method="fa_cartesian",
            frame_avg_angles=list(ANGLES), rotate_deg=0.0, batch_size=64,
            cond_autocast="bf16", ckpt_sha256="nope")


# --------------------------------------------------------------------------- #
# 4. end-to-end through evaluate_model
# --------------------------------------------------------------------------- #
class _FakeModule:
    def __init__(self):
        self.diffusion = types.SimpleNamespace(
            model=types.SimpleNamespace(diffusion_objective="rectified_flow"),
            pretransform=None, conditioner=lambda md, dev, **kw: {},
            io_channels=1, dist_shift=1.0, get_conditioning_inputs=lambda c: {})
        self.device = "cpu"

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def to(self, device):
        return self


def _stub(monkeypatch, tmp_path, ckpt_payload, file_training=None):
    """Toy configs + a checkpoint whose embedded config the test chooses.

    ``create_model_from_config`` is replaced by a FORBIDDEN stub: the binding is
    required to refuse before any model exists, so a test that reaches
    construction fails with a message that says so rather than with whatever the
    stub would have returned.
    """
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1,
        "training": file_training or {"use_ema": False, "cond_method": "fa_cartesian"},
    }))
    dataset_cfg = tmp_path / "dataset.json"
    dataset_cfg.write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = tmp_path / "toy.ckpt"
    torch.save(ckpt_payload, str(ckpt))

    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config",
        lambda cfg, model: _FakeModule())
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: [])
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config",
        lambda *a, **k: types.SimpleNamespace(
            update_metrics=lambda *a, **k: None,
            compute_metrics=lambda split: {"T60": 1.0}))
    return str(model_cfg), str(dataset_cfg), str(ckpt)


def _allow_construction(monkeypatch):
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda cfg: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], [])))


def _forbid_construction(monkeypatch):
    def _boom(cfg):
        raise AssertionError(
            "create_model_from_config was reached: the trained-as binding did not "
            "refuse before model/GPU construction")
    monkeypatch.setattr(eval_FLAC, "create_model_from_config", _boom)


def _run(model_cfg, dataset_cfg, ckpt, **kw):
    args = dict(steps=1, cfg_scale=1.0, device="cpu", eval_name="e",
                cond_method="fa_cartesian", frame_avg_angles=tuple(ANGLES),
                frame_avg_max_fwd_samples=EVAL_CAP, cond_autocast="bf16",
                batch_size=64)
    args.update(kw)
    return eval_FLAC.evaluate_model(model_cfg, dataset_cfg, ckpt, **args)


def test_the_binding_refuses_before_any_model_is_constructed(tmp_path, monkeypatch):
    """The whole point of binding at the checkpoint: it costs no GPU."""
    paths = _stub(monkeypatch, tmp_path,
                  {"state_dict": {}, "model_config": embedded(cond_method="fa_invariant")})
    _forbid_construction(monkeypatch)
    with pytest.raises(ValueError) as e:
        _run(*paths)
    assert "fa_invariant" in str(e.value)


def test_a_checkpoint_with_no_embedded_config_refuses_fa_cartesian(tmp_path, monkeypatch):
    paths = _stub(monkeypatch, tmp_path, {"state_dict": {}})
    _forbid_construction(monkeypatch)
    with pytest.raises(ValueError) as e:
        _run(*paths)
    assert "model_config" in str(e.value)


def test_the_bound_happy_path_records_the_receipt_and_the_real_digest(
        tmp_path, monkeypatch):
    paths = _stub(monkeypatch, tmp_path, {"state_dict": {}, "model_config": embedded()})
    _allow_construction(monkeypatch)
    _run(*paths, eval_name="exp21_BFC_S40000_K8_s42")
    out = tmp_path / "toy_metrics_1_1.0_exp21_BFC_S40000_K8_s42_fa_cartesian_a4.json"
    rec = json.loads(out.read_text())
    assert rec["trained_cond_method"] == "fa_cartesian"
    assert rec["ckpt_sha256"] == eval_FLAC.file_sha256(paths[2])


@pytest.mark.parametrize("method", ["vanilla", "fa_invariant"])
def test_vanilla_and_fa_invariant_are_not_bound_and_still_run(
        tmp_path, monkeypatch, method):
    """NO-CHANGE PIN (round-5 scope). The binding is fa_cartesian-scoped: a
    checkpoint carrying NO embedded config at all still evaluates under the two
    legacy methods, exactly as every committed row was produced."""
    paths = _stub(monkeypatch, tmp_path, {"state_dict": {}},
                  file_training={"use_ema": False})
    _allow_construction(monkeypatch)
    _run(*paths, cond_method=method, eval_name="legacy",
         frame_avg_angles=tuple(ANGLES) if method == "fa_invariant" else None)
    suffix = "_fa_invariant_a4" if method == "fa_invariant" else ""
    rec = json.loads((tmp_path / f"toy_metrics_1_1.0_legacy{suffix}.json").read_text())
    assert rec["cond_method"] == method
    assert "trained_cond_method" not in rec        # nothing was embedded to report
    assert rec["ckpt_sha256"] == eval_FLAC.file_sha256(paths[2])   # digest is uniform


@pytest.mark.parametrize("method,trained", [("vanilla", "vanilla"),
                                            ("fa_invariant", "fa_invariant")])
def test_the_comparator_arms_report_their_own_trained_method(
        tmp_path, monkeypatch, method, trained):
    """D6: B-F and P1 re-evaluated at this pin must carry the same after-the-fact
    proof as BFC. Their exp_07 checkpoints DO embed a config (verified on disk:
    B-F's names cond_method 'fa_invariant' with C4 angles and NO cap key; P1's
    names no cond_method at all, i.e. the factory default, vanilla), so the
    receipt is reported for them too — while the ABORT stays fa_cartesian-only."""
    cfg = embedded(cond_method=trained, drop=("frame_avg_max_fwd_samples",))
    if trained == "vanilla":
        cfg = embedded(drop=("cond_method", "frame_avg_angles",
                             "frame_avg_max_fwd_samples"))
    paths = _stub(monkeypatch, tmp_path, {"state_dict": {}, "model_config": cfg},
                  file_training={"use_ema": False})
    _allow_construction(monkeypatch)
    _run(*paths, cond_method=method, eval_name="repin",
         frame_avg_angles=tuple(ANGLES) if method == "fa_invariant" else None)
    suffix = "_fa_invariant_a4" if method == "fa_invariant" else ""
    rec = json.loads((tmp_path / f"toy_metrics_1_1.0_repin{suffix}.json").read_text())
    assert rec["trained_cond_method"] == trained
