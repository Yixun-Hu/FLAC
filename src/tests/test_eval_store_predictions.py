"""Tests for ``--store_predictions`` storing the *exactly-as-scored* tensor
(exp_14 round C; announcement 08 ``worklog/worklog_yixun/announcement/08_save_pred_waveforms.md``).

At pin ``7bbd8aa`` ``eval_FLAC``'s per-batch loop appended ``fakes.cpu()`` to
``decoded_samples`` **before** the clamp/pad block, so the stored bundle was the
raw decoder output, not the tensor the metric callback scored. Round C extracts
that block into :func:`eval_FLAC.clamp_and_pad` and moves the append after it.

RED first, one cycle per commit:
  C1 ``eval_FLAC.clamp_and_pad`` does not exist (AttributeError).
  C2 the stored tensor is the pre-clamp/pad one (not bitwise the scored input).
  C3 the bundle meta lacks ``ckpt_path`` / ``n_items`` / ``eval_name`` / ``steps``
     / ``cfg_scale`` / ``stored_after_clamp_pad`` / ``artifact_contract``.

The artifact contract pinned here: the stored tensor is the **clamped/padded
callback input** -- the metric callback's own ``float()`` cast and its
``max_len`` (8000-sample) crop happen *inside* scoring and are deliberately not
part of the artifact.
"""
import hashlib
import json
import os
import types

import pytest
import torch

import eval_FLAC  # noqa: E402  (heavy but side-effect-free at import)
import src.inference.sampling as sampling  # patched in place: the loop imports it inline


# --------------------------------------------------------------------------- #
# Pinned reference: the clamp/pad block as it stood at 7bbd8aa
# (``git show 7bbd8aa:eval_FLAC.py``, lines 305-311), copied verbatim into a
# function. Only whitespace is normalized (the pin has a trailing space after
# ``clamp(-1.0, 1.0)``); no token differs. DO NOT EDIT -- this is the
# equivalence oracle for the extraction.
# --------------------------------------------------------------------------- #
def _pinned_clamp_and_pad(fakes, reals):
    # Clamp and pad if necessary
    fakes = fakes.clamp(-1.0, 1.0)
    if fakes.shape != reals.shape:
        if fakes.shape[-1] < reals.shape[-1]:
            fakes = torch.nn.functional.pad(fakes, (0, reals.shape[-1] - fakes.shape[-1]))
        else:
            reals = torch.nn.functional.pad(reals, (0, fakes.shape[-1] - reals.shape[-1]))
    return fakes, reals


# --------------------------------------------------------------------------- #
# C1: clamp_and_pad semantics
# --------------------------------------------------------------------------- #
def test_clamp_and_pad_clamps_out_of_range_values():
    """Values outside [-1, 1] are clamped; in-range values pass through."""
    fakes = torch.tensor([[[-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 7.5]]])
    reals = torch.zeros(1, 1, 7)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert torch.equal(
        out_f, torch.tensor([[[-1.0, -1.0, -0.25, 0.0, 0.5, 1.0, 1.0]]])
    )
    assert out_f.min() >= -1.0 and out_f.max() <= 1.0
    assert torch.equal(out_r, reals)


def test_clamp_and_pad_short_fakes_are_zero_padded_to_reals():
    """fakes shorter than reals -> fakes zero-padded on the right; reals untouched."""
    fakes = torch.full((2, 1, 5), 2.0)
    reals = torch.arange(2 * 1 * 8, dtype=torch.float32).reshape(2, 1, 8)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.shape == reals.shape
    assert torch.equal(out_f[..., :5], torch.ones(2, 1, 5))   # clamped 2.0 -> 1.0
    assert torch.equal(out_f[..., 5:], torch.zeros(2, 1, 3))  # zero padding
    assert out_r is reals


def test_clamp_and_pad_long_fakes_pad_reals():
    """fakes longer than reals -> reals zero-padded instead; fakes keep their length."""
    fakes = torch.full((2, 1, 11), -4.0)
    reals = torch.ones(2, 1, 8)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.shape == (2, 1, 11)
    assert out_r.shape == (2, 1, 11)
    assert torch.equal(out_f, torch.full((2, 1, 11), -1.0))
    assert torch.equal(out_r[..., :8], torch.ones(2, 1, 8))
    assert torch.equal(out_r[..., 8:], torch.zeros(2, 1, 3))


def test_clamp_and_pad_equal_lengths_leaves_reals_identical():
    """Equal shapes -> no padding at all; reals is returned as the same object."""
    fakes = torch.rand(3, 2, 16) * 0.5
    reals = torch.rand(3, 2, 16)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_r is reals
    assert torch.equal(out_f, fakes)  # already in range: clamp is a no-op


def test_clamp_and_pad_does_not_mutate_its_inputs():
    """Pure: the caller's tensors are unchanged (clamp/pad both return copies)."""
    fakes = torch.tensor([[[-3.0, 0.5, 9.0]]])
    reals = torch.zeros(1, 1, 5)
    fakes_before, reals_before = fakes.clone(), reals.clone()
    eval_FLAC.clamp_and_pad(fakes, reals)
    assert torch.equal(fakes, fakes_before)
    assert torch.equal(reals, reals_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clamp_and_pad_preserves_dtype_and_device(dtype):
    fakes = (torch.randn(2, 1, 5) * 3).to(dtype)
    reals = torch.randn(2, 1, 8).to(dtype)
    out_f, out_r = eval_FLAC.clamp_and_pad(fakes, reals)
    assert out_f.dtype == dtype and out_r.dtype == dtype
    assert out_f.device == fakes.device and out_r.device == reals.device


@pytest.mark.parametrize(
    "shape_f,shape_r",
    [
        ((2, 1, 8), (2, 1, 8)),    # equal
        ((2, 1, 5), (2, 1, 8)),    # fakes shorter -> pad fakes
        ((2, 1, 11), (2, 1, 8)),   # fakes longer  -> pad reals
        ((1, 1, 1), (1, 1, 4)),    # degenerate single sample
        ((3, 2, 4), (3, 2, 4)),    # multi-channel, equal
    ],
)
def test_clamp_and_pad_matches_pinned_block(shape_f, shape_r):
    """Bitwise equivalence with the verbatim 7bbd8aa block on random tensors
    (scaled x3 so clamping actually fires)."""
    torch.manual_seed(0)
    fakes = torch.randn(*shape_f) * 3.0
    reals = torch.randn(*shape_r)
    got_f, got_r = eval_FLAC.clamp_and_pad(fakes.clone(), reals.clone())
    exp_f, exp_r = _pinned_clamp_and_pad(fakes.clone(), reals.clone())
    assert torch.equal(got_f, exp_f)
    assert torch.equal(got_r, exp_r)


# --------------------------------------------------------------------------- #
# C2: loop-level proof that the STORED tensor is the SCORED tensor
#
# evaluate_model is driven end-to-end on CPU with every heavy piece stubbed at
# the eval_FLAC namespace (the pattern of test_eval_paths.py's wiring test),
# but -- unlike that test -- with a NON-empty dataloader, so the per-batch loop
# really runs. The sampler is patched on ``src.inference.sampling`` because the
# loop imports it inside the body.
# --------------------------------------------------------------------------- #
DOWNSAMPLING_RATIO = 8
SAMPLE_SIZE = 64
DECODED_LEN = 5      # shorter than REAL_LEN -> the pad branch fires
REAL_LEN = 8
DECODE_GAIN = 5.0    # pushes the decoder output well outside [-1, 1]


class _RecordingPretransform:
    """Stand-in VAE decoder: returns an out-of-range, too-short waveform and
    keeps a copy of every raw output (the pre-round-C artifact)."""

    downsampling_ratio = DOWNSAMPLING_RATIO

    def __init__(self):
        self.raw = []

    def decode(self, latents):
        out = latents[:, :1, :DECODED_LEN] * DECODE_GAIN
        self.raw.append(out.detach().clone())
        return out


class _FakeDiffusion:
    def __init__(self, pretransform, io_channels=4):
        self.model = types.SimpleNamespace(diffusion_objective="rectified_flow")
        self.pretransform = pretransform
        self.io_channels = io_channels
        self.dist_shift = None
        self.conditioner = lambda metadata, device: {}

    def get_conditioning_inputs(self, conditioning):
        return {}


class _FakeLoopModule:
    """PL-wrapper stand-in whose .diffusion drives the real per-batch loop."""

    def __init__(self, pretransform):
        self.diffusion = _FakeDiffusion(pretransform)
        self.device = "cpu"

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def to(self, device):
        return self


class _RecordingMetricCallback:
    """Records the exact tensor handed to update_metrics -- the scored input."""

    def __init__(self):
        self.scored = []

    def update_metrics(self, stage, pred, ref, scene=None, filename=None,
                       eval_type=None, depth=None):
        self.scored.append(pred.detach().clone())

    def compute_metrics(self, stage):
        return {"T60": 1.0}


def _md(i):
    g = torch.Generator().manual_seed(i)
    return {
        "scene": f"scene{i}",
        "depth": torch.randn(3, 4, 6, generator=g),
        "source": torch.randn(3, generator=g),
    }


def _fake_batches(sizes=(2, 3)):
    batches, idx = [], 0
    for n in sizes:
        g = torch.Generator().manual_seed(100 + n)
        reals = torch.randn(n, 1, REAL_LEN, generator=g)
        metadata = [_md(idx + j) for j in range(n)]
        batches.append((reals, metadata))
        idx += n
    return batches


def _run_loop(tmp_path, monkeypatch, store_predictions=True, eval_name="c2",
              batch_sizes=(2, 3), expect_ckpt_sha256=None):
    """Drive evaluate_model over ``batch_sizes`` fake batches; return
    (pretransform, metric_callback, output paths, n_items)."""
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": SAMPLE_SIZE,
        "sample_rate": 22050, "audio_channels": 1, "training": {"use_ema": False},
    }))
    dataset_cfg = tmp_path / "dataset.json"
    dataset_cfg.write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {}}, str(ckpt))

    pretransform = _RecordingPretransform()
    metric_callback = _RecordingMetricCallback()
    batches = _fake_batches(batch_sizes)

    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda cfg: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], [])),
    )
    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config",
        lambda cfg, model: _FakeLoopModule(pretransform),
    )
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: batches)
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config", lambda *a, **k: metric_callback
    )
    # The loop imports the sampler inside the body -> patch the module attribute.
    monkeypatch.setattr(
        sampling, "sample_discrete_euler",
        lambda model, x, steps=None, **kw: x,
    )

    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt),
        steps=1, cfg_scale=1.0, batch_size=2, device="cpu", eval_name=eval_name,
        seed=42, store_predictions=store_predictions, cond_autocast="off",
        expect_ckpt_sha256=expect_ckpt_sha256,
    )

    paths = eval_FLAC.build_output_paths(
        str(ckpt), steps=1, cfg_scale=1.0, eval_name=eval_name,
        cond_method="vanilla", rotate_deg=0.0,
    )
    return pretransform, metric_callback, paths, sum(batch_sizes)


def _load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_stored_predictions_are_bitwise_the_scored_tensor(tmp_path, monkeypatch):
    """The saved bundle == torch.cat of exactly what update_metrics received."""
    pretransform, metric_callback, paths, n_items = _run_loop(tmp_path, monkeypatch)

    bundle = _load_bundle(paths["predictions"])
    stored = bundle["predictions"]
    scored = torch.cat(metric_callback.scored, dim=0)

    assert len(metric_callback.scored) == 2, "the per-batch loop did not run twice"
    assert stored.shape == scored.shape == (n_items, 1, REAL_LEN)
    assert torch.equal(stored, scored)


def test_stored_predictions_differ_from_the_raw_decoder_output(tmp_path, monkeypatch):
    """Fixture sanity + regression pin: the raw decode really was out-of-range
    and too short, so storing it (the pre-round-C behaviour) is detectable."""
    pretransform, _, paths, n_items = _run_loop(tmp_path, monkeypatch, eval_name="c2raw")

    raw = torch.cat(pretransform.raw, dim=0)
    assert raw.shape == (n_items, 1, DECODED_LEN)
    assert raw.abs().max() > 1.0, "fixture no longer exercises the clamp"

    stored = _load_bundle(paths["predictions"])["predictions"]
    assert stored.shape[-1] == REAL_LEN                    # padded
    assert stored.min() >= -1.0 and stored.max() <= 1.0    # clamped
    assert torch.equal(stored[..., :DECODED_LEN], raw.clamp(-1.0, 1.0))
    assert torch.equal(stored[..., DECODED_LEN:], torch.zeros(n_items, 1, REAL_LEN - DECODED_LEN))


# --------------------------------------------------------------------------- #
# C3: bundle meta carries the run's provenance (plan §2 completion contract)
# --------------------------------------------------------------------------- #
CONTRACT = ("clamped/padded callback input (float32 cast and 8000-sample crop "
            "are scoring-internal)")


def _meta_kwargs(**over):
    kwargs = dict(
        seed=42, n_samples=7, cond_method="fa_invariant",
        frame_avg_angles=[0.0], rotate_deg=0.0, batch_size=32,
        cond_autocast="bf16", ckpt_path="ckpts/step=40000.ckpt",
        eval_name="dc_cyl_f025_K8_s42", steps=1, cfg_scale=1.0,
    )
    kwargs.update(over)
    return kwargs


def test_predictions_meta_carries_the_new_provenance_fields():
    meta = eval_FLAC.build_predictions_meta("ds.json", **_meta_kwargs())
    assert meta["ckpt_path"] == "ckpts/step=40000.ckpt"
    assert meta["eval_name"] == "dc_cyl_f025_K8_s42"
    assert meta["steps"] == 1
    assert meta["cfg_scale"] == 1.0
    assert meta["stored_after_clamp_pad"] is True
    assert meta["artifact_contract"] == CONTRACT


def test_predictions_meta_n_items_mirrors_n_samples():
    """``n_items`` is the announcement-08 name; ``n_samples`` stays for the
    exp_02 comparator. They must never disagree."""
    for n in (0, 1, 6337):
        meta = eval_FLAC.build_predictions_meta("ds.json", **_meta_kwargs(n_samples=n))
        assert meta["n_items"] == meta["n_samples"] == n


def test_predictions_meta_keeps_every_legacy_key():
    """The exp_02 comparator's guarded keys (and rotate_deg) survive unchanged."""
    meta = eval_FLAC.build_predictions_meta("ds.json", **_meta_kwargs())
    assert meta["dataset_config"] == "ds.json"
    assert meta["seed"] == 42
    assert meta["n_samples"] == 7
    assert meta["cond_method"] == "fa_invariant"
    assert meta["frame_avg_angles"] == [0.0]
    assert meta["rotate_deg"] == 0.0
    assert meta["batch_size"] == 32
    assert meta["cond_autocast"] == "bf16"


def test_predictions_meta_new_fields_are_optional():
    """Legacy call sites (positional, no provenance) still work: the new
    provenance fields default to None, the two constants are always present."""
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", 42, 7, "vanilla", None, 0.0, 32, "default",
    )
    assert meta["ckpt_path"] is None and meta["eval_name"] is None
    assert meta["steps"] is None and meta["cfg_scale"] is None
    assert meta["stored_after_clamp_pad"] is True
    assert meta["artifact_contract"] == CONTRACT


def test_saved_bundle_meta_matches_the_run(tmp_path, monkeypatch):
    """End-to-end: the meta in the file on disk describes the run that wrote it."""
    _, _, paths, n_items = _run_loop(tmp_path, monkeypatch, eval_name="c3meta")
    meta = _load_bundle(paths["predictions"])["meta"]

    assert meta["ckpt_path"] == str(tmp_path / "toy.ckpt")
    assert meta["eval_name"] == "c3meta"
    assert meta["steps"] == 1
    assert meta["cfg_scale"] == 1.0
    assert meta["n_items"] == meta["n_samples"] == n_items
    assert meta["stored_after_clamp_pad"] is True
    assert meta["artifact_contract"] == CONTRACT
    # protocol fields the completion contract checks per cell
    assert meta["dataset_config"] == str(tmp_path / "dataset.json")
    assert meta["seed"] == 42
    assert meta["cond_method"] == "vanilla"
    assert meta["frame_avg_angles"] is None
    assert meta["rotate_deg"] == 0.0
    assert meta["batch_size"] == 2
    assert meta["cond_autocast"] == "off"


def test_saved_bundle_n_items_counts_the_stored_rows(tmp_path, monkeypatch):
    """n_items is the real row count of the stored tensor, not a config echo."""
    _, _, paths, n_items = _run_loop(
        tmp_path, monkeypatch, eval_name="c3count", batch_sizes=(3, 1, 2)
    )
    bundle = _load_bundle(paths["predictions"])
    assert n_items == 6
    assert bundle["meta"]["n_items"] == bundle["predictions"].shape[0] == 6


# --------------------------------------------------------------------------- #
# G: the checkpoint digest travels with the artifact (codex full-r2 finding 1)
# --------------------------------------------------------------------------- #
def test_predictions_meta_carries_the_checkpoint_digest():
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", **_meta_kwargs(ckpt_sha256="cd" * 32))
    assert meta["ckpt_sha256"] == "cd" * 32


def test_predictions_meta_digest_defaults_to_none_for_legacy_callers():
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", 42, 7, "vanilla", None, 0.0, 32, "default",
    )
    assert meta["ckpt_sha256"] is None
    # ... and the legacy keys are still all there
    assert meta["dataset_config"] == "ds.json" and meta["stored_after_clamp_pad"] is True


def _ckpt_digest(tmp_path):
    return hashlib.sha256((tmp_path / "toy.ckpt").read_bytes()).hexdigest()


def test_both_artifacts_carry_the_digest_of_the_checkpoint_actually_loaded(
        tmp_path, monkeypatch):
    """No flag passed -- a legacy call site -- and the digest is embedded anyway,
    in the bundle meta AND in the metrics JSON the results table reads."""
    _, _, paths, _ = _run_loop(tmp_path, monkeypatch, eval_name="gdigest")
    digest = _ckpt_digest(tmp_path)

    assert _load_bundle(paths["predictions"])["meta"]["ckpt_sha256"] == digest
    record = json.loads(open(paths["metrics"]).read())
    assert record["ckpt_sha256"] == digest
    assert record["ckpt_path"] == str(tmp_path / "toy.ckpt")


def test_a_matching_pin_proceeds_and_embeds_the_same_digest(tmp_path, monkeypatch):
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {}}, str(ckpt))       # the digest _run_loop will pin
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    _, metric_callback, paths, n_items = _run_loop(
        tmp_path, monkeypatch, eval_name="gpinned", expect_ckpt_sha256=digest)

    assert len(metric_callback.scored) == 2         # the run really happened
    assert _load_bundle(paths["predictions"])["meta"]["ckpt_sha256"] == digest
    assert json.loads(open(paths["metrics"]).read())["ckpt_sha256"] == digest


def test_a_mismatched_pin_writes_no_artifact_at_all(tmp_path, monkeypatch):
    with pytest.raises(eval_FLAC.CheckpointDigestMismatch):
        _run_loop(tmp_path, monkeypatch, eval_name="gstale",
                  expect_ckpt_sha256="00" * 32)
    paths = eval_FLAC.build_output_paths(
        str(tmp_path / "toy.ckpt"), steps=1, cfg_scale=1.0, eval_name="gstale",
        cond_method="vanilla", rotate_deg=0.0,
    )
    assert not os.path.exists(paths["metrics"])
    assert not os.path.exists(paths["predictions"])


# --------------------------------------------------------------------------- #
# H: hash and load are ONE descriptor, end to end (codex full-r3 finding 1)
#
# Round G's window: the evaluator hashed pathname A, closed it, and reopened A to
# load. Between the two, A could be replaced by another contract-valid step-40000
# file B -- B was scored, A's digest was stamped, and restoring A before the gates
# ran left both of them satisfied. The two ways the file can change under us are
# tested separately, because they must end differently: a *rename* over the
# pathname cannot touch the inode an open descriptor already names (so the
# ORIGINAL loads and the run stands), while an in-place *mutation* of that inode
# is real and must stop the run before anything is scored or written.
# --------------------------------------------------------------------------- #
def _copy_bytes(src, dst):
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.read())


def test_a_pathname_swapped_between_hash_and_load_never_reaches_the_scorer(
        tmp_path, monkeypatch):
    ckpt = str(tmp_path / "toy.ckpt")            # _run_loop writes A here
    impostor = str(tmp_path / "impostor.ckpt")   # B: loadable, different bytes
    torch.save({"state_dict": {}, "marker": "IMPOSTOR"}, impostor)
    swapped, loaded = [], []
    real_load = torch.load

    def swapping_load(handle, *args, **kwargs):
        if hasattr(handle, "fileno") and not swapped:    # the evaluator's ckpt load
            swapped.append(True)
            backup, staged = str(tmp_path / "A.bytes"), str(tmp_path / "B.staged")
            _copy_bytes(ckpt, backup)
            _copy_bytes(impostor, staged)
            os.replace(staged, ckpt)             # B renamed over A's pathname
            obj = real_load(handle, *args, **kwargs)
            loaded.append(obj)
            os.replace(backup, ckpt)             # A restored before the gates look
            return obj
        return real_load(handle, *args, **kwargs)

    monkeypatch.setattr(eval_FLAC.torch, "load", swapping_load)
    _, metric_callback, paths, _ = _run_loop(tmp_path, monkeypatch, eval_name="hswap")

    assert swapped == [True], "the swap never happened: the test proves nothing"
    assert "marker" not in loaded[0], "the impostor's bytes were loaded and scored"
    assert len(metric_callback.scored) == 2                  # the run itself stands
    digest = _ckpt_digest(tmp_path)                          # A's bytes, restored
    assert json.loads(open(paths["metrics"]).read())["ckpt_sha256"] == digest
    assert _load_bundle(paths["predictions"])["meta"]["ckpt_sha256"] == digest


def test_an_in_place_mutation_during_the_load_writes_no_artifact_at_all(
        tmp_path, monkeypatch):
    ckpt = str(tmp_path / "toy.ckpt")
    real_load = torch.load

    def mutating_load(handle, *args, **kwargs):
        obj = real_load(handle, *args, **kwargs)
        if hasattr(handle, "fileno"):
            with open(ckpt, "r+b") as other:     # same inode, a different handle
                other.seek(0, os.SEEK_END)
                other.write(b"mutated in place under the evaluator")
        return obj

    monkeypatch.setattr(eval_FLAC.torch, "load", mutating_load)
    with pytest.raises(eval_FLAC.CheckpointMutatedDuringLoad):
        _run_loop(tmp_path, monkeypatch, eval_name="hmutate")

    paths = eval_FLAC.build_output_paths(
        ckpt, steps=1, cfg_scale=1.0, eval_name="hmutate",
        cond_method="vanilla", rotate_deg=0.0,
    )
    assert not os.path.exists(paths["metrics"])
    assert not os.path.exists(paths["predictions"])
