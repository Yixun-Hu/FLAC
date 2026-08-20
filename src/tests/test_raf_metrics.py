"""RAF policy in the metric stack (exp_19, contract section F, TDD cycle 12).

Covers ``AcousticMetricsCallback`` and ``RT60Error`` for ``dataset_name="RAF"``,
the fail-closed FD/retrieval guard (no AGREE model was ever trained on RAF), and
regression tests pinning that the AcousticRooms and HAA paths are untouched.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.metrics.metric_callback import SUPPORTED_DATASETS, AcousticMetricsCallback
from src.metrics.modules.RT60 import RT60Error

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_MODEL_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC",
                                 "RAF", "FLAC_RAF_finetune.json")

# The default flag set omits eval_l1_distance_multires so most tests stay cheap;
# R10 fixed its CUDA-only window, and the RAF-enabled set is exercised in full by
# test_multires_l1_runs_on_cpu_for_the_enabled_raf_metric_set.
_METRIC_FLAGS = dict(eval_T60=True, eval_C50=True, eval_EDT=True, eval_env=True,
                     eval_l1_distance=True)


def _decaying_rir(seed, n=10240, sr=22050, tau=0.08):
    """A synthetic RIR with a real exponential decay, so T60/EDT are measurable."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / sr
    return (torch.randn(n, generator=g) * torch.exp(-t / tau)).view(1, 1, n) * 0.5


def _callback(dataset_name="RAF", **kwargs):
    flags = dict(_METRIC_FLAGS)
    flags.update(kwargs)
    return AcousticMetricsCallback(dataset_name=dataset_name, device="cpu", **flags)


# --------------------------------------------------------------------------- #
# RAF support
# --------------------------------------------------------------------------- #
def test_raf_is_a_supported_dataset():
    assert "RAF" in SUPPORTED_DATASETS


def test_raf_uses_the_haa_metric_window():
    cb = _callback("RAF")
    assert cb.max_len == 9600
    assert cb.max_len_magenv == 9600


def test_raf_accumulates_per_scene_without_an_explicit_flag():
    """Two rooms of unequal size: RAF headline numbers are per-scene means."""
    cb = _callback("RAF")
    assert cb.eval_by_scene is True


def test_raf_update_compute_and_by_scene():
    cb = _callback("RAF")
    pred = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    ref = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    cb.update_metrics("test", pred, ref, scene=["EmptyRoom", "FurnishedRoom"])
    metrics = cb.compute_metrics("test")
    for key in ("T60", "Invalid T60", "C50", "EDT", "Env", "L1_STFT"):
        assert key in metrics, key
        assert np.isfinite(float(metrics[key] if not isinstance(metrics[key], tuple)
                                 else metrics[key][0]))
    assert set(metrics["by_scene"]) == {"EmptyRoom", "FurnishedRoom"}
    for scene in metrics["by_scene"].values():
        for key in ("T60", "C50", "EDT", "Env", "L1_STFT"):
            assert key in scene, key
    assert "FD" not in metrics and "R@1" not in metrics


def test_raf_registers_the_multires_metric():
    """The RAF config asks for it, and since R10 its update path runs anywhere."""
    cb = _callback("RAF", eval_l1_distance_multires=True)
    assert "test" in cb.l1_stft_multires
    assert "EmptyRoom" not in cb.scene_metrics["test"]   # created lazily, per scene


def test_raf_compute_resets_the_accumulators():
    cb = _callback("RAF")
    pred, ref = _decaying_rir(1), _decaying_rir(2)
    preds = torch.cat([pred, pred], dim=0)
    refs = torch.cat([ref, ref], dim=0)
    cb.update_metrics("test", preds, refs, scene=["EmptyRoom", "FurnishedRoom"])
    first = cb.compute_metrics("test")
    cb.update_metrics("test", preds, preds, scene=["EmptyRoom", "FurnishedRoom"])
    second = cb.compute_metrics("test")
    assert second["C50"] != first["C50"]      # not contaminated by the first batch
    assert abs(float(second["C50"])) < 1e-6   # pred == ref


def test_raf_rejects_a_wrong_sample_rate_or_channel_count():
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", sample_rate=44100, device="cpu")
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", audio_channels=2, device="cpu")


# --------------------------------------------------------------------------- #
# FD / retrieval guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["eval_FD", "eval_retrieval"])
def test_raf_refuses_agree_based_metrics(flag):
    """No AGREE model exists for RAF: FD/Recall are unavailable, never zero."""
    with pytest.raises(ValueError) as exc:
        AcousticMetricsCallback(dataset_name="RAF", device="cpu",
                                **{flag: True, "AGREE_ckpt": "weights/AGREE_HAA.pt"})
    assert "RAF" in str(exc.value)


@pytest.mark.parametrize("flag", ["eval_FD", "eval_retrieval"])
def test_raf_guard_fires_even_without_a_checkpoint(flag):
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", device="cpu", **{flag: True})


def test_unsupported_dataset_still_raises():
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="Matterport", device="cpu")


# --------------------------------------------------------------------------- #
# RT60 policy
# --------------------------------------------------------------------------- #
def test_rt60_raf_mirrors_the_haa_decay_window():
    """RAF RIRs are real, truncated measurements like HAA's, so RAF adopts HAA's
    T30 window (decay_db=30) and its invalid-sample masking rather than AR's T20
    on clean simulated RIRs."""
    assert RT60Error(dataset_name="RAF").decay_db == 30
    assert RT60Error(dataset_name="HAA").decay_db == 30
    assert RT60Error(dataset_name="AcousticRooms").decay_db == 20


def test_rt60_unknown_dataset_still_raises():
    with pytest.raises(NotImplementedError):
        RT60Error(dataset_name="Matterport")


def test_rt60_raf_counts_invalid_samples_like_haa():
    silent = torch.zeros(1, 1, 9600)
    raf = RT60Error(dataset_name="RAF")
    raf.update(silent, silent)
    error, invalid = raf.compute()
    assert invalid == 1.0
    assert error == 100.0     # invalid samples are charged the maximum error

    ar = RT60Error(dataset_name="AcousticRooms")
    ar.update(silent, silent)
    assert ar.compute()[1] == 0.0   # AR path records no invalids (unchanged)


# --------------------------------------------------------------------------- #
# AR / HAA regressions
# --------------------------------------------------------------------------- #
def test_acousticrooms_behaviour_is_unchanged():
    cb = _callback("AcousticRooms")
    assert cb.max_len == 8000
    assert cb.max_len_magenv == 9600
    assert cb.eval_by_scene is False


def test_acousticrooms_per_scene_flag_still_works():
    assert _callback("AcousticRooms", eval_per_scene=True).eval_by_scene is True


def test_haa_behaviour_is_unchanged():
    cb = _callback("HAA")
    assert cb.max_len == 9600
    assert cb.eval_by_scene is True


def test_haa_still_allows_agree_metrics_when_a_checkpoint_is_given():
    """The RAF guard must not leak into HAA/AR: only the missing-checkpoint rule
    applies there."""
    with pytest.raises(ValueError) as exc:
        AcousticMetricsCallback(dataset_name="HAA", device="cpu", eval_FD=True)
    assert "AGREE_ckpt" in str(exc.value)


def test_haa_dampened_scene_is_still_excluded_from_t60():
    cb = _callback("HAA")
    pred, ref = _decaying_rir(1), _decaying_rir(2)
    cb.update_metrics("test", pred, ref, scene=["dampenedBase"])  # noqa: E501
    assert len(cb.RT60["test"].t60_error) == 0


# --------------------------------------------------------------------------- #
# config integration
# --------------------------------------------------------------------------- #
def test_callback_builds_from_the_raf_model_config():
    with open(_RAF_MODEL_CONFIG) as f:
        metrics_config = json.load(f)["training"]["metrics"]
    cb = AcousticMetricsCallback(
        dataset_name=metrics_config["dataset_name"],
        device="cpu",
        eval_T60=metrics_config.get("eval_T60", False),
        eval_C50=metrics_config.get("eval_C50", False),
        eval_EDT=metrics_config.get("eval_EDT", False),
        eval_l1_distance=metrics_config.get("eval_l1_distance", False),
        eval_l1_distance_multires=metrics_config.get("eval_l1_distance_multires", False),
        eval_FD=metrics_config.get("eval_FD", False),
        eval_retrieval=metrics_config.get("eval_retrieval", False),
        eval_env=metrics_config.get("eval_env", False),
        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
    )
    assert cb.dataset_name == "RAF"
    assert cb.max_len == 9600
    assert cb.eval_FD is False and cb.eval_retrieval is False
    assert cb.AGREE_model is None


# --------------------------------------------------------------------------- #
# r2 R8 + R10 (phase 2b: these land with the src/metrics change)
# --------------------------------------------------------------------------- #
def _two_room_batch():
    """Room A gets 3 items, room B gets 1: unequal sizes make the two
    aggregations differ, which is exactly what R8 is about."""
    preds = torch.cat([_decaying_rir(i) for i in (1, 2, 3)] + [_decaying_rir(9)], dim=0)
    refs = torch.cat([_decaying_rir(i + 100) for i in (1, 2, 3)]
                     + [_decaying_rir(109)], dim=0)
    scenes = ["EmptyRoom", "EmptyRoom", "EmptyRoom", "FurnishedRoom"]
    return preds, refs, scenes


def test_raf_top_level_metrics_are_the_equal_room_macro_mean():
    cb = _callback("RAF")
    preds, refs, scenes = _two_room_batch()
    cb.update_metrics("test", preds, refs, scene=scenes)
    metrics = cb.compute_metrics("test")
    assert metrics["aggregation"] == "macro_room"
    assert metrics["n_rooms"] == 2
    for key in ("C50", "EDT", "Env", "L1_STFT"):
        per_room = [metrics["by_scene"][s][key] for s in ("EmptyRoom", "FurnishedRoom")]
        assert metrics[key] == pytest.approx(sum(per_room) / 2.0, rel=1e-6), key


def test_raf_macro_mean_differs_from_the_per_item_mean():
    """With 3 items in one room and 1 in the other, the per-item mean weights the
    larger room 3:1; the paper's protocol weights rooms equally."""
    cb = _callback("RAF")
    preds, refs, scenes = _two_room_batch()
    cb.update_metrics("test", preds, refs, scene=scenes)
    macro = cb.compute_metrics("test")

    flat = _callback("AcousticRooms")          # per-item aggregation, same inputs
    flat.update_metrics("test", preds, refs, scene=scenes)
    per_item = flat.compute_metrics("test")
    assert macro["C50"] != pytest.approx(per_item["C50"], rel=1e-9)


def test_raf_reports_invalid_t60_as_both_count_and_rate():
    cb = _callback("RAF")
    good, silent = _decaying_rir(1), torch.zeros(1, 1, 10240)
    preds = torch.cat([good, silent, silent], dim=0)
    refs = torch.cat([_decaying_rir(2), silent, silent], dim=0)
    cb.update_metrics("test", preds, refs,
                      scene=["EmptyRoom", "EmptyRoom", "FurnishedRoom"])
    metrics = cb.compute_metrics("test")
    assert metrics["Invalid T60 count"] == 2.0
    assert metrics["T60 items"] == 3
    assert metrics["Invalid T60 rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["Invalid T60"] == pytest.approx(metrics["Invalid T60 rate"])


def test_rt60_invalid_stats_are_additive_to_the_existing_contract():
    metric = RT60Error(dataset_name="RAF")
    metric.update(torch.zeros(1, 1, 9600), torch.zeros(1, 1, 9600))
    metric.update(_decaying_rir(1), _decaying_rir(2))
    error, rate = metric.compute()          # unchanged 2-tuple
    count, rate2, n = metric.invalid_stats()
    assert n == 2 and count == 1.0
    assert rate2 == pytest.approx(0.5) and rate == pytest.approx(rate2)
    assert isinstance(error, float)


def test_acousticrooms_and_haa_aggregation_is_untouched():
    for name in ("AcousticRooms", "HAA"):
        cb = _callback(name)
        preds, refs, scenes = _two_room_batch()
        cb.update_metrics("test", preds, refs, scene=scenes)
        metrics = cb.compute_metrics("test")
        assert "aggregation" not in metrics
        assert "Invalid T60 count" not in metrics
        assert "n_rooms" not in metrics


def test_multires_l1_runs_on_cpu_for_the_enabled_raf_metric_set():
    """R10: the Hann window follows the signal's device, so the exact metric set
    the RAF config enables is exercisable on CPU (and on any GPU index)."""
    cb = AcousticMetricsCallback(dataset_name="RAF", device="cpu", eval_T60=True,
                                 eval_C50=True, eval_EDT=True, eval_env=True,
                                 eval_l1_distance=True, eval_l1_distance_multires=True)
    preds, refs, scenes = _two_room_batch()
    cb.update_metrics("test", preds, refs, scene=scenes)
    metrics = cb.compute_metrics("test")
    assert np.isfinite(metrics["L1_STFT_MultiRes"])
    assert metrics["L1_STFT_MultiRes"] == pytest.approx(
        sum(metrics["by_scene"][s]["L1_STFT_MultiRes"]
            for s in ("EmptyRoom", "FurnishedRoom")) / 2.0, rel=1e-6)


def test_multires_l1_window_follows_the_input_dtype_and_device():
    from src.metrics.modules.l1_stft_multires import get_stft

    x = torch.randn(4096)
    assert torch.isfinite(get_stft(x, n_fft=64)).all()
    assert torch.isfinite(get_stft(x.double(), n_fft=64)).all()


# --------------------------------------------------------------------------- #
# r3 S4: per-item STFT attribution + fail-closed RAF scene attribution
# --------------------------------------------------------------------------- #
def _single_room_reference(dataset_name, items):
    """Global metrics for ONE room, via a callback that has no RAF macro path.

    Deliberately runs under HAA (same 9600 window, same metric objects, no
    equal-room macro and no RAF room-set constraint), so the value it returns is
    produced by the global accumulators alone and cannot be the macro number the
    caller is checking. With a single update the legacy and per-item L1 weightings
    coincide, so this is also a valid per-item reference.
    """
    dataset_name = "HAA" if dataset_name == "RAF" else dataset_name
    cb = _callback(dataset_name, eval_l1_distance_multires=True)
    preds = torch.cat([p for p, _ in items], dim=0)
    refs = torch.cat([r for _, r in items], dim=0)
    cb.update_metrics("test", preds, refs, scene=["OneRoom"] * len(items))
    return cb.compute_metrics("test")


_MACRO_KEYS = ("T60", "C50", "EDT", "Env", "L1_STFT", "L1_STFT_MultiRes")


def _unequal_rooms():
    """3 items in EmptyRoom, 1 in FurnishedRoom, with distinct signals."""
    a = [(_decaying_rir(i), _decaying_rir(i + 100)) for i in (1, 2, 3)]
    b = [(_decaying_rir(9, tau=0.16), _decaying_rir(109, tau=0.16))]
    return a, b


def test_per_scene_l1_stft_uses_only_that_scene_items():
    """S4: the batch STFT was appended to every item's scene accumulator, so each
    room's L1_STFT silently contained the other room's items."""
    a, b = _unequal_rooms()
    cb = _callback("RAF", eval_l1_distance_multires=True)
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs,
                      scene=["EmptyRoom"] * 3 + ["FurnishedRoom"])
    metrics = cb.compute_metrics("test")

    ref_a = _single_room_reference("RAF", a)
    ref_b = _single_room_reference("RAF", b)
    assert metrics["by_scene"]["EmptyRoom"]["L1_STFT"] == pytest.approx(
        ref_a["L1_STFT"], rel=1e-5)
    assert metrics["by_scene"]["FurnishedRoom"]["L1_STFT"] == pytest.approx(
        ref_b["L1_STFT"], rel=1e-5)
    # the two rooms really do differ, so the assertion above has teeth
    assert ref_a["L1_STFT"] != pytest.approx(ref_b["L1_STFT"], rel=1e-3)


def test_every_macro_metric_matches_an_independent_unequal_room_oracle():
    """S4: each macro value is the mean of two SEPARATELY computed room values."""
    a, b = _unequal_rooms()
    cb = _callback("RAF", eval_l1_distance_multires=True)
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs,
                      scene=["EmptyRoom"] * 3 + ["FurnishedRoom"])
    metrics = cb.compute_metrics("test")

    ref_a = _single_room_reference("RAF", a)
    ref_b = _single_room_reference("RAF", b)
    for key in _MACRO_KEYS:
        oracle = (ref_a[key] + ref_b[key]) / 2.0
        assert metrics[key] == pytest.approx(oracle, rel=1e-5), key
        # and the macro mean is NOT the per-item mean, since the rooms are unequal
        per_item = (3 * ref_a[key] + ref_b[key]) / 4.0
        if abs(ref_a[key] - ref_b[key]) > 1e-6:
            assert metrics[key] != pytest.approx(per_item, rel=1e-9), key


def test_global_l1_stft_is_the_per_item_mean():
    a, b = _unequal_rooms()
    cb = _callback("AcousticRooms", eval_l1_distance_multires=True)
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs, scene=["A", "A", "A", "B"])
    metrics = cb.compute_metrics("test")
    ref_a = _single_room_reference("AcousticRooms", a)
    ref_b = _single_room_reference("AcousticRooms", b)
    assert metrics["L1_STFT"] == pytest.approx(
        (3 * ref_a["L1_STFT"] + ref_b["L1_STFT"]) / 4.0, rel=1e-5)


def test_raf_requires_scene_attribution():
    """S4: a RAF call without scenes used to fall back to per-item metrics and
    silently drop the invalid count -- i.e. report a different estimand."""
    cb = _callback("RAF")
    pred, ref = _decaying_rir(1), _decaying_rir(2)
    with pytest.raises(ValueError) as exc:
        cb.update_metrics("test", pred, ref)
    assert "scene" in str(exc.value).lower()


@pytest.mark.parametrize("scene", [
    ["EmptyRoom"],                        # too short for a 2-item batch
    ["EmptyRoom", "EmptyRoom", "x"],      # too long
    ["EmptyRoom", None],                  # incomplete
    ["EmptyRoom", ""],                    # empty label
    "EmptyRoom",                          # a bare string is not per-item attribution
])
def test_raf_rejects_incomplete_scene_attribution(scene):
    cb = _callback("RAF")
    preds = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    refs = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    with pytest.raises(ValueError):
        cb.update_metrics("test", preds, refs, scene=scene)


def test_non_raf_datasets_still_accept_a_missing_scene():
    """AR runs without per-scene attribution; that path must not move."""
    cb = _callback("AcousticRooms")
    cb.update_metrics("test", _decaying_rir(1), _decaying_rir(2))
    assert np.isfinite(cb.compute_metrics("test")["C50"])


# --------------------------------------------------------------------------- #
# r3 S7: the registered autocast behaviour of the multires-l1 window
# --------------------------------------------------------------------------- #
def test_multires_window_under_cpu_bf16_autocast():
    """R10's contract is 'the window follows the signal'; evaluation runs under
    bf16 autocast, so that is where it has to hold."""
    from src.metrics.modules.l1_stft_multires import get_stft

    x = torch.randn(4096)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = get_stft(x, n_fft=64)
    assert torch.isfinite(out).all()
    assert out.device == x.device


def test_multires_metric_update_under_cpu_bf16_autocast():
    cb = _callback("RAF", eval_l1_distance_multires=True)
    a, b = _unequal_rooms()
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cb.update_metrics("test", preds, refs,
                          scene=["EmptyRoom"] * 3 + ["FurnishedRoom"])
    metrics = cb.compute_metrics("test")
    assert np.isfinite(metrics["L1_STFT_MultiRes"])
    assert np.isfinite(metrics["L1_STFT"])


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("RAF_TEST_CUDA") != "1",
    reason="CUDA case is opt-in (RAF_TEST_CUDA=1): this box's GPUs carry other "
           "sessions' runs, so the suite must not allocate a CUDA context by default")
def test_multires_window_on_a_non_default_cuda_device():
    """--device cuda:1 with both GPUs visible was the second half of the R10 bug."""
    from src.metrics.modules.l1_stft_multires import get_stft

    device = f"cuda:{torch.cuda.device_count() - 1}"
    x = torch.randn(4096, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = get_stft(x, n_fft=64)
    assert out.device.type == "cuda"
    assert out.device.index == torch.device(device).index


# --------------------------------------------------------------------------- #
# r4 T8: L1_STFT weighting — legacy for AR/HAA, corrected for RAF
# --------------------------------------------------------------------------- #
def _independent_log_mag_stft(wave):
    """Re-implementation of metric_callback.stft from its documented constants.

    Deliberately NOT the callback's own module: an oracle that imports the code
    under test cannot witness a change in that code.
    """
    window = torch.hann_window(62)
    spec = torch.stft(wave, 124, 31, 62, window, return_complex=False,
                      pad_mode='constant')
    mag = torch.sqrt(torch.clamp(spec[..., 0] ** 2 + spec[..., 1] ** 2, min=1e-7))
    return torch.log(mag + 1e-8)


def _independent_l1_items(preds, refs, max_len_magenv=9600):
    """Per-item L1_STFT values, exactly as L1_STFT.update computes them."""
    p = _independent_log_mag_stft(preds.squeeze(1)[..., :max_len_magenv])
    r = _independent_log_mag_stft(refs.squeeze(1)[..., :max_len_magenv])
    return torch.mean((p - r) ** 2, dim=(1, 2))


def _unequal_update_batches():
    """Two updates of size 3 and 1 — the shape of a drop_last=False eval epoch."""
    first = ([_decaying_rir(i) for i in (1, 2, 3)],
             [_decaying_rir(i + 100) for i in (1, 2, 3)])
    second = ([_decaying_rir(9, tau=0.16)], [_decaying_rir(109, tau=0.16)])
    return first, second


def _legacy_global_l1(batches):
    """Pre-S4 arithmetic: each item's update appended the WHOLE batch's vector, so
    a batch of size B contributed B*B entries and every item in it was weighted B
    times relative to an item from a size-1 batch."""
    entries = []
    for preds, refs in batches:
        items = _independent_l1_items(torch.cat(preds, dim=0), torch.cat(refs, dim=0))
        entries.extend(items.tolist() * len(preds))
    return round(float(np.mean(entries)), 4)


def _corrected_global_l1(batches):
    """Per-item arithmetic: one entry per item, every item weighted once."""
    entries = []
    for preds, refs in batches:
        items = _independent_l1_items(torch.cat(preds, dim=0), torch.cat(refs, dim=0))
        entries.extend(items.tolist())
    return round(float(np.mean(entries)), 4)


def test_the_two_weightings_actually_differ_on_unequal_batches():
    """If they agreed there would be nothing to preserve; with 3+1 they do not."""
    batches = _unequal_update_batches()
    assert _legacy_global_l1(batches) != _corrected_global_l1(batches)


@pytest.mark.parametrize("dataset_name", ["AcousticRooms", "HAA"])
def test_ar_and_haa_global_l1_keep_the_legacy_weighting(dataset_name):
    """T8: AR/HAA numbers are a published record. The S4 attribution fix must not
    silently re-weight them, so their global L1_STFT stays bug-compatible."""
    batches = _unequal_update_batches()
    cb = _callback(dataset_name)
    for preds, refs in batches:
        cb.update_metrics("test", torch.cat(preds, dim=0), torch.cat(refs, dim=0),
                          scene=["A"] * len(preds))
    assert cb.compute_metrics("test")["L1_STFT"] == _legacy_global_l1(batches)


def test_raf_reports_the_macro_mean_over_per_item_room_values():
    """RAF is a new metric with no record to preserve, so its accumulator uses the
    corrected per-item weighting -- and what it REPORTS is the equal-room macro
    mean over per-item room values, which the macro path supersedes the global
    accumulator with. Golden derived outside the callback."""
    batches = _unequal_update_batches()
    rooms = (["EmptyRoom"] * 3, ["FurnishedRoom"])   # both rooms, unequal batches
    cb = _callback("RAF")
    for (preds, refs), scene in zip(batches, rooms):
        cb.update_metrics("test", torch.cat(preds, dim=0), torch.cat(refs, dim=0),
                          scene=scene)
    metrics = cb.compute_metrics("test")

    room_values = []
    for preds, refs in batches:
        items = _independent_l1_items(torch.cat(preds, dim=0), torch.cat(refs, dim=0))
        room_values.append(float(items.mean()))
    assert metrics["by_scene"]["EmptyRoom"]["L1_STFT"] == pytest.approx(
        room_values[0], rel=1e-4)
    assert metrics["by_scene"]["FurnishedRoom"]["L1_STFT"] == pytest.approx(
        room_values[1], rel=1e-4)
    assert metrics["L1_STFT"] == pytest.approx(sum(room_values) / 2.0, rel=1e-4)
    # neither the legacy nor the corrected GLOBAL number is what RAF reports
    assert metrics["L1_STFT"] != pytest.approx(_legacy_global_l1(batches), rel=1e-9)
    assert metrics["L1_STFT"] != pytest.approx(_corrected_global_l1(batches), rel=1e-9)


def test_equal_batches_are_unaffected_by_the_weighting_choice():
    """Why the r3 single-update regression could not see this."""
    equal = (([_decaying_rir(1)], [_decaying_rir(101)]),
             ([_decaying_rir(2)], [_decaying_rir(102)]))
    assert _legacy_global_l1(equal) == _corrected_global_l1(equal)


def test_macro_metrics_match_hand_derived_goldens_outside_the_callback():
    """T8: the r3 oracle re-ran the same callback. These goldens are computed from
    the raw signals only."""
    a, b = _unequal_rooms()
    cb = _callback("RAF")
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs,
                      scene=["EmptyRoom"] * 3 + ["FurnishedRoom"])
    metrics = cb.compute_metrics("test")

    items = _independent_l1_items(preds, refs)
    room_a = float(items[:3].mean())
    room_b = float(items[3:].mean())
    assert metrics["by_scene"]["EmptyRoom"]["L1_STFT"] == pytest.approx(room_a, rel=1e-4)
    assert metrics["by_scene"]["FurnishedRoom"]["L1_STFT"] == pytest.approx(room_b, rel=1e-4)
    assert metrics["L1_STFT"] == pytest.approx((room_a + room_b) / 2.0, rel=1e-4)
    # ... and that is NOT the per-item mean over the four items
    assert metrics["L1_STFT"] != pytest.approx(float(items.mean()), rel=1e-9)


def test_per_scene_l1_is_per_item_for_every_dataset():
    """The S4 attribution fix stands for HAA too: a room's value may never contain
    another room's items, whatever the global weighting does."""
    a, b = _unequal_rooms()
    cb = _callback("HAA")
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs, scene=["roomA"] * 3 + ["roomB"])
    metrics = cb.compute_metrics("test")
    items = _independent_l1_items(preds, refs)
    assert metrics["by_scene"]["roomA"]["L1_STFT"] == pytest.approx(
        float(items[:3].mean()), rel=1e-4)
    assert metrics["by_scene"]["roomB"]["L1_STFT"] == pytest.approx(
        float(items[3:].mean()), rel=1e-4)


# --------------------------------------------------------------------------- #
# r4 T6: scene labels are normalised, validated and completeness-checked
# --------------------------------------------------------------------------- #
def test_tuple_scene_labels_do_not_collapse_into_one_pseudo_room():
    """T6: a tuple passed the length gate but attribution only indexed lists, so
    every item was filed under one tuple-valued key and the macro became a mean
    over a single pseudo-room."""
    a, b = _unequal_rooms()
    cb = _callback("RAF")
    preds = torch.cat([p for p, _ in a + b], dim=0)
    refs = torch.cat([r for _, r in a + b], dim=0)
    cb.update_metrics("test", preds, refs,
                      scene=("EmptyRoom", "EmptyRoom", "EmptyRoom", "FurnishedRoom"))
    metrics = cb.compute_metrics("test")
    assert set(metrics["by_scene"]) == {"EmptyRoom", "FurnishedRoom"}
    assert metrics["n_rooms"] == 2


@pytest.mark.parametrize("labels", [
    ["EmptyRoom", "Kitchen"],
    ["EmptyRoom", "emptyroom"],
    ["EmptyRoom", "classroomBase"],
])
def test_raf_rejects_labels_outside_the_registered_room_set(labels):
    cb = _callback("RAF")
    preds = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    refs = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    with pytest.raises(ValueError) as exc:
        cb.update_metrics("test", preds, refs, scene=labels)
    assert "EmptyRoom" in str(exc.value) and "FurnishedRoom" in str(exc.value)


def test_raf_macro_requires_both_rooms():
    """A run that only ever saw one room cannot report an equal-room macro mean."""
    cb = _callback("RAF")
    cb.update_metrics("test", _decaying_rir(1), _decaying_rir(2), scene=["EmptyRoom"])
    with pytest.raises(ValueError) as exc:
        cb.compute_metrics("test")
    assert "FurnishedRoom" in str(exc.value)


def test_raf_macro_over_both_rooms_still_computes():
    cb = _callback("RAF")
    preds = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    refs = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    cb.update_metrics("test", preds, refs, scene=["EmptyRoom", "FurnishedRoom"])
    assert cb.compute_metrics("test")["n_rooms"] == 2


def test_non_raf_datasets_accept_any_scene_label():
    """HAA's four base rooms are not the RAF room set; that path must not move."""
    cb = _callback("HAA")
    preds = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    refs = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    cb.update_metrics("test", preds, refs, scene=("classroomBase", "hallwayBase"))
    assert set(cb.compute_metrics("test")["by_scene"]) == {"classroomBase", "hallwayBase"}
