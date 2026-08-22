"""exp_21 (bf_fa_cartesian) — the EVALUATION side: ``eval_FLAC.py`` + ``eval_pl.py``.

Rounds 1-2 made ``fa_cartesian`` trainable. ``eval_FLAC.py`` still rejects it, so
the arm cannot be measured; this round makes it a first-class ``cond_method``
there and closes the ``eval_pl.py`` provenance hole the r2 review flagged
(nit 3). The contracts pinned here are plan §3d, §3h item 10 and §5.

RED first. Before the implementation ``eval_FLAC.evaluate_model`` raises
``Unknown cond_method`` for ``fa_cartesian``, ``orbit_provenance`` returns the
NO-ORBIT pair ``("n/a", None)`` for it, argparse rejects the choice, and
``eval_pl`` has no guard at all.

**Not every test here is red, and the passing ones are the point of half the
round** (r2 review nit 2 convention — a red-phase description must match the
recorded evidence). Three groups are DELIBERATE NO-CHANGE PINS, marked
``NO-CHANGE PIN`` in their docstrings:

* the output-filename rule. ``build_output_paths`` already inserts
  ``_<cond_method>_a<n>`` for every non-vanilla method, so ``_fa_cartesian_a4``
  falls out unmodified. The plan says verify, do not fork (§3d); these tests are
  what makes a later fork fail here.
* the ``fa_invariant`` / ``vanilla`` records. Widening an equality test to a
  membership test can silently change the OTHER branch, and every row already in
  ``model_comparison.md`` was written by these two paths. The golden below is a
  literal capture of the pre-change record, key ORDER included.
* the eval chunk plan (§3h item 10). ``fa_cartesian_conditioning`` and the orbit
  executor already implement it; what this round adds is that eval REACHES them,
  so the batch-64/cap-64 vs batch-64/cap-32 split is pinned against the function
  the eval path will call.

The conditioner fixtures come from ``test_fa_cartesian`` — the sibling module
that owns them (pytest prepends ``src/tests``; same import route as
``test_finetune_cond`` -> ``test_cond_dispatch``). ``test_eval_paths.py`` is NOT
imported and NOT edited: it is exp_03/exp_11/exp_14's audited record of the
vanilla + fa_invariant eval contract and part of this round's regression set, so
leaving it byte-unchanged is what makes its green run evidence that this round
disturbed neither method.
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import eval_FLAC
import eval_pl
from src.data import yaw_rotation as yr
from test_fa_cartesian import _batch, _build_cond, C4, POSE_KEYS  # noqa: E402

REPO_ROOT = str(Path(__file__).resolve().parents[2])
CKPT = "weights/FLAC/FLAC_EMA.ckpt"
ANGLES = [0.0, 90.0, 180.0, 270.0]

# The registered EVAL cap (plan §2 chunk plan / §5 template): batch 64 needs a
# cap of at least 64, because an orbit chunk is whole angles and cannot be split
# below one. Eval mode draws no RoPE rescale, so the cap is protocol identity,
# not numerics — which is itself asserted in test_eval_cap_is_a_plan_knob_only.
EVAL_CAP = 64


@pytest.fixture(autouse=True)
def single_thread():
    """One intra-op thread (restored afterwards): the batch-64 orbit below runs
    256 tiny conditioner rows, where torch's default pool spends its time
    launching threads."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# --------------------------------------------------------------------------- #
# 1. filename suffix — NO-CHANGE PIN (plan §3d: verify, do not fork)
# --------------------------------------------------------------------------- #
def test_fa_cartesian_suffix_falls_out_of_the_existing_rule():
    """NO-CHANGE PIN. ``_fa_cartesian_a4`` must come from the generic
    non-vanilla rule, not from a method-specific branch: a fork would be a second
    place where an arm's artifacts get their identity, and the exp_02 bug (two
    runs sharing one predictions file) came from exactly that kind of split."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="exp21_BFC_S40000_K8_s42",
        cond_method="fa_cartesian", rotate_deg=0.0, n_angles=4,
    )
    assert paths["metrics"] == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp21_BFC_S40000_K8_s42_fa_cartesian_a4.json"
    )
    assert paths["predictions"] == (
        "weights/FLAC/FLAC_EMA_predictions_1_1.0_exp21_BFC_S40000_K8_s42_fa_cartesian_a4.pt"
    )


def test_fa_cartesian_suffix_composes_with_rotation_and_angle_count():
    """NO-CHANGE PIN. The invariance grid (§5) writes five cells of ONE
    checkpoint that differ only by ``--rotate-deg``; the method tag must precede
    the rot suffix and the angle count must be the real one, or the 45-degree
    negative control overwrites the 0-degree registered cell."""
    paths = eval_FLAC.build_output_paths(
        CKPT, steps=1, cfg_scale=1.0, eval_name="e",
        cond_method="fa_cartesian", rotate_deg=45.0, n_angles=2,
    )
    assert paths["metrics"].endswith("_e_fa_cartesian_a2_rot45.json")
    assert paths["metrics"].index("_fa_cartesian_a2") < paths["metrics"].index("_rot45")


def test_the_suffix_rule_is_shared_not_per_method():
    """NO-CHANGE PIN. The two frame-averaged methods' names must differ ONLY by
    the method token — the proof that one rule serves both."""
    kw = dict(steps=1, cfg_scale=1.0, eval_name="e", rotate_deg=90.0, n_angles=4)
    inv = eval_FLAC.build_output_paths(CKPT, cond_method="fa_invariant", **kw)
    car = eval_FLAC.build_output_paths(CKPT, cond_method="fa_cartesian", **kw)
    for kind in ("metrics", "predictions"):
        assert inv[kind] == car[kind].replace("fa_cartesian", "fa_invariant")


# --------------------------------------------------------------------------- #
# 2. provenance — fa_cartesian rows must carry POPULATED orbit fields
# --------------------------------------------------------------------------- #
def test_orbit_provenance_treats_fa_cartesian_like_fa_invariant():
    """``fa_cartesian`` executes the SAME batched orbit executor, so its rows must
    say so. ``("n/a", None)`` would be false provenance in the other direction
    from the vanilla case: it would make a frame-averaged row look like a
    single-pass one, and the admission validator (§3g) reads these two fields."""
    assert eval_FLAC.orbit_provenance("fa_cartesian", EVAL_CAP) == (yr.ORBIT_EXECUTION, 64)
    assert eval_FLAC.orbit_provenance("fa_cartesian", 32) == (yr.ORBIT_EXECUTION, 32)
    # omitted cap -> the module default, exactly as for fa_invariant
    assert (eval_FLAC.orbit_provenance("fa_cartesian")
            == eval_FLAC.orbit_provenance("fa_invariant")
            == (yr.ORBIT_EXECUTION, yr.FRAME_AVG_MAX_FWD_SAMPLES))


def test_metrics_record_carries_the_real_orbit_fields_for_fa_cartesian():
    """The plan's records schema (§3d): cond_method, the REAL angle list, the
    batched execution and the REAL cap. Nulls here are the specific failure the
    round-3 brief names — the admission validator would reject the row, after the
    GPU hours that produced it."""
    rec = eval_FLAC.build_metrics_record(
        {"T60": 1.0}, CKPT, rotate_deg=0.0, cond_method="fa_cartesian",
        frame_avg_angles=list(ANGLES), cond_autocast="bf16", batch_size=64,
        n_samples=6337, frame_avg_max_fwd_samples=EVAL_CAP,
    )
    assert rec["cond_method"] == "fa_cartesian"
    assert rec["frame_avg_angles"] == ANGLES
    assert rec["orbit_execution"] == yr.ORBIT_EXECUTION == "batched"
    assert rec["frame_avg_fwd_cap"] == EVAL_CAP
    assert isinstance(rec["source_sha"], str) and rec["source_sha"]
    json.loads(json.dumps(rec))


def test_predictions_meta_carries_the_real_orbit_fields_for_fa_cartesian():
    """Same schema on the predictions sidecar: the exp_02 comparator guard reads
    it, and a stored prediction set must name the execution that produced it."""
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", seed=42, n_samples=7, cond_method="fa_cartesian",
        frame_avg_angles=list(ANGLES), rotate_deg=0.0, batch_size=64,
        cond_autocast="bf16", frame_avg_max_fwd_samples=EVAL_CAP,
    )
    assert meta["cond_method"] == "fa_cartesian"
    assert meta["frame_avg_angles"] == ANGLES
    assert meta["orbit_execution"] == yr.ORBIT_EXECUTION
    assert meta["frame_avg_fwd_cap"] == EVAL_CAP
    json.loads(json.dumps(meta))


# A literal capture of the pre-change fa_invariant record, KEY ORDER INCLUDED
# (json.dump preserves insertion order, so the order is part of the artifact).
# Every published row was written by this path; widening an equality test to a
# membership test must not move a byte of it.
_GOLDEN_FA_INVARIANT_RECORD = {
    "metrics": {"T60": 1.0},
    "ckpt_path": CKPT,
    "rotate_deg": 0.0,
    "cond_method": "fa_invariant",
    "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
    "cond_autocast": "bf16",
    "orbit_execution": "batched",
    "frame_avg_fwd_cap": 64,
    "source_sha": "<sha>",
    "batch_size": 64,
    "n_samples": 6337,
    "dataset_config": "ds.json",
    "seed": 42,
    "cfg_scale": 1.0,
    "steps": 1,
    "eval_name": "e",
    "weights_source": "ema",
    "device": "cpu",
}

_GOLDEN_FA_INVARIANT_META = {
    "dataset_config": "ds.json",
    "seed": 42,
    "n_samples": 7,
    "cond_method": "fa_invariant",
    "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
    "rotate_deg": 0.0,
    "batch_size": 64,
    "cond_autocast": "bf16",
    "orbit_execution": "batched",
    "frame_avg_fwd_cap": 64,
    "source_sha": "<sha>",
}


def test_fa_invariant_record_is_byte_identical_to_the_pre_change_golden():
    """NO-CHANGE PIN (regression). Captured from the implementation BEFORE this
    round, key order included: the B-F rows in ``model_comparison.md`` and the
    D6-a re-evaluation must remain the same artifact."""
    rec = eval_FLAC.build_metrics_record(
        {"T60": 1.0}, CKPT, rotate_deg=0.0, cond_method="fa_invariant",
        frame_avg_angles=list(ANGLES), cond_autocast="bf16", batch_size=64,
        n_samples=6337, dataset_config="ds.json", seed=42, cfg_scale=1.0,
        steps=1, eval_name="e", weights_source="ema", device="cpu",
        frame_avg_max_fwd_samples=64,
    )
    rec["source_sha"] = "<sha>"
    assert list(rec) == list(_GOLDEN_FA_INVARIANT_RECORD)      # order is part of the file
    assert rec == _GOLDEN_FA_INVARIANT_RECORD

    meta = eval_FLAC.build_predictions_meta(
        "ds.json", seed=42, n_samples=7, cond_method="fa_invariant",
        frame_avg_angles=list(ANGLES), rotate_deg=0.0, batch_size=64,
        cond_autocast="bf16", frame_avg_max_fwd_samples=64,
    )
    meta["source_sha"] = "<sha>"
    assert list(meta) == list(_GOLDEN_FA_INVARIANT_META)
    assert meta == _GOLDEN_FA_INVARIANT_META


def test_vanilla_still_records_no_orbit_and_no_angles():
    """NO-CHANGE PIN (regression). A vanilla evaluation executes no orbit; a
    membership test that accidentally admitted it would label P1's rows as
    protocol-compatible with the frame-averaged arms."""
    assert eval_FLAC.orbit_provenance("vanilla", EVAL_CAP) == ("n/a", None)
    rec = eval_FLAC.build_metrics_record(
        {"T60": 1.0}, CKPT, rotate_deg=0.0, cond_method="vanilla",
        frame_avg_angles=None, frame_avg_max_fwd_samples=EVAL_CAP,
    )
    assert rec["orbit_execution"] == "n/a"
    assert rec["frame_avg_fwd_cap"] is None
    assert rec["frame_avg_angles"] is None


# --------------------------------------------------------------------------- #
# 3. evaluate_model — acceptance, flag resolution, and what lands in the record
# --------------------------------------------------------------------------- #
class _RecordingConditioner:
    """The raw ``MultiConditioner`` slot: records every direct call so a test can
    prove the VANILLA path ran (or, for the frame-averaged methods, that it
    did not)."""

    def __init__(self):
        self.calls = []

    def __call__(self, metadata, device, **kwargs):
        self.calls.append({"metadata": metadata, "device": device, "kwargs": kwargs})
        return {"stub": [torch.zeros(len(metadata), 1), None]}


class _FakeEvalModule:
    """Minimal stand-in for the PL wrapper ``evaluate_model`` drives: chainable
    ``eval()``/``requires_grad_()``/``to()``, a ``.diffusion`` with no
    pretransform, and a ``.device``. With an EMPTY dataloader the eval loop body
    never runs, so no conditioner, sampler or GPU is touched — the run exercises
    exactly the validation, resolution and record-writing stages under test.

    It also carries the handful of attributes the loop BODY reads
    (``io_channels``, ``dist_shift``, ``get_conditioning_inputs``, a model with a
    ``diffusion_objective``), so the dispatch tests below can drive one real
    batch through it with the sampler stubbed.

    Kept local rather than imported from ``test_eval_paths``: importing that
    module inserts the exp_02 worklog directory into ``sys.path`` and imports its
    comparator, which would make this file's collection depend on a worklog
    folder a pinned worktree need not carry (the trap ``eval_FLAC._load_exp16_readback``
    documents)."""

    def __init__(self):
        self.conditioner = _RecordingConditioner()
        self.diffusion = types.SimpleNamespace(
            model=types.SimpleNamespace(diffusion_objective="rectified_flow"),
            pretransform=None,
            conditioner=self.conditioner,
            io_channels=1,
            dist_shift=1.0,
            get_conditioning_inputs=lambda conditioning: {},
        )
        self.device = "cpu"

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def to(self, device):
        return self


def _stub_eval_deps(monkeypatch, tmp_path):
    """Write a toy model/dataset/ckpt triple and stub the four factories
    ``evaluate_model`` calls, at the ``eval_FLAC`` namespace."""
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1, "training": {"use_ema": False, "cond_method": "fa_cartesian"},
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
        eval_FLAC, "create_training_wrapper_from_config",
        lambda cfg, model: _FakeEvalModule(),
    )
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: [])
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config",
        lambda *a, **k: types.SimpleNamespace(
            update_metrics=lambda *a, **k: None,
            compute_metrics=lambda split: {"T60": 1.0},
        ),
    )
    return str(model_cfg), str(dataset_cfg), str(ckpt)


def test_evaluate_model_accepts_fa_cartesian_and_records_the_real_protocol(
        tmp_path, monkeypatch):
    """End-to-end through the real resolution and record-writing stages: the
    method is accepted, the frame angles and the cap resolve exactly as they do
    for fa_invariant, the artifact lands at the ``_fa_cartesian_a4`` path, and the
    record carries POPULATED orbit fields rather than the nulls a non-frame-averaged
    method would write."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_deps(monkeypatch, tmp_path)

    eval_FLAC.evaluate_model(
        model_cfg, dataset_cfg, ckpt, steps=1, cfg_scale=1.0, device="cpu",
        eval_name="exp21_BFC_S40000_K8_s42", cond_method="fa_cartesian",
        frame_avg_angles=(0.0, 90.0, 180.0, 270.0),
        frame_avg_max_fwd_samples=EVAL_CAP, cond_autocast="bf16", batch_size=64,
    )

    out = tmp_path / ("toy_metrics_1_1.0_exp21_BFC_S40000_K8_s42_fa_cartesian_a4.json")
    assert out.is_file(), sorted(p.name for p in tmp_path.iterdir())
    saved = json.loads(out.read_text())
    assert saved["cond_method"] == "fa_cartesian"
    assert saved["frame_avg_angles"] == ANGLES          # the REAL list, not null
    assert saved["orbit_execution"] == "batched"
    assert saved["frame_avg_fwd_cap"] == EVAL_CAP       # the REAL cap, not null
    assert saved["cond_autocast"] == "bf16"
    assert saved["batch_size"] == 64


def test_evaluate_model_resolves_non_default_angles_for_fa_cartesian(
        tmp_path, monkeypatch):
    """The angle list is READ, not assumed: a two-angle orbit must reach both the
    filename (``_a2``) and the record. Pins that the resolution is shared with
    fa_invariant rather than hardcoded to C4 for the new method."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_deps(monkeypatch, tmp_path)

    eval_FLAC.evaluate_model(
        model_cfg, dataset_cfg, ckpt, steps=1, cfg_scale=1.0, device="cpu",
        eval_name="two", cond_method="fa_cartesian", frame_avg_angles=(0.0, 180.0),
        frame_avg_max_fwd_samples=32,
    )

    out = tmp_path / "toy_metrics_1_1.0_two_fa_cartesian_a2.json"
    assert out.is_file(), sorted(p.name for p in tmp_path.iterdir())
    saved = json.loads(out.read_text())
    assert saved["frame_avg_angles"] == [0.0, 180.0]
    assert saved["frame_avg_fwd_cap"] == 32


def test_evaluate_model_defaults_the_cap_for_fa_cartesian(tmp_path, monkeypatch):
    """An omitted cap resolves to the module default and is RECORDED as the
    number, never as null: the row must state the cap the run applied."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_deps(monkeypatch, tmp_path)

    eval_FLAC.evaluate_model(
        model_cfg, dataset_cfg, ckpt, steps=1, cfg_scale=1.0, device="cpu",
        eval_name="dflt", cond_method="fa_cartesian",
    )

    saved = json.loads((tmp_path / "toy_metrics_1_1.0_dflt_fa_cartesian_a4.json").read_text())
    assert saved["frame_avg_fwd_cap"] == yr.FRAME_AVG_MAX_FWD_SAMPLES == 64
    assert saved["frame_avg_angles"] == list(yr.DEFAULT_FRAME_ANGLES)


def test_evaluate_model_shares_the_cap_validator_with_fa_invariant(tmp_path, monkeypatch):
    """A misconfigured cap must abort BEFORE any file or model work, naming the
    operator-facing flag — the same fail-fast fa_invariant gets. Every path
    argument points at a nonexistent file, so late validation would surface
    FileNotFoundError instead, and model construction trips a loud sentinel."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"),
    )
    with pytest.raises(ValueError, match="frame-avg-max-fwd-samples"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "missing_model.json"), str(tmp_path / "missing_ds.json"),
            str(tmp_path / "missing.ckpt"), steps=1, cfg_scale=1.0, device="cpu",
            cond_method="fa_cartesian", frame_avg_max_fwd_samples=0,
        )


def test_evaluate_model_still_rejects_an_unknown_method(tmp_path, monkeypatch):
    """NO-CHANGE PIN. Widening the fail-fast list must not make it permissive:
    an unregistered method still aborts before model construction, or a typo'd
    launcher would silently evaluate vanilla and record the typo."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"),
    )
    for bad in ("canon", "fa_cartesian_", "FA_CARTESIAN"):
        with pytest.raises(ValueError, match="cond_method"):
            eval_FLAC.evaluate_model(
                str(tmp_path / "m.json"), str(tmp_path / "d.json"),
                str(tmp_path / "c.ckpt"), steps=1, cfg_scale=1.0, device="cpu",
                cond_method=bad,
            )


# --------------------------------------------------------------------------- #
# 3b. the CONDITIONING DISPATCH itself — which function the eval loop calls
# --------------------------------------------------------------------------- #
# The tests above drive evaluate_model with an EMPTY dataloader, so they never
# execute the loop body: deleting the fa_cartesian dispatch branch would leave
# every one of them green while the arm silently evaluated as VANILLA — a
# single-pass conditioning reported under a record that claims a C4 orbit, which
# is the announcement-05 failure in its worst form (plausible numbers, wrong
# arm). That is the exact coverage defect the r2 review caught on the training
# side, so the branch gets its own pin: one real batch through the loop, with
# only the sampler stubbed.
def _one_batch_loader(n=2):
    """A dataloader of exactly one batch: (reals, metadata) with real depth
    panoramas and poses, since the loop's metric stage reads md['depth'],
    md['source'] and md['scene']."""
    return [(torch.zeros(n, 1, 64), _batch(n))]


def _spy_conditioning(monkeypatch):
    """Replace both frame-average entry points with recording spies (at the
    eval_FLAC namespace, which is where the dispatch resolves them)."""
    calls = []

    def make(name):
        def spy(*args, **kwargs):
            calls.append({"fn": name, "args": args, "kwargs": kwargs})
            return {"stub": [torch.zeros(len(args[1]), 1), None]}
        return spy

    monkeypatch.setattr(eval_FLAC, "fa_cartesian_conditioning", make("fa_cartesian"))
    monkeypatch.setattr(eval_FLAC, "invariant_conditioning", make("fa_invariant"))
    monkeypatch.setattr(
        "src.inference.sampling.sample_discrete_euler",
        lambda model, noise, steps, **kw: torch.zeros_like(noise),
    )
    return calls


def _run_one_batch(tmp_path, monkeypatch, cond_method, **kwargs):
    model_cfg, dataset_cfg, ckpt = _stub_eval_deps(monkeypatch, tmp_path)
    module = _FakeEvalModule()
    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config", lambda cfg, model: module)
    monkeypatch.setattr(
        eval_FLAC, "create_dataloader_from_config", lambda *a, **k: _one_batch_loader())
    calls = _spy_conditioning(monkeypatch)
    eval_FLAC.evaluate_model(
        model_cfg, dataset_cfg, ckpt, steps=1, cfg_scale=1.0, device="cpu",
        eval_name="disp", cond_method=cond_method, cond_autocast="bf16",
        frame_avg_max_fwd_samples=EVAL_CAP, **kwargs)
    return module, calls


def test_the_eval_loop_calls_fa_cartesian_conditioning_for_fa_cartesian(
        tmp_path, monkeypatch):
    """The dispatch branch, executed. Delete it and this fails: the call list
    would show the raw conditioner instead. The call SHAPE is pinned too —
    conditioner, metadata and device positionally, then the angle tuple
    positionally, then ``max_fwd_samples`` as a keyword — the same form the
    fa_invariant branch issues, so the arms differ in the function and in nothing
    about how it is invoked (round-2 call-shape discipline, extended to eval)."""
    module, calls = _run_one_batch(tmp_path, monkeypatch, "fa_cartesian")

    assert [c["fn"] for c in calls] == ["fa_cartesian"], calls
    args, kwargs = calls[0]["args"], calls[0]["kwargs"]
    assert args[0] is module.diffusion.conditioner
    assert len(args[1]) == 2 and all("depth" in md for md in args[1])  # batch metadata
    assert args[2] == "cpu"
    assert args[3] == tuple(ANGLES)                    # angles POSITIONAL, as for fa_invariant
    assert kwargs == {"max_fwd_samples": EVAL_CAP}     # cap by KEYWORD, nothing else
    # ...and the raw single-pass conditioner was never used: a frame-averaged arm
    # that quietly ran vanilla would still write a record claiming a C4 orbit.
    assert module.conditioner.calls == []


def test_the_eval_loop_still_calls_invariant_conditioning_for_fa_invariant(
        tmp_path, monkeypatch):
    """NO-CHANGE PIN (regression). Adding a branch must not steal the existing
    one: B-F still routes to invariant_conditioning, with the same call shape."""
    module, calls = _run_one_batch(tmp_path, monkeypatch, "fa_invariant")
    assert [c["fn"] for c in calls] == ["fa_invariant"], calls
    assert calls[0]["args"][3] == tuple(ANGLES)
    assert calls[0]["kwargs"] == {"max_fwd_samples": EVAL_CAP}
    assert module.conditioner.calls == []


def test_the_eval_loop_still_runs_a_single_pass_for_vanilla(tmp_path, monkeypatch):
    """NO-CHANGE PIN (regression). Vanilla takes neither branch: exactly one
    direct conditioner call per batch, and no orbit."""
    module, calls = _run_one_batch(tmp_path, monkeypatch, "vanilla")
    assert calls == []
    assert len(module.conditioner.calls) == 1
    assert module.conditioner.calls[0]["kwargs"] == {}


# --------------------------------------------------------------------------- #
# 4. the CLI: argparse choices and the plan §5 registered templates
# --------------------------------------------------------------------------- #
def test_parser_accepts_fa_cartesian_and_still_rejects_junk():
    """argparse must admit the new choice (and only it): a bad value exits 2 with
    'invalid choice', a good one gets past choices to the required-args error."""
    bad = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--cond-method", "fa_cartesian_typo"],
        capture_output=True, cwd=REPO_ROOT,
    )
    assert bad.returncode == 2
    assert b"invalid choice" in bad.stderr

    good = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--cond-method", "fa_cartesian"],
        capture_output=True, cwd=REPO_ROOT,
    )
    assert good.returncode == 2                     # still missing required args...
    assert b"invalid choice" not in good.stderr     # ...but not on choices


# The plan §5 registered templates, verbatim except for the two paths a dry run
# must NOT have (the model config is opened first, so a nonexistent one stops the
# run right after every flag has been parsed AND every fail-fast validator has
# passed — no ckpt read, no model, no GPU).
_TEMPLATE_TAIL = [
    "--dataset-config", "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
    "--cond-method", "fa_cartesian",
    "--frame-avg-angles", "0,90,180,270",
    "--frame-avg-max-fwd-samples", "64",
    "--rotate-mode", "fixed",
    "--cond-autocast", "bf16",
    "--batch-size", "64", "--cfg-scale", "1.0", "--steps", "1",
    "--record-per-scene", "--seed", "42",
]


@pytest.mark.parametrize("extra,name", [
    (["--rotate-deg", "0"], "exp21_BFC_S40000_K8_s42"),
    (["--rotate-deg", "45", "--record-stream", "--expected-stream-count", "6337"],
     "exp21_BFC_S40000_K8_s42_rot45"),
])
def test_registered_template_parses_and_passes_every_fail_fast(tmp_path, extra, name):
    """The §5 templates must be executable as written. This is a DRY check: the
    process dies at the first file open, which proves argparse accepted every
    flag and that the cond_method / cap / rotation / autocast validators all
    passed — the four gates that would otherwise fail a real cell after it had
    already been queued behind 40k training steps."""
    argv = [sys.executable, "eval_FLAC.py",
            "--model-config", str(tmp_path / "no_such_model_config.json"),
            "--ckpt-path", str(tmp_path / "no_such.ckpt"),
            "--eval-name", name] + _TEMPLATE_TAIL + extra
    proc = subprocess.run(argv, capture_output=True, cwd=REPO_ROOT)
    err = proc.stderr.decode()
    assert proc.returncode != 0
    assert "FileNotFoundError" in err, err[-3000:]
    assert "no_such_model_config.json" in err
    for forbidden in ("invalid choice", "unrecognized arguments",
                      "Unknown cond_method", "Unknown cond_autocast",
                      "frame_avg_max_fwd_samples"):
        assert forbidden not in err, f"{forbidden!r} in stderr:\n{err[-3000:]}"


# --------------------------------------------------------------------------- #
# 5. the EVAL chunk plan (plan §3h item 10) — NO-CHANGE PIN on §2's split
# --------------------------------------------------------------------------- #
def test_eval_batch_64_needs_cap_64_and_cap_32_is_refused():
    """NO-CHANGE PIN. The plan declares TRAINING and EVAL chunk plans separately:
    training runs cap 32 at micro-batch 32 (one angle per chunk), evaluation runs
    batch 64 and therefore REQUIRES cap >= 64 — a chunk is whole angles and cannot
    be split below one. Pinned against the function the eval path calls, so the
    §5 template's ``--frame-avg-max-fwd-samples 64`` is a checked requirement
    rather than a convention: at cap 32 the registered command would abort on the
    first batch, hours into a queued cell."""
    md = _batch(64)
    cond = _build_cond()
    out = yr.fa_cartesian_conditioning(cond, md, "cpu", C4, max_fwd_samples=64)
    # batch 64, cap 64 -> one angle per chunk: the base pass plus three chunks.
    assert cond.conditioners["source"].batch_sizes == [64, 64, 64, 64]
    for key in POSE_KEYS:
        assert out[key][0].shape[0] == 64
        assert torch.isfinite(out[key][0]).all()

    with pytest.raises(ValueError, match="exceeds"):
        yr.fa_cartesian_conditioning(_build_cond(), md, "cpu", C4, max_fwd_samples=32)


def test_eval_cap_is_a_plan_knob_only():
    """NO-CHANGE PIN. Eval mode draws no RoPE rescale, so a larger cap only
    regroups the forwards — the numbers must be identical. This is what makes
    ``--frame-avg-max-fwd-samples 64`` a protocol-identity pin rather than a
    result-changing hyperparameter (plan §2 EVAL chunk plan)."""
    md = _batch(64)
    small, big = _build_cond(), _build_cond()
    a = yr.fa_cartesian_conditioning(small, md, "cpu", C4, max_fwd_samples=64)
    b = yr.fa_cartesian_conditioning(big, md, "cpu", C4, max_fwd_samples=256)
    assert small.conditioners["source"].batch_sizes == [64, 64, 64, 64]
    assert big.conditioners["source"].batch_sizes == [64, 192]   # all 3 angles at once
    for key in POSE_KEYS:
        assert torch.allclose(a[key][0], b[key][0], atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. eval_pl.py — fail closed on an arm it cannot give provenance for (r2 nit 3)
# --------------------------------------------------------------------------- #
class _Sentinel(Exception):
    """Raised by a stubbed factory to prove execution got PAST the guard."""


def _pl_args(model_config_path):
    return types.SimpleNamespace(
        seed=42, model_config=str(model_config_path),
        val_dataset_config="unused.json", batch_size=1, num_workers=0,
        cfg_scale=1.0, steps=1, store_predictions=False, num_gpus=1, num_nodes=1,
        precision="32", accum_batches=1, gradient_clip_val=0.0, ckpt_path="unused.ckpt",
    )


def _write_pl_config(tmp_path, cond_method):
    cfg = tmp_path / "model.json"
    training = {"use_ema": False, "metrics": {}}
    if cond_method is not None:
        training["cond_method"] = cond_method
    cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1, "training": training,
    }))
    return cfg


def test_eval_pl_rejects_fa_cartesian_before_any_construction(tmp_path, monkeypatch):
    """eval_pl writes ``{metrics, ckpt_path}`` and nothing else — no cond_method,
    no angles, no orbit execution or cap, no source SHA. It inherits the widened
    wrapper dispatch, so without a guard it would happily RUN the arm and emit a
    row indistinguishable from a vanilla one. It must abort before the model or
    the wrapper is built (both trip a loud sentinel here)."""
    cfg = _write_pl_config(tmp_path, "fa_cartesian")
    monkeypatch.setattr(eval_pl, "get_all_args", lambda: _pl_args(cfg))
    monkeypatch.setattr(
        eval_pl, "create_model_from_config",
        lambda *a, **k: pytest.fail("eval_pl reached model construction"))
    monkeypatch.setattr(
        eval_pl, "create_training_wrapper_from_config",
        lambda *a, **k: pytest.fail("eval_pl reached wrapper construction"))

    with pytest.raises(ValueError) as excinfo:
        eval_pl.main()
    msg = str(excinfo.value)
    assert "fa_cartesian" in msg
    assert "eval_FLAC.py" in msg          # names the entry point that IS registered


@pytest.mark.parametrize("cond_method", ["vanilla", "fa_invariant", None])
def test_eval_pl_does_not_guard_the_registered_methods(tmp_path, monkeypatch, cond_method):
    """SELECTIVITY. The guard is for the one arm with no registered-eval
    provenance; vanilla and fa_invariant runs (and configs that declare no
    cond_method at all) must be untouched. Proven by a sentinel from the stubbed
    model factory: reaching it means the guard let the run through."""
    cfg = _write_pl_config(tmp_path, cond_method)
    monkeypatch.setattr(eval_pl, "get_all_args", lambda: _pl_args(cfg))

    def _boom(*a, **k):
        raise _Sentinel("past the guard")

    monkeypatch.setattr(eval_pl, "create_model_from_config", _boom)
    with pytest.raises(_Sentinel):
        eval_pl.main()
