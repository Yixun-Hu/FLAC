"""exp_14 (fa_drawshare) — the frame-average forward cap is a CONFIG key.

Main-session seat for this round: **claude-fable-5 (xhigh)** (model-change flag);
code by the Opus 5 Coder seat per SOP §Roles.

`src/data/yaw_rotation.py:501` derives the orbit partition from a MODULE GLOBAL:

    angles_per_chunk = max(1, FRAME_AVG_MAX_FWD_SAMPLES // micro_batch)

which is why announcement 06 has to call the chunk plan "derived, not declared".
exp_14's treatment is that partition, so the cap must become a declared,
checkpoint-embedded property of the arm (`training.frame_avg_max_fwd_samples`)
rather than an ambient constant — an env override would be invisible to the
checkpoint and would leak across sessions (plan §3.1 / review H4).

The contract these tests pin, in the order the plan states it:

1. **Default = today, exactly.** No key, no kwarg -> the module default 64 and a
   byte-identical chunk plan. Every recipe already in the record must keep the
   number it was run under.
2. **Explicit value honoured, no global mutation.** The passed cap drives the
   partition; ``yr.FRAME_AVG_MAX_FWD_SAMPLES`` is never rebound (the exp_10
   dispersion probe mutates the global and restores it — a training path must
   never do that, because a crash mid-step would leave the process serving a cap
   nobody declared).
3. **Fail closed.** bool / non-int / <1 / cap < micro-batch abort at the factory,
   at direct wrapper construction AND at the call site.
4. **The partition is asserted by EXECUTION**, not by arithmetic: caps 32/64/96
   at micro-32 must produce 1/2/3 angles per chunk, counted as real conditioner
   forwards. Arithmetic on the cap is exactly what cannot catch a threading bug.
5. **Round-trip**: the cap survives into the embedded ``model_config`` of a
   checkpoint, which is what the launcher's resume gate compares.
6. **Evaluation is a separate, pinned quantity**: ``--frame-avg-max-fwd-samples``
   reaches the ``invariant_conditioning`` call and is recorded in the metrics
   record next to ``cond_method`` / ``frame_avg_angles``.
"""
import copy
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import eval_FLAC
import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr
from src.models.conditioners import MultiConditioner
from src.training.factory import create_training_wrapper_from_config

DEV = "cpu"
_REPO = Path(__file__).resolve().parents[2]

# The exp_14 arms. DS-PA reproduces exp_07/exp_10's per-angle July path (1 angle
# per chunk at micro-32); DS-CS3 reproduces exp_11's C4L draw-sharing TOPOLOGY
# (all three non-identity angles in one chunk).
ARM_DIR = _REPO / "worklog/worklog_yixun/exp_14_fa_drawshare_claude"
BF_CONFIG = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
ARM_CONFIGS = {"DSPA": (ARM_DIR / "FLAC_AR_BF_DSPA.json", 32),
               "DSCS3": (ARM_DIR / "FLAC_AR_BF_DSCS3.json", 96)}

CAP_KEY = "frame_avg_max_fwd_samples"


# --------------------------------------------------------------------------- #
# tiny counting conditioners (no DINOv3, no pretrained weights)
# --------------------------------------------------------------------------- #
class _CountingViT(nn.Module):
    """GeometryConditioner stand-in: records the batch size of every forward.

    The chunk plan is READ OFF THIS LIST, so the tests observe the partition the
    production code actually executed rather than recomputing ``cap // batch``.
    """

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.name = "GeometryConditioner"
        self.out_dim = out_dim
        self.batch_sizes = []

    def forward(self, coord_list, device=DEV):
        self.batch_sizes.append(len(coord_list))
        coords = [c["coord"] for c in coord_list]
        depths = [c["depth"] for c in coord_list]
        coord = torch.stack(coords, dim=0)
        if coord.ndim == 2:
            coord = coord.unsqueeze(1)
        depth = torch.stack(depths, dim=0)                     # [B, 3, H, W]
        B, N, _ = coord.shape
        pooled = depth.mean(dim=(2, 3))                        # [B, 3]
        feat = torch.tanh(coord + pooled[:, None, :])          # [B, N, 3]
        out = feat.sum(-1, keepdim=True).expand(B, N, self.out_dim).contiguous()
        return [out, torch.ones(B, 1, device=device)]


class _CountingPose(nn.Module):
    """dist_embedder stand-in (runs exactly once per invariant_conditioning)."""

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.name = "DistEmbedderConditioner"
        self.out_dim = out_dim
        self.batch_sizes = []

    def forward(self, x_list, device=DEV):
        self.batch_sizes.append(len(x_list))
        x = torch.stack(x_list, dim=0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, N, _ = x.shape
        out = torch.tanh(x.sum(-1, keepdim=True)).expand(B, N, self.out_dim).contiguous()
        return [out, torch.ones(B, N, device=device)]


VIT_IDS = ("source_vit", "context_poses_vit")


def _build_cond():
    return MultiConditioner({
        "source": _CountingPose(),
        "context_poses": _CountingPose(),
        "source_vit": _CountingViT(),
        "context_poses_vit": _CountingViT(),
    })


def _depth(H: int = 2, W: int = 16) -> torch.Tensor:
    """A small non-axisymmetric panorama: W is a multiple of 4 so every C4 angle
    is an exact column roll, and the content varies with azimuth so a rotation is
    observable at all."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    d = (3.0 + torch.sin(theta)).view(1, W).expand(H, W)
    return torch.stack([d * torch.cos(theta), d * torch.sin(theta),
                        torch.linspace(-1.0, 1.0, H).view(H, 1).expand(H, W)], dim=0).contiguous()


def _md(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(2, 3, generator=g),
        "context_poses_vit": torch.randn(2, 3, generator=g),
        "depth": _depth(),
    }


def _batch(n: int) -> list:
    return [_md(s) for s in range(n)]


C4 = (0.0, 90.0, 180.0, 270.0)


def _plans(cond):
    return {vit: cond.conditioners[vit].batch_sizes for vit in VIT_IDS}


# --------------------------------------------------------------------------- #
# 1. default -> 64, behaviour unchanged
# --------------------------------------------------------------------------- #
def test_default_cap_is_64_and_partition_is_unchanged():
    """No kwarg, ``None`` and an explicit 64 must be the SAME execution."""
    assert yr.FRAME_AVG_MAX_FWD_SAMPLES == 64

    md = _batch(32)
    outs, plans = [], []
    for kwargs in ({}, {"max_fwd_samples": None}, {"max_fwd_samples": 64}):
        cond = _build_cond()
        outs.append(yr.invariant_conditioning(cond, md, DEV, C4, **kwargs))
        plans.append(_plans(cond))

    # micro-32, cap 64 -> 2 angles per chunk -> base(32) + chunk(64) + chunk(32)
    assert plans[0] == {v: [32, 64, 32] for v in VIT_IDS}, plans[0]
    assert plans[1] == plans[0] and plans[2] == plans[0]
    for key in VIT_IDS:
        assert torch.equal(outs[0][key][0], outs[1][key][0])
        assert torch.equal(outs[0][key][0], outs[2][key][0])


def test_default_call_does_not_touch_the_module_global():
    before = yr.FRAME_AVG_MAX_FWD_SAMPLES
    yr.invariant_conditioning(_build_cond(), _batch(4), DEV, C4)
    assert yr.FRAME_AVG_MAX_FWD_SAMPLES == before == 64


# --------------------------------------------------------------------------- #
# 2. explicit cap -> the exp_14 partitions, asserted by EXECUTION
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap,angles_per_chunk,plan", [
    (32, 1, [32, 32, 32, 32]),   # DS-PA  : per-angle draws (the July path)
    (64, 2, [32, 64, 32]),       # today's default
    (96, 3, [32, 96]),           # DS-CS3 : exp_11's 3/3 sharing topology
])
def test_micro32_partitions_for_the_exp14_caps(cap, angles_per_chunk, plan):
    """The registered chunk plans (announcement 06) — counted, not computed.

    ``angles_per_chunk`` is stated in the parametrisation and checked against the
    executed forwards: chunk sizes must be ``k * 32`` with ``k <= angles/chunk``,
    and the three non-identity angles must be covered exactly once."""
    batch = 32
    cond = _build_cond()
    out = yr.invariant_conditioning(cond, _batch(batch), DEV, C4, max_fwd_samples=cap)

    for vit in VIT_IDS:
        sizes = cond.conditioners[vit].batch_sizes
        assert sizes == plan, f"cap {cap}: {vit} plan {sizes} != {plan}"
        assert sizes[0] == batch, "the base (angle-0) pass is one plain batch"
        rotated = sizes[1:]
        assert sum(rotated) == 3 * batch, "the three rotated angles must run exactly once"
        assert max(rotated) == angles_per_chunk * batch
        assert all(s <= cap for s in rotated), f"cap {cap} exceeded: {sizes}"
    # the non-ViT conditioners still run exactly once
    for pid in ("source", "context_poses"):
        assert cond.conditioners[pid].batch_sizes == [batch]
    assert set(out) == {"source", "context_poses", *VIT_IDS}


def test_cap_choice_does_not_change_the_averaging_arithmetic():
    """Different partitions, same frame average (these stand-in conditioners are
    deterministic, so only the GROUPING differs — exactly the exp_11 disclosure:
    the arithmetic is identical, the augmentation schedule is not)."""
    md = _batch(8)
    ref = yr.invariant_conditioning(_build_cond(), md, DEV, C4, max_fwd_samples=8)
    for cap in (16, 24, 32, 64):
        got = yr.invariant_conditioning(_build_cond(), md, DEV, C4, max_fwd_samples=cap)
        for key in VIT_IDS:
            assert torch.allclose(ref[key][0], got[key][0], atol=1e-6), (cap, key)


def test_explicit_cap_never_mutates_the_module_global():
    for cap in (32, 96):
        yr.invariant_conditioning(_build_cond(), _batch(4), DEV, C4, max_fwd_samples=cap)
        assert yr.FRAME_AVG_MAX_FWD_SAMPLES == 64, (
            "the cap must be PASSED, never assigned to the module global: a crash "
            "mid-step would otherwise leave the process serving an undeclared cap"
        )


# --------------------------------------------------------------------------- #
# 3. fail-closed validation at the call site
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, False, 1.0, 32.0, "32", [32], object()])
def test_call_site_rejects_non_integer_caps(bad):
    with pytest.raises(ValueError, match="max_fwd_samples"):
        yr.invariant_conditioning(_build_cond(), _batch(2), DEV, C4, max_fwd_samples=bad)


@pytest.mark.parametrize("bad", [0, -1, -64])
def test_call_site_rejects_non_positive_caps(bad):
    with pytest.raises(ValueError, match="max_fwd_samples"):
        yr.invariant_conditioning(_build_cond(), _batch(2), DEV, C4, max_fwd_samples=bad)


def test_cap_below_the_batch_is_rejected_in_the_existing_style():
    """A chunk is whole angles, so it cannot be split below one angle. The message
    names the knob that produced the cap, so the operator knows where to fix it."""
    with pytest.raises(ValueError) as e:
        yr.invariant_conditioning(_build_cond(), _batch(33), DEV, C4, max_fwd_samples=32)
    msg = str(e.value)
    assert "33" in msg and "32" in msg
    assert "training.frame_avg_max_fwd_samples" in msg
    assert "cannot be split below one angle" in msg

    # the DEFAULT path keeps its historical wording (the constant is named)
    with pytest.raises(ValueError) as e2:
        yr.invariant_conditioning(_build_cond(), _batch(65), DEV, C4)
    assert "FRAME_AVG_MAX_FWD_SAMPLES" in str(e2.value)


def test_cap_equal_to_the_batch_is_accepted():
    """cap == micro-batch is the DS-PA arm itself (one angle per chunk)."""
    cond = _build_cond()
    yr.invariant_conditioning(cond, _batch(32), DEV, C4, max_fwd_samples=32)
    assert cond.conditioners["source_vit"].batch_sizes == [32, 32, 32, 32]


def test_validation_runs_even_when_there_is_nothing_to_chunk():
    """A single-angle orbit short-circuits before the partition; an invalid cap
    must still abort, or a typo'd arm could train silently for a week whenever
    the orbit happened to be trivial."""
    with pytest.raises(ValueError, match="max_fwd_samples"):
        yr.invariant_conditioning(_build_cond(), _batch(2), DEV, (0.0,), max_fwd_samples=0)


# --------------------------------------------------------------------------- #
# 4. propagation: factory AND direct wrapper construction
# --------------------------------------------------------------------------- #
def _training_config(**extra):
    cfg = {
        "model_type": "diffusion_cond",
        "sample_rate": 22050,
        "training": {
            "learning_rate": 5e-5,
            "cond_method": "fa_invariant",
            "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
        },
    }
    cfg["training"].update(extra)
    return cfg


@pytest.fixture()
def stub_wrapper(monkeypatch):
    """Capture the factory's kwargs without building DINOv3."""
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    return captured


def test_factory_passes_the_cap_through(stub_wrapper):
    create_training_wrapper_from_config(_training_config(**{CAP_KEY: 96}), object())
    assert stub_wrapper["frame_avg_max_fwd_samples"] == 96


def test_factory_omits_the_kwarg_when_the_key_is_absent(stub_wrapper):
    """Absent key -> the construction call is literally the pre-change call, so
    every arm already in the record is unaffected."""
    create_training_wrapper_from_config(_training_config(), object())
    assert "frame_avg_max_fwd_samples" not in stub_wrapper


@pytest.mark.parametrize("bad", [True, False, 32.0, "32", None, [32], {}])
def test_factory_rejects_non_integer_caps(bad, stub_wrapper):
    with pytest.raises(ValueError, match=CAP_KEY):
        create_training_wrapper_from_config(_training_config(**{CAP_KEY: bad}), object())


@pytest.mark.parametrize("bad", [0, -1])
def test_factory_rejects_non_positive_caps(bad, stub_wrapper):
    with pytest.raises(ValueError, match=CAP_KEY):
        create_training_wrapper_from_config(_training_config(**{CAP_KEY: bad}), object())


# ---- direct construction (as fail-closed as the config path) --------------- #
class _TinyDiffusion(nn.Module):
    """Minimum surface DiffusionCondTrainingWrapper touches at construction."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.conditioner = _build_cond()
        self.diffusion_objective = "rectified_flow"
        self.pretransform = None


def _direct(**kwargs):
    return tdiff.DiffusionCondTrainingWrapper(
        _TinyDiffusion(), lr=5e-5, use_ema=False, cond_method="fa_invariant", **kwargs)


def test_direct_construction_default_is_none_meaning_module_default():
    wrapper = _direct()
    assert wrapper.frame_avg_max_fwd_samples is None


def test_direct_construction_honours_an_explicit_cap():
    assert _direct(frame_avg_max_fwd_samples=32).frame_avg_max_fwd_samples == 32


@pytest.mark.parametrize("bad", [True, False, 32.0, "32", [32]])
def test_direct_construction_rejects_non_integer_caps(bad):
    with pytest.raises(ValueError, match="frame_avg_max_fwd_samples"):
        _direct(frame_avg_max_fwd_samples=bad)


@pytest.mark.parametrize("bad", [0, -5])
def test_direct_construction_rejects_non_positive_caps(bad):
    with pytest.raises(ValueError, match="frame_avg_max_fwd_samples"):
        _direct(frame_avg_max_fwd_samples=bad)


def test_wrapper_forwards_the_cap_to_invariant_conditioning(monkeypatch):
    """The treatment must arrive at the call site as a KEYWORD (the 5th positional
    parameter is ``vit_ids``: a positional slip there would silently disable the
    frame average instead of changing the cap)."""
    seen = {}

    def _spy(conditioner, metadata, device, angles=None, vit_ids=None, **kwargs):
        seen["args"] = (angles, vit_ids)
        seen["kwargs"] = kwargs
        return conditioner(metadata, device)

    monkeypatch.setattr(tdiff, "invariant_conditioning", _spy)
    wrapper = _direct(frame_avg_max_fwd_samples=96)
    wrapper._compute_conditioning(_batch(2))
    assert seen["kwargs"] == {"max_fwd_samples": 96}
    assert seen["args"][1] is None, "vit_ids must not be filled positionally"

    monkeypatch.setattr(tdiff, "invariant_conditioning", _spy)
    _direct()._compute_conditioning(_batch(2))
    assert seen["kwargs"] == {"max_fwd_samples": None}


def test_wrapper_cap_reaches_the_real_partition():
    """End-to-end through the real ``_compute_conditioning``: config value 32 ->
    one angle per chunk at micro-32, counted on the conditioner."""
    wrapper = _direct(frame_avg_max_fwd_samples=32)
    wrapper._compute_conditioning(_batch(32))
    assert wrapper.diffusion.conditioner.conditioners["source_vit"].batch_sizes == [32] * 4


# --------------------------------------------------------------------------- #
# 5. config round-trip: the cap is embedded in the checkpoint
# --------------------------------------------------------------------------- #
def test_cap_survives_the_model_config_checkpoint_round_trip(tmp_path):
    """``train.py``'s ModelConfigEmbedderCallback writes ``model_config`` into the
    checkpoint; the launcher's resume gate compares that embedded object against
    the arm's current JSON. Both halves are exercised here."""
    from train import ModelConfigEmbedderCallback

    cfg = _training_config(**{CAP_KEY: 96})
    ckpt = {"global_step": 2500, "state_dict": {}}
    ModelConfigEmbedderCallback(cfg).on_save_checkpoint(None, None, ckpt)

    path = tmp_path / "round_trip.ckpt"
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded["model_config"]["training"][CAP_KEY] == 96
    assert loaded["model_config"] == cfg, "the embedded object must equal the config"

    drifted = copy.deepcopy(cfg)
    drifted["training"][CAP_KEY] = 32
    assert loaded["model_config"] != drifted, (
        "a changed cap MUST show up as a parsed-object mismatch — this is the "
        "comparison the launcher's resume gate makes (the wrapper is rebuilt from "
        "the CURRENT json at train.py:160 before PL loads the ckpt at :230, so a "
        "silently changed cap would take effect on resume)"
    )


# --------------------------------------------------------------------------- #
# 6. the arm configs are BF + exactly one key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", sorted(ARM_CONFIGS))
def test_arm_config_is_bf_plus_only_the_cap_key(arm):
    path, cap = ARM_CONFIGS[arm]
    assert path.is_file(), f"arm config not found: {path}"
    arm_cfg = json.loads(path.read_text())
    bf = json.loads(BF_CONFIG.read_text())

    assert arm_cfg["training"][CAP_KEY] == cap
    stripped = copy.deepcopy(arm_cfg)
    stripped["training"].pop(CAP_KEY)
    assert stripped == bf, f"{path.name} differs from FLAC_AR_BF.json beyond {CAP_KEY}"

    # the rest of the arm identity the plan pins
    assert arm_cfg["training"]["cond_method"] == "fa_invariant"
    assert arm_cfg["training"]["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
    assert arm_cfg["training"]["use_ema"] is True


@pytest.mark.parametrize("arm", sorted(ARM_CONFIGS))
def test_arm_config_declares_the_intended_chunk_plan(arm):
    """announcement 06: cap + micro-batch -> angles per chunk. DS-PA must be 1/3
    (per-angle, the July path) and DS-CS3 3/3 (exp_11's sharing topology)."""
    path, cap = ARM_CONFIGS[arm]
    micro = 32
    expected = {"DSPA": 1, "DSCS3": 3}[arm]
    assert max(1, cap // micro) == expected

    cfg = json.loads(path.read_text())
    cond = _build_cond()
    yr.invariant_conditioning(cond, _batch(micro), DEV,
                              tuple(cfg["training"]["frame_avg_angles"]),
                              max_fwd_samples=cfg["training"][CAP_KEY])
    sizes = cond.conditioners["source_vit"].batch_sizes[1:]
    assert max(sizes) == expected * micro
    assert len(sizes) == math.ceil(3 / expected)


@pytest.mark.parametrize("arm", sorted(ARM_CONFIGS))
def test_arm_config_builds_through_the_factory(arm, stub_wrapper):
    path, cap = ARM_CONFIGS[arm]
    create_training_wrapper_from_config(json.loads(path.read_text()), object())
    assert stub_wrapper["frame_avg_max_fwd_samples"] == cap
    assert stub_wrapper["cond_method"] == "fa_invariant"


def test_the_two_arms_differ_in_exactly_one_key():
    """One delta, or the experiment is a two-factor comparison nothing can untangle."""
    a = json.loads(ARM_CONFIGS["DSPA"][0].read_text())
    b = json.loads(ARM_CONFIGS["DSCS3"][0].read_text())
    assert a["training"][CAP_KEY] == 32 and b["training"][CAP_KEY] == 96
    a["training"][CAP_KEY] = b["training"][CAP_KEY]
    assert a == b


# --------------------------------------------------------------------------- #
# 7. evaluation: a SEPARATE, explicitly pinned cap
# --------------------------------------------------------------------------- #
def test_orbit_provenance_default_is_unchanged():
    assert eval_FLAC.orbit_provenance("fa_invariant") == (yr.ORBIT_EXECUTION, 64)
    assert eval_FLAC.orbit_provenance("vanilla") == ("n/a", None)


def test_orbit_provenance_records_an_explicit_cap():
    assert eval_FLAC.orbit_provenance("fa_invariant", 32) == (yr.ORBIT_EXECUTION, 32)
    # a vanilla row executes NO orbit: it must stay null even if a cap was passed
    assert eval_FLAC.orbit_provenance("vanilla", 32) == ("n/a", None)


def test_metrics_record_carries_the_evaluation_cap():
    rec = eval_FLAC.build_metrics_record(
        {"T60": 1.0}, "c.ckpt", rotate_deg=0.0, cond_method="fa_invariant",
        frame_avg_angles=[0.0, 90.0, 180.0, 270.0], frame_avg_max_fwd_samples=32)
    assert rec["frame_avg_fwd_cap"] == 32
    assert rec["cond_method"] == "fa_invariant"
    assert rec["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
    json.loads(json.dumps(rec))

    meta = eval_FLAC.build_predictions_meta(
        "ds.json", 42, 7, "fa_invariant", [0.0, 90.0, 180.0, 270.0], 0.0, 64, "bf16",
        frame_avg_max_fwd_samples=96)
    assert meta["frame_avg_fwd_cap"] == 96
    json.loads(json.dumps(meta))


def test_metrics_record_default_is_byte_identical_to_today():
    """The pinned common evaluation cap is 64 — the module default — so every row
    already in the record keeps its exact shape AND its exact values."""
    kwargs = dict(rotate_deg=0.0, cond_method="fa_invariant",
                  frame_avg_angles=[0.0, 90.0, 180.0, 270.0], batch_size=64,
                  n_samples=6337, seed=42)
    legacy = eval_FLAC.build_metrics_record({"T60": 1.0}, "c.ckpt", **kwargs)
    pinned = eval_FLAC.build_metrics_record({"T60": 1.0}, "c.ckpt",
                                            frame_avg_max_fwd_samples=64, **kwargs)
    legacy.pop("source_sha"), pinned.pop("source_sha")
    assert list(legacy) == list(pinned), "key ORDER must not change either"
    assert legacy == pinned
    assert legacy["frame_avg_fwd_cap"] == 64


@pytest.mark.parametrize("bad", [True, 0, -1, 32.0, "32"])
def test_evaluate_model_rejects_a_bad_cap_before_any_work(bad, tmp_path, monkeypatch):
    """The cap is validated before file/model/dataloader work: a misconfigured
    evaluation must never reach a GPU (the resolve_rotation_plan precedent)."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"))
    with pytest.raises(ValueError, match="frame_avg_max_fwd_samples"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", cond_method="fa_invariant",
            frame_avg_max_fwd_samples=bad)


# ---- CLI: parse -> call kwarg -> record ----------------------------------- #
def test_cli_exposes_the_frame_avg_max_fwd_samples_flag():
    bad = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--frame-avg-max-fwd-samples", "abc"],
        capture_output=True, cwd=str(_REPO))
    assert bad.returncode == 2 and b"invalid int value" in bad.stderr

    good = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--frame-avg-max-fwd-samples", "64"],
        capture_output=True, cwd=str(_REPO))
    assert good.returncode == 2                        # still missing required args...
    assert b"unrecognized arguments" not in good.stderr  # ...but the flag parsed
    assert b"invalid int value" not in good.stderr


def test_cli_default_is_the_pinned_evaluation_cap():
    """`--frame-avg-max-fwd-samples` defaults to 64: the value every historical
    evaluation ran under, and the common cap both exp_14 arms are evaluated at."""
    import argparse
    import inspect

    src = inspect.getsource(eval_FLAC)
    assert "--frame-avg-max-fwd-samples" in src
    assert "frame_avg_max_fwd_samples" in inspect.signature(
        eval_FLAC.evaluate_model).parameters

    # the declared default, read off a parser built the same way as the CLI's
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-avg-max-fwd-samples", type=int, default=64)
    assert parser.parse_args([]).frame_avg_max_fwd_samples == 64
    block = src.split('parser.add_argument("--frame-avg-max-fwd-samples"')[1].split(")\n")[0]
    assert "type=int" in block and "default=64" in block
    main_block = src.split('if __name__ == "__main__":')[1]
    assert "frame_avg_max_fwd_samples=args.frame_avg_max_fwd_samples" in main_block


class _OneBatchLoader:
    """A dataloader yielding exactly one real batch, so the eval loop's
    conditioning call actually executes."""

    def __init__(self, batch):
        self._batch = batch
        self.dataset = [None] * len(batch[1])

    def __iter__(self):
        return iter([self._batch])


def _stub_eval_stack(monkeypatch, tmp_path, loader):
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1, "training": {"use_ema": False},
    }))
    dataset_cfg = tmp_path / "dataset.json"
    dataset_cfg.write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {}}, str(ckpt))

    class _FakeModule:
        def __init__(self):
            self.diffusion = types.SimpleNamespace(
                model=types.SimpleNamespace(diffusion_objective="rectified_flow"),
                pretransform=None, conditioner=_build_cond(), io_channels=1,
                dist_shift=None, get_conditioning_inputs=lambda cond: {})
            self.device = "cpu"

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda cfg: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], []),
                                          diffusion_objective="rectified_flow"))
    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config", lambda cfg, model: _FakeModule())
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: loader)
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config",
        lambda *a, **k: types.SimpleNamespace(
            update_metrics=lambda *a, **k: None,
            compute_metrics=lambda split: {"T60": 1.0}))
    import src.inference.sampling as sampling
    monkeypatch.setattr(sampling, "sample_discrete_euler",
                        lambda model, noise, steps, **kw: noise)
    return model_cfg, dataset_cfg, ckpt


def _eval_batch(n=2):
    md = []
    for i in range(n):
        m = _md(i)
        m["scene"] = f"Toy/Toy_idx_{i}"
        md.append(m)
    return torch.zeros(n, 1, 64), md


@pytest.mark.parametrize("cap,plan", [(None, [2, 6]), (2, [2, 2, 2, 2]), (4, [2, 4, 2])])
def test_eval_cli_cap_reaches_the_conditioning_call_and_the_record(
        cap, plan, tmp_path, monkeypatch):
    """parse -> call kwarg -> record, on the REAL eval loop: the cap drives the
    executed partition and the same number lands in the metrics JSON."""
    batch = _eval_batch(2)
    loader = _OneBatchLoader(batch)
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path, loader)

    seen = {}
    real = yr.invariant_conditioning

    def _spy(conditioner, metadata, device, angles=yr.DEFAULT_FRAME_ANGLES, **kwargs):
        seen.update(kwargs)
        seen["conditioner"] = conditioner
        return real(conditioner, metadata, device, angles, **kwargs)

    monkeypatch.setattr(eval_FLAC, "invariant_conditioning", _spy)

    kwargs = {} if cap is None else {"frame_avg_max_fwd_samples": cap}
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        batch_size=2, device="cpu", eval_name="exp14_capplumbing", seed=42,
        cond_method="fa_invariant", **kwargs)

    assert seen["max_fwd_samples"] == (64 if cap is None else cap)
    assert seen["conditioner"].conditioners["source_vit"].batch_sizes == plan

    written = json.loads(
        (tmp_path / "toy_metrics_1_1.0_exp14_capplumbing_fa_invariant_a4.json").read_text())
    assert written["frame_avg_fwd_cap"] == (64 if cap is None else cap)
    assert written["cond_method"] == "fa_invariant"
    assert written["orbit_execution"] == "batched"
