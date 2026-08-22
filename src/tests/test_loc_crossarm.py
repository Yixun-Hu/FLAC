"""exp_20 cross-arm machinery (`src/localization/crossarm.py`).

The admission tests build real (tiny) checkpoints rather than mocking torch:
the contract is about what a file on disk contains, so a fixture that cannot be
loaded by ``torch.load`` would prove nothing about the gate.
"""
import copy
import json
import os

import numpy as np
import pytest
import torch

from src.localization import crossarm as ca

# --------------------------------------------------------------------------- #
# checkpoint fixtures
# --------------------------------------------------------------------------- #
_CONFIG = {
    "model": {"conditioning": {"configs": [{"id": "a", "config": {}},
                                           {"id": "b", "config": {"gradient_checkpointing": True}}]}},
    "sample_size": 64,
    "training": {"use_ema": True, "cfg_dropout_prob": 0.1},
}


def _state_dict(n=3, partial_ema=False, wrong_shape=False, wrong_dtype=False):
    online = {f"diffusion.model.block{i}.weight": torch.zeros(2, 3) for i in range(n)}
    ema = {f"diffusion_ema.ema_model.block{i}.weight": torch.zeros(2, 3) for i in range(n)}
    if partial_ema:
        ema.pop(f"diffusion_ema.ema_model.block{n - 1}.weight")
    if wrong_shape:
        ema[f"diffusion_ema.ema_model.block0.weight"] = torch.zeros(4, 3)
    if wrong_dtype:
        ema[f"diffusion_ema.ema_model.block0.weight"] = torch.zeros(2, 3, dtype=torch.float64)
    other = {"diffusion.conditioner.x": torch.zeros(1),
             "diffusion_ema.initted": torch.tensor(True),
             "diffusion_ema.step": torch.tensor(40000)}
    return {**online, **ema, **other}


def _write_ckpt(path, config=None, step=40000, **state_kwargs):
    torch.save({"global_step": step, "epoch": 7,
                "state_dict": _state_dict(**state_kwargs),
                "model_config": copy.deepcopy(config if config is not None else _CONFIG)},
               str(path))
    return str(path)


def _write_config(path, config=None):
    payload = copy.deepcopy(config if config is not None else _CONFIG)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


@pytest.fixture()
def arm_files(tmp_path):
    return (_write_ckpt(tmp_path / "arm.ckpt"), _write_config(tmp_path / "arm.json"))


# --------------------------------------------------------------------------- #
# B2 -- checkpoint admission
# --------------------------------------------------------------------------- #
def test_admission_accepts_a_clean_checkpoint(arm_files):
    ckpt, config = arm_files
    record = ca.admit_checkpoint(ckpt, config, arm="P1", expect_step=40000,
                                 check_load_integrity=False)
    assert record["admitted"] is True and record["reasons"] == []
    assert record["arm"] == "P1" and record["global_step"] == 40000
    assert len(record["sha256"]) == 64 and len(record["config_sha256"]) == 64
    assert record["ema_key_count"] == 3 and record["online_model_key_count"] == 3
    assert record["embedded_config_canonical_sha256"] == record["config_canonical_sha256"]
    assert record["load_integrity"]["checked"] is False
    assert record["cond_method"] == "vanilla"


def test_admission_refuses_a_partial_ema_family(arm_files, tmp_path):
    _ckpt, config = arm_files
    partial = _write_ckpt(tmp_path / "partial.ckpt", partial_ema=True)
    record = ca.admit_checkpoint(partial, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("mirror" in reason.lower() for reason in record["reasons"]), record["reasons"]
    assert record["ema_key_count"] is None


@pytest.mark.parametrize("kwargs,fragment", [({"wrong_shape": True}, "shape"),
                                             ({"wrong_dtype": True}, "dtype")])
def test_admission_refuses_a_drifted_ema_family(tmp_path, arm_files, kwargs, fragment):
    _ckpt, config = arm_files
    drifted = _write_ckpt(tmp_path / "drift.ckpt", **kwargs)
    record = ca.admit_checkpoint(drifted, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any(fragment in reason for reason in record["reasons"]), record["reasons"]


@pytest.mark.parametrize("step", [39999, 40001, 0])
def test_admission_refuses_a_step_mismatch(tmp_path, arm_files, step):
    _ckpt, config = arm_files
    wrong = _write_ckpt(tmp_path / f"step{step}.ckpt", step=step)
    record = ca.admit_checkpoint(wrong, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("global_step" in reason for reason in record["reasons"])


def test_admission_refuses_a_step_that_is_not_a_plain_int(tmp_path, arm_files):
    """40000.0 and True both equal 40000 under int(); neither IS the endpoint."""
    _ckpt, config = arm_files
    for value in (40000.0, True):
        path = _write_ckpt(tmp_path / f"step_{type(value).__name__}.ckpt", step=value)
        record = ca.admit_checkpoint(path, config, arm="P1", check_load_integrity=False)
        assert record["admitted"] is False
        assert any("plain int" in reason for reason in record["reasons"]), record["reasons"]


def test_admission_refuses_a_config_the_checkpoint_was_not_trained_with(tmp_path, arm_files):
    ckpt, _config = arm_files
    other = copy.deepcopy(_CONFIG)
    other["training"]["cfg_dropout_prob"] = 0.2
    record = ca.admit_checkpoint(ckpt, _write_config(tmp_path / "other.json", other),
                                 arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("canonical" in reason for reason in record["reasons"]), record["reasons"]


def test_admission_is_type_sensitive_about_the_config(tmp_path, arm_files):
    """True == 1 in Python; the canonical bytes are `true` and `1`."""
    ckpt, _config = arm_files
    coerced = copy.deepcopy(_CONFIG)
    coerced["training"]["use_ema"] = 1
    record = ca.admit_checkpoint(ckpt, _write_config(tmp_path / "coerced.json", coerced),
                                 arm="P1", check_load_integrity=False)
    assert record["admitted"] is False


# --------------------------------------------------------------------------- #
# B2 -- the arm IDENTITY embedded in the checkpoint
# --------------------------------------------------------------------------- #
def _fa_config():
    config = copy.deepcopy(_CONFIG)
    config["training"]["cond_method"] = "fa_invariant"
    config["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    return config


def _yaw_config():
    config = copy.deepcopy(_CONFIG)
    config["training"]["yaw_aug"] = {"enabled": True, "img_w": 512, "seed": 42}
    return config


def test_admission_reads_the_conditioning_method_out_of_the_checkpoint(tmp_path):
    """The embedded training config names the arm's conditioning method, so the
    refusal does not have to rest on the manifest alone."""
    fa = _write_ckpt(tmp_path / "bf.ckpt", config=_fa_config())
    record = ca.admit_checkpoint(fa, _write_config(tmp_path / "bf.json", _fa_config()),
                                 arm="BF", check_load_integrity=False)
    assert record["admitted"] is True
    assert record["cond_method"] == "fa_invariant"
    assert record["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]


def test_admission_refuses_an_arm_whose_embedded_identity_is_wrong(tmp_path):
    vanilla_ckpt = _write_ckpt(tmp_path / "p1.ckpt")
    vanilla_config = _write_config(tmp_path / "p1.json")
    # a vanilla checkpoint offered as the frame-averaged arm
    record = ca.admit_checkpoint(vanilla_ckpt, vanilla_config, arm="BF",
                                 check_load_integrity=False)
    assert record["admitted"] is False
    assert any("cond_method" in reason for reason in record["reasons"]), record["reasons"]

    # the frame-averaged checkpoint offered as the vanilla arm
    fa_ckpt = _write_ckpt(tmp_path / "bf.ckpt", config=_fa_config())
    fa_config = _write_config(tmp_path / "bf.json", _fa_config())
    record = ca.admit_checkpoint(fa_ckpt, fa_config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("cond_method" in reason for reason in record["reasons"])

    # the yaw arm must carry its augmentation block, and the others must not
    yaw_ckpt = _write_ckpt(tmp_path / "yaw.ckpt", config=_yaw_config())
    yaw_config = _write_config(tmp_path / "yaw.json", _yaw_config())
    assert ca.admit_checkpoint(yaw_ckpt, yaw_config, arm="YAW",
                               check_load_integrity=False)["admitted"] is True
    record = ca.admit_checkpoint(yaw_ckpt, yaw_config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("yaw_aug" in reason for reason in record["reasons"])
    record = ca.admit_checkpoint(vanilla_ckpt, vanilla_config, arm="YAW",
                                 check_load_integrity=False)
    assert record["admitted"] is False
    assert any("yaw_aug" in reason for reason in record["reasons"])


def test_admission_record_is_json_serialisable_and_names_its_inputs(arm_files):
    ckpt, config = arm_files
    record = ca.admit_checkpoint(ckpt, config, arm="P1", check_load_integrity=False)
    text = json.dumps(record, sort_keys=True)
    assert ckpt in text and config in text
    assert record["expect_step"] == 40000
    assert record["created_utc"].endswith("+00:00")


def _exp15_kit():
    import importlib.util
    import pathlib

    kit = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
           "exp_15_yaw_aug_claude" / "yaw_aug_record_control.py")
    if not kit.is_file():
        pytest.skip("exp_15 kit not present")
    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", kit)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hashing_reads_the_held_inode_not_whatever_the_name_now_points_at(tmp_path):
    """r1 review F5: hashing by PATH and holding a descriptor are two different
    lookups. If the name is re-pointed after the open, path-hashing measures the
    new file while the load (and every later identity check) sees the old one."""
    real = _write_ckpt(tmp_path / "real.ckpt")
    decoy = _write_ckpt(tmp_path / "decoy.ckpt", step=1)
    real_sha, decoy_sha = ca.sha256_file(real), ca.sha256_file(decoy)
    assert real_sha != decoy_sha

    fd = os.open(real, os.O_RDONLY)
    try:
        os.replace(decoy, real)              # the NAME moves; the descriptor does not
        assert ca._sha256_fd(fd) == real_sha, "the held inode was not what was hashed"
        assert ca.sha256_file(real) == decoy_sha, "path hashing would have measured the decoy"
    finally:
        os.close(fd)


def test_snapshot_hashes_through_the_descriptor(tmp_path, monkeypatch):
    """The snapshot must not fall back to path hashing."""
    path = _write_ckpt(tmp_path / "held.ckpt")
    expected = ca.sha256_file(path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("snapshot_checkpoint hashed by path")

    monkeypatch.setattr(ca, "sha256_file", refuse)
    _checkpoint, digest, identity = ca.snapshot_checkpoint(path)
    assert digest == expected and identity["inode"] > 0


@pytest.mark.parametrize("kwargs,fragment", [
    ({"partial_ema": True}, "mirror"),
    ({"wrong_shape": True}, "shape"),
    ({"wrong_dtype": True}, "dtype"),
])
def test_both_implementations_refuse_the_same_ema_pathologies(kwargs, fragment):
    """The port must share exp_15's REFUSALS, not only its happy path."""
    reference = _exp15_kit()
    state = _state_dict(**kwargs)
    with pytest.raises(ValueError) as ours:
        ca.summarize_ema(state)
    with pytest.raises(ValueError) as theirs:
        reference.summarize_ema(state)
    assert fragment in str(ours.value) and fragment in str(theirs.value)


def test_both_implementations_refuse_an_extra_ema_key():
    reference = _exp15_kit()
    state = _state_dict()
    state["diffusion_ema.ema_model.stray.weight"] = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="mirror"):
        ca.summarize_ema(state)
    with pytest.raises(ValueError, match="mirror"):
        reference.summarize_ema(state)


def test_both_implementations_refuse_a_missing_family():
    reference = _exp15_kit()
    online_only = {k: v for k, v in _state_dict().items()
                   if not k.startswith(ca.EMA_WEIGHT_PREFIX)}
    for module in (ca, reference):
        with pytest.raises(ValueError, match="EMA"):
            module.summarize_ema(online_only)
    ema_only = {k: v for k, v in _state_dict().items()
                if not k.startswith(ca.ONLINE_MODEL_PREFIX)}
    for module in (ca, reference):
        with pytest.raises(ValueError, match="online"):
            module.summarize_ema(ema_only)


def test_both_implementations_snapshot_the_same_facts(tmp_path):
    reference = _exp15_kit()
    path = _write_ckpt(tmp_path / "snap.ckpt")
    _ours, our_sha, our_identity = ca.snapshot_checkpoint(path)
    _theirs, their_sha, their_identity = reference.snapshot_checkpoint(path)
    assert our_sha == their_sha and our_identity == their_identity


def test_admission_primitives_agree_with_the_exp15_kit(arm_files):
    """The ported semantics must BE the exp_15 semantics, not merely resemble
    them: both implementations are run over the same fixture."""
    import importlib.util
    import pathlib

    kit = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
           "exp_15_yaw_aug_claude" / "yaw_aug_record_control.py")
    if not kit.is_file():
        pytest.skip("exp_15 kit not present")
    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", kit)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    ckpt, _config = arm_files
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert ca.canonical_bytes(_CONFIG) == reference.canonical_bytes(_CONFIG)
    assert ca.canonical_sha256(_CONFIG) == reference.canonical_sha256(_CONFIG)
    assert ca.summarize_ema(payload["state_dict"]) == reference.summarize_ema(
        payload["state_dict"])
    for bad in ({1: "int key"}, {"x": float("nan")}, {"x": {1, 2}}):
        with pytest.raises(ValueError):
            ca.canonical_bytes(bad)


def test_admission_refuses_a_checkpoint_that_moved_while_it_was_read(tmp_path, monkeypatch):
    ckpt = _write_ckpt(tmp_path / "moving.ckpt")
    config = _write_config(tmp_path / "moving.json")
    original = ca.safe_load_checkpoint

    def replace_then_load(path):
        payload = original(path)
        _write_ckpt(tmp_path / "moving.ckpt", step=1)      # the file changes mid-read
        return payload

    monkeypatch.setattr(ca, "safe_load_checkpoint", replace_then_load)
    record = ca.admit_checkpoint(ckpt, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("changed" in r or "replaced" in r for r in record["reasons"]), record["reasons"]


def test_load_integrity_uses_the_registered_whitelist(monkeypatch, arm_files):
    """The registered contract (eval_FLAC.LOAD_WHITELIST_PREFIXES) is 0 missing /
    0 STRAY unexpected: every real checkpoint carries diffusion_ema bookkeeping
    and the training loss module, and refusing those would refuse every arm."""
    from eval_FLAC import LOAD_WHITELIST_PREFIXES

    assert LOAD_WHITELIST_PREFIXES == ("diffusion_ema.", "losses.")
    benign = ["diffusion_ema.initted", "diffusion_ema.step", "losses.losses.0.weight"]
    verdict = ca.classify_load_integrity(missing=[], unexpected=benign)
    assert verdict["n_missing"] == 0 and verdict["n_stray"] == 0
    assert verdict["n_whitelisted"] == 3 and verdict["clean"] is True

    stray = ca.classify_load_integrity(missing=[], unexpected=benign + ["model.blocks.9.w"])
    assert stray["n_stray"] == 1 and stray["clean"] is False
    assert ca.classify_load_integrity(missing=["model.x"], unexpected=[])["clean"] is False

    ckpt, config = arm_files
    monkeypatch.setattr(ca, "load_integrity",
                        lambda model_config, state_dict: {"checked": True, "missing": [],
                                                          "unexpected": benign})
    record = ca.admit_checkpoint(ckpt, config, arm="P1", check_load_integrity=True)
    assert record["admitted"] is True, record["reasons"]
    assert record["load_integrity"]["n_whitelisted"] == 3


# --------------------------------------------------------------------------- #
# B1 -- FA protocol binding: the chunk plan is DECLARED, not inherited
# --------------------------------------------------------------------------- #
def _fa_metadata(n=3, img_w=16, height=8):
    """Minimal metadata the frame-average path accepts: a depth panorama and the
    pose fields the ViT conditioners read."""
    out = []
    generator = torch.Generator().manual_seed(11)
    for i in range(n):
        depth = torch.randn(3, height, img_w, generator=generator)
        out.append({
            "depth": depth,
            "source": torch.tensor([1.0 + i, 2.0, 0.5]),
            "source_vit": torch.tensor([1.0 + i, 2.0, 0.5]),
            "context_poses": torch.tensor([[0.5, 1.5, 0.4], [2.0, 0.5, 0.6]]),
            "context_poses_vit": torch.tensor([[0.5, 1.5, 0.4], [2.0, 0.5, 0.6]]),
            "scene": f"room{i}",
        })
    return out


class _RecordingConditioner:
    """A deterministic stand-in that records the batch size of every forward."""

    def __init__(self, ids=("source_vit", "context_poses_vit", "context_audio")):
        self.ids, self.calls = ids, []

    def __call__(self, metadata, device, only_ids=None):
        ids = only_ids or self.ids
        self.calls.append({"batch": len(metadata), "ids": tuple(ids)})
        out = {}
        for name in ids:
            rows = []
            for md in metadata:
                pose = md.get("source_vit" if "source" in name else "context_poses_vit")
                value = torch.as_tensor(pose).reshape(-1)[:3].sum() + float(md["depth"].sum())
                rows.append(torch.stack([value, value * 2.0]))
            out[name] = [torch.stack(rows), torch.ones(len(metadata), 1)]
        return out


def test_per_angle_cap_gives_one_forward_per_angle():
    """cap = candidate micro-batch => angles_per_chunk == 1 (plan B1)."""
    from src.data.yaw_rotation import invariant_conditioning

    metadata = _fa_metadata(n=3)
    per_angle = _RecordingConditioner()
    invariant_conditioning(per_angle, metadata, "cpu", ca.FA_ANGLES,
                           max_fwd_samples=ca.fa_max_fwd_samples(metadata))
    orbit_calls = [c for c in per_angle.calls if c["batch"] != 3 or c["ids"] != per_angle.ids]
    assert [c["batch"] for c in orbit_calls] == [3, 3, 3], per_angle.calls

    batched = _RecordingConditioner()
    invariant_conditioning(batched, metadata, "cpu", ca.FA_ANGLES)      # module default 64
    orbit_calls = [c for c in batched.calls if c["ids"] != batched.ids]
    assert [c["batch"] for c in orbit_calls] == [9], batched.calls


def test_fa_conditioning_helper_matches_the_reference_orbit():
    """The driver's FA call must equal the per-angle reference accumulation."""
    from src.data.yaw_rotation import invariant_conditioning

    metadata = _fa_metadata(n=2)
    driver = ca.fa_conditioning(_RecordingConditioner(), metadata, "cpu", ca.FA_ANGLES)
    reference = invariant_conditioning(_RecordingConditioner(), metadata, "cpu", ca.FA_ANGLES,
                                       max_fwd_samples=len(metadata))
    for key in ("source_vit", "context_poses_vit"):
        assert torch.equal(driver[key][0], reference[key][0])


def test_fa_run_state_declares_every_locked_field():
    state = ca.fa_run_state(cond_method="fa_invariant", frame_avg_angles=[0, 90, 180, 270],
                            rotate_deg=0.0, cond_autocast="default")
    assert state["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
    assert state["frame_avg_chunk_plan"] == ca.FA_CHUNK_PLAN == "per_angle"
    assert state["orbit_execution"] == "per_angle"
    assert set(ca.FA_LOCKED_FIELDS) <= set(state)


def test_fa_registration_lock_detects_every_mutation():
    state = ca.fa_run_state(cond_method="fa_invariant", frame_avg_angles=ca.FA_ANGLES,
                            rotate_deg=0.0, cond_autocast="default")
    manifest = dict(state)
    assert ca.fa_reasons(manifest, state) == []
    for field, mutated in (("frame_avg_angles", [0.0, 120.0, 240.0]),
                           ("rotate_deg", 90.0),
                           ("cond_autocast", "off"),
                           ("frame_avg_chunk_plan", "batched"),
                           ("cond_method", "vanilla")):
        broken = dict(manifest, **{field: mutated})
        reasons = ca.fa_reasons(broken, state)
        assert any(field in reason for reason in reasons), (field, reasons)
    missing = {k: v for k, v in manifest.items() if k != "frame_avg_angles"}
    assert any("frame_avg_angles" in r for r in ca.fa_reasons(missing, state))


# --------------------------------------------------------------------------- #
# B1(c) -- the refusal matrix, and what it can honestly bind to
# --------------------------------------------------------------------------- #
def test_cond_method_binds_to_the_checkpoint_when_the_checkpoint_says(tmp_path):
    fa_ckpt = torch.load(_write_ckpt(tmp_path / "bf.ckpt", config=_fa_config()),
                         map_location="cpu", weights_only=True)
    vanilla_ckpt = torch.load(_write_ckpt(tmp_path / "p1.ckpt"), map_location="cpu",
                              weights_only=True)

    ok = ca.cond_method_binding(fa_ckpt, "fa_invariant")
    assert ok["binding"] == "checkpoint" and ok["reasons"] == []
    bad = ca.cond_method_binding(fa_ckpt, "vanilla")
    assert bad["reasons"] and "fa_invariant" in bad["reasons"][0]
    bad = ca.cond_method_binding(vanilla_ckpt, "fa_invariant")
    assert bad["reasons"] and "vanilla" in bad["reasons"][0]
    assert ca.cond_method_binding(vanilla_ckpt, "vanilla")["reasons"] == []


def test_cond_method_binding_is_honest_about_a_stripped_release():
    """The released EMA checkpoint carries no model_config, so the method is not
    detectable from the file. Without a VERIFIED manifest that is `unbound`, not
    a manifest binding (r1 review F6)."""
    verdict = ca.cond_method_binding({"state_dict": {}}, "vanilla")
    assert verdict["binding"] == "unbound" and verdict["reasons"] == []
    assert verdict["stamped"] is True and "no model_config" in verdict["note"]
    with_manifest = ca.cond_method_binding({"state_dict": {}}, "vanilla",
                                           manifest={"cond_method": "vanilla"},
                                           manifest_verified=True)
    assert with_manifest["binding"] == "manifest"


# --------------------------------------------------------------------------- #
# B1(b) -- the fa parity gate
# --------------------------------------------------------------------------- #
def test_fa_parity_gate_passes_on_matched_paths():
    metadata = _fa_metadata(n=2)
    verdict = ca.fa_parity_gate(lambda: _RecordingConditioner(), metadata, device="cpu",
                                angles=ca.FA_ANGLES, autocast=False)
    assert verdict["match"] is True and verdict["max_abs_diff"] == 0.0
    assert verdict["bitwise"] is True and verdict["autocast"] is False
    assert set(verdict["ids"]) >= {"source_vit", "context_poses_vit"}


def test_fa_parity_gate_detects_a_divergent_replay():
    """A replay that averages a different orbit is exactly what the gate exists
    to catch."""
    metadata = _fa_metadata(n=2)
    verdict = ca.fa_parity_gate(lambda: _RecordingConditioner(), metadata, device="cpu",
                                angles=ca.FA_ANGLES, replay_angles=(0.0, 90.0),
                                autocast=False)
    assert verdict["match"] is False and verdict["max_abs_diff"] > 0.0


def test_fa_parity_gate_records_a_tolerance_under_autocast():
    metadata = _fa_metadata(n=2)
    verdict = ca.fa_parity_gate(lambda: _RecordingConditioner(), metadata, device="cpu",
                                angles=ca.FA_ANGLES, autocast=True, tolerance=1e-3)
    assert verdict["autocast"] is True and verdict["tolerance"] == 1e-3
    assert verdict["match"] is True and verdict["max_abs_diff"] <= 1e-3


# --------------------------------------------------------------------------- #
# the driver side of B1
# --------------------------------------------------------------------------- #
def test_driver_conditioning_call_routes_fa_through_the_declared_plan():
    import eval_localization as el

    metadata = _fa_metadata(n=3)
    recorder = _RecordingConditioner()
    out = el.conditioning_call("fa_invariant", recorder, metadata, "cpu", ca.FA_ANGLES)
    orbit = [c for c in recorder.calls if c["ids"] != recorder.ids]
    assert [c["batch"] for c in orbit] == [3, 3, 3]          # per angle, not one chunk
    assert "source_vit" in out

    plain = _RecordingConditioner()
    el.conditioning_call("vanilla", plain, metadata, "cpu", ca.FA_ANGLES)
    assert [c["batch"] for c in plain.calls] == [3]          # one ordinary forward


def test_driver_refuses_a_checkpoint_conditioned_the_way_it_was_not_trained(tmp_path):
    import eval_localization as el

    fa_ckpt = _write_ckpt(tmp_path / "bf.ckpt", config=_fa_config())
    args = el.parse_args(["--model-config", _write_config(tmp_path / "bf.json", _fa_config()),
                          "--dataset-config", "d.json", "--ckpt-path", fa_ckpt,
                          "--agree-ckpt", "a.pt", "--cond-method", "vanilla"])
    with pytest.raises(SystemExit, match="fa_invariant"):
        el.load_checkpoint_and_validate(args, _fa_config())

    ok = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                        "--ckpt-path", fa_ckpt, "--agree-ckpt", "a.pt",
                        "--cond-method", "fa_invariant"])
    assert el.load_checkpoint_and_validate(ok, _fa_config()) is not None
    assert ok.cond_method_binding["binding"] == "checkpoint"


def test_driver_refuses_a_vanilla_checkpoint_asked_to_frame_average(tmp_path):
    import eval_localization as el

    vanilla = _write_ckpt(tmp_path / "p1.ckpt")
    args = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                          "--ckpt-path", vanilla, "--agree-ckpt", "a.pt",
                          "--cond-method", "fa_invariant"])
    with pytest.raises(SystemExit, match="vanilla"):
        el.load_checkpoint_and_validate(args, _CONFIG)


def test_driver_fa_registration_gate_refuses_every_mutation(tmp_path):
    import eval_localization as el

    args = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt",
                          "--cond-method", "fa_invariant", "--cond-autocast", "default",
                          "--rotate-deg", "0"])
    state = el.fa_protocol_state(args)
    assert state["frame_avg_chunk_plan"] == "per_angle"
    el.assert_fa_registration(dict(state), args)                     # matched manifest passes

    for field, mutated in (("frame_avg_angles", [0.0, 180.0]), ("rotate_deg", 15.0),
                           ("cond_autocast", "off"), ("frame_avg_chunk_plan", "batched")):
        with pytest.raises(SystemExit, match=field):
            el.assert_fa_registration(dict(state, **{field: mutated}), args)
    with pytest.raises(SystemExit, match="frame_avg_angles"):
        el.assert_fa_registration({k: v for k, v in state.items()
                                   if k != "frame_avg_angles"}, args)


def test_vanilla_runs_are_untouched_by_the_fa_gate():
    """exp_18's committed manifests lock cond_method='vanilla' and carry no FA
    block; the new gate must be inert for them."""
    import eval_localization as el

    args = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt"])
    assert args.cond_method == "vanilla"
    assert el.fa_protocol_state(args) is None
    el.assert_fa_registration({"cond_method": "vanilla"}, args)       # no FA fields needed


def test_provenance_records_the_declared_plan_and_the_binding(tmp_path):
    """A vanilla row keeps exp_18's exact orbit fields; an FA row states the plan."""
    import eval_localization as el

    vanilla = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                             "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt",
                             "--num-samples", "8"])
    record = el.build_provenance(vanilla, "ck", "ag", "split", "ema", 1)
    assert record["orbit_execution"] == "n/a" and record["frame_avg_fwd_cap"] is None
    # the FA-only key stays absent on a vanilla row; the binding is on every row
    assert "frame_avg_chunk_plan" not in record
    assert "cond_method_binding" in record

    fa = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                        "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt",
                        "--num-samples", "8", "--cond-method", "fa_invariant"])
    fa.cond_method_binding = {"binding": "checkpoint", "checkpoint_cond_method": "fa_invariant"}
    record = el.build_provenance(fa, "ck", "ag", "split", "ema", 1)
    assert record["orbit_execution"] == "per_angle"
    assert record["frame_avg_fwd_cap"] == "candidate_micro_batch"
    assert record["frame_avg_chunk_plan"] == "per_angle"
    assert record["cond_method_binding"]["binding"] == "checkpoint"
    assert record["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]


# --------------------------------------------------------------------------- #
# B3 -- the paired-inference gate
# --------------------------------------------------------------------------- #
def _facts(arm="P1", **over):
    facts = {
        "arm": arm, "regime": "K8", "seed": 42,
        "query_ids": ["q0", "q1", "q2"],
        "context_stream_digest": "c" * 64,
        "split_hash": "s" * 64,
        "split_file_sha256": "f" * 64,
        "candidate_manifest_sha256": "m" * 64,
        "loader": {"batch_size": 4, "num_workers": 4, "shuffle": False, "drop_last": False},
        "noise_keys": {"q0": [1, 2], "q1": [3, 4], "q2": [5, 6]},
    }
    facts.update(over)
    return facts


def test_pairing_accepts_arms_that_scored_the_same_queries():
    verdict = ca.validate_pairing([_facts("P1"), _facts("BF"), _facts("YAW")])
    assert verdict["paired"] is True and verdict["mismatches"] == []
    assert verdict["n_arms"] == 3 and verdict["reference_arm"] == "BF"
    assert set(verdict["fields_checked"]) == set(ca.PAIRING_FIELDS)
    assert verdict["n_queries"] == 3


@pytest.mark.parametrize("field,mutation", [
    ("query_ids", ["q0", "q2", "q1"]),                       # same set, different ORDER
    ("context_stream_digest", "d" * 64),
    ("split_hash", "x" * 64),
    ("split_file_sha256", "y" * 64),
    ("candidate_manifest_sha256", "z" * 64),
    ("loader", {"batch_size": 8, "num_workers": 4, "shuffle": False, "drop_last": False}),
    ("noise_keys", {"q0": [1, 2], "q1": [3, 4], "q2": [7, 6]}),
])
def test_pairing_detects_every_field(field, mutation):
    verdict = ca.validate_pairing([_facts("P1"), _facts("BF", **{field: mutation})])
    assert verdict["paired"] is False
    assert any(m["field"] == field for m in verdict["mismatches"]), verdict["mismatches"]
    assert "unpaired" in verdict["fallback"]


def test_pairing_refuses_fewer_than_two_arms_and_mixed_cells():
    with pytest.raises(ValueError, match="two arms"):
        ca.validate_pairing([_facts("P1")])
    verdict = ca.validate_pairing([_facts("P1"), _facts("BF", seed=43)])
    assert verdict["paired"] is False
    assert any(m["field"] == "cell" for m in verdict["mismatches"])


def test_pairing_facts_are_read_from_published_artifacts(tmp_path):
    rows = tmp_path / "rows.jsonl"
    with open(rows, "w") as handle:
        for i in range(3):
            handle.write(json.dumps({"query_id": f"q{i}", "room_id": "R0",
                                     "noise_keys": [i, i + 1]}) + "\n")
    summary = tmp_path / "summary.json"
    with open(summary, "w") as handle:
        json.dump({"provenance": {"seed": 42, "context_stream_digest": "c" * 64,
                                  "split_hash": "s" * 64, "split_file_sha256": "f" * 64,
                                  "candidate_manifest_sha256": "m" * 64,
                                  "batch_size": 4, "num_workers": 4,
                                  "loader_shuffle": False, "loader_drop_last": False}},
                  handle)
    facts = ca.pairing_facts(str(rows), str(summary), arm="P1", regime="K8")
    assert facts["query_ids"] == ["q0", "q1", "q2"] and facts["seed"] == 42
    assert facts["loader"] == {"batch_size": 4, "num_workers": 4, "shuffle": False,
                               "drop_last": False}
    assert facts["noise_keys"]["q1"] == [1, 2]
    assert ca.validate_pairing([facts, dict(facts, arm="BF")])["paired"] is True


# --------------------------------------------------------------------------- #
# B3 -- seeds are replicates, and the confirmatory family is exactly four tests
# --------------------------------------------------------------------------- #
def test_seed_aggregation_is_per_query_then_clustered():
    per_seed = {
        42: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0},
             {"query_id": "q1", "room_id": "R1", "top1": 0.0, "e_loc": 2.0}],
        43: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0},
             {"query_id": "q1", "room_id": "R1", "top1": 1.0, "e_loc": 1.0}],
        44: [{"query_id": "q0", "room_id": "R0", "top1": 0.0, "e_loc": 3.0},
             {"query_id": "q1", "room_id": "R1", "top1": 1.0, "e_loc": 1.0}],
    }
    records = ca.aggregate_seeds_per_query(per_seed)
    assert [r["query_id"] for r in records] == ["q0", "q1"]
    assert records[0]["top1"] == pytest.approx(2 / 3) and records[0]["n_seeds"] == 3
    assert records[0]["e_loc"] == pytest.approx(1.0)
    assert records[1]["top1"] == pytest.approx(2 / 3)
    assert records[1]["room_id"] == "R1"


def test_seed_aggregation_refuses_an_incomplete_cell():
    per_seed = {42: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0},
                     {"query_id": "q1", "room_id": "R0", "top1": 1.0, "e_loc": 0.0}],
                43: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0}]}
    with pytest.raises(ValueError, match="q1"):
        ca.aggregate_seeds_per_query(per_seed)
    mixed = {42: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0}],
             43: [{"query_id": "q0", "room_id": "ELSEWHERE", "top1": 1.0, "e_loc": 0.0}]}
    with pytest.raises(ValueError, match="room"):
        ca.aggregate_seeds_per_query(mixed)


def test_holm_family_is_exactly_the_four_registered_contrasts():
    assert len(ca.CONFIRMATORY_CONTRASTS) == 4
    assert set(ca.CONFIRMATORY_CONTRASTS) == {("BF", "P1", "K8"), ("BF", "P1", "K1"),
                                              ("YAW", "P1", "K8"), ("YAW", "P1", "K1")}
    p_values = {"BF_vs_P1_K8": 0.001, "BF_vs_P1_K1": 0.30,
                "YAW_vs_P1_K8": 0.02, "YAW_vs_P1_K1": 0.60}
    family = ca.build_holm_family(p_values)
    assert family["n_tests"] == 4 and family["endpoint"] == "top1"
    assert [t["label"] for t in family["tests"]][0] == "BF_vs_P1_K8"
    assert family["tests"][0]["rejected"] is True

    with pytest.raises(ValueError, match="exactly"):
        ca.build_holm_family({k: v for k, v in list(p_values.items())[:3]})
    with pytest.raises(ValueError, match="not a registered contrast"):
        ca.build_holm_family(dict(p_values, BF_vs_YAW_K8=0.01))


# --------------------------------------------------------------------------- #
# M5 / registration tooling: the per-arm manifest generator
# --------------------------------------------------------------------------- #
def _generator():
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
            "exp_20_loc_crossarm_claude" / "gen_arm_manifests.py")
    spec = importlib.util.spec_from_file_location("gen_arm_manifests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_R4_METRIC_MANIFEST = ("worklog/worklog_yixun/exp_18_loc_invert_claude/"
                       "loc_invert_R4_metric_registration.json")


def test_generator_emits_six_protocol_and_three_metric_manifests(tmp_path):
    gen = _generator()
    admissions = {arm: {"sha256": f"{arm.lower()}" * 8 + "0" * (64 - 8 * len(arm)),
                        "config_sha256": "c" * 64, "global_step": 40000, "arm": arm}
                  for arm in ("P1", "BF", "YAW")}
    written = gen.generate(str(tmp_path), admissions=admissions,
                           metric_source=_R4_METRIC_MANIFEST)
    protocol = [p for p in written if "_registration.json" in p and "metric" not in p]
    metric = [p for p in written if "metric_registration.json" in p]
    assert len(protocol) == 6 and len(metric) == 3
    for arm in ("P1", "BF", "YAW"):
        for regime in ("R2", "R2b"):
            assert any(f"{arm}_{regime}_registration.json" in p for p in protocol)


def test_generated_protocol_manifest_locks_every_exp18_field(tmp_path):
    gen = _generator()
    payload = gen.protocol_manifest("P1", "R2", ckpt_sha256="a" * 64,
                                    model_config_sha256="b" * 64)
    exp18 = json.load(open("worklog/worklog_yixun/exp_18_loc_invert_claude/"
                           "loc_invert_R2_registration.json"))
    assert set(exp18) <= set(payload), sorted(set(exp18) - set(payload))
    assert payload["seeds"] == [42, 43, 44] and payload["cond_method"] == "vanilla"
    assert payload["ckpt_sha256"] == "a" * 64
    assert payload["split_file_sha256"] == exp18["split_file_sha256"]
    assert payload["tau"] == exp18["tau"] and payload["agg"] == exp18["agg"]


def test_generated_bf_manifest_carries_the_fa_block_and_others_do_not(tmp_path):
    gen = _generator()
    bf = gen.protocol_manifest("BF", "R2", ckpt_sha256="a" * 64, model_config_sha256="b" * 64)
    assert bf["cond_method"] == "fa_invariant"
    for field in ca.FA_LOCKED_FIELDS:
        assert field in bf, field
    assert bf["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]
    assert bf["frame_avg_chunk_plan"] == "per_angle" and bf["rotate_deg"] == 0.0

    for arm in ("P1", "YAW"):
        payload = gen.protocol_manifest(arm, "R2", ckpt_sha256="a" * 64,
                                        model_config_sha256="b" * 64)
        assert payload["cond_method"] == "vanilla"
        assert "frame_avg_angles" not in payload and "frame_avg_chunk_plan" not in payload


def test_metric_manifest_inherits_the_scorer_subdocument_by_deep_equality(tmp_path):
    gen = _generator()
    source = json.load(open(_R4_METRIC_MANIFEST))
    payload = gen.metric_manifest("BF", metric_source=_R4_METRIC_MANIFEST,
                                  ckpt_sha256="a" * 64, protocol_digests={"R2": "d" * 64})
    assert payload["metric_config"] == source["metric_config"], "scorer subdoc drifted"
    assert payload["registerable"] == source["registerable"]
    assert payload["inherited_from"]["path"] == _R4_METRIC_MANIFEST
    assert payload["inherited_from"]["metric_config_canonical_sha256"] == \
        ca.canonical_sha256(source["metric_config"])
    assert payload["seeds"] == [42, 43, 44]
    assert payload["ckpt_sha256"] == "a" * 64
    assert payload["protocol_manifest_digests"] == {"R2": "d" * 64}
    assert "recalibration" in payload["transport_caveat"].lower() or \
        "calibrated" in payload["transport_caveat"].lower()


def test_metric_manifest_refuses_a_mutated_scorer_subdocument(tmp_path):
    gen = _generator()
    source = json.load(open(_R4_METRIC_MANIFEST))
    mutated = copy.deepcopy(source)
    mutated["metric_config"]["delta_max"] = 32
    path = tmp_path / "mutated.json"
    with open(path, "w") as handle:
        json.dump(mutated, handle)
    with pytest.raises(ValueError, match="delta_max|deep"):
        gen.metric_manifest("BF", metric_source=str(path), ckpt_sha256="a" * 64,
                            protocol_digests={"R2": "d" * 64},
                            expect_metric_config=source["metric_config"])


def test_committed_admission_records_are_admitted_and_bind_the_staged_checkpoints():
    """The three records this round produced are evidence for the ladder; they
    must say admitted, at the registered step, with the arm identity read from
    the file itself."""
    import pathlib

    folder = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
              "exp_20_loc_crossarm_claude")
    expected_cond = {"P1": "vanilla", "BF": "fa_invariant", "YAW": "vanilla"}
    for arm, cond in expected_cond.items():
        path = folder / f"loc_crossarm_admission_{arm}.json"
        if not path.is_file():
            pytest.skip(f"{path.name} not present")
        record = json.loads(path.read_text())
        assert record["admitted"] is True and record["reasons"] == []
        assert record["global_step"] == ca.REGISTERED_STEP
        assert record["cond_method"] == cond
        assert record["config_path"] == ca.ARMS[arm]["config_rel"]
        assert record["embedded_config_canonical_sha256"] == record["config_canonical_sha256"]
        assert record["ema_key_count"] == record["online_model_key_count"] == 210
        assert record["load_integrity"]["clean"] is True
        assert record["load_integrity"]["n_missing"] == 0
        assert record["load_integrity"]["n_stray"] == 0
        assert len(record["sha256"]) == 64
    shas = {arm: json.loads((folder / f"loc_crossarm_admission_{arm}.json").read_text())["sha256"]
            for arm in expected_cond}
    assert len(set(shas.values())) == 3, "two arms hash to the same checkpoint"


# --------------------------------------------------------------------------- #
# r2 F6 -- the binding is never assumed: checkpoint, manifest, or UNBOUND
# --------------------------------------------------------------------------- #
def test_binding_states_cover_the_whole_matrix(tmp_path):
    fa = torch.load(_write_ckpt(tmp_path / "bf.ckpt", config=_fa_config()),
                    map_location="cpu", weights_only=True)
    vanilla = torch.load(_write_ckpt(tmp_path / "p1.ckpt"), map_location="cpu",
                         weights_only=True)
    configless = {"state_dict": {}}

    # 1. the checkpoint answers -> bound to the file, agreement required
    assert ca.cond_method_binding(fa, "fa_invariant")["binding"] == "checkpoint"
    assert ca.cond_method_binding(vanilla, "vanilla")["binding"] == "checkpoint"
    assert ca.cond_method_binding(fa, "vanilla")["reasons"]
    assert ca.cond_method_binding(vanilla, "fa_invariant")["reasons"]

    # 2. configless + a verified manifest that agrees -> bound to the manifest
    bound = ca.cond_method_binding(configless, "vanilla",
                                   manifest={"cond_method": "vanilla"}, manifest_verified=True)
    assert bound["binding"] == "manifest" and bound["reasons"] == []

    # 3. configless + a manifest that disagrees -> refusal
    clash = ca.cond_method_binding(configless, "fa_invariant",
                                   manifest={"cond_method": "vanilla"}, manifest_verified=True)
    assert clash["reasons"] and "manifest" in clash["reasons"][0]

    # 4. configless + an UNverified manifest is not a binding
    unverified = ca.cond_method_binding(configless, "vanilla",
                                        manifest={"cond_method": "vanilla"},
                                        manifest_verified=False)
    assert unverified["binding"] == "unbound"

    # 5. configless + nothing at all -> unbound, and that is a REFUSAL when the
    #    run is registered, a stamped state when it is a smoke/dev run
    unbound = ca.cond_method_binding(configless, "vanilla")
    assert unbound["binding"] == "unbound" and unbound["reasons"] == []
    assert unbound["stamped"] is True
    registered = ca.cond_method_binding(configless, "vanilla", registered=True)
    assert registered["binding"] == "unbound" and registered["reasons"]
    assert "registered" in registered["reasons"][0]


def test_driver_records_the_binding_for_every_row(tmp_path):
    """r1 review F6: P1/YAW checkpoint-bound decisions were absent from
    provenance because the key was FA-conditional."""
    import eval_localization as el

    vanilla_ckpt = _write_ckpt(tmp_path / "p1.ckpt")
    args = el.parse_args(["--model-config", "m.json", "--dataset-config", "d.json",
                          "--ckpt-path", vanilla_ckpt, "--agree-ckpt", "a.pt",
                          "--num-samples", "8"])
    el.load_checkpoint_and_validate(args, _CONFIG)
    record = el.build_provenance(args, "ck", "ag", "split", "ema", 1)
    assert record["cond_method_binding"]["binding"] == "checkpoint"
    assert record["cond_method_binding"]["checkpoint_cond_method"] == "vanilla"


# --------------------------------------------------------------------------- #
# r2 F2 -- pairing and aggregation are fail-closed
# --------------------------------------------------------------------------- #
def test_pairing_refuses_evidence_that_is_missing_or_empty():
    """The reviewer's probe: two runs carrying only arm/regime/seed returned
    paired=True with n_queries=0."""
    bare = [{"arm": "P1", "regime": "K8", "seed": 42},
            {"arm": "BF", "regime": "K8", "seed": 42}]
    verdict = ca.validate_pairing(bare)
    assert verdict["paired"] is False
    fields = {m["field"] for m in verdict["mismatches"]}
    assert {"query_ids", "context_stream_digest", "noise_keys"} <= fields

    for field in ("query_ids", "noise_keys"):
        empty = [_facts("P1", **{field: type(_facts()[field])()}),
                 _facts("BF", **{field: type(_facts()[field])()})]
        verdict = ca.validate_pairing(empty)
        assert verdict["paired"] is False
        assert any(m["field"] == field for m in verdict["mismatches"]), field


def test_pairing_refuses_duplicate_ids_and_unkeyed_noise_and_repeated_arms():
    duped = _facts("P1", query_ids=["q0", "q0", "q1"])
    verdict = ca.validate_pairing([duped, _facts("BF", query_ids=["q0", "q0", "q1"])])
    assert verdict["paired"] is False
    assert any("duplicate" in m["detail"] for m in verdict["mismatches"])

    unkeyed = _facts("P1", noise_keys={"q0": [1, 2]})
    verdict = ca.validate_pairing([unkeyed, _facts("BF", noise_keys={"q0": [1, 2]})])
    assert any("noise" in m["field"] for m in verdict["mismatches"])

    verdict = ca.validate_pairing([_facts("P1"), _facts("P1")])
    assert verdict["paired"] is False
    assert any(m["field"] == "arms" for m in verdict["mismatches"])


def test_pairing_facts_refuse_rows_without_noise_keys(tmp_path):
    rows = tmp_path / "rows.jsonl"
    with open(rows, "w") as handle:
        handle.write(json.dumps({"query_id": "q0", "room_id": "R0"}) + "\n")
    summary = tmp_path / "summary.json"
    with open(summary, "w") as handle:
        json.dump({"provenance": {"seed": 42}}, handle)
    with pytest.raises(ValueError, match="noise"):
        ca.pairing_facts(str(rows), str(summary), arm="P1", regime="K8")


def test_seed_aggregation_requires_exactly_the_registered_seeds():
    """A one-seed cell aggregated happily before (r1 review F2)."""
    def cell(seeds):
        return {seed: [{"query_id": "q0", "room_id": "R0", "top1": 1.0, "e_loc": 0.0}]
                for seed in seeds}

    assert len(ca.aggregate_seeds_per_query(cell([42, 43, 44]),
                                            registered_seeds=(42, 43, 44))) == 1
    for seeds in ([42], [42, 43], [42, 43, 44, 45], [42, 43, 45]):
        with pytest.raises(ValueError, match="seed"):
            ca.aggregate_seeds_per_query(cell(seeds), registered_seeds=(42, 43, 44))


def test_registered_seeds_come_from_the_manifest_not_a_constant(tmp_path):
    manifest = {"seeds": [42, 43, 44]}
    assert ca.registered_seeds(manifest) == (42, 43, 44)
    with pytest.raises(ValueError, match="seeds"):
        ca.registered_seeds({})
    with pytest.raises(ValueError, match="seeds"):
        ca.registered_seeds({"seeds": []})
    assert ca.registered_seeds({"seeds": [7, 8]}) == (7, 8)
