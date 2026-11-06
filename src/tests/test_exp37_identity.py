"""exp_37 fix round (Codex code review 1, HIGH 1): ONE registered run identity, enforced by every loader.

Every field G0 checks is registered: the model-config, checkpoint, AGREE and context-manifest digests, the audit and
selection digests, the query list, conditioning, orbit, K_ctx, K_gen, tau, the sample seed, the batch, the sampler,
the SDPA backend, the no-flash launcher / overlay digests, the mesh-root digest map, and the timing flags; optional
modes (K_ctx 1, stage split, subsets, c2f, rotation, waveform dump, replay) must be present exactly when expected.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import hashlib
import json
from pathlib import Path

import pytest

from exp37_fakes import example_identity
from src.localization import exp37_identity as ident
from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import load_pilot_manifest

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"


@pytest.fixture(scope="module")
def frozen():
    return json.loads((E9 / "geometry_audit.json").read_text()), load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_registered_constants_are_this_branch_s_files():
    assert ident.MODEL_CONFIG_SHA256 == _sha(REPO / "src/configs/model_configs/FLAC/AR/FLAC_AR.json")
    assert ident.CONTEXT_MANIFEST_SHA256 == load_context_manifest(E9 / "context_manifest_exp01_seed42.json")["sha256"]
    assert ident.OVERLAY_SHA256 == _sha(REPO / "overlays/no_flash/flash_attn.py")
    assert ident.launcher_sha256() == _sha(REPO / "overlays/no_flash/exp37_no_flash_launch.py")
    assert ident.AGREE_SHA256.startswith("3a13243d") and set(ident.ARMS) == {"P1", "YAW", "BF"}


def test_a_complete_runner_identity_passes(frozen):
    audit, pilot = frozen
    expected = ident.registered_identity("BF", selection=pilot, audit=audit, seed=42, timing=True)
    assert ident.identity_reasons(example_identity(expected, audit), expected, audit) == []


@pytest.mark.parametrize("key, value", [
    ("model_config_sha256", "x"), ("checkpoint_sha256", "x"), ("agree_checkpoint_sha256", "x"),
    ("context_manifest_sha256", "x"), ("geometry_audit_sha256", "x"), ("pilot_manifest_sha256", "x"),
    ("query_indices", [1]), ("conditioning_method", "vanilla"), ("frame_average_angles", [0.0]), ("n_context", 1),
    ("score_sample_counts", [1, 4, 8]), ("tau", 0.2), ("sample_seed", 43), ("candidate_batch_size", 32),
    ("sampler", {"type": "x"}), ("attention_backend", "flash_attn"), ("synchronize_timing", False),
    ("warmup_query_count", 0), ("measure_core_forward", False)])
def test_every_registered_field_is_enforced(frozen, key, value):
    audit, pilot = frozen
    expected = ident.registered_identity("BF", selection=pilot, audit=audit, seed=42, timing=True)
    identity = {**example_identity(expected, audit), key: value}
    assert key in " ".join(ident.identity_reasons(identity, expected, audit))


def test_the_launcher_and_the_mesh_digest_map_are_enforced(frozen):
    audit, pilot = frozen
    expected = ident.registered_identity("P1", selection=pilot, audit=audit, seed=42, timing=True)
    good = example_identity(expected, audit)
    bad_launch = {**good, "no_flash_launch": {**good["no_flash_launch"], "launcher_sha256": "0" * 64}}
    room = next(iter(audit["rooms"]))
    bad_mesh = json.loads(json.dumps(good))
    bad_mesh["mesh_root_override"]["rooms"][room]["mesh_sha256"] = "0" * 64
    missing = {k: v for k, v in good.items() if k != "mesh_root_override"}
    for identity, needle in ((bad_launch, "no_flash_launch"), (bad_mesh, "mesh_root_override"),
                             (missing, "mesh_root_override")):
        assert needle in " ".join(ident.identity_reasons(identity, expected, audit))


@pytest.mark.parametrize("mode", [
    dict(context_count=1), dict(stage_split=True), dict(candidate_subset="nested_1m"),
    dict(candidate_subset="c2f", c2f_top_k=5), dict(rotate_deg=90.0, timing=False, waveform_mode="inline"),
    dict(timing=False, waveform_mode="replay", replay_of_sha256="r" * 64)])
def test_each_optional_mode_is_required_exactly_when_expected(frozen, mode):
    audit, pilot = frozen
    kwargs = {"timing": True, **mode}
    expected = ident.registered_identity("BF", selection=pilot, audit=audit, seed=42, **kwargs)
    identity = example_identity(expected, audit)
    assert ident.identity_reasons(identity, expected, audit) == []
    plain = ident.registered_identity("BF", selection=pilot, audit=audit, seed=42, timing=kwargs["timing"])
    assert ident.identity_reasons(identity, plain, audit)                       # the mode is not the plain run
    assert ident.identity_reasons(example_identity(plain, audit), expected, audit)


def test_timing_and_waveform_dumps_never_combine(frozen):
    audit, pilot = frozen
    with pytest.raises(ValueError, match="never dump"):
        ident.registered_identity("P1", selection=pilot, audit=audit, seed=42, timing=True, waveform_mode="inline")
    with pytest.raises(ValueError, match="replay"):
        ident.registered_identity("P1", selection=pilot, audit=audit, seed=42, timing=False, waveform_mode="replay")
