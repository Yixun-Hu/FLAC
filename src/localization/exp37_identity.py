"""exp_37 registered run identity (Codex code review 1, HIGH 1): one definition, enforced by every loader.

``registered_identity`` returns the exact run-manifest identity a registered exp_37 run must carry -- every field the
G0 comparator checks (model-config, checkpoint, AGREE, context-manifest, audit and selection digests; queries;
conditioning and orbit; K_ctx; K_gen; tau; the sample seed; the batch; the sampler; the SDPA backend; the no-flash
launcher / overlay digests; the mesh-root digest map; the timing flags) plus the optional modes, each of which must be
present exactly when expected (``ABSENT`` otherwise). ``identity_reasons`` lists every deviation. No heavy imports.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_SHA256 = "f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9"   # FLAC_AR.json
AGREE_SHA256 = "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"          # AGREE_fullAR.pt
CONTEXT_MANIFEST_SHA256 = "b757da281dcde3ffc310aac67279a240dac5cb1ff1d9966bf918f69c4dde6f58"
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"
SENTINEL = "EXP37_NO_FLASH_LAUNCHER"
LAUNCHER_REL = ("overlays", "no_flash", "exp37_no_flash_launch.py")
SAMPLER = {"type": "rectified_flow_discrete_euler", "steps": 1, "cfg_scale": 1.0}
C4 = [0.0, 90.0, 180.0, 270.0]
ARMS = {
    "P1": {"cond": "vanilla", "ckpt": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c"},
    "YAW": {"cond": "vanilla", "ckpt": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7"},
    "BF": {"cond": "fa_invariant", "ckpt": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328"},
}
K1_CONTEXT_SELECTION = {
    "model_contexts": "contexts[0] of the frozen context manifest (its exp_09 seed-42 loader order)",
    "candidate_filter_contexts": "all 8 manifest contexts (candidate_filter_context_sources_global)",
}


class _Absent:
    def __repr__(self):
        return "ABSENT"


ABSENT = _Absent()          # the key must not be in the identity


def launcher_sha256(repo_root: Path = REPO_ROOT) -> str:
    return hashlib.sha256(Path(repo_root).joinpath(*LAUNCHER_REL).read_bytes()).hexdigest()


def registered_identity(arm: str, *, selection: dict, audit: dict, seed: int, timing: bool, context_count: int = 8,
                        frame_angles=None, candidate_subset: str | None = None, c2f_top_k: int | None = None,
                        rotate_deg: float = 0.0, stage_split: bool = False, waveform_mode: str | None = None,
                        replay_of_sha256: str | None = None, repo_root: Path = REPO_ROOT) -> dict:
    if timing and waveform_mode:
        raise ValueError("timing passes never dump waveforms (use an artifact replay)")
    if (waveform_mode == "replay") != bool(replay_of_sha256):
        raise ValueError("a replay dump names exactly the run it replays")
    if rotate_deg and timing:
        raise ValueError("rotated runs are accuracy-only")
    cond = ARMS[arm]["cond"]
    return {
        "model_config_sha256": MODEL_CONFIG_SHA256, "checkpoint_sha256": ARMS[arm]["ckpt"],
        "agree_checkpoint_sha256": AGREE_SHA256, "context_manifest_sha256": CONTEXT_MANIFEST_SHA256,
        "geometry_audit_sha256": audit["sha256"], "pilot_manifest_sha256": selection["sha256"],
        "query_indices": [int(r["index"]) for r in selection["records"]], "conditioning_method": cond,
        "frame_average_angles": ([float(a) for a in frame_angles] if frame_angles else list(C4))
        if cond == "fa_invariant" else None,
        "n_context": int(context_count), "score_sample_counts": [1], "tau": 0.1, "sample_seed": int(seed),
        "candidate_batch_size": 64, "synchronize_timing": bool(timing), "warmup_query_count": 1 if timing else 0,
        "measure_core_forward": bool(timing), "sampler": dict(SAMPLER), "attention_backend": "sdpa",
        "no_flash_launch": {"overlay_sha256": OVERLAY_SHA256, "launcher_sha256": launcher_sha256(repo_root),
                            "sentinel": SENTINEL},
        "mesh_root_override": "the audit's digest map",
        "context_selection": dict(K1_CONTEXT_SELECTION) if int(context_count) != 8 else ABSENT,
        "stage_split": True if stage_split else ABSENT,
        "candidate_subset": candidate_subset or ABSENT,
        "c2f_top_k": int(c2f_top_k) if candidate_subset == "c2f" else ABSENT,
        "c2f_chebyshev_radius_m": 0.5 if candidate_subset == "c2f" else ABSENT,
        "rotate_deg": float(rotate_deg) if rotate_deg else ABSENT,
        "waveform_dump": waveform_mode or ABSENT,
        "replay_of": replay_of_sha256 or ABSENT,
    }


def identity_reasons(identity: dict, expected: dict, audit: dict) -> list[str]:
    bad = []
    for key, want in expected.items():
        have = identity.get(key, ABSENT)
        if key == "mesh_root_override":
            override = have if isinstance(have, dict) else {}
            digests = {room: (row or {}).get("mesh_sha256") for room, row in (override.get("rooms") or {}).items()}
            if override.get("audit_mesh_root") != audit["mesh_root"] or digests != {
                    room: row["mesh_sha256"] for room, row in audit["rooms"].items()}:
                bad.append(key)
        elif key == "waveform_dump":
            if (have.get("mode") if isinstance(have, dict) else have) != want:
                bad.append(key)
        elif key == "replay_of":
            if (have.get("run_manifest_sha256") if isinstance(have, dict) else have) != want:
                bad.append(key)
        elif have != want:
            bad.append(key)
    return [f"identity field {key} is not the registered value" for key in bad] + [
        f"identity field {key} is not registered" for key in sorted(set(identity) - set(expected))]
