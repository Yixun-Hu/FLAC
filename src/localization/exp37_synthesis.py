"""exp_37 B3 synthesis evidence (Codex code review 2, BLOCKING 5): the 12 eval_FLAC cells, each bound by its OWN
artifacts, never by labels in the spec.

The cells are B-F@40k x orbit {0} / {0,180} / C4 x theta {0, 90} x K_ctx {1, 8}, seed 42, no-flash. The 8 {0} /
{0,180} cells are NEW (the exp_36 Coder's H35-compatible leaf, block ``BF-ORB``, cells
``exp37_BF_ORB{1,2}_S40000_K{k}_s42_th{theta}``, run at the leaf commit); the 4 C4 cells are REUSED from exp_35 phase
(i) (block ``BF-pin``: ``exp35_BF_T_S40000_K{k}_s42`` at theta 0, ``exp35_BF_C4_S40000_K{k}_s42_th90`` at theta 90,
run at H35). Each cell supplies the exp_25 runner's three artifacts, all hashed in the spec and all distinct across
cells: the evaluator's metrics JSON, its schema-2 receipt (``<stem>.cell.json``) and its scored-waveform dump manifest.
``cell_reasons`` checks that the artifacts THEMSELVES carry the cell: the checkpoint (sha256 measured / loaded), the
SDPA backend with the pinned no-flash overlay and launcher, the frame-average orbit, theta, K (the registered dataset
config and its sha256), the seed, the evaluator identity (pin = the leaf commit or H35, the tree digests, the model
config, the full argv), the source type and the stem, and that the dump manifest is the one the receipt recorded and
binds the metrics file. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from src.localization import exp37_identity as ident

H35_PIN = "5b065801e9d1c1844461ec84cd0558702c714326"
CKPT_SHA256 = ident.ARMS["BF"]["ckpt"]
CKPT_LINK = "outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt"
MODEL_CONFIG = ("worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328")
DATASET = {1: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
               "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"),
           8: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
               "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55")}
STREAMS = 6337
DUMP_ROOT = "/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms"
NO_FLASH = {"attention_backend": "sdpa", "no_flash_overlay_sha256": ident.OVERLAY_SHA256,
            "no_flash_launcher_sha256": "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de"}
RECEIPT_KEYS = ("schema_version", "cell", "stem", "block", "arm", "step", "ckpt_link", "ckpt_target", "ckpt_sha256",
                "ckpt_cache", "ckpt_sha256_measured", "ckpt_sha256_loaded", "ckpt_bytes_loaded", "model_config",
                "model_config_sha256", "dataset_config", "dataset_config_sha256", "expected_stream_count", "dump",
                "dump_root", "argv", "expect_head", "campaign_digest", "spawn_digest", "post_digest", "artifact_sha256",
                "dump_evidence", "lane", "log", "rc", "attention_backend", "no_flash_overlay_sha256",
                "no_flash_launcher_sha256")
DUMP_MANIFEST_KEYS = ("schema_version", "n", "dtype", "shape", "sha256", "input_hash", "metrics_stem", "source_sha",
                      "ckpt_path", "metrics_path", "metrics_sha256", "stream_sidecar_path", "stream_sidecar_sha256",
                      "canon_sidecar_path", "canon_sidecar_sha256")
ORBITS = {"0": [0.0], "0_180": [0.0, 180.0], "c4": [0.0, 90.0, 180.0, 270.0]}
SHA_KEY = {"metrics_json": "metrics_sha256", "receipt": "receipt_sha256", "dump_manifest": "dump_manifest_sha256"}
CELL_KEYS = ("orbit", "theta", "k_context", "seed", "source", "cell", "metrics_json", "metrics_sha256", "receipt",
             "receipt_sha256", "dump_manifest", "dump_manifest_sha256")
FIXED_ARGS = {"--cond-method": "fa_invariant", "--frame-avg-max-fwd-samples": "64", "--cond-autocast": "bf16",
              "--cfg-scale": "1.0", "--steps": "1", "--batch-size": "64", "--device": "cuda", "--seed": "42",
              "--rotate-mode": "fixed", "--expected-stream-count": str(STREAMS)}


def cell_name(orbit: str, theta: int, k: int) -> tuple[str, str, str]:
    """(cell id, block, source) of one registered synthesis cell."""
    if orbit == "c4":
        name = f"exp35_BF_T_S40000_K{k}_s42" if theta == 0 else f"exp35_BF_C4_S40000_K{k}_s42_th90"
        return name, "BF-pin", "reused_exp35"
    return f"exp37_BF_ORB{1 if orbit == '0' else 2}_S40000_K{k}_s42_th{theta}", "BF-ORB", "new"


def cell_stem(orbit: str, theta: int, k: int) -> str:
    return (f"epoch=8-step=40000_metrics_1_1.0_{cell_name(orbit, theta, k)[0]}_fa_invariant_a{len(ORBITS[orbit])}"
            + (f"_rot{theta}" if theta else ""))


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hex64(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _flags(argv) -> dict:
    out, items = {}, list(argv or [])
    for i, token in enumerate(items):
        if isinstance(token, str) and token.startswith("--"):
            out.setdefault(token, []).append(items[i + 1] if i + 1 < len(items) else None)
    return out


def cell_reasons(cell: dict, *, leaf_commit: str, metrics_keys) -> tuple[list[str], dict | None]:
    """Every identity the cell's own artifacts must carry; returns (reasons, the tabulated row)."""
    orbit, theta, k = cell.get("orbit"), cell.get("theta"), cell.get("k_context")
    if orbit not in ORBITS or theta not in (0, 90) or k not in (1, 8) or cell.get("seed") != 42:
        return [f"cell {orbit} / {theta} / K {k}: not a registered synthesis cell"], None
    name, block, source = cell_name(orbit, theta, k)
    pin, stem, (dataset, dataset_sha) = (leaf_commit if source == "new" else H35_PIN), cell_stem(orbit, theta, k), DATASET[k]
    bad = [] if (cell.get("source"), cell.get("cell")) == (source, name) else [f"{name}: source / cell id mismatch"]
    try:
        files = {key: Path(cell[key]) for key in ("metrics_json", "receipt", "dump_manifest")}
        for key, path in files.items():
            if _sha(path) != cell[SHA_KEY[key]]:
                bad.append(f"{name}: {key} {path} does not hash to the spec's sha256")
        metrics, receipt = json.loads(files["metrics_json"].read_text()), json.loads(files["receipt"].read_text())
        manifest_bytes = files["dump_manifest"].read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError, KeyError) as error:
        return bad + [f"{name}: an artifact is unreadable ({error})"], None
    metrics_sha = _sha(files["metrics_json"])
    # ---- the receipt: what ran, on which checkpoint, at which pin, on which backend, with which argv
    if not isinstance(receipt, dict) or tuple(receipt) != RECEIPT_KEYS:
        bad.append(f"{name}: the receipt is not the schema-2 key set")
        receipt = receipt if isinstance(receipt, dict) else {}
    want = {"schema_version": 2, "cell": name, "stem": stem, "block": block, "arm": "BF", "step": 40000,
            "ckpt_link": CKPT_LINK, "ckpt_sha256": CKPT_SHA256, "ckpt_sha256_measured": CKPT_SHA256,
            "ckpt_sha256_loaded": CKPT_SHA256, "ckpt_cache": f"outputs_FLAC/exp25_depth_canon/.ckpt_cache/{CKPT_SHA256}.ckpt",
            "model_config": MODEL_CONFIG[0], "model_config_sha256": MODEL_CONFIG[1], "dataset_config": dataset,
            "dataset_config_sha256": dataset_sha, "expected_stream_count": STREAMS, "dump": True, "dump_root": DUMP_ROOT,
            "expect_head": pin, "rc": 0, **NO_FLASH}
    bad += [f"{name}: receipt {key} is {receipt.get(key)!r}, not {value!r}" for key, value in want.items()
            if receipt.get(key) != value]
    digests = [receipt.get(key) for key in ("campaign_digest", "spawn_digest", "post_digest")]
    if not _hex64(digests[0]) or len(set(digests)) != 1:
        bad.append(f"{name}: the receipt's tree digests are not one sha256 throughout")
    if (receipt.get("artifact_sha256") or {}).get("metrics") != metrics_sha:
        bad.append(f"{name}: the receipt does not bind this metrics file")
    flags = _flags(receipt.get("argv"))
    expected_flags = {**FIXED_ARGS, "--frame-avg-angles": ",".join(str(int(a)) for a in ORBITS[orbit]),
                      "--rotate-deg": str(theta), "--eval-name": name, "--model-config": MODEL_CONFIG[0],
                      "--dataset-config": dataset, "--ckpt-path": CKPT_LINK, "--dump-scored-waveforms": DUMP_ROOT}
    bad += [f"{name}: argv {flag} is {flags.get(flag)}, not [{value!r}]" for flag, value in expected_flags.items()
            if flags.get(flag) != [value]]
    # ---- the evaluator's own record of the run
    want_metrics = {"eval_name": name, "ckpt_path": CKPT_LINK, "ckpt_sha256_loaded": CKPT_SHA256,
                    "cond_method": "fa_invariant", "frame_avg_angles": ORBITS[orbit], "rotate_deg": float(theta),
                    "seed": 42, "dataset_config": dataset, "source_sha": pin, "n_samples": STREAMS, "cfg_scale": 1.0,
                    "steps": 1, "batch_size": 64, "cond_autocast": "bf16", "weights_source": "ema"}
    bad += [f"{name}: metrics {key} is {metrics.get(key)!r}, not {value!r}" for key, value in want_metrics.items()
            if metrics.get(key) != value]
    scenes = metrics.get("by_scene") if isinstance(metrics.get("by_scene"), dict) else {}
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not scenes or not all(numeric((metrics.get("metrics") or {}).get(key)) and all(
            numeric((row or {}).get(key)) for row in scenes.values()) for key in metrics_keys):
        bad.append(f"{name}: the metrics file lacks a finite {list(metrics_keys)} value (overall or per scene)")
    # ---- the dump manifest: the one the receipt recorded, binding this metrics file at this pin
    evidence = receipt.get("dump_evidence") or {}
    if hashlib.sha256(manifest_bytes).hexdigest() != evidence.get("manifest_sha256") or manifest != evidence.get("manifest"):
        bad.append(f"{name}: the dump manifest is not the one its receipt recorded")
    if not isinstance(manifest, dict) or tuple(manifest) != DUMP_MANIFEST_KEYS:
        bad.append(f"{name}: the dump manifest is not the schema-1 key set")
        manifest = manifest if isinstance(manifest, dict) else {}
    want_manifest = {"schema_version": 1, "n": STREAMS, "dtype": "float32", "metrics_stem": stem, "source_sha": pin,
                     "ckpt_path": CKPT_LINK, "metrics_sha256": metrics_sha}
    bad += [f"{name}: dump manifest {key} is {manifest.get(key)!r}, not {value!r}" for key, value in want_manifest.items()
            if manifest.get(key) != value]
    if evidence.get("marker") != f"{manifest.get('sha256')}\n" or evidence.get("waveforms_sidecar") != (
            f"{manifest.get('sha256')}  waveforms.npy\n") or not _hex64(manifest.get("sha256")):
        bad.append(f"{name}: the dump's COMPLETE marker / sidecar do not hold its waveform sha256")
    if bad:
        return bad, None
    return [], {"orbit": orbit, "theta": theta, "k_context": k, "seed": 42, "source": source, "cell": name, "pin": pin,
                "metrics": {key: float(metrics["metrics"][key]) for key in metrics_keys},
                "per_scene_mean": {key: float(statistics.fmean(row[key] for row in scenes.values())) for key in metrics_keys},
                "artifacts": {key: cell[SHA_KEY[key]] for key in SHA_KEY}}


def validate_section(section: dict) -> dict:
    """The complete synthesis section: 12 unique cells whose own artifacts carry their identities; RuntimeError
    otherwise (a pending section is the caller's diagnostic)."""
    cells, keys = section["cells"], list(section["metrics_keys"])
    paths = [Path(c[key]).resolve() for c in cells for key in ("metrics_json", "receipt", "dump_manifest")]
    digests = [c[SHA_KEY[key]] for c in cells for key in SHA_KEY]
    if len(set(paths)) != len(paths) or len(set(digests)) != len(digests):
        raise RuntimeError("synthesis artifacts are reused across cells (every file and every digest must be unique)")
    rows, reasons = [], []
    for cell in cells:
        bad, row = cell_reasons(cell, leaf_commit=section["leaf_commit"], metrics_keys=keys)
        reasons += bad
        rows.append(row)
    if reasons:
        raise RuntimeError("synthesis evidence refused: " + "; ".join(reasons))
    return {"status": "complete", "leaf_commit": section["leaf_commit"], "runbook": section["runbook"],
            "metrics_keys": keys, "cells": rows}
