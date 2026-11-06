"""exp_37 predicted-waveform artifacts (announcement 08; Codex code review 1, BLOCKING 1).

Every generative exp_37 run that reports accuracy keeps the exact tensors it scored: per query one ``.npy``
``[M, 1, 10240]`` float32 -- the ``prepare_generated_audio`` output (decoded, clamped) -- in the candidate order of the
run's query record, with the sha256 of the rows' full-grid indices; and a hashed ``dump_manifest.json`` bound to the
run manifest, the checkpoint, the seed, the backend and the complete query coverage. The dump directory is
``<root>/<run stem>``, created by the run itself and never overwritten. Timing passes never dump: an untimed
deterministic replay (``mode = replay``) regenerates and dumps them, bound to the timing run it reproduces.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.localization.pilot import canonical_sha256

DUMP_ROOT = Path("/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms")
SAMPLES = 10240
TENSOR = ("the exact prepare_generated_audio output (decoded, clamped) that was scored; [M, K=1, 10240] float32; "
          "rows in the candidate order of the run's query record")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WaveformDump:
    """The write-once store of one run's scored waveforms."""

    def __init__(self, dump_dir: Path, *, run_stem: str, root: Path = DUMP_ROOT):
        dump_dir, root = Path(dump_dir), Path(root)
        if dump_dir.parent.resolve() != root.resolve():
            raise ValueError(f"the dump directory must sit directly under the registered root {root}")
        if dump_dir.name != run_stem:
            raise ValueError(f"the dump directory must be named after the run stem {run_stem!r}")
        if dump_dir.exists():
            raise RuntimeError(f"{dump_dir} exists (waveform dumps are write-once)")
        root.mkdir(parents=True, exist_ok=True)
        dump_dir.mkdir()
        self.dir, self.files = dump_dir, {}

    def write(self, query_index: int, waveforms, full_indices) -> dict:
        array = np.asarray(waveforms)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[1] != 1 or array.shape[2] != SAMPLES:
            raise ValueError("waveforms must be the scored [M, 1, 10240] float32 tensor")
        indices = np.asarray(full_indices)
        if indices.shape != (array.shape[0],):
            raise ValueError("one full-grid index per waveform row is required")
        path = self.dir / f"query_{int(query_index):05d}.npy"
        if int(query_index) in self.files or path.exists():
            raise RuntimeError(f"query {query_index} was already written (write-once)")
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array))
        os.replace(temporary, path)
        entry = {"path": path.name, "sha256": _file_sha256(path), "shape": list(array.shape), "dtype": "float32",
                 "candidate_count": int(array.shape[0]),
                 "full_indices_sha256": hashlib.sha256(indices.astype("<u4").tobytes()).hexdigest()}
        self.files[int(query_index)] = entry
        return entry

    def finish(self, *, run_manifest_sha256: str, identity: dict, mode: str, query_indices, replay_of=None) -> Path:
        if sorted(self.files) != sorted(int(i) for i in query_indices):
            raise RuntimeError("waveform coverage is incomplete: not every query of the run was dumped")
        if mode not in ("inline", "replay") or (mode == "replay") != (replay_of is not None):
            raise ValueError("mode is inline, or replay together with the run it replays")
        body = {"schema_version": 1, "record_type": "exp37_waveform_dump", "mode": mode, "dump_dir": str(self.dir),
                "tensor": TENSOR, "run_manifest_sha256": run_manifest_sha256,
                "checkpoint_sha256": identity["checkpoint_sha256"], "sample_seed": identity["sample_seed"],
                "attention_backend": identity["attention_backend"],
                "conditioning_method": identity.get("conditioning_method"),
                "frame_average_angles": identity.get("frame_average_angles"), "rotate_deg": identity.get("rotate_deg", 0.0),
                "replay_of": replay_of,
                "coverage": {"query_indices": [int(i) for i in query_indices],
                             "files": {str(i): self.files[int(i)] for i in query_indices}}}
        path = self.dir / "dump_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return path


def load_dump_manifest(dump_dir: Path, *, run_manifest_sha256: str, query_counts: dict, verify_sha: bool = True) -> dict:
    """The validated manifest of one run's dump: hashed, bound to the run, covering exactly its queries with
    [count, 1, 10240] float32 files whose bytes still hash to the recorded sha256."""
    dump_dir = Path(dump_dir)
    payload = json.loads((dump_dir / "dump_manifest.json").read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}) or \
            payload.get("record_type") != "exp37_waveform_dump":
        raise RuntimeError(f"{dump_dir}: the dump manifest fails its own sha256 / record type")
    if payload["run_manifest_sha256"] != run_manifest_sha256:
        raise RuntimeError(f"{dump_dir}: the dump belongs to another run manifest")
    files = payload["coverage"]["files"]
    if sorted(int(i) for i in files) != sorted(int(i) for i in query_counts):
        raise RuntimeError(f"{dump_dir}: the dump coverage is not exactly the run's queries")
    for index, count in query_counts.items():
        entry = files[str(int(index))]
        array = np.load(dump_dir / entry["path"], mmap_mode="r")
        if list(array.shape) != [int(count), 1, SAMPLES] or array.dtype != np.float32 or entry["shape"] != list(array.shape):
            raise RuntimeError(f"{dump_dir}: query {index} waveform shape {list(array.shape)} is not [{count}, 1, {SAMPLES}]")
        if verify_sha and _file_sha256(dump_dir / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"{dump_dir}: query {index} waveform file does not match its sha256")
    return payload
