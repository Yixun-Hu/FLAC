"""exp_37 predicted-waveform artifacts (announcement 08; Codex code reviews 1 / 2, BLOCKING 1 / 2).

Every generative exp_37 run that reports accuracy keeps the exact tensors it scored: per query one ``.npy``
``[M, 1, 10240]`` float32 -- the ``prepare_generated_audio`` output (decoded, clamped) -- in the candidate order of the
run's query record, its ACTUAL row -> full-grid index vector (``<u4``, strictly increasing, a hashed companion file;
``arange(M)`` for a full-grid run, the subset / union indices otherwise), and a hashed ``dump_manifest.json`` bound to
the run manifest, the checkpoint, the seed, the backend and the complete query coverage.

Crash safety (reviews 2 / 3): everything is written into a uniquely owned staging directory
``<root>/.staging-<stem>-<pid>-<token>`` (``.owner.json``: pid + /proc start time), each file created exclusively,
fsync'd and made read-only, directories fsync'd; ``finish`` writes the manifest exclusively LAST and publishes by ONE
``renameat2(RENAME_NOREPLACE)`` into ``<root>/<stem>`` (refused when the name exists; a second ``finish`` refused).
There is NO fallback (review 3, B-01): a libc or filesystem without the atomic no-replace rename, and any directory
fsync that fails or is unsupported, raise ``DurabilityError`` -- no weaker filesystem protocol is authorized -- and
the staging stays (a root fsync failure after the rename unpublishes the dump back to its staging name). A crash or
a refusal therefore leaves only the staging directory; the stem is refused until ``tools/exp37_quarantine.py`` has
moved it away.
Timing passes never dump: an untimed deterministic replay (``mode = replay``) regenerates and dumps them.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix rounds 1-2; the final fix round (fixed after review
3, not re-reviewed (cap)).
"""

from __future__ import annotations

import ctypes
import datetime
import errno
import hashlib
import io
import json
import os
import secrets
from pathlib import Path

import numpy as np

from src.localization.pilot import canonical_sha256

DUMP_ROOT = Path("/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms")
SAMPLES = 10240
OWNER = ".owner.json"
MANIFEST = "dump_manifest.json"
TENSOR = ("the exact prepare_generated_audio output (decoded, clamped) that was scored; [M, K=1, 10240] float32; "
          "rows in the candidate order of the run's query record; query_XXXXX.indices.npy = the rows' full-grid "
          "indices (<u4, strictly increasing)")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def start_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{int(pid)}/stat") as handle:
            return int(handle.read().rpartition(")")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class DurabilityError(RuntimeError):
    """The filesystem cannot give the atomic no-replace rename or the directory fsync that publication requires."""


def fsync_dir(path: Path) -> str:
    """fsync a directory and return "ok"; ANY failure -- an unsupported directory fsync (EINVAL / ENOTSUP / EBADF)
    included -- is fatal (review 3, B-01: no weaker filesystem protocol is authorized)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as error:
        raise DurabilityError(f"cannot open the directory {path} for its fsync: {error}") from error
    try:
        os.fsync(fd)
    except OSError as error:
        raise DurabilityError(f"the directory fsync of {path} failed or is unsupported "
                              f"({errno.errorcode.get(error.errno, error.errno)}): publication refused") from error
    finally:
        os.close(fd)
    return "ok"


def write_exclusive(path: Path, data: bytes) -> None:
    """Create ``path`` (never an existing one), write, fsync, close, make it read-only (best effort)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o444)
    except OSError:
        pass


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array))
    return buffer.getvalue()


def rename_noreplace(source: Path, target: Path) -> str:
    """Atomically rename ``source`` to ``target`` unless ``target`` exists: ``renameat2(RENAME_NOREPLACE)`` and
    nothing else (review 3, B-01). An existing target is FileExistsError; a libc without renameat2 or a filesystem
    refusing the flag (ENOSYS / EINVAL / ENOTSUP) is DurabilityError -- never a check-then-rename fallback."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (OSError, AttributeError) as error:
        raise DurabilityError("renameat2(RENAME_NOREPLACE) is unavailable: publication refused (no check-then-rename "
                              "fallback)") from error
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:         # AT_FDCWD, RENAME_NOREPLACE
        return "renameat2(RENAME_NOREPLACE)"
    code = ctypes.get_errno()
    if code in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(target))
    if code in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
        raise DurabilityError(f"the filesystem refuses renameat2(RENAME_NOREPLACE) for {target} "
                              f"({errno.errorcode.get(code, code)}): publication refused (no fallback)")
    raise OSError(code, os.strerror(code), str(target))


def staging_leftovers(root: Path, run_stem: str) -> list[Path]:
    return sorted(Path(root).glob(f".staging-{run_stem}-*")) if Path(root).is_dir() else []


class WaveformDump:
    """The write-once, crash-safe store of one run's scored waveforms."""

    def __init__(self, dump_dir: Path, *, run_stem: str, root: Path = DUMP_ROOT):
        dump_dir, root = Path(dump_dir), Path(root)
        if dump_dir.parent.resolve() != root.resolve():
            raise ValueError(f"the dump directory must sit directly under the registered root {root}")
        if dump_dir.name != run_stem:
            raise ValueError(f"the dump directory must be named after the run stem {run_stem!r}")
        if os.path.lexists(dump_dir):
            raise RuntimeError(f"{dump_dir} exists (waveform dumps are write-once)")
        leftovers = staging_leftovers(root, run_stem)
        if leftovers:
            raise RuntimeError(f"{leftovers[0]} is an unpublished dump of this stem: quarantine it first "
                               "(tools/exp37_quarantine.py)")
        root.mkdir(parents=True, exist_ok=True)
        self.token = f"{os.getpid()}-{secrets.token_hex(6)}"
        self.staging = root / f".staging-{run_stem}-{self.token}"
        os.mkdir(self.staging)                                       # exclusive
        owner = {"pid": os.getpid(), "pid_start_ticks": start_ticks(os.getpid()), "run_stem": run_stem,
                 "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_exclusive(self.staging / OWNER, json.dumps(owner, sort_keys=True).encode())
        self.directory_fsync = {fsync_dir(self.staging), fsync_dir(root)}
        self.dir, self.root, self.files, self.finished = dump_dir, root, {}, False

    def path(self, name: str) -> Path:
        """Where a written file is now (staging before publication, the final directory after)."""
        return (self.dir if self.finished else self.staging) / name

    def write(self, query_index: int, waveforms, full_indices) -> dict:
        if self.finished:
            raise RuntimeError("the dump is published (write-once)")
        array = np.asarray(waveforms)
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[1] != 1 or array.shape[2] != SAMPLES:
            raise ValueError("waveforms must be the scored [M, 1, 10240] float32 tensor")
        indices = np.asarray(full_indices)
        if indices.shape != (array.shape[0],):
            raise ValueError("one full-grid index per waveform row is required")
        if not np.issubdtype(indices.dtype, np.integer) or (len(indices) and (indices[0] < 0 or np.any(
                np.diff(indices) <= 0) or indices[-1] >= 2 ** 32)):
            raise ValueError("the full-grid indices must be strictly increasing nonnegative integers")
        name = f"query_{int(query_index):05d}"
        if int(query_index) in self.files or os.path.lexists(self.staging / f"{name}.npy"):
            raise RuntimeError(f"query {query_index} was already written (write-once)")
        rows = indices.astype("<u4")
        write_exclusive(self.staging / f"{name}.npy", _npy_bytes(array))
        write_exclusive(self.staging / f"{name}.indices.npy", _npy_bytes(rows))
        self.directory_fsync.add(fsync_dir(self.staging))
        entry = {"path": f"{name}.npy", "sha256": _file_sha256(self.staging / f"{name}.npy"),
                 "shape": list(array.shape), "dtype": "float32", "candidate_count": int(array.shape[0]),
                 "indices_path": f"{name}.indices.npy", "indices_dtype": "<u4",
                 "indices_sha256": _file_sha256(self.staging / f"{name}.indices.npy"),
                 "full_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest()}
        self.files[int(query_index)] = entry
        return entry

    def finish(self, *, run_manifest_sha256: str, identity: dict, mode: str, query_indices, replay_of=None) -> Path:
        if self.finished:
            raise RuntimeError("finish() was already called: the dump is published (write-once)")
        if sorted(self.files) != sorted(int(i) for i in query_indices):
            raise RuntimeError("waveform coverage is incomplete: not every query of the run was dumped")
        if mode not in ("inline", "replay") or (mode == "replay") != (replay_of is not None):
            raise ValueError("mode is inline, or replay together with the run it replays")
        body = {"schema_version": 2, "record_type": "exp37_waveform_dump", "mode": mode, "dump_dir": str(self.dir),
                "tensor": TENSOR, "run_manifest_sha256": run_manifest_sha256,
                "checkpoint_sha256": identity["checkpoint_sha256"], "sample_seed": identity["sample_seed"],
                "attention_backend": identity["attention_backend"],
                "conditioning_method": identity.get("conditioning_method"),
                "frame_average_angles": identity.get("frame_average_angles"), "rotate_deg": identity.get("rotate_deg", 0.0),
                "replay_of": replay_of, "staged_as": self.staging.name, "directory_fsync": sorted(self.directory_fsync),
                "coverage": {"query_indices": [int(i) for i in query_indices],
                             "files": {str(i): self.files[int(i)] for i in query_indices}}}
        write_exclusive(self.staging / MANIFEST,
                        (json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n").encode())
        fsync_dir(self.staging)
        try:
            self.publish_method = rename_noreplace(self.staging, self.dir)
        except FileExistsError as error:
            raise RuntimeError(f"{self.dir} exists: the dump is not published over it (staging kept)") from error
        try:
            fsync_dir(self.root)
        except DurabilityError as error:                     # not durable: unpublish (the stem stays refused)
            try:
                rename_noreplace(self.dir, self.staging)
            except BaseException as undo:
                raise DurabilityError(f"{error}; and unpublishing {self.dir} failed ({undo})") from error
            raise DurabilityError(f"{error}; the dump was unpublished back to {self.staging.name}") from error
        self.finished = True
        return self.dir / MANIFEST


def dump_indices(dump_dir: Path, entry: dict) -> np.ndarray:
    return np.load(Path(dump_dir) / entry["indices_path"])


def load_dump_manifest(dump_dir: Path, *, run_manifest_sha256: str, query_counts: dict, verify_sha: bool = True) -> dict:
    """The validated manifest of one run's dump: hashed, bound to the run, covering exactly its queries with
    [count, 1, 10240] float32 waveform files and their strictly increasing <u4 index vectors, every file's bytes still
    hashing to the recorded sha256."""
    dump_dir = Path(dump_dir)
    payload = json.loads((dump_dir / MANIFEST).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}) or \
            payload.get("record_type") != "exp37_waveform_dump" or payload.get("schema_version") != 2:
        raise RuntimeError(f"{dump_dir}: the dump manifest fails its own sha256 / record type / schema")
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
        rows = dump_indices(dump_dir, entry)
        if rows.dtype != np.dtype("<u4") or rows.shape != (int(count),) or np.any(np.diff(rows.astype(np.int64)) <= 0) \
                or hashlib.sha256(rows.tobytes()).hexdigest() != entry["full_indices_sha256"]:
            raise RuntimeError(f"{dump_dir}: query {index} index vector is not the recorded increasing mapping")
        if verify_sha and (_file_sha256(dump_dir / entry["path"]) != entry["sha256"]
                           or _file_sha256(dump_dir / entry["indices_path"]) != entry["indices_sha256"]):
            raise RuntimeError(f"{dump_dir}: query {index} waveform / index file does not match its sha256")
    return payload
