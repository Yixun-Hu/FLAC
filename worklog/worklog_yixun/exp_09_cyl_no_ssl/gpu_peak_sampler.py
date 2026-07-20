#!/usr/bin/env python3
"""exp-09 C1 external GPU peak-VRAM sampler (integrative-review finding 2, item 1).

Wraps a CHILD command (the C1 fit/smoke training invocation) and samples
``nvidia-smi --query-gpu=index,memory.used`` EXTERNALLY at a fixed interval for BOTH
physical GPUs for the whole life of the child. It is the robust replacement for the
in-shell 1 s samplers the exp_07 fit probe used (``m1_ddp_fit_probe.sh:71-72``): a
single external process owns the sampling, records per-GPU peaks + per-GPU valid-sample
counts, and FAILS CLOSED if EITHER GPU obtained no valid sample (a dead sampler must
never masquerade as "peak 0 MiB, gate is tiny").

Contract (verbatim where the review quotes it)
-----------------------------------------------
* samples ``nvidia-smi --query-gpu=index,memory.used`` externally at a fixed interval
  for both physical GPUs during a child command;
* "fail if either sampler obtains no valid sample";
* outputs atomic JSON ``{per_gpu_peaks_mib, samples_per_gpu, child_exit}``;
* the derived gate ``= max(peak_gpu0, peak_gpu1) + 4096 MiB`` is computed and serialised
  (COMPUTATION ONLY — freezing the number into records happens in the C1 records step,
  not here).

Exit code: ``3`` if the sampler failed closed (any requested GPU got zero valid
samples); otherwise the CHILD's own exit code is propagated (so the fit script sees the
real training exit). The JSON is ALWAYS written (atomically) recording what happened.

CPU-testable: the ``nvidia-smi`` binary is resolved through the ``NVIDIA_SMI`` env var
(default ``"nvidia-smi"``), so a test points it at a fake that emits controlled CSV. No
GPU is ever required by this module or its tests.

Run:  gpu_peak_sampler.py --out <peak.json> [--interval 1.0] [--gpus 0,1]
                          [--gate-margin-mib 4096] -- <child argv...>
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import typing as tp

# Atomic-writer temp naming (mirrors audit_convention.py / aggregate_gate.py so the
# three atomic writers never drift). Temps are ``<out_dir>/.gpu_peak_sampler.<rand>.tmp``.
OUTPUT_TMP_PREFIX = ".gpu_peak_sampler."
OUTPUT_TMP_SUFFIX = ".tmp"

SCHEMA_VERSION = 1
DEFAULT_INTERVAL_S = 1.0
DEFAULT_GATE_MARGIN_MIB = 4096   # plan §3 / review: MIN_FREE_MB = measured peak + 4,096 MiB.
SAMPLER_FAIL_EXITCODE = 3


# ============================================================================================
# Atomic serialisation (mutation target: a non-atomic write leaves a partial peak record).
# ============================================================================================
def atomic_write_json(path: str, record: tp.Mapping[str, tp.Any]) -> None:
    """Serialise ``record`` to ``path`` atomically: temp file in the SAME directory,
    ``fsync``, then ``os.replace`` (atomic on POSIX). ``allow_nan=False`` -> any
    non-finite number raises BEFORE the destination is touched and the temp is removed.
    On ANY failure the destination is left untouched (no partial write)."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=OUTPUT_TMP_PREFIX, suffix=OUTPUT_TMP_SUFFIX)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, allow_nan=False, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic rename
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ============================================================================================
# nvidia-smi sampling.
# ============================================================================================
def _nvidia_smi_cmd() -> str:
    """The nvidia-smi binary. Resolved through ``NVIDIA_SMI`` (default ``nvidia-smi``) so a
    CPU test can inject a fake that emits controlled CSV — the mocked-nvidia-smi seam."""
    return os.environ.get("NVIDIA_SMI", "nvidia-smi")


def query_used_mib(gpus: tp.Sequence[int]) -> tp.Dict[int, int]:
    """ONE external ``nvidia-smi --query-gpu=index,memory.used`` call. Returns
    ``{gpu_index: used_mib}`` for exactly the requested indices that came back with a
    PARSEABLE integer. A failed call, a missing index, or an unparseable value simply
    yields no entry for that GPU this tick (that tick is not a valid sample for it) —
    the module never fabricates a reading. nvidia-smi reports PHYSICAL indices regardless
    of ``CUDA_VISIBLE_DEVICES``, so the physical GPUs are always the ones sampled."""
    want = set(gpus)
    try:
        out = subprocess.run(
            [_nvidia_smi_cmd(), "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    result: tp.Dict[int, int] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            used = int(parts[1])
        except ValueError:
            continue
        if idx in want:
            result[idx] = used
    return result


class _Sampler:
    """Background per-tick sampler. Tracks the running MAX used-MiB and the VALID-sample
    count for each requested GPU. Thread-safe enough for one sampler thread + one reader."""

    def __init__(self, gpus: tp.Sequence[int], interval_s: float):
        self.gpus = list(gpus)
        self.interval_s = interval_s
        self.peaks: tp.Dict[int, tp.Optional[int]] = {g: None for g in self.gpus}
        self.counts: tp.Dict[int, int] = {g: 0 for g in self.gpus}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-peak-sampler", daemon=True)

    def _sample_once(self) -> None:
        used = query_used_mib(self.gpus)
        for g, u in used.items():
            self.counts[g] += 1
            cur = self.peaks[g]
            if cur is None or u > cur:
                self.peaks[g] = u

    def _run(self) -> None:
        # Sample IMMEDIATELY (no initial sleep) so even a short-lived child yields >=1
        # sample when nvidia-smi works; then sample every interval until stopped.
        self._sample_once()
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def start(self) -> None:
        self._thread.start()

    def stop_and_join(self) -> None:
        self._stop.set()
        self._thread.join()
        # One final tick captures a peak that may have landed after the last periodic
        # sample (max() discards it if the child already freed its memory).
        self._sample_once()


# ============================================================================================
# Orchestration.
# ============================================================================================
def assemble_record(
    gpus: tp.Sequence[int],
    peaks: tp.Mapping[int, tp.Optional[int]],
    counts: tp.Mapping[int, int],
    child_exit: tp.Optional[int],
    interval_s: float,
    gate_margin_mib: int,
    child_argv: tp.Sequence[str],
    started_at: float,
    ended_at: float,
) -> tp.Dict[str, tp.Any]:
    """Pure record assembly (no I/O, no timing) — the fail-closed rule and the derived-gate
    math live HERE so they can be unit-tested deterministically.

    * ``sampler_ok`` is True IFF EVERY requested GPU obtained at least one valid sample
      (mutation target: relaxing this to ``any``/``True`` lets a dead sampler through).
    * ``derived_gate_mib = max(peak_gpu0, peak_gpu1) + gate_margin_mib`` — computed ONLY
      when ``sampler_ok`` (peaks are all non-None then); ``None`` otherwise, never a bogus
      small gate. (Mutation target: dropping ``+ gate_margin_mib`` under-provisions the gate.)
    """
    samples_per_gpu = {g: int(counts[g]) for g in gpus}
    per_gpu_peaks = {g: peaks[g] for g in gpus}
    sampler_ok = all(samples_per_gpu[g] >= 1 for g in gpus)

    derived_gate_mib: tp.Optional[int] = None
    if sampler_ok:
        derived_gate_mib = max(per_gpu_peaks[g] for g in gpus) + gate_margin_mib

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "gpu_peak_sampler.py",
        "gpus": list(gpus),
        "interval_s": interval_s,
        "gate_margin_mib": gate_margin_mib,
        "child_command": list(child_argv),
        # keys are stringified so the JSON is portable (JSON object keys are strings).
        "per_gpu_peaks_mib": {str(g): per_gpu_peaks[g] for g in gpus},
        "samples_per_gpu": {str(g): samples_per_gpu[g] for g in gpus},
        "child_exit": child_exit,
        "sampler_ok": sampler_ok,
        "derived_gate_mib": derived_gate_mib,
        "started_at": started_at,
        "ended_at": ended_at,
    }


def run_with_sampling(
    child_argv: tp.Sequence[str],
    gpus: tp.Sequence[int],
    interval_s: float,
    gate_margin_mib: int,
) -> tp.Dict[str, tp.Any]:
    """Start the sampler, run the child to completion, stop the sampler, and assemble the
    record via :func:`assemble_record`."""
    sampler = _Sampler(gpus, interval_s)
    started_at = time.time()
    sampler.start()
    try:
        child = subprocess.Popen(list(child_argv))
        child_exit = child.wait()
    finally:
        sampler.stop_and_join()
    ended_at = time.time()

    return assemble_record(
        gpus, sampler.peaks, sampler.counts, child_exit, interval_s,
        gate_margin_mib, child_argv, started_at, ended_at,
    )


def _split_child_argv(argv: tp.Sequence[str]) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """Split ``argv`` at the FIRST ``--`` into (sampler_args, child_argv)."""
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    own, child_argv = _split_child_argv(raw)

    parser = argparse.ArgumentParser(
        description="exp-09 C1 external GPU peak-VRAM sampler (wraps a child command)")
    parser.add_argument("--out", required=True, help="output peak JSON path (atomic)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="sampling interval in seconds (default 1.0)")
    parser.add_argument("--gpus", default="0,1",
                        help="comma-separated physical GPU indices to sample (default 0,1)")
    parser.add_argument("--gate-margin-mib", type=int, default=DEFAULT_GATE_MARGIN_MIB,
                        help="MiB added to the peak for the derived gate (default 4096)")
    args = parser.parse_args(own)

    if not child_argv:
        parser.error("no child command given: put it after `--`")
    if args.interval <= 0:
        parser.error(f"--interval must be > 0 (got {args.interval})")
    try:
        gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    except ValueError:
        parser.error(f"--gpus must be comma-separated integers (got {args.gpus!r})")
    if not gpus:
        parser.error("--gpus resolved to an empty list")

    record = run_with_sampling(child_argv, gpus, args.interval, args.gate_margin_mib)
    atomic_write_json(args.out, record)

    peaks = record["per_gpu_peaks_mib"]
    if not record["sampler_ok"]:
        print(
            "gpu_peak_sampler: FAIL-CLOSED — a GPU obtained NO valid sample "
            f"(samples_per_gpu={record['samples_per_gpu']}, peaks={peaks}); "
            "refusing to report a peak. Is nvidia-smi alive?",
            file=sys.stderr,
        )
        return SAMPLER_FAIL_EXITCODE
    print(
        f"gpu_peak_sampler: peaks(MiB)={peaks} samples={record['samples_per_gpu']} "
        f"derived_gate_mib={record['derived_gate_mib']} child_exit={record['child_exit']} "
        f"-> {args.out}"
    )
    return record["child_exit"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
