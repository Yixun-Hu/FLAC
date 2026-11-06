"""exp_37 run provenance (plan_loc_efficiency.md §10 "CPU co-tenancy" and "Provenance").

- ``attention_backend()``: the DiT attention backend actually selected in THIS process
  (``src.models.transformer.flash_attn_func is None`` -> ``"sdpa"``); ``importlib.util.find_spec`` cannot see the
  no-flash shadow, so the module attribute is the only truthful source.
- ``attention_identity()``: what a run manifest records -- always the backend; with ``require="sdpa"`` also the
  verified, pid-free part of the no-flash launcher's in-process hand-off (``EXP37_NO_FLASH_LAUNCH``), refusing
  unless this very process was started by ``overlays/no_flash/exp37_no_flash_launch.py`` on this checkout's
  overlay and launcher bytes and runs SDPA.
- ``host_snapshot()`` / ``write_host_record()``: CPU affinity, torch intra- / inter-op threads, OMP / MKL / BLAS
  variables, the CUDA card, and every co-tenant (GPU compute apps, top CPU processes) -- outside every timer.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import torch

LAUNCH_ENV = "EXP37_NO_FLASH_LAUNCH"
SENTINEL = "EXP37_NO_FLASH_LAUNCHER"
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"   # the reviewed exp_35 bytes
OVERLAY_REL = ("overlays", "no_flash", "flash_attn.py")
LAUNCHER_REL = ("overlays", "no_flash", "exp37_no_flash_launch.py")
HANDOFF_KEYS = frozenset({"attention_backend", "flash_attn_func_is_none", "overlay_sha256", "launcher_sha256",
                          "pid", "target", "sentinel"})
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")
TOP_PROCESSES = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attention_backend() -> str:
    from src.models import transformer

    return "sdpa" if transformer.flash_attn_func is None else "flash_attn"


def no_flash_launch_handoff(repo_root: Path | str, *, environ=None) -> dict:
    """The launcher's hand-off, verified against THIS process and THIS checkout; RuntimeError otherwise."""
    environ = os.environ if environ is None else environ
    raw = environ.get(LAUNCH_ENV)
    if raw is None:
        raise RuntimeError(f"no {LAUNCH_ENV} hand-off: not launched through {'/'.join(LAUNCHER_REL)}")
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise RuntimeError(f"the {LAUNCH_ENV} hand-off is not JSON") from error
    if not isinstance(data, dict) or set(data) != HANDOFF_KEYS:
        raise RuntimeError(f"the {LAUNCH_ENV} hand-off has unexpected keys")
    if data["pid"] != os.getpid():
        raise RuntimeError(f"the {LAUNCH_ENV} hand-off belongs to another process (pid {data['pid']})")
    if data["attention_backend"] != "sdpa" or data["flash_attn_func_is_none"] is not True:
        raise RuntimeError("the hand-off's asserted backend is not sdpa")
    if data["sentinel"] != SENTINEL:
        raise RuntimeError(f"the hand-off's sentinel is not {SENTINEL}")
    root = Path(repo_root)
    overlay = _sha256(root.joinpath(*OVERLAY_REL))
    if not data["overlay_sha256"] == overlay == OVERLAY_SHA256:
        raise RuntimeError("the overlay digest differs from the hand-off or from the reviewed pin")
    if data["launcher_sha256"] != _sha256(root.joinpath(*LAUNCHER_REL)):
        raise RuntimeError("the launcher digest differs from this checkout's launcher")
    if attention_backend() != "sdpa":
        raise RuntimeError("src.models.transformer.flash_attn_func is not None: the DiT would run flash attention")
    return data


def attention_identity(repo_root: Path | str, require: str | None, *, environ=None) -> dict:
    """The run-manifest part: the backend, plus (when SDPA is required) the pid-free launcher provenance."""
    if require is None:
        return {"attention_backend": attention_backend()}
    if require != "sdpa":
        raise ValueError("only the sdpa attention backend can be required")
    handoff = no_flash_launch_handoff(repo_root, environ=environ)
    return {"attention_backend": attention_backend(),
            "no_flash_launch": {key: handoff[key] for key in ("overlay_sha256", "launcher_sha256", "sentinel")}}


def _command(run, argv: list[str], limit: int | None = None) -> dict:
    try:
        done = run(argv, capture_output=True, text=True, timeout=30, check=False)
    except Exception as error:                     # noqa: BLE001 -- a missing tool is recorded, never fatal
        return {"argv": argv, "error": f"{type(error).__name__}: {error}"}
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return {"argv": argv, "returncode": done.returncode, "lines": lines[:limit] if limit else lines}


def _cuda_device(device: str) -> dict | None:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(target)
    return {"index": target.index, "name": props.name, "uuid": str(getattr(props, "uuid", "")),
            "total_memory_bytes": int(props.total_memory)}


def host_snapshot(device: str, *, run=subprocess.run) -> dict:
    return {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
        "cuda_device": _cuda_device(device),
        "gpu_compute_apps": _command(run, ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                           "--format=csv,noheader"]),
        "top_cpu_processes": _command(run, ["ps", "-eo", "pid,ppid,user,pcpu,pmem,etime,comm", "--sort=-pcpu",
                                            "--no-headers"], limit=TOP_PROCESSES),
    }


def write_host_record(output_dir: Path | str, label: str, *, device: str, handoff: dict | None,
                      run_manifest_sha256: str, run=subprocess.run) -> Path:
    """``<output>/host_records/<label>_<UTC>_pid<pid>.json`` (one file per process and moment; never overwritten)."""
    snapshot = host_snapshot(device, run=run)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_dir) / "host_records" / f"{label}_{stamp}_pid{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"label": label, "run_manifest_sha256": run_manifest_sha256, "no_flash_launch_handoff": handoff,
               "snapshot": snapshot}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path
