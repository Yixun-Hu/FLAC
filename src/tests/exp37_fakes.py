"""exp_37 shared CPU fakes for the localization executors (a tiny conditioner / DiT / pretransform / retrieval and
a frozen query) -- test support only, never imported by production code.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c (moved out of test_exp37_stage_split.py)."""
from pathlib import Path

import numpy as np
import torch


class FakeConditioner:
    def __call__(self, metadata, device, only_ids=None):
        out = {}
        for key in only_ids:
            values = torch.stack([item[key].float().reshape(-1)[:3] for item in metadata]).unsqueeze(1).to(device)
            out[key] = [values, torch.ones(len(metadata), 1, device=device)]
        return out


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, cond=None, **sampler_kwargs):      # cfg_scale, batch_cfg (cfg 1.0: unused)
        return x * self.w + cond.mean(dim=1).view(-1, 1, 1)


class FakePretransform:
    downsampling_ratio = 1024

    def decode(self, z):
        return z.mean(dim=1, keepdim=True).repeat_interleave(1024, dim=-1) * 0.1


class FakeDiffusion:
    diffusion_objective, io_channels, dist_shift = "rectified_flow", 2, None

    def __init__(self):
        self.conditioner, self.model, self.pretransform = FakeConditioner(), FakeDiT(), FakePretransform()

    def get_conditioning_inputs(self, conditioning):
        return {"cond": torch.cat([conditioning[k][0].reshape(conditioning[k][0].shape[0], -1)
                                   for k in sorted(conditioning)], dim=1)}


class FakeModule:
    device = "cpu"

    def __init__(self):
        self.diffusion = FakeDiffusion()


class FakeRetrieval:
    def compute_audio_features(self, waveforms):
        return torch.nn.functional.normalize(waveforms[:, 0, :16] + torch.arange(16.0), dim=1)


def _fake_query(record, root):
    k = len(record["contexts"])
    gen = torch.Generator().manual_seed(7)
    poses = torch.arange(k * 3, dtype=torch.float32).reshape(k, 3) / 10 + 0.1
    metadata = {"scene": "S", "source": torch.tensor([1.0, 0.5, 0.2]), "source_vit": torch.tensor([[1.0, 0.5, 0.2]]),
                "context_poses": poses, "context_poses_vit": poses.clone(),
                "context_audio": torch.linspace(0, 1, k * 16).reshape(k, 1, 16), "depth": torch.rand(3, 4, 8, generator=gen)}
    return torch.linspace(-0.5, 0.5, 10240).reshape(1, 10240), metadata


CANDIDATES = np.array([[0.25 * i, 0.5, 1.0] for i in range(70)])                     # two batches of 64 + 6
GRID = np.array([[x, y, 1.0] for x in np.arange(0.0, 4.01, 0.5) for y in np.arange(0.0, 3.01, 0.5)])   # 63 points


def example_identity(expected: dict, audit: dict) -> dict:
    """The run-manifest identity a correct runner writes for ``expected`` (test support; exp37_identity)."""
    import copy

    from src.localization.exp37_identity import ABSENT
    out = {}
    for key, want in expected.items():
        if want is ABSENT:
            continue
        if key == "mesh_root_override":
            out[key] = {"argument": "AcousticRooms/room_mesh_obj_format", "resolved": "/x", "audit_mesh_root": audit["mesh_root"],
                        "rooms": {room: {"mesh_path": f"/x/{room}.obj", "mesh_sha256": row["mesh_sha256"]}
                                  for room, row in audit["rooms"].items()}}
        elif key == "waveform_dump":
            out[key] = {"mode": want, "dir": "/dump"}
        elif key == "replay_of":
            out[key] = {"run_manifest_sha256": want, "run_dir": "/run"}
        else:
            out[key] = copy.deepcopy(want)
    return out


# --------------------------------------------------------------------------- #
# co-tenancy attestations (Codex code review 1, BLOCKING 3): what the runner / selector + the monitor child write
# --------------------------------------------------------------------------- #
CARD = "6ac2604a-1111"
HOST = "exp37-host"
START = 123
PRODUCER = {"pid": 42, "pid_start_ticks": START, "hostname": HOST, "card_uuid": CARD}


def host_snapshot(pid=42, card=CARD, hostname=HOST, **override):
    """A host snapshot as run_provenance.host_snapshot records it (both host commands successful)."""
    return {"pid": pid, "hostname": hostname, "cuda_device": {"uuid": card},
            "gpu_compute_apps": {"argv": ["nvidia-smi"], "returncode": 0, "lines": [f"GPU-{card}, {pid}, python, 9 MiB"]},
            "top_cpu_processes": {"argv": ["ps"], "returncode": 0, "lines": [f"{pid} 1 me 99.0 1.0 10:00 python"]},
            **override}


def _monitor_script():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "exp37_cotenancy_monitor.py"
    spec = importlib.util.spec_from_file_location("exp37_cotenancy_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_attestation(directory, binding_sha256, *, pid=42, card=CARD, co_tenant=False, exp36=False, start=START,
                      hostname=HOST):
    """A monitor stream (one sample per second over t = 0..20 s) and the markers 2 / 3 / 18 s; ``co_tenant`` puts
    another pid on the card at t = 10 s, ``exp36`` an exp_36 process on the host at t = 5 s."""
    from src.localization import exp37_cotenancy as cot

    directory = Path(directory)
    directory.mkdir(parents=True)
    out = _monitor_script().ChainWriter(directory / "monitor.jsonl", binding_sha256=binding_sha256)
    out.write({"kind": "start", "target_pid": pid, "target_start": start, "target_comm": "python", "card_uuid": card,
               "proc": "/proc", "interval_seconds": 1.0, "hostname": hostname, "exp36_roots": list(cot.EXP36_ROOTS)})
    for t in range(21):
        lines = [f"GPU-{card}, {pid}, python, 900 MiB"] + ([f"GPU-{card}, 77, python, 9 MiB"] if co_tenant and t == 10
                                                           else [])
        out.write({"kind": "sample", "t_monotonic": float(t), "target_alive": True, "gpu": {"returncode": 0, "lines": lines},
                   "exp36": [{"pid": 5, "cwd": "/home/yixunhu/codespace/exp-36-yaw-c4-spread"}] if exp36 and t == 5 else [],
                   "cpu": {"interval_start": t - 1.0, "interval_end": float(t), "interval_seconds": 1.0,
                           "foreign_cores": 0.2},
                   "foreign_processes": [{"pid": 9, "comm": "kworker", "uid": 0, "cores": 0.2}]})
    out.write({"kind": "stop", "reason": "stop_file"})
    markers = cot.ChainWriter(directory / "markers.jsonl", binding_sha256=binding_sha256)
    for name, t in zip(cot.MARKERS, (2.0, 3.0, 18.0)):
        markers.write({"marker": name, "pid": pid, "t_monotonic": t})
    return directory


def attest_run(run_dir, *, loaded_extra=None, **kwargs):
    """The producer record, hashed start / loaded / end host records bound to the run's manifest, and its monitor
    attestation -- all naming the same pid, /proc start time, host and card."""
    import json

    from src.localization.pilot import canonical_sha256

    run_dir = Path(run_dir)
    binding = json.loads((run_dir / "run_manifest.json").read_text())["sha256"]
    producer = {"record_type": "exp37_producer", "run_manifest_sha256": binding, "pid": kwargs.get("pid", 42),
                "pid_start_ticks": kwargs.get("start", START), "hostname": kwargs.get("hostname", HOST),
                "card_uuid": kwargs.get("card", CARD)}
    (run_dir / "producer.json").write_text(json.dumps({**producer, "sha256": canonical_sha256(producer)}))
    for label in ("start", "loaded", "end"):
        body = {"label": label, "run_manifest_sha256": binding, "no_flash_launch_handoff": None,
                "snapshot": host_snapshot(pid=kwargs.get("pid", 42), card=kwargs.get("card", CARD),
                                          hostname=kwargs.get("hostname", HOST)),
                "extra": loaded_extra if label == "loaded" else None}
        path = run_dir / "host_records" / f"{label}_20261106T000000000000Z_pid42.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    write_attestation(run_dir / "cotenancy", binding, **kwargs)
    return run_dir


def selector_binding(payload_fields: dict, output) -> dict:
    """The selector run's binding as tools/exp37_selector_latency.py writes it (before the measurement)."""
    from src.localization.exp37_latency import SELECTOR_BOUND
    from src.localization.pilot import canonical_sha256

    body = {"record_type": "exp37_selector_binding", "output": str(output), "pid": 42, "pid_start_ticks": START,
            "hostname": HOST, "card_uuid": CARD, "started_at_utc": "2026-11-06T00:00:00+00:00",
            **{key: payload_fields[key] for key in SELECTOR_BOUND}}
    return {**body, "sha256": canonical_sha256(body)}


def attest_selector(path, binding: dict, **kwargs):
    return write_attestation(Path(str(path) + ".cotenancy"), binding["sha256"], **kwargs)


# --------------------------------------------------------------------------- #
# selector vectors with the full repeat evidence (Codex code review 2, HIGH 3)
# --------------------------------------------------------------------------- #
SELECTOR_FIELDS = ("observed_encode_seconds", "generated_encode_seconds", "similarity_seconds", "argmax_seconds",
                   "scoring_total_seconds")


def selector_row(record: dict, seconds: float, peak: int = 5) -> dict:
    """One query row exactly as tools/exp37_selector_latency.py writes it: three repeats of the bundle's four parts,
    their per-repeat total, and every per-field median."""
    import statistics
    parts = {"observed_encode_seconds": 0.01, "generated_encode_seconds": seconds - 0.013, "similarity_seconds": 0.002,
             "argmax_seconds": 0.001}
    repeats = [{**parts, "scoring_total_seconds": parts["observed_encode_seconds"] + parts["generated_encode_seconds"]
                + parts["similarity_seconds"] + parts["argmax_seconds"]} for _ in range(3)]
    return {"query_index": record["index"], "query_id": record["query_id"], "room": record["room"],
            "candidate_count": record["candidate_count"], "candidate_indices_sha256": record["candidate_indices_sha256"],
            "agree": {"repeat_seconds": {f: [r[f] for r in repeats] for f in SELECTOR_FIELDS},
                      "median_seconds": {f: float(statistics.median(float(r[f]) for r in repeats)) for f in SELECTOR_FIELDS}},
            "peak_memory_allocated_bytes": peak, "peak_memory_reserved_bytes": peak}


def selector_fields(selection: dict, seconds: float, **over) -> dict:
    """A complete exp_37 selector payload body (before its binding and sha256)."""
    from src.localization import exp37_identity as ident
    return {"schema_version": 1, "record_type": "exp37_agree_selector_latency", "device": "cuda:0",
            "timed_function": "tools/benchmark_localization_selector_latency.py::benchmark_agree_query",
            "repeat_count": 3, "candidate_batch_size": 64, "selection_sha256": selection["sha256"],
            "agree_checkpoint_file_sha256": ident.AGREE_SHA256, "context_manifest_sha256": ident.CONTEXT_MANIFEST_SHA256,
            "query_count": len(selection["records"]), "host": host_snapshot(),
            "queries": [selector_row(r, seconds) for r in selection["records"]], **over}
