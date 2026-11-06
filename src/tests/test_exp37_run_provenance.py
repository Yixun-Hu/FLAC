"""exp_37 round B-a: run provenance -- the attention backend actually selected, the verification of the no-flash
launcher's in-process hand-off, and the host / co-tenancy snapshot every run records (plan_loc_efficiency.md §10
"CPU co-tenancy" and "Provenance"). CPU only; no model is built.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""
import hashlib
import json
import os
import shutil
import subprocess

import pytest

import src.models.transformer as transformer
from src.localization import run_provenance as rp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERLAY_SHA256 = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_attention_backend_reads_the_transformer_module(monkeypatch):
    monkeypatch.setattr(transformer, "flash_attn_func", None)
    assert rp.attention_backend() == "sdpa"
    monkeypatch.setattr(transformer, "flash_attn_func", object())
    assert rp.attention_backend() == "flash_attn"


@pytest.fixture
def launch_repo(tmp_path, monkeypatch):
    """A repo holding the real overlay + launcher, a valid hand-off for THIS pid, SDPA in-process."""
    ov = tmp_path / "overlays" / "no_flash"
    ov.mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "overlays", "no_flash", "flash_attn.py"), ov / "flash_attn.py")
    shutil.copy(os.path.join(REPO, "overlays", "no_flash", "exp37_no_flash_launch.py"), ov / "exp37_no_flash_launch.py")
    handoff = {"attention_backend": "sdpa", "flash_attn_func_is_none": True, "overlay_sha256": OVERLAY_SHA256,
               "launcher_sha256": _sha(ov / "exp37_no_flash_launch.py"), "pid": os.getpid(),
               "target": "localize_FLAC.py", "sentinel": "EXP37_NO_FLASH_LAUNCHER"}
    monkeypatch.setattr(transformer, "flash_attn_func", None)
    return tmp_path, handoff


def _env(handoff):
    return {rp.LAUNCH_ENV: json.dumps(handoff)}


def test_a_valid_handoff_is_verified_and_returned(launch_repo):
    root, handoff = launch_repo
    assert rp.no_flash_launch_handoff(root, environ=_env(handoff)) == handoff


def test_the_stable_identity_part_excludes_the_pid(launch_repo):
    root, handoff = launch_repo
    got = rp.attention_identity(root, "sdpa", environ=_env(handoff))
    assert got == {"attention_backend": "sdpa",
                   "no_flash_launch": {"overlay_sha256": OVERLAY_SHA256, "launcher_sha256": handoff["launcher_sha256"],
                                       "sentinel": "EXP37_NO_FLASH_LAUNCHER"}}


def test_without_a_requirement_only_the_backend_is_recorded(launch_repo, monkeypatch):
    root, _handoff = launch_repo
    assert rp.attention_identity(root, None, environ={}) == {"attention_backend": "sdpa"}
    monkeypatch.setattr(transformer, "flash_attn_func", object())
    assert rp.attention_identity(root, None, environ={}) == {"attention_backend": "flash_attn"}


@pytest.mark.parametrize("mutate, needle", [
    (lambda h: None, "no EXP37_NO_FLASH_LAUNCH hand-off"),
    (lambda h: "{not json", "not JSON"),
    (lambda h: {**h, "extra": 1}, "unexpected keys"),
    (lambda h: {k: v for k, v in h.items() if k != "target"}, "unexpected keys"),
    (lambda h: {**h, "pid": h["pid"] + 1}, "another process"),
    (lambda h: {**h, "attention_backend": "flash_attn"}, "asserted backend"),
    (lambda h: {**h, "flash_attn_func_is_none": False}, "asserted backend"),
    (lambda h: {**h, "sentinel": "EXP35_NO_FLASH_LAUNCHER"}, "sentinel"),
    (lambda h: {**h, "overlay_sha256": "0" * 64}, "overlay"),
    (lambda h: {**h, "launcher_sha256": "0" * 64}, "launcher"),
], ids=["absent", "garbage", "extra-key", "missing-key", "other-pid", "flash-claimed", "flag-false",
        "wrong-sentinel", "overlay-digest", "launcher-digest"])
def test_a_bad_handoff_is_refused(launch_repo, mutate, needle):
    root, handoff = launch_repo
    value = mutate(dict(handoff))
    environ = {} if value is None else {rp.LAUNCH_ENV: value if isinstance(value, str) else json.dumps(value)}
    with pytest.raises(RuntimeError, match=needle):
        rp.attention_identity(root, "sdpa", environ=environ)


def test_an_edited_overlay_file_is_refused_even_with_a_matching_claim(launch_repo):
    root, handoff = launch_repo
    path = root / "overlays" / "no_flash" / "flash_attn.py"
    path.write_text(path.read_text() + "# edited\n")
    with pytest.raises(RuntimeError, match="overlay"):
        rp.attention_identity(root, "sdpa", environ=_env({**handoff, "overlay_sha256": _sha(path)}))


def test_flash_in_process_is_refused_whatever_the_handoff_says(launch_repo, monkeypatch):
    root, handoff = launch_repo
    monkeypatch.setattr(transformer, "flash_attn_func", object())
    with pytest.raises(RuntimeError, match="flash"):
        rp.attention_identity(root, "sdpa", environ=_env(handoff))


def test_only_sdpa_can_be_required(launch_repo):
    root, handoff = launch_repo
    with pytest.raises(ValueError, match="sdpa"):
        rp.attention_identity(root, "flash_attn", environ=_env(handoff))


def _fake_run(outputs):
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        result = outputs[argv[0]]
        if isinstance(result, Exception):
            raise result
        return subprocess.CompletedProcess(argv, 0, stdout=result, stderr="")
    return run, calls


def test_host_snapshot_records_threads_affinity_environment_and_co_tenants(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    run, calls = _fake_run({"nvidia-smi": "GPU-abc, 123, /usr/bin/python, 1000 MiB\n",
                            "ps": "".join(f"{i} 1 user 9.0 0.1 01:00 proc{i}\n" for i in range(30))})
    snap = rp.host_snapshot("cpu", run=run)
    assert snap["pid"] == os.getpid() and snap["cpu_affinity"] == sorted(os.sched_getaffinity(0))
    assert snap["environment"]["OMP_NUM_THREADS"] == "7" and snap["environment"]["MKL_NUM_THREADS"] is None
    assert isinstance(snap["torch_num_threads"], int) and isinstance(snap["torch_num_interop_threads"], int)
    assert snap["gpu_compute_apps"]["lines"] == ["GPU-abc, 123, /usr/bin/python, 1000 MiB"]
    assert len(snap["top_cpu_processes"]["lines"]) == 20 and snap["top_cpu_processes"]["lines"][0].endswith("proc0")
    assert snap["cuda_device"] is None                                    # a CPU device records no card
    assert [c[0][0] for c in calls] == ["nvidia-smi", "ps"]
    assert all(c[1].get("timeout") and c[1].get("check") is False for c in calls)


def test_a_failing_tool_is_recorded_not_raised():
    run, _ = _fake_run({"nvidia-smi": FileNotFoundError("nvidia-smi"), "ps": "1 0 root 0.0 0.0 00:01 init\n"})
    snap = rp.host_snapshot("cpu", run=run)
    assert "FileNotFoundError" in snap["gpu_compute_apps"]["error"] and snap["top_cpu_processes"]["lines"]


def test_a_host_record_is_written_atomically_under_host_records(tmp_path):
    run, _ = _fake_run({"nvidia-smi": "", "ps": ""})
    path = rp.write_host_record(tmp_path, "start", device="cpu", handoff={"pid": os.getpid()},
                                run_manifest_sha256="abc", run=run)
    assert path.parent == tmp_path / "host_records" and path.name.startswith("start_")
    assert path.name.endswith(f"_pid{os.getpid()}.json")
    payload = json.loads(path.read_text())
    assert payload["label"] == "start" and payload["run_manifest_sha256"] == "abc"
    assert payload["no_flash_launch_handoff"] == {"pid": os.getpid()} and payload["snapshot"]["pid"] == os.getpid()
    assert not list((tmp_path / "host_records").glob("*.tmp"))


def test_a_host_record_can_carry_extra_measurements(tmp_path):
    run, _ = _fake_run({"nvidia-smi": "", "ps": ""})
    path = rp.write_host_record(tmp_path, "loaded", device="cpu", handoff=None, run_manifest_sha256="abc", run=run,
                                extra={"memory_after_flac_load": None})
    assert json.loads(path.read_text())["extra"] == {"memory_after_flac_load": None}


def test_the_weights_footprint_is_none_off_cuda():
    assert rp.weights_footprint("cpu") is None
