"""exp_37 fix round 2 (Codex code review 2, HIGH 8): the runbook's step receipts (``tools/exp37_receipt.py``).

``write`` records a finished step (status ok / skipped, HEAD, the exact command, the log, start / end times) as a
hashed, write-once JSON; ``check`` refuses (exit 2) a missing, tampered or foreign receipt, another status, another
HEAD, or one older than ``--max-age-hours``. Standard library only. CPU.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = [sys.executable, str(REPO / "tools" / "exp37_receipt.py")]


def _write(path, **over):
    args = {"step": "B1.0", "phase": "B1", "status": "ok", "head": "a" * 40, "command": "bash x", "log": "l.log",
            "started": "2026-11-06T00:00:00+00:00", "finished": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.update(over)
    argv = TOOL + ["write", "--path", str(path)] + [x for k, v in args.items() for x in (f"--{k}", v)]
    return subprocess.run(argv, capture_output=True, text=True)


def _check(path, *extra):
    return subprocess.run(TOOL + ["check", "--path", str(path), "--step", "B1.0", *extra], capture_output=True, text=True)


def test_a_receipt_is_written_once_and_checked(tmp_path):
    path = tmp_path / "receipts" / "B1.0.json"
    assert _write(path).returncode == 0
    payload = json.loads(path.read_text())
    assert payload["record_type"] == "exp37_runbook_receipt" and payload["status"] == "ok" and len(payload["sha256"]) == 64
    assert _write(path).returncode == 2                                     # write-once
    assert _check(path, "--status", "ok", "--head", "a" * 40, "--max-age-hours", "6").returncode == 0


def test_every_mismatch_is_refused(tmp_path):
    path = tmp_path / "B1.0.json"
    _write(path, finished="2026-01-01T00:00:00+00:00")
    assert _check(path, "--status", "skipped").returncode == 2
    assert _check(path, "--head", "b" * 40).returncode == 2
    assert _check(path, "--max-age-hours", "6").returncode == 2             # stale
    assert _check(tmp_path / "missing.json").returncode == 2
    body = json.loads(path.read_text())
    path.write_text(json.dumps({**body, "status": "skipped"}))              # tampered
    assert _check(path).returncode == 2
    other = tmp_path / "other.json"
    _write(other, step="B2.0")
    assert _check(other).returncode == 2                                    # another step's receipt
