"""exp_22 A7 (plan §9 A7 r3 (b), (i)): the launcher's OWN sealed admission evidence -- 31 polls of the chosen card
60 s apart, each appended (time, the compute-app output and its sha256) to an O_EXCL capture record -- and its
verifier: the count, the cadence, every poll free, every table re-parsed, one index, the freshness bound."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from conftest import C, GPU0, GPU1, MAP, FakeWorld, cfg_for, load, sandbox_tools

E = load("exp22_r2c_evidence")
T0 = 1_800_000_000.0
DONE = "EVAL_DONE ACEattn s43 cells=40 2026-11-06T15:00:00-05:00\n"


class Stalling(FakeWorld):
    """apps answers by time; ``stall`` adds seconds inside the n-th apps query; ``unreadable_at`` raises there."""

    def __init__(self, apps_at=lambda t: "", stall=None, unreadable_at=None):
        super().__init__(t=T0)
        self.apps_at, self.stall, self.bad, self.n = apps_at, dict(stall or {}), unreadable_at, 0
        self.answers["apps"] = self._apps

    def _apps(self):
        self.n += 1
        if self.n == self.bad:
            raise C.Unreadable("nvidia-smi --query-compute-apps=pid,gpu_uuid: rc 15")
        self.t += self.stall.pop(self.n, 0)
        return self.apps_at(self.t)


def capture(tmp_path, world, card="GPU0"):
    cfg = cfg_for(tmp_path)
    with open(cfg.queue_log, "a") as f:
        f.write(DONE)
    rec = tmp_path / "capture_polls.jsonl"
    rc, info = E.capture_polls(world, cfg, card, str(rec))
    return rc, info, rec


def test_31_free_polls_make_a_record_the_verifier_admits(tmp_path):
    rc, info, rec = capture(tmp_path, Stalling(apps_at=lambda t: f"9, {GPU1}\n"))
    assert rc == 0, info
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    assert len(rows) == 31 and [r["k"] for r in rows] == list(range(31)) and rows[-1]["t"] - rows[0]["t"] == 1800
    assert rows[3]["apps_raw"] == f"9, {GPU1}\n" and rows[3]["apps_sha256"] == hashlib.sha256(rows[3]["apps_raw"].encode()).hexdigest()
    why, got = E.capture_reasons(rec.read_bytes(), "GPU0")
    assert why == [] and got == {"card": "GPU0", "uuid": GPU0, "index": 0, "first": T0, "last": T0 + 1800, "polls": 31}


def test_gpu1_is_captured_on_its_own_rule(tmp_path):
    rc, info, rec = capture(tmp_path, Stalling(apps_at=lambda t: f"9, {GPU0}\n"), card="GPU1")
    assert rc == 0 and E.capture_reasons(rec.read_bytes(), "GPU1")[0] == []
    assert E.capture_reasons(rec.read_bytes(), "GPU0")[0]                  # a record of the other card


def test_a_busy_poll_ends_the_capture_and_the_record_shows_it(tmp_path):
    rc, why, rec = capture(tmp_path, Stalling(apps_at=lambda t: f"5, {GPU0}\n" if t >= T0 + 17 * 60 else ""))
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    assert rc == 1 and "poll 17" in why and len(rows) == 18 and rows[-1]["free"] is False
    assert E.capture_reasons(rec.read_bytes(), "GPU0")[0]


def test_a_late_poll_ends_the_capture(tmp_path):
    rc, why, rec = capture(tmp_path, Stalling(stall={9: 90}))             # the 9th apps query (poll 8) stalls 90 s
    assert rc == 1 and "late" in why and len(rec.read_text().splitlines()) == 10


def test_an_unreadable_input_ends_the_capture(tmp_path):
    rc, why, rec = capture(tmp_path, Stalling(unreadable_at=4))
    rows = [json.loads(x) for x in rec.read_text().splitlines()]
    assert rc == 1 and "unreadable" in why and "unreadable" in rows[-1] and len(rows) == 4


def test_the_capture_record_is_never_overwritten(tmp_path):
    (tmp_path / "capture_polls.jsonl").write_text("prior\n")
    with pytest.raises(FileExistsError):
        capture(tmp_path, Stalling())
    assert (tmp_path / "capture_polls.jsonl").read_text() == "prior\n"


def _rows(tmp_path):
    rc, info, rec = capture(tmp_path, Stalling())
    assert rc == 0
    return [json.loads(x) for x in rec.read_text().splitlines()]


def _raw(rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


@pytest.mark.parametrize("tamper", [
    lambda rs: rs.pop(10),
    lambda rs: rs.append(dict(rs[-1], k=31, due=rs[-1]["due"] + 60, t=rs[-1]["t"] + 60)),
    lambda rs: rs.reverse(),
    lambda rs: rs[5].update(apps_raw=f"77, {GPU0}\n", apps_sha256=hashlib.sha256(f"77, {GPU0}\n".encode()).hexdigest()),
    lambda rs: rs[5].update(apps_raw=f"77, {GPU1}\n"),                     # the sha256 no longer binds it
    lambda rs: rs[6].update(due=rs[6]["due"] + 30, t=rs[6]["t"] + 30),
    lambda rs: rs[7].update(t=rs[7]["due"] + 16),
    lambda rs: rs[8].update(free=False),
    lambda rs: rs[9].update(reasons=["x"]),
    lambda rs: rs[11].update(rule="exp35 lock HELD"),
    lambda rs: rs[12].update(index=1),
    lambda rs: rs[13].update(gpu_map_raw=f"0, {GPU1}\n1, {GPU0}\n"),
    lambda rs: rs[14].update(reservation="present (holder exp35)"),
    lambda rs: rs[15].update(uuid=GPU1),
    lambda rs: rs[16].update(unreadable="x"),
], ids=["dropped", "extra", "reordered", "card-listed", "unbound-table", "off-cadence", "late", "busy", "reasons",
        "lock-held", "index", "map", "reserved", "uuid", "unreadable"])
def test_a_tampered_record_is_refused(tmp_path, tamper):
    rows = _rows(tmp_path)
    assert E.capture_reasons(_raw(rows), "GPU0")[0] == []
    tamper(rows)
    assert E.capture_reasons(_raw(rows), "GPU0")[0]


def test_non_json_is_refused():
    assert E.capture_reasons(b"\xff\xfe", "GPU0")[0] and E.capture_reasons(b"{not json\n", "GPU0")[0]
    assert E.capture_reasons(b"", "GPU0")[0] and E.capture_reasons(b"[]\n" * 31, "GPU0")[0]


# --------------------------------------------------------------------------- #
# the CLI in a sandbox copy: capture-polls, then check-capture (sha256 + freshness)
# --------------------------------------------------------------------------- #
def _sandbox(tmp_path, apps=""):
    cfg = cfg_for(tmp_path)
    with open(cfg.queue_log, "a") as f:
        f.write(DONE)
    nv = tmp_path / "nvidia-smi"
    (tmp_path / "apps.txt").write_text(apps)
    nv.write_text(f"#!/bin/sh\ncase \"$1\" in --query-gpu=*) printf '%s' '{MAP}';; *) cat {tmp_path / 'apps.txt'};; esac\n")
    nv.chmod(0o755)
    tools = sandbox_tools(tmp_path / "W", common=dict(
        RESERVATIONS=cfg.reservations, EXP35_LOCK=cfg.exp35_lock, QUEUE_LOG=cfg.queue_log, PROC=cfg.proc,
        NVIDIA_SMI=str(nv), POLL_S=0.05, WINDOW_POLLS=4, POLL_SLACK_S=5, FRESH_S=2))
    return os.path.join(tools, "exp22_r2c_evidence.py")


def _cli(script, *args):
    return subprocess.run([sys.executable, "-B", script, *args], capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


def test_the_cli_captures_and_checks_with_the_sha256_and_the_freshness_bound(tmp_path):
    ev = _sandbox(tmp_path)
    rec = tmp_path / "capture_polls.jsonl"
    p = _cli(ev, "capture-polls", "--card", "GPU0", "--record", str(rec))
    assert p.returncode == 0 and p.stdout.startswith("CAPTURE POLLS OK"), p.stdout + p.stderr
    sha = hashlib.sha256(rec.read_bytes()).hexdigest()
    p = _cli(ev, "check-capture", "--card", "GPU0", "--record", str(rec), "--sha256", sha, "--fresh")
    assert p.returncode == 0 and p.stdout.startswith(f"CAPTURE OK card GPU0 uuid {GPU0} index 0 first "), p.stdout
    assert " last " in p.stdout and " age " in p.stdout
    p = _cli(ev, "check-capture", "--card", "GPU0", "--record", str(rec), "--sha256", "0" * 64, "--fresh")
    assert p.returncode == 1 and "sha256" in p.stdout
    p = _cli(ev, "check-capture", "--card", "GPU1", "--record", str(rec), "--sha256", sha)
    assert p.returncode == 1
    import time
    time.sleep(2.2)                                                         # past the sandbox's 2 s bound
    p = _cli(ev, "check-capture", "--card", "GPU0", "--record", str(rec), "--sha256", sha, "--fresh")
    assert p.returncode == 1 and "stale" in p.stdout
    p = _cli(ev, "capture-polls", "--card", "GPU0", "--record", str(rec))
    assert p.returncode == 1 and "exists" in p.stdout                     # never overwritten


def test_the_cli_refuses_a_busy_card(tmp_path):
    ev = _sandbox(tmp_path, apps=f"5, {GPU0}\n")
    p = _cli(ev, "capture-polls", "--card", "GPU0", "--record", str(tmp_path / "c.jsonl"))
    assert p.returncode == 1 and p.stdout.startswith("CAPTURE POLLS REFUSED")
