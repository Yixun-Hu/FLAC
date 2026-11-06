"""exp_36 addendum (the Planner, after review II; Yixun's announcement 08 + A1 "exactly the flags of its existing
cells"): every exp_36 cell DUMPS its scored waveforms into the manifest's default dump root, so before each runner step
the chain runs ``--exp36-dump-preflight --pin <EXPECT_HEAD>``: every selected cell's dump directory (and any
``.staging-*`` sibling) must be absent -- a collision with a dump this run checkout did not publish refuses everything.
Codex code review III HIGH 1: a dump is this checkout's (a resumed chain; left to the runner's reviewed predicate: SKIP
when complete and verified, else QUARANTINE + rerun, which also recovers a damaged dump) only when its receipt
AUTHENTICATES: a regular, non-symlink, parseable JSON receipt of exactly the selected cell (cell, stem) at the current
launch pin, under the farm root and the dump root, with ``dump: true`` -- a dangling or ordinary symlink, unrelated or
malformed JSON, another cell, pin, farm root or dump root, or ``dump: false`` collides.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + A1 + the Planner's addendum.
"""
import json
import os
import signal

import pytest

import test_depth_canon_collect as T

C, MANIFEST = T.C, T.MANIFEST
NAMES = ["exp36_YAW_C4_S40000_K8_s42_th90", "exp36_CANON_C4_S40000_K8_s42_th90", "exp35_BF_OFF_S40000_K1_s42_th315",
         "exp25_YAW_C4_S40000_K1_s43_th90"]


def _row(name):
    (r,) = [c for c in MANIFEST["cells"] if c["cell"] == name]
    return r


def _paths(r, dump_root):
    return C.cell_paths(r, os.path.join(T.FARM, r["arm"], T.ckpt_base(r) + ".ckpt"), dump_root)


PIN = "b" * 40


def preflight(tmp, names=NAMES, pin=PIN, dump_root=None):
    return C.exp36_dump_preflight(MANIFEST, names, pin=pin, base_dir=str(tmp), dump_root=dump_root or str(tmp / "nas"),
                                  canon_index=T.canon_index())


def test_absent_dump_directories_pass(tmp_path):
    (tmp_path / "nas").mkdir()
    assert preflight(tmp_path) == []


@pytest.mark.parametrize("name", NAMES)
def test_a_foreign_dump_directory_refuses_everything(tmp_path, name):
    os.makedirs(_paths(_row(name), str(tmp_path / "nas")).dump_dir)
    reasons = preflight(tmp_path)
    assert len(reasons) == 1 and name in reasons[0] and "collides" in reasons[0], reasons


def test_a_staging_leftover_refuses(tmp_path):
    d = _paths(_row(NAMES[1]), str(tmp_path / "nas")).dump_dir
    os.makedirs(d + ".staging-4242")
    reasons = preflight(tmp_path)
    assert len(reasons) == 1 and NAMES[1] in reasons[0], reasons


def _owned(tmp, name=NAMES[0]):
    """A cell this checkout published at PIN: its dump under tmp/nas and its authenticated receipt in the farm."""
    r = _row(name)
    T.write_cell(tmp, r, pin=PIN, dump=True, dump_root=str(tmp / "nas"))
    p = _paths(r, str(tmp / "nas"))
    assert os.path.isfile(os.path.join(p.dump_dir, "COMPLETE"))
    return r, p, tmp / p.receipt


def test_a_dump_of_this_checkouts_own_cell_is_left_to_the_runner(tmp_path):
    _owned(tmp_path)
    assert preflight(tmp_path) == []


def test_a_damaged_dump_owned_by_an_authenticated_receipt_is_left_to_the_runners_recovery(tmp_path):
    _, p, _ = _owned(tmp_path)
    os.unlink(os.path.join(p.dump_dir, "COMPLETE"))                        # incomplete: the runner quarantines + reruns
    os.makedirs(p.dump_dir + ".staging-77")
    assert preflight(tmp_path) == []


def _edit(path, **kw):
    doc = json.load(open(path))
    doc.update(kw)
    with open(path, "w") as f:
        json.dump(doc, f)


def _relink(receipt, target):
    os.rename(receipt, str(receipt) + ".real")
    os.symlink(target, receipt)


def _replace(receipt, make):
    os.unlink(receipt)
    make(str(receipt))


UNAUTHENTICATED = [
    (lambda r, p, rc: _relink(rc, str(rc) + ".real"), {}),                       # an ordinary symlink
    (lambda r, p, rc: _relink(rc, "/nonexistent/receipt.json"), {}),             # a dangling symlink
    (lambda r, p, rc: rc.write_text("{}"), {}),                                  # unrelated JSON
    (lambda r, p, rc: rc.write_text("not json {"), {}),                          # malformed
    (lambda r, p, rc: _edit(rc, cell=NAMES[3]), {}),                             # another cell
    (lambda r, p, rc: _edit(rc, stem="x"), {}),
    (lambda r, p, rc: None, {"pin": "c" * 40}),                                  # another launch pin
    (lambda r, p, rc: _edit(rc, ckpt_link="elsewhere/YAW/epoch=8-step=40000.ckpt"), {}),   # another farm root
    (lambda r, p, rc: _edit(rc, dump_root="/elsewhere"), {}),                    # another dump root
    (lambda r, p, rc: _edit(rc, dump=False), {}),
    (lambda r, p, rc: _replace(rc, os.mkdir), {}),                               # not a regular file: a directory
    (lambda r, p, rc: _replace(rc, os.mkfifo), {}),                              # ... a FIFO (never blocks)
]


class _Blocked(Exception):
    """Not an OSError (a TimeoutError would be swallowed as an open failure): a blocked open fails the test."""


def _alarm(signum, frame):
    raise _Blocked("the preflight blocked on the receipt path")


@pytest.mark.parametrize("edit, kw", UNAUTHENTICATED, ids=["symlink", "dangling", "json", "malformed", "cell", "stem",
                                                          "pin", "farm_root", "dump_root", "dump_false", "directory",
                                                          "fifo"])
def test_an_unauthenticated_receipt_never_claims_an_existing_dump(tmp_path, edit, kw):
    r, p, receipt = _owned(tmp_path)
    edit(r, p, receipt)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        reasons = preflight(tmp_path, **kw)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert len(reasons) == 1 and NAMES[0] in reasons[0] and "collides" in reasons[0], reasons


def test_a_bad_pin_is_refused(tmp_path):
    (tmp_path / "nas").mkdir()
    for pin in (None, "b" * 39, "B" * 40):
        assert preflight(tmp_path, pin=pin)


def test_an_unknown_cell_or_an_empty_selection_is_refused(tmp_path):
    (tmp_path / "nas").mkdir()
    assert preflight(tmp_path, ["exp36_nope"]) and preflight(tmp_path, [])
    assert preflight(tmp_path, ["exp35_P1_OFF_S40000_K1_s42_th135"]) == []   # chain I's H35 rows are checked too


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    index = tmp_path / "canon.sha256"
    index.write_text("".join(f"{sha}  {path}\n" for _, (path, sha) in sorted(T.canon_index().items())))
    argv = ["--exp36-dump-preflight", "--only", ",".join(NAMES), "--cells", str(cells), "--base-dir", str(tmp_path),
            "--dump-root", str(tmp_path / "nas"), "--canon-ckpt-manifest", str(index)]
    (tmp_path / "nas").mkdir()
    assert C.main(argv) == 2 and "pin" in capsys.readouterr().out                  # review III: the launch pin
    argv += ["--pin", PIN]
    assert C.main(argv) == 0 and "EXP36 DUMP PREFLIGHT OK: 4 cell(s)" in capsys.readouterr().out
    os.makedirs(_paths(_row(NAMES[3]), str(tmp_path / "nas")).dump_dir)
    assert C.main(argv) == 2 and "EXP36 DUMP PREFLIGHT REFUSED" in capsys.readouterr().out
