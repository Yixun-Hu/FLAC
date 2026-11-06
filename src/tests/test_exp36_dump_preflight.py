"""exp_36 addendum (the Planner, after review II; Yixun's announcement 08 + A1 "exactly the flags of its existing
cells"): every exp_36 cell DUMPS its scored waveforms into the manifest's default dump root, so before each runner step
the chain runs ``--exp36-dump-preflight``: every selected cell's dump directory (and any ``.staging-*`` sibling) must be
absent -- a collision with a dump this run checkout did not publish refuses everything. A dump with this checkout's own
receipt (a resumed chain) is left to the runner's reviewed predicate (SKIP when complete and verified, else QUARANTINE).

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + A1 + the Planner's addendum.
"""
import json
import os

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


def preflight(tmp, names=NAMES):
    return C.exp36_dump_preflight(MANIFEST, names, base_dir=str(tmp), dump_root=str(tmp / "nas"),
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


def test_a_dump_of_this_checkouts_own_cell_is_left_to_the_runner(tmp_path):
    r = _row(NAMES[0])
    T.write_cell(tmp_path, r, pin="b" * 40, dump=True, dump_root=str(tmp_path / "nas"))   # dump + its receipt
    assert os.path.isdir(_paths(r, str(tmp_path / "nas")).dump_dir)
    assert preflight(tmp_path) == []


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
    assert C.main(argv) == 0 and "EXP36 DUMP PREFLIGHT OK: 4 cell(s)" in capsys.readouterr().out
    os.makedirs(_paths(_row(NAMES[3]), str(tmp_path / "nas")).dump_dir)
    assert C.main(argv) == 2 and "EXP36 DUMP PREFLIGHT REFUSED" in capsys.readouterr().out
