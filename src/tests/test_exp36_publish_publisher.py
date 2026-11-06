"""exp_36, cycle 12 (Codex code review II HIGH 3, NIT 1, NIT 3): the publisher's own provenance, and the parsed CLI.

* ``publisher_closure(repo, pin)``: the publishing checkout is at the full reviewed commit ``--publisher-pin`` and its
  publication / replay-verification closure (eval_FLAC.py, src/ but src/tests, this collector, the cells manifest) is
  committed and clean -- a modified, deleted or untracked closure file, a missing git HEAD and a wrong or short pin are
  refused; the manifest records the HEAD, the collector's path + sha256 and the closure digest BESIDE the artifacts'
  provenance (pin H36_I + its campaign digest), never instead of it.
* the parsed CLI requires ``--publisher-pin`` and the collector's own cells manifest, and routes each set to its own
  default manifest (A: 48 entries, OFF: 12; no override); its help names both sets.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + A1 + Codex code review II.
"""
import hashlib
import json
import os
import subprocess

import pytest

import test_depth_canon_collect as T
from test_exp36_publish import (CELLS24, PUB, YAWOFF, _cli, _listing, pub, pub_off, publish)  # noqa: F401
from test_exp36_replay_check import H36, _row, check, world  # noqa: F401

C = T.C


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture()
def repo(tmp_path):
    """A throwaway git checkout holding the closure's paths (their contents do not matter: the check is about clean,
    committed bytes at the pin), plus a test file and an unrelated file outside the closure."""
    r = tmp_path / "pubrepo"
    for rel in ("eval_FLAC.py", "src/data/x.py", "src/tests/test_x.py", C.EXP36_COLLECTOR_REL, C.EXP36_CELLS_REL,
                "other.txt"):
        os.makedirs(os.path.dirname(r / rel), exist_ok=True)
        (r / rel).write_text(rel + "\n")
    _git(tmp_path, "init", "-q", str(r))
    _git(r, "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-q", "-m", "closure")
    return r


def test_a_clean_closure_at_the_pin_is_recorded(repo):
    head = _git(repo, "rev-parse", "HEAD")
    info, why = C.publisher_closure(str(repo), head)
    assert why == [] and (info["repo"], info["head"], info["collector"]) == (str(repo), head, C.EXP36_COLLECTOR_REL)
    assert info["collector_sha256"] == _sha(repo / C.EXP36_COLLECTOR_REL)
    files = sorted(["eval_FLAC.py", "src/data/x.py", C.EXP36_COLLECTOR_REL, C.EXP36_CELLS_REL])   # src/tests apart
    lines = "".join(f"{_sha(repo / f)}  {f}\n" for f in files)
    assert info["closure_digest"] == hashlib.sha256(lines.encode()).hexdigest()


@pytest.mark.parametrize("edit", [
    lambda r: (r / C.EXP36_COLLECTOR_REL).write_text("dirty collector\n"),
    lambda r: (r / "src/data/x.py").write_text("dirty dependency\n"),
    lambda r: (r / "src/data/new.py").write_text("an untracked dependency\n"),
    lambda r: os.unlink(r / "eval_FLAC.py"),
    lambda r: (r / C.EXP36_CELLS_REL).write_text("{}\n")], ids=["collector", "dependency", "untracked", "deleted",
                                                                "cells"])
def test_dirty_closure_bytes_are_refused(repo, edit):
    head = _git(repo, "rev-parse", "HEAD")
    edit(repo)
    info, why = C.publisher_closure(str(repo), head)
    assert info is None and why and any("not clean" in x for x in why), why


def test_test_files_and_files_outside_the_closure_do_not_count(repo):
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "src/tests/test_x.py").write_text("edited\n")
    (repo / "other.txt").write_text("edited\n")
    assert C.publisher_closure(str(repo), head)[1] == []


def test_a_wrong_or_short_pin_or_a_missing_head_is_refused(repo, tmp_path):
    head = _git(repo, "rev-parse", "HEAD")
    for pin in ("0" * 40, head[:12], None, head.upper()):
        info, why = C.publisher_closure(str(repo), pin)
        assert info is None and why, pin
    empty = tmp_path / "nohead"
    _git(tmp_path, "init", "-q", str(empty))
    info, why = C.publisher_closure(str(empty), head)
    assert info is None and any("HEAD" in x for x in why), why


def test_the_closure_names_this_collector_and_its_cells_manifest():
    assert os.path.join(C.REPO, C.EXP36_COLLECTOR_REL) == os.path.abspath(C.__file__)
    assert os.path.join(C.REPO, C.EXP36_CELLS_REL) == C.CELLS_JSON
    assert C.EXP36_PUBLISHER_CLOSURE == ("eval_FLAC.py", "src", ":(exclude)src/tests", C.EXP36_COLLECTOR_REL,
                                         C.EXP36_CELLS_REL)


def test_a_refused_publisher_refuses_the_publication_before_any_write(pub):
    before = _listing(pub["dest"])
    status, doc = publish(pub, publisher_check=lambda repo, pin: (None, ["the publication closure is not clean"]))
    assert status == "refused" and doc["reasons"] == ["the publication closure is not clean"], doc
    assert _listing(pub["dest"]) == before and not os.path.exists(pub["out"])


def test_the_cli_requires_the_publisher_pin_and_the_collectors_own_cells(pub, tmp_path, monkeypatch, capsys):
    assert _cli(pub, tmp_path, monkeypatch, "--exp36-set", "A", publisher_pin=None) == 2
    assert "--publisher-pin" in capsys.readouterr().out and not os.path.exists(pub["out"])
    other = tmp_path / "other_cells.json"
    other.write_text(json.dumps(T.MANIFEST))
    assert _cli(pub, tmp_path, monkeypatch, "--exp36-set", "A", "--cells", str(other)) == 2
    assert "cells" in capsys.readouterr().out and not os.path.exists(pub["out"])


@pytest.fixture()
def both(world, tmp_path):
    """The replay world + BOTH sets' cells (the 24 + the 6) + an equal YAW verdict on disk."""
    for name in CELLS24 + YAWOFF:
        T.write_cell(world["new"], _row(name), pin=H36, dump=False)
    status, verdict = check(world)
    assert status == "equal", verdict["reasons"]
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(verdict, indent=1))
    heads = {str(world["new"]): H36, str(world["ref"]): "5" * 40}
    return dict(world, verdict=str(path), dest=str(world["ref"] / T.FARM / "YAW"), heads=heads,
                out=str(tmp_path / "publication_A.json"))


def test_the_parsed_cli_routes_each_set_to_its_own_default_manifest(both, tmp_path, monkeypatch, capsys):
    """Review II NIT 1: no --publish-manifest-out (it no longer exists); EXP36_PUBLICATION redirected to a temp
    pattern; A -> 48 entries, OFF -> 12, two distinct default files."""
    assert _cli(both, tmp_path, monkeypatch, "--exp36-set", "A") == 0
    assert _cli(both, tmp_path, monkeypatch, "--exp36-set", "OFF") == 0
    out = capsys.readouterr().out
    assert "EXP36 PUBLISHED 48 file(s) (set A)" in out and "EXP36 PUBLISHED 12 file(s) (set OFF)" in out
    a, off = (json.load(open(tmp_path / f"publication_{s}.json")) for s in ("A", "OFF"))
    assert (a["set"], len(a["entries"]), off["set"], len(off["entries"])) == ("A", 48, "OFF", 12)
    assert sorted(p.name for p in tmp_path.glob("publication_*.json")) == ["publication_A.json", "publication_OFF.json"]
    with pytest.raises(SystemExit) as gone:                                       # the override no longer exists
        _cli(both, tmp_path, monkeypatch, "--exp36-set", "A", "--publish-manifest-out", "x.json")
    assert gone.value.code == 2


def test_the_cli_help_names_both_sets(capsys):
    with pytest.raises(SystemExit):
        C.main(["--help"])
    text = "".join(capsys.readouterr().out.split())                # argparse wraps (and may break at hyphens)
    assert "A=the24YAW-C4Scells(48files)" in text and "OFF=the6YAWs42off-orbitcells(12files)" in text
    assert "--publisher-pin" in text
