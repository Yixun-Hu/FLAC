"""exp_46 fix round 2 -- Codex code r2 finding 8: the staged NAS publication is atomic and no-clobber.

The staging directory <dest root>/<run>.incoming is created EXCLUSIVELY before rsync (an existing entry or a link
refuses), and the admitted staging directory reaches <dest root>/<run> through a no-replace rename: a final path
created meanwhile refuses -- never replaced, never nested (an ordinary ``mv`` nests). Temporary directories only:
the NAS is never touched.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402
import exp46_nas_publish as nas  # noqa: E402

RUN = "exp46_C4Ff100_l40_2x32"


def _listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), str(d)) for a, ds, fs in os.walk(str(d)) for f in ds + fs)


def _staged(root):
    """stage, then what rsync + the admission leave behind: a sealed run's files in the staging directory."""
    incoming = nas.stage(str(root), RUN)
    for name in (reg.TRAINING_RECEIPT, reg.COMMIT_MARKER, "CHECKPOINTS.sha256"):
        open(os.path.join(incoming, name), "w").write(name)
    return incoming


def test_stage_then_publish_moves_the_admitted_run_into_place(tmp_path):
    root = tmp_path / "nas"
    root.mkdir()
    incoming = _staged(root)
    assert incoming == str(root / f"{RUN}.incoming") and os.path.isdir(incoming) and not os.path.islink(incoming)
    final, how = nas.publish(str(root), RUN)
    assert final == str(root / RUN) and how in ("renameat2", "rename")
    assert _listing(root) == sorted([RUN] + [f"{RUN}/{n}" for n in (reg.TRAINING_RECEIPT, reg.COMMIT_MARKER,
                                                                      "CHECKPOINTS.sha256")])


@pytest.mark.parametrize("entry", ["staging dir", "staging file", "staging link", "dangling staging link", "final dir",
                                   "final link"])
def test_stage_refuses_any_existing_staging_or_published_entry_before_any_copy(tmp_path, entry):
    root = tmp_path / "nas"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = root / (RUN if entry.startswith("final") else f"{RUN}.incoming")
    if entry.endswith("dir"):
        target.mkdir()
    elif entry == "staging file":
        target.write_text("x")
    else:
        target.symlink_to(elsewhere if entry != "dangling staging link" else tmp_path / "absent")
    before = _listing(root)
    with pytest.raises(reg.GateError, match="exists"):
        nas.stage(str(root), RUN)
    assert _listing(root) == before and _listing(elsewhere) == []
    assert nas.main(["stage", "--dest-root", str(root), "--run", RUN]) == 2


def test_stage_never_creates_or_follows_its_root(tmp_path, monkeypatch):
    with pytest.raises(reg.GateError, match="not a real directory"):
        nas.stage(str(tmp_path / "unmounted"), RUN)
    assert not os.path.lexists(tmp_path / "unmounted")
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    with pytest.raises(reg.GateError, match="not a real directory"):
        nas.stage(str(tmp_path / "linked"), RUN)
    prot = tmp_path / "protected_exp45_nas"
    prot.mkdir()
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", (str(prot),))
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        nas.stage(str(prot), RUN)
    assert _listing(tmp_path / "real") == [] and _listing(prot) == []


@pytest.mark.parametrize("noreplace", [True, False], ids=["renameat2", "rename fallback"])
@pytest.mark.parametrize("final", ["a published run", "an empty directory", "a file", "a link"])
def test_publish_refuses_a_final_path_created_meanwhile_never_replacing_or_nesting(tmp_path, monkeypatch, final,
                                                                                noreplace):
    """The final path appears AFTER every check, just before the move (another publisher): the no-replace rename
    refuses -- renameat2(RENAME_NOREPLACE), or the fallback's fresh check -- and nothing is replaced or nested; the
    same final path present before publish is refused up front."""
    root = tmp_path / "nas"
    root.mkdir()
    incoming = _staged(root)
    dst = root / RUN

    def make_final():
        if final == "a published run":
            dst.mkdir()
            (dst / "theirs").write_text("theirs")
        elif final == "an empty directory":
            dst.mkdir()
        elif final == "a file":
            dst.write_text("theirs")
        else:
            dst.symlink_to(tmp_path / "absent")
    if not noreplace:                                   # a libc / filesystem without renameat2
        monkeypatch.setattr(nas.ctypes, "CDLL", lambda *a, **k: object())
    real, moves = nas._rename_noreplace, []

    def meanwhile(src, dst_):
        make_final()                                    # created after every check
        moves.append(real(src, dst_))
    monkeypatch.setattr(nas, "_rename_noreplace", meanwhile)
    with pytest.raises(reg.GateError, match="never replaced or nested"):
        nas.publish(str(root), RUN)
    monkeypatch.setattr(nas, "_rename_noreplace", real)
    after = _listing(root)
    assert moves == [] and os.path.isdir(incoming) and set(os.listdir(incoming)) >= {reg.COMMIT_MARKER}
    assert not os.path.lexists(dst / f"{RUN}.incoming") if dst.is_dir() and not dst.is_symlink() else True
    assert (dst / "theirs").read_text() == "theirs" if final == "a published run" else True
    with pytest.raises(reg.GateError, match="never replaced or nested"):      # present before publish: refused
        nas.publish(str(root), RUN)
    assert _listing(root) == after


@pytest.mark.parametrize("case", ["a staging link", "no staging", "unsealed"])
def test_publish_refuses_anything_but_a_real_sealed_staging_directory(tmp_path, case):
    root = tmp_path / "nas"
    root.mkdir()
    if case == "a staging link":
        (tmp_path / "elsewhere").mkdir()
        (root / f"{RUN}.incoming").symlink_to(tmp_path / "elsewhere")
    elif case == "unsealed":
        nas.stage(str(root), RUN)
    before = _listing(root)
    with pytest.raises(reg.GateError, match="staging directory|sealed"):
        nas.publish(str(root), RUN)
    assert _listing(root) == before
    assert nas.main(["publish", "--dest-root", str(root), "--run", RUN]) == 2
