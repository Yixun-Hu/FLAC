"""exp_36 yaw_c4_spread, cycle 4: the publication ``--exp36-publish`` (plan_yaw_c4_spread.md r0 §2 item 3 as amended
by the Codex plan review B4 / H4). A SEPARATE command, never run by the runner or the chain.

Gates before any write: this worktree at the pin (H36) and the exp-25 worktree at its expected HEAD; the destination
the plain YAW farm directory of the exp-25 worktree; the replay verdict ``equal`` at H36 AND re-computed identically;
the 24 cells verified here, dump-free, receipt 2 sdpa, under the verdict's campaign digest; none of the 48 destination
paths present (``lexists``). Then each of the 48 files (24 metrics JSONs + 24 stream sidecars; never the replay, never
a ``.cell.json``) is written to a temp file, fsynced, hard-linked to its final name (no replace) and re-hashed; the
publication manifest records sources, destinations, hashes, receipts, H36, the digest, the exp-25 HEAD and the archived
reference. A stop part-way is ``partial``; ``resume`` accepts only byte-identical destinations and removes only this
tool's own temp files.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import json
import os

import pytest

import test_depth_canon_collect as T
from test_exp36_replay_check import H36, REPLAY, _rel, _row, _sha, check, world  # noqa: F401

C, MANIFEST = T.C, T.MANIFEST
EXP25_HEAD = "5" * 40
CELLS24 = [f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)]


@pytest.fixture()
def pub(world, tmp_path):
    """The world of the replay check + the 24 cells published in the exp_36 dir + an equal verdict on disk."""
    for name in CELLS24:
        T.write_cell(world["new"], _row(name), pin=H36, dump=False)
    status, verdict = check(world)
    assert status == "equal", verdict["reasons"]
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(verdict, indent=1))
    dest = world["ref"] / T.FARM / "YAW"
    heads = {str(world["new"]): H36, str(world["ref"]): EXP25_HEAD}
    return dict(world, verdict=str(path), dest=str(dest), out=str(tmp_path / "publication.json"), heads=heads)


def publish(p, **kw):
    args = dict(cell_set="A", pin=H36, verdict_path=p["verdict"], manifest=MANIFEST, out=p["out"], base_dir=str(p["new"]),
                ref_base_dir=str(p["ref"]), ref_dump_root=str(p["ref"] / "dumps"), dump_root=str(p["new"] / "dumps"),
                dest=p["dest"], exp25_head=EXP25_HEAD, reference=p["pins"], git_head=lambda d: p["heads"].get(d))
    args.update(kw)
    return C.exp36_publish(**args)


def _listing(d):
    return sorted(os.listdir(d))


def test_the_48_files_are_published_and_recorded(pub):
    before = _listing(pub["dest"])
    status, doc = publish(pub)
    assert status == "published" and doc["status"] == "complete", doc.get("reasons")
    added = sorted(set(_listing(pub["dest"])) - set(before))
    want = sorted(os.path.basename(_rel(_row(n), k)) for n in CELLS24 for k in ("metrics", "stream"))
    assert added == want and len(added) == 48 and not [n for n in added if n.endswith(".cell.json") or REPLAY in n]
    assert set(before) <= set(_listing(pub["dest"]))                    # nothing removed or renamed
    for e in doc["entries"]:
        assert e["state"] == "published" and e["dest"] == os.path.join(pub["dest"], os.path.basename(e["source"]))
        assert _sha(e["dest"]) == e["dest_sha256"] == e["source_sha256"] == _sha(pub["new"] / e["source"])
        assert e["receipt_sha256"] == _sha(pub["new"] / e["receipt"]) and e["receipt"].endswith(".cell.json")
    assert [(e["cell"], e["kind"]) for e in doc["entries"]] == [(n, k) for n in CELLS24 for k in ("metrics", "stream")]
    assert (doc["pin"], doc["campaign_digest"], doc["exp25_head"], doc["dest"]) == (H36, T.DIGEST, EXP25_HEAD,
                                                                                    pub["dest"])
    assert doc["verdict"] == {"path": pub["verdict"], "sha256": _sha(pub["verdict"])}
    ref = doc["archived_reference"]
    assert {k: ref[k] for k in pub["pins"]} == pub["pins"] and ref["reference"] == "exp25_YAW_C4_S40000_K8_s42_th90"
    assert "--exp36-publish-resume" in doc["recovery"]
    assert json.loads(open(pub["out"]).read()) == doc
    assert not [n for n in _listing(pub["dest"]) if n.startswith(C.EXP36_TMP_PREFIX)]


def test_any_present_destination_refuses_everything(pub):
    taken = os.path.join(pub["dest"], os.path.basename(_rel(_row(CELLS24[7]), "stream")))
    open(taken, "w").close()
    before = _listing(pub["dest"])
    status, doc = publish(pub)
    assert status == "refused" and any("present" in x for x in doc["reasons"]), doc
    assert _listing(pub["dest"]) == before and not os.path.exists(pub["out"])        # nothing written at all
    os.symlink("/nonexistent", taken + ".x")                                         # a dangling symlink counts too
    os.unlink(taken)
    os.rename(taken + ".x", taken)
    assert publish(pub)[0] == "refused" and not os.path.exists(pub["out"])


def _edit_json(path, fn):
    doc = json.load(open(path))
    fn(doc)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


GATES = [
    (lambda p: p["heads"].update({str(p["new"]): "c" * 40}), "not at the pin"),
    (lambda p: p["heads"].update({str(p["ref"]): "c" * 40}), "exp-25 worktree"),
    (lambda p: p.update(dest=str(p["ref"] / T.FARM / "P1")), "destination"),
    (lambda p: _edit_json(p["verdict"], lambda d: d.update(result="mismatch")), "not an equal verdict"),
    (lambda p: _edit_json(p["verdict"], lambda d: d.update(pin="c" * 40)), "not an equal verdict"),
    (lambda p: _edit_json(p["verdict"], lambda d: d.update(campaign_digest="e" * 64)), "re-run"),
    (lambda p: _edit_json(p["verdict"], lambda d: d["replay"]["files"]["stream"].update(sha256="0" * 64)), "re-run"),
    (lambda p: os.unlink(p["verdict"]), "verdict"),
    (lambda p: os.unlink(p["new"] / _rel(_row(CELLS24[11]), "receipt")), CELLS24[11]),
    (lambda p: T.write_receipt(p["new"], _row(CELLS24[3]), f"{T.FARM}/YAW/epoch=8-step=40000.ckpt",
                               _rel(_row(CELLS24[3]), "metrics"), False, "c" * 40), CELLS24[3]),
    (lambda p: _edit_json(p["new"] / _rel(_row(CELLS24[0]), "receipt"), lambda d: d.update(
        campaign_digest="e" * 64, spawn_digest="e" * 64, post_digest="e" * 64)), CELLS24[0]),
    (lambda p: os.makedirs(p["new"] / "dumps" / os.path.basename(_rel(_row(CELLS24[9]), "metrics"))[:-5]), CELLS24[9]),
    (lambda p: open(p["out"], "w").close(), "publication manifest"),
]


@pytest.mark.parametrize("edit, needle", GATES, ids=[f"g{i}" for i in range(len(GATES))])
def test_every_gate_refuses_before_any_write(pub, edit, needle):
    edit(pub)
    before = _listing(pub["dest"]) if os.path.isdir(pub["dest"]) else None
    status, doc = publish(pub)
    assert status == "refused" and any(needle in x for x in doc["reasons"]), doc
    assert before is None or _listing(pub["dest"]) == before


def test_a_symlinked_destination_is_refused(pub, tmp_path):
    link = tmp_path / "YAWlink"
    os.symlink(pub["dest"], link)
    assert publish(pub, dest=str(link))[0] == "refused"
    os.rename(pub["dest"], pub["dest"] + ".real")
    os.symlink(pub["dest"] + ".real", pub["dest"])
    status, doc = publish(pub)
    assert status == "refused" and any("destination" in x for x in doc["reasons"]), doc


def test_a_stop_part_way_is_partial_and_resume_completes_it(pub, monkeypatch):
    real, calls = os.link, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 6:
            raise OSError("simulated crash at the sixth link")
        return real(src, dst)

    monkeypatch.setattr(C.os, "link", flaky)
    status, doc = publish(pub)
    monkeypatch.setattr(C.os, "link", real)
    assert status == "partial" and doc["status"] == "partial" and "sixth link" in doc["reasons"][0], doc
    assert json.loads(open(pub["out"]).read())["status"] == "partial"
    done = [e for e in doc["entries"] if e.get("state") == "published"]
    assert len(done) == 5 and all(_sha(e["dest"]) == e["source_sha256"] for e in done)
    assert not os.path.lexists(calls[5])                                    # no truncated or partial final file
    assert not [n for n in _listing(pub["dest"]) if n.startswith(C.EXP36_TMP_PREFIX)]
    status, doc = publish(pub)                                              # without resume: refused, untouched
    assert status == "refused" and json.loads(open(pub["out"]).read())["status"] == "partial"
    leftover = os.path.join(pub["dest"], f"{C.EXP36_TMP_PREFIX}999-{os.path.basename(calls[5])}")
    open(leftover, "w").close()                                             # a crash between write and link
    status, doc = publish(pub, resume=True)
    assert status == "published" and doc["status"] == "complete", doc.get("reasons")
    assert sum(e["state"] == "present (resume: byte-identical to its source)" for e in doc["entries"]) == 5
    assert all(_sha(e["dest"]) == e["source_sha256"] for e in doc["entries"]) and not os.path.lexists(leftover)


def test_resume_refuses_a_foreign_destination_or_temp_file(pub):
    name = os.path.basename(_rel(_row(CELLS24[2]), "metrics"))
    with open(os.path.join(pub["dest"], name), "w") as f:
        f.write("{}")
    status, doc = publish(pub, resume=True)
    assert status == "refused" and any("byte-identical" in x for x in doc["reasons"]), doc
    os.unlink(os.path.join(pub["dest"], name))
    open(os.path.join(pub["dest"], C.EXP36_TMP_PREFIX + "1-not_ours.json"), "w").close()
    status, doc = publish(pub, resume=True)
    assert status == "refused" and any("temp" in x for x in doc["reasons"]), doc


def _cli(p, tmp_path, monkeypatch, *extra):
    monkeypatch.setitem(C.EXP36_REPLAYS, REPLAY, dict(C.EXP36_REPLAYS[REPLAY], **p["pins"]))
    monkeypatch.setattr(C, "EXP25_WORKTREE", str(p["ref"]))
    monkeypatch.setattr(C, "EXP36_PUBLISH_DEST", p["dest"])
    monkeypatch.setattr(C, "EXP25_WORKTREE_HEAD", EXP25_HEAD)
    monkeypatch.setattr(C, "_git_head", lambda d: p["heads"].get(d))
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    return C.main(["--exp36-publish", "--pin", H36, "--exp36-verdict", p["verdict"], "--cells", str(cells),
                   "--base-dir", str(p["new"]), "--exp25-dump-root", str(p["ref"] / "dumps"),
                   "--dump-root", str(p["new"] / "dumps"), "--publish-manifest-out", p["out"], *extra])


def test_cli_publish_needs_a_set(pub, tmp_path, monkeypatch, capsys):
    """amendment A1 §5: two separate invocations, --exp36-set A (the 24) and --exp36-set OFF (the 6 YAW s42 off-orbit
    cells); without a set nothing is published."""
    assert _cli(pub, tmp_path, monkeypatch) == 2 and "--exp36-set" in capsys.readouterr().out
    assert not os.path.exists(pub["out"])


def test_cli_publish_exit_codes(pub, tmp_path, monkeypatch, capsys):
    A = ("--exp36-set", "A")
    assert _cli(pub, tmp_path, monkeypatch, *A, "--exp25-base-dir", str(tmp_path)) == 2   # the pinned worktree only
    assert "exp-25" in capsys.readouterr().out and not os.path.exists(pub["out"])
    assert _cli(pub, tmp_path, monkeypatch, *A) == 0 and "EXP36 PUBLISHED 48" in capsys.readouterr().out
    assert _cli(pub, tmp_path, monkeypatch, *A) == 2 and "REFUSED" in capsys.readouterr().out
    assert _cli(pub, tmp_path, monkeypatch, *A, "--exp36-publish-resume") == 0            # all 48 byte-identical


def test_cli_a_crashed_publication_exits_3_never_0(pub, tmp_path, monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(C, "exp36_publish", boom)
    assert _cli(pub, tmp_path, monkeypatch, "--exp36-set", "A") == 3 and "CRASHED" in capsys.readouterr().out


# ---- amendment A1 §5: the OFF set -- the 6 YAW s42 off-orbit cells, its own invocation ---- #
YAWOFF = [f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" for k in (1, 8) for d in (135, 225, 315)]


@pytest.fixture()
def pub_off(world, tmp_path):
    """The replay world + the 6 YAW off-orbit cells (chain I) + an equal YAW verdict on disk."""
    for name in YAWOFF:
        T.write_cell(world["new"], _row(name), pin=H36, dump=False)
    status, verdict = check(world)
    assert status == "equal", verdict["reasons"]
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(verdict, indent=1))
    heads = {str(world["new"]): H36, str(world["ref"]): EXP25_HEAD}
    return dict(world, verdict=str(path), dest=str(world["ref"] / T.FARM / "YAW"), out=str(tmp_path / "pub_off.json"),
                heads=heads)


def test_the_off_set_publishes_its_12_files_only(pub_off):
    before = _listing(pub_off["dest"])
    status, doc = publish(pub_off, cell_set="OFF")
    assert status == "published" and doc["set"] == "OFF", doc.get("reasons")
    added = sorted(set(_listing(pub_off["dest"])) - set(before))
    assert added == sorted(os.path.basename(_rel(_row(n), k)) for n in YAWOFF for k in ("metrics", "stream"))
    assert [e["cell"] for e in doc["entries"]][::2] == YAWOFF


def test_each_set_needs_its_own_cells_and_the_yaw_verdict(pub_off, tmp_path):
    status, doc = publish(pub_off, cell_set="A")                         # the 24 are not there
    assert status == "refused" and any("exp25_YAW_C4_S40000_K1_s43_th90" in x for x in doc["reasons"]), doc
    os.unlink(pub_off["new"] / _rel(_row(YAWOFF[4]), "receipt"))
    status, doc = publish(pub_off, cell_set="OFF")
    assert status == "refused" and any(YAWOFF[4] in x for x in doc["reasons"]), doc
    with pytest.raises(KeyError):
        publish(pub_off, cell_set="B")


def test_a_verdict_of_another_replay_is_refused(pub_off):
    doc = json.load(open(pub_off["verdict"]))
    doc["replay"]["cell"] = "exp36_P1_C4_S40000_K8_s42_th90"
    with open(pub_off["verdict"], "w") as f:
        json.dump(doc, f)
    status, doc = publish(pub_off, cell_set="OFF")
    assert status == "refused" and any("YAW replay" in x for x in doc["reasons"]), doc


def test_a_relative_plan_path_resolves_against_the_base_dir(world, tmp_path, monkeypatch):
    """The chain passes the runner's plan path relative to the checkout; the publish re-runs the check from another
    cwd (--base-dir = the run worktree) -- the plan resolves against the base dir, never the cwd."""
    rel = os.path.relpath(world["plan"], world["new"])
    monkeypatch.chdir(tmp_path)
    status, v = check(world, plan_json=rel)
    assert status == "equal" and v["plan_json"] == rel, v["reasons"]


def test_the_final_collector_reproduces_a_verdict_of_the_chain_i_collector(world):
    """The publication re-computes the replay verdict and requires it byte-for-byte equal (as JSON) to the recorded one;
    chain I records its verdicts with the collector at H36_I 461c58e0 -- this collector must produce the same dict."""
    import subprocess
    import types
    path = os.path.join(T.EXPDIR, "depth_canon_collect.py")
    src = subprocess.run(["git", "-C", T.REPO, "show", f"461c58e0:{os.path.relpath(path, T.REPO)}"], check=True,
                         capture_output=True, text=True).stdout
    old = types.ModuleType("depth_canon_collect_at_461c58e0")
    old.__file__ = path
    exec(compile(src, f"{path}@461c58e0", "exec"), old.__dict__)
    kw = dict(pin=H36, plan_json=world["plan"], manifest=MANIFEST, base_dir=str(world["new"]),
              ref_base_dir=str(world["ref"]), ref_dump_root=str(world["ref"] / "dumps"), reference=world["pins"],
              canon_index=T.canon_index())
    assert old.exp36_replay_check(**kw) == C.exp36_replay_check(**kw)
    assert old.exp36_replay_check(**kw)[0] == "equal"
