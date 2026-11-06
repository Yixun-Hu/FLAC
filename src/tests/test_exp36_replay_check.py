"""exp_36 yaw_c4_spread, cycle 3: the replay hard stop ``--exp36-replay-check`` (plan_yaw_c4_spread.md r0 §2 item 2
as amended by the Codex plan review B3 / H2).

The replay ``exp36_YAW_C4_S40000_K8_s42_th90`` (this worktree, at H36, dump-free, receipt schema 2 on the SDPA
backend, its receipt's campaign digest == the replay run plan's) vs the archived exp_25 cell
``exp25_YAW_C4_S40000_K8_s42_th90`` (the exp-25 worktree, at d6ca307a, with its dump): the three archived raw digests
and the archived evaluator log first, then both cells verified, then type-strict equality of ``metrics``, ``by_scene``
and every other record key but ``source_sha`` / ``eval_name``, the raw stream sidecar and the loaded checkpoint.
Exit 0 equal / 1 mismatch / 2 refused; the verdict JSON binds H36, the campaign digest and every artifact hash.

Synthetic cells (N = 8) are written with the REAL evaluator functions (test_depth_canon_collect's fixtures); the
pinned archived digests are swapped for the synthetic reference's (``reference=``). One test reads the REAL archived
cell in ~/codespace/exp-25-depth-canon (read-only; skipped where it is absent).

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import hashlib
import json
import os

import pytest

import test_depth_canon_collect as T

C, MANIFEST = T.C, T.MANIFEST
H36 = "b" * 40
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
REPLAY, TEMPLATE = "exp36_YAW_C4_S40000_K8_s42_th90", "exp25_YAW_C4_S40000_K8_s42_th90"
FALLBACK = "Using EMA model\nNo module named 'flash_attn'\nflash_attn not installed, disabling Flash Attention\nok\n"


def _row(name):
    (r,) = [c for c in MANIFEST["cells"] if c["cell"] == name]
    return r


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _rel(r, kind):
    base = f"{T.FARM}/YAW/epoch=8-step=40000_metrics_{r['metrics_suffix']}"
    return base + {"metrics": ".json", "stream": ".stream.json", "receipt": ".cell.json"}[kind]


def _plan(new, digest=T.DIGEST, dump_root=None, **edit):
    """The replay runner's plan JSON as depth_canon_collect.py --plan writes it (one entry: the replay, WITH its dump --
    the Planner's addendum: every exp_36 cell dumps, like its template)."""
    dump_root = dump_root or str(new / "dumps")
    entry = C.plan_cells(MANIFEST, [_row(REPLAY)], farm_root=T.FARM, dump_root=dump_root, canon_index=None,
                         no_dump_blocks=[])[0]
    entry.update(edit)
    path = new / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "campaign_digest": digest, "farm_root": T.FARM,
                                "dump_root": dump_root, "n_selected": 1, "cells": [entry]}))
    return str(path)


def dump_pins(receipt_path):
    """The reference's dump evidence as its receipt binds it: the dump manifest's and the waveforms' sha256."""
    ev = json.loads(open(receipt_path).read())["dump_evidence"]
    return {"dump_manifest_sha256": ev["manifest_sha256"], "waveform_sha256": ev["manifest"]["sha256"]}


def replay(w, root=None, **kw):
    """(Re)publish the replay WITH its dump under a fresh dump root (a dump directory is never overwritten) and
    re-plan it there."""
    w["n"] = w.get("n", 0) + 1
    w["root"] = root or str(w["new"] / f"dumps{w['n']}")
    T.write_cell(w["new"], _row(REPLAY), pin=H36, dump=True, dump_root=w["root"], **kw)
    w["plan"] = _plan(w["new"], dump_root=w["root"])


@pytest.fixture()
def world(tmp_path):
    """The archived reference (pin d6ca307a, dump, receipt 1, its fallback log) and the replay (H36, WITH its dump,
    receipt 2 + its backend log), each in its own worktree-like dir; the reference pins are the synthetic files'."""
    ref, new = tmp_path / "exp25", tmp_path / "exp36"
    T.write_cell(ref, _row(TEMPLATE), pin=EXP25_PIN)
    (ref / "logs").mkdir()
    (ref / "logs" / "x.log").write_text(FALLBACK)
    pins = {f"{k}_sha256": _sha(ref / _rel(_row(TEMPLATE), k)) for k in ("metrics", "stream", "receipt")}
    pins.update(log_sha256=_sha(ref / "logs" / "x.log"), loaded=[MANIFEST["arms"]["YAW"]["ckpt"]["sha256"], T.CKPT_BYTES],
                **dump_pins(ref / _rel(_row(TEMPLATE), "receipt")))
    w = {"ref": ref, "new": new, "pins": pins}
    replay(w)
    return w


def check(w, **kw):
    args = dict(pin=H36, plan_json=w["plan"], manifest=MANIFEST, base_dir=str(w["new"]), ref_base_dir=str(w["ref"]),
                ref_dump_root=str(w["ref"] / "dumps"), reference=w["pins"], canon_index=T.canon_index())
    args.update(kw)
    return C.exp36_replay_check(**args)


def test_the_replay_equal_to_the_archived_cell(world):
    status, v = check(world)
    assert status == "equal" and v["result"] == "equal" and v["reasons"] == [], v["reasons"]
    assert (v["pin"], v["campaign_digest"], v["plan_sha256"]) == (H36, T.DIGEST, _sha(world["plan"]))
    ref, new = v["reference"], v["replay"]
    assert (ref["cell"], ref["pin"], ref["dump"], new["cell"], new["pin"], new["dump"]) == (
        TEMPLATE, EXP25_PIN, True, REPLAY, H36, False)
    for kind in ("metrics", "stream", "receipt"):
        assert ref["files"][kind] == {"path": _rel(_row(TEMPLATE), kind), "sha256": world["pins"][f"{kind}_sha256"]}
        assert new["files"][kind] == {"path": _rel(_row(REPLAY), kind),
                                      "sha256": _sha(world["new"] / _rel(_row(REPLAY), kind))}
    assert ref["files"]["log"] == {"path": "logs/x.log", "sha256": world["pins"]["log_sha256"]}
    assert new["files"]["log"] == {"path": "logs/x.log", "sha256": _sha(world["new"] / "logs" / "x.log")}
    assert new["receipt_schema"] == 2 and {k: new[k] for k in C.RECEIPT_V2_KEYS} == C.backend_fields()
    assert v["loaded"] == world["pins"]["loaded"] and json.loads(json.dumps(v)) == v
    assert ref["files"]["dump"] == {"manifest_sha256": world["pins"]["dump_manifest_sha256"],
                                    "waveform_sha256": world["pins"]["waveform_sha256"]}
    assert (new["dump"], new["files"]["dump"]["waveform_sha256"]) == (True, world["pins"]["waveform_sha256"])
    assert new["files"]["dump"]["dir"] == C.cell_paths(_row(REPLAY), f"{T.FARM}/YAW/epoch=8-step=40000.ckpt",
                                                       world["root"]).dump_dir
    assert any("dump" in x for x in v["compared"])


def _rewrite(w, **kw):
    """The replay re-published with a changed evaluator output (receipt + dump rebuilt, so it still verifies)."""
    replay(w, **kw)


MISMATCH = [
    (dict(metrics=dict(T.default_metrics(_row(REPLAY)), T60=7.5)), "metrics"),
    (dict(metrics=dict(T.default_metrics(_row(REPLAY)), C50=1)), "metrics"),
    (dict(by_scene={s: dict(T.default_metrics(_row(REPLAY)), EDT=40.0 + i) for i, s in enumerate(T.SCENES)}),
     "by_scene"),
]


MISMATCH.append((dict(waves=T.waveforms_for(_row(REPLAY)) * 0.5), "dump waveform sha256"))   # the addendum


@pytest.mark.parametrize("kw, needle", MISMATCH, ids=["T60", "C50_int", "by_scene", "waveforms"])
def test_a_different_replay_is_a_mismatch(world, kw, needle):
    _rewrite(world, **kw)
    status, v = check(world)
    assert status == "mismatch" and v["result"] == "mismatch" and any(needle in x for x in v["reasons"]), v["reasons"]


def test_equality_is_type_strict(world, tmp_path):
    """8 is not 8.0: a reference published with T60 = 8.0, the replay with T60 = 8 (then 8.0: equal)."""
    m, ref = T.default_metrics(_row(TEMPLATE)), tmp_path / "exp25b"
    T.write_cell(ref, _row(TEMPLATE), pin=EXP25_PIN, metrics=dict(m, T60=8.0))
    (ref / "logs").mkdir()
    (ref / "logs" / "x.log").write_text(FALLBACK)
    world["ref"] = ref
    world["pins"].update({f"{k}_sha256": _sha(ref / _rel(_row(TEMPLATE), k)) for k in ("metrics", "stream", "receipt")},
                         **dump_pins(ref / _rel(_row(TEMPLATE), "receipt")))
    _rewrite(world, metrics=dict(m, T60=8))
    status, v = check(world)
    assert status == "mismatch" and any("type int != float" in x for x in v["reasons"]), v["reasons"]
    _rewrite(world, metrics=dict(m, T60=8.0))
    assert check(world)[0] == "equal"


def test_the_loaded_checkpoint_must_be_the_pinned_one(world):
    """A verified cell's loaded digest + length already equal its receipt's (the runner refuses otherwise); the
    comparison binds both records and both receipts to the pinned (sha256, bytes) of the archived load."""
    world["pins"]["loaded"] = [world["pins"]["loaded"][0], T.CKPT_BYTES + 1]
    status, v = check(world)
    assert status == "mismatch" and sum("loaded checkpoint" in x for x in v["reasons"]) == 4, v["reasons"]


def test_the_raw_stream_sidecar_must_be_the_archived_one(world):
    """With a dump a re-serialized stream cannot verify (the dump manifest binds the stream's bytes), so the raw
    comparison is exercised through the pin: any other archived stream digest is a mismatch."""
    world["pins"]["stream_sha256"] = "0" * 64
    status, v = check(world)
    assert status == "mismatch" and any("stream sidecar sha256" in x for x in v["reasons"]), v["reasons"]


def test_the_dump_checks_are_ctl_style(world):
    """The exp_35 CTL checks against the archived dump evidence: the waveforms' sha256 and the COMPLETE marker equal the
    pinned reference's, n / dtype / shape and the input hash equal the reference dump manifest's."""
    world["pins"]["waveform_sha256"] = "0" * 64
    status, v = check(world)
    assert status == "refused" and any("dump evidence" in x for x in v["reasons"]), v["reasons"]   # the ref pin


def _tamper(path, data=b" "):
    with open(path, "ab") as f:
        f.write(data)


REFUSED = [
    (lambda w: _tamper(w["ref"] / _rel(_row(TEMPLATE), "metrics")), "archived reference metrics"),
    (lambda w: _tamper(w["ref"] / _rel(_row(TEMPLATE), "stream")), "archived reference stream"),
    (lambda w: _tamper(w["ref"] / _rel(_row(TEMPLATE), "receipt")), "archived reference receipt"),
    (lambda w: os.unlink(w["ref"] / "logs" / "x.log"), "archived evaluator log"),
    (lambda w: _tamper(w["ref"] / "logs" / "x.log"), "archived evaluator log"),
    (lambda w: w["pins"].update(log_sha256=_sha_text(FALLBACK.replace("No module", "Module"))) or
     (w["ref"] / "logs" / "x.log").write_text(FALLBACK.replace("No module", "Module")), "fallback"),
    (lambda w: os.unlink(w["new"] / _rel(_row(REPLAY), "metrics")), "replay does not verify"),
    (lambda w: os.unlink(w["new"] / "logs" / "x.log"), "replay does not verify"),
    (lambda w: w.update(plan=_plan(w["new"], digest="e" * 64, dump_root=w["root"])), "replay does not verify"),
    (lambda w: w.update(plan=_plan(w["new"], dump=False, dump_root=w["root"])), "replay run plan"),
    (lambda w: w.update(plan=_plan(w["new"], cell=TEMPLATE, dump_root=w["root"])), "replay run plan"),
    (lambda w: w.update(plan=_plan(w["new"], digest="nope", dump_root=w["root"])), "replay run plan"),
    (lambda w: w.update(plan=_plan(w["new"], dump_root=w["root"] + "_other")), "replay does not verify"),
    (lambda w: __import__("shutil").rmtree(C.cell_paths(_row(REPLAY), f"{T.FARM}/YAW/epoch=8-step=40000.ckpt",
                                                         w["root"]).dump_dir), "replay does not verify"),
    (lambda w: w.update(plan=str(w["new"] / "absent.json")), "replay run plan"),
]


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("edit, needle", REFUSED, ids=[f"r{i}" for i in range(len(REFUSED))])
def test_an_undecidable_comparison_is_refused(world, edit, needle):
    edit(world)
    status, v = check(world)
    assert status == "refused" and v["result"] == "refused" and any(needle in x for x in v["reasons"]), v["reasons"]


def test_a_replay_run_without_a_dump_or_with_a_schema1_receipt_is_refused(world):
    T.write_cell(world["new"], _row(REPLAY), pin=H36, dump=False, dump_root=str(world["new"] / "nodump"))
    world["plan"] = _plan(world["new"], dump_root=str(world["new"] / "nodump"))
    status, v = check(world)
    assert status == "refused" and any("replay does not verify" in x for x in v["reasons"]), v["reasons"]
    replay(world)
    assert check(world)[0] == "equal"                                            # with its dump: equal again
    receipt = world["new"] / _rel(_row(REPLAY), "receipt")
    doc = {k: v for k, v in json.loads(receipt.read_text()).items() if k not in C.RECEIPT_V2_KEYS}
    receipt.write_text(json.dumps(dict(doc, schema_version=1)))
    status, v = check(world)
    assert status == "refused" and any("receipt keys" in x for x in v["reasons"]), v["reasons"]


def test_a_bad_pin_is_refused(world):
    for pin in ("b" * 39, None, EXP25_PIN.upper()):
        assert check(world, pin=pin)[0] == "refused"
    assert check(world, pin="c" * 40)[0] == "refused"                       # the replay ran at H36, not at c...c


def _cli(w, tmp_path, monkeypatch, *extra):
    monkeypatch.setitem(C.EXP36_REPLAYS, REPLAY, dict(C.EXP36_REPLAYS[REPLAY], **w["pins"]))
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    return C.main(["--exp36-replay-check", "--exp36-replay", REPLAY, "--pin", H36, "--plan-json", w["plan"],
                   "--cells", str(cells), "--base-dir", str(w["new"]), "--exp36-ref-dir", str(w["ref"]),
                   "--exp25-dump-root", str(w["ref"] / "dumps"), *extra])


def test_cli_exit_codes_and_the_verdict_file(world, tmp_path, monkeypatch, capsys):
    out = tmp_path / "verdict.json"
    assert _cli(world, tmp_path, monkeypatch, "--verdict-out", str(out)) == 0
    assert "EXP36 REPLAY EQUAL" in capsys.readouterr().out
    assert json.loads(out.read_text()) == check(world)[1]
    before = out.read_bytes()
    assert _cli(world, tmp_path, monkeypatch, "--verdict-out", str(out)) == 2      # never overwritten
    assert out.read_bytes() == before and "exists" in capsys.readouterr().out
    _rewrite(world, metrics=dict(T.default_metrics(_row(REPLAY)), FD=0.5))
    out2 = tmp_path / "verdict2.json"
    assert _cli(world, tmp_path, monkeypatch, "--verdict-out", str(out2)) == 1
    assert json.loads(out2.read_text())["result"] == "mismatch" and "MISMATCH" in capsys.readouterr().out
    _tamper(world["ref"] / _rel(_row(TEMPLATE), "stream"))
    assert _cli(world, tmp_path, monkeypatch) == 2 and "REFUSED" in capsys.readouterr().out


ARCHIVE = os.path.expanduser("~/codespace/exp-25-depth-canon")


@pytest.mark.skipif(not os.path.isfile(os.path.join(ARCHIVE, _rel(_row(TEMPLATE), "receipt"))),
                    reason="the archived exp_25 cell is not on this machine")
def test_the_real_archived_reference_verifies_against_the_pinned_digests():
    """Read-only: the archived exp_25 YAW C4 K8 s42 th90 cell in the exp-25 worktree hashes to the pinned digests,
    its evaluator log shows the SDPA fallback, and it verifies offline at d6ca307a under this branch's manifest."""
    files, reasons = C.exp36_reference(C.load_cells(C.CELLS_JSON), replay=REPLAY)
    assert reasons == [], reasons
    pins = C.EXP36_REPLAYS[REPLAY]
    assert [files[k]["sha256"] for k in ("metrics", "stream", "receipt", "log")] == [
        pins["metrics_sha256"], pins["stream_sha256"], pins["receipt_sha256"], pins["log_sha256"]]
    assert pins["metrics_sha256"].startswith("fc8cb212") and pins["stream_sha256"].startswith("e671270f")
    assert pins["receipt_sha256"].startswith("ee7d624a") and pins["log_sha256"].startswith("8cbf6e24")


def test_cli_a_crashed_check_is_refused_never_a_mismatch(world, tmp_path, monkeypatch, capsys):
    """An uncaught exception would exit 1 -- the mismatch code; the CLI maps any crash to 2 (refused)."""
    def boom(**kw):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(C, "exp36_replay_check", boom)
    out = tmp_path / "v.json"
    assert _cli(world, tmp_path, monkeypatch, "--verdict-out", str(out)) == 2 and not out.exists()
    assert "REFUSED" in capsys.readouterr().out
