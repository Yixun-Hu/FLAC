"""exp_36 amendment A1: the four replay hard stops of the off-orbit chain (the YAW one of test_exp36_replay_check.py
plus P1, B-F and CANON), one registry ``EXP36_REPLAYS`` (reference cell, reference worktree, its pin, its raw digests,
its backend proof), the same equality rule; the CANON replay's canon sidecar is compared too, the B-F reference (exp_35
phase (i), read-only in the exp-35 run worktree at H35) proves its backend by the no-flash launcher's log, the exp_25
references by the flash_attn-not-installed fallback pair. A replay run plan may hold several replays (chain I runs
the four in one runner call).

Synthetic cells (N = 8, the REAL evaluator writers); the real archived references are read read-only where present.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + amendment A1.
"""
import json
import os

import pytest

import test_depth_canon_collect as T
from test_exp36_replay_check import FALLBACK, H36, _sha, dump_pins

C, MANIFEST = T.C, T.MANIFEST
H35 = "5b065801e9d1c1844461ec84cd0558702c714326"
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
C4 = "C4_S40000_K8_s42_th90"
ARMS = {"YAW": (f"exp25_YAW_{C4}", EXP25_PIN), "P1": (f"exp25_P1_{C4}", EXP25_PIN),
        "BF": (f"exp35_BF_{C4}", H35), "CANON": (f"exp25_CANON_{C4}", EXP25_PIN)}
REAL = {   # amendment A1 + plan r0 §1: the archived raw digests (read 2026-11-06)
    "YAW": ("fc8cb212a9a5b852d1ad3bc31e01ff17c1b0b93b79141d92d0c016751eb11130",
            "ee7d624a0c5a5df589fa7e47690aca85e2d6297007a466ad69820bf3c7bf9177", None),
    "P1": ("88bb41c190da815e8212a2c8a45b559e9eda7df9f8384a04727f7bf83956c74d",
           "eb5a588c3b2a3ad6262c8b1edd4b504100ed9b4997803b6c721f2488baa61929", None),
    "BF": ("ef2840c746b871268bddf8ae36609d1ee33186074a9afb01612c5db97606d82f",
           "431ceba60189490e85d53d942f43025c493a863cddec928ec7555d34ed5b16fd", None),
    "CANON": ("51025f3b36075efb3966f757f17cdf43f4348046a55d28b58a1f0b726cbea8aa",
              "3fc4ed172ac0e8b7efefe15cb035f55de08f76c2591b9f8498d8b5cb4c6a87ac",
              "5e3d6c4f1904038dd0b8eeec49db8c18474a6df992ec5bd3994ebb33f1536700")}
STREAM = "e671270fc3cd61f5a47dd68f6f41707db68d6e62b4f2750ae52e9e7f9b7085a8"
DUMPS = {   # the Planner's addendum: each reference receipt's dump evidence (dump manifest, waveforms), read 2026-11-06
    "YAW": ("6a702f0007654f5b4a23e463207e4d494889d34a1b63263821213c21d43d2002",
            "1485ef84c6753959effa202b7cfe3b1fd364c1d66d995c864a1d22ac4b33d6ee"),
    "P1": ("99de989732a26dbd8b3047e45651129f2b824b5c72a19bc7cb505a02f251a92b",
           "34dfa66d5c67dc1e945420e8f6aa1aac11ae2f4cb0e43b4fb33e3f660f3eb9b4"),
    "BF": ("975cb72125977b6724cda82501bff4b356accb80278db7f8341c69566630fd5c",
           "d9f46debf8ebf7739a7a1dc4267e1a1ad5a280716b77c9dd8af61f311c39dac6"),
    "CANON": ("ca864c12d0ca9058c95f5e129777f3ac12f152d3c526acdf5bcb869d409c311a",
              "441330a0b8f5159d8cd5c07baf1a8112bb6a296b00e98e703ca7409e0c293180")}


def _row(name):
    (r,) = [c for c in MANIFEST["cells"] if c["cell"] == name]
    return r


def _paths(r):
    return C.cell_paths(r, os.path.join(T.FARM, r["arm"], T.ckpt_base(r) + ".ckpt"), "/d")


def _plan(new, replays, digest=T.DIGEST):
    entries = C.plan_cells(MANIFEST, [_row(n) for n in replays], farm_root=T.FARM, dump_root=str(new / "dumps"),
                           canon_index=T.canon_index(), no_dump_blocks=[])           # every replay dumps (addendum)
    path = new / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "campaign_digest": digest, "farm_root": T.FARM,
                                "dump_root": str(new / "dumps"), "n_selected": len(entries), "cells": entries}))
    return str(path)


def make(tmp, arm, replays=None):
    """The reference (its pin, a dump, its backend evidence) and the replay (H36, WITH its dump, receipt 2)."""
    ref_name, pin = ARMS[arm]
    replay = f"exp36_{arm}_{C4}"
    ref, new = tmp / f"ref_{arm}", tmp / f"new_{arm}"
    T.write_cell(ref, _row(ref_name), pin=pin)                    # B-F: receipt 2 + the launcher's log (the helper)
    if arm != "BF":
        (ref / "logs").mkdir()
        (ref / "logs" / "x.log").write_text(FALLBACK)
    T.write_cell(new, _row(replay), pin=H36, dump=True, dump_root=str(new / "dumps"))
    p = _paths(_row(ref_name))
    pins = {f"{k}_sha256": _sha(ref / getattr(p, k)) for k in ("metrics", "stream", "receipt")}
    pins.update(log_sha256=_sha(ref / "logs" / "x.log"), canon_sha256=_sha(ref / p.canon) if arm == "CANON" else None,
                loaded=[T.expected_target(_row(ref_name))[1], T.CKPT_BYTES], **dump_pins(ref / p.receipt))
    return {"replay": replay, "ref": ref, "new": new, "pins": pins, "plan": _plan(new, replays or [replay])}


def check(w, **kw):
    args = dict(replay=w["replay"], pin=H36, plan_json=w["plan"], manifest=MANIFEST, base_dir=str(w["new"]),
                ref_base_dir=str(w["ref"]), ref_dump_root=str(w["ref"] / "dumps"), reference=w["pins"],
                canon_index=T.canon_index())
    args.update(kw)
    return C.exp36_replay_check(**args)


def test_the_registry_holds_the_four_replays_with_the_amendment_digests():
    assert list(C.EXP36_REPLAYS) == [f"exp36_{a}_{C4}" for a in ("YAW", "P1", "BF", "CANON")]
    for arm, (metrics, receipt, canon) in REAL.items():
        e = C.EXP36_REPLAYS[f"exp36_{arm}_{C4}"]
        assert (e["dump_manifest_sha256"], e["waveform_sha256"]) == DUMPS[arm]
        assert (e["reference"], e["pin"]) == ARMS[arm] and e["backend"] == ("launcher" if arm == "BF" else "fallback")
        assert (e["metrics_sha256"], e["receipt_sha256"], e["canon_sha256"], e["stream_sha256"]) == (
            metrics, receipt, canon, STREAM)
        assert e["worktree"] == os.path.expanduser("~/codespace/exp-35-orbitrir-yaw" if arm == "BF"
                                                   else "~/codespace/exp-25-depth-canon")
    assert C.EXP36_REFERENCE is C.EXP36_REPLAYS[C.EXP36_REPLAY]


@pytest.mark.parametrize("arm", ["P1", "BF", "CANON"])
def test_each_replay_equal_to_its_reference(tmp_path, arm):
    w = make(tmp_path, arm)
    status, v = check(w)
    assert status == "equal" and v["reasons"] == [], v["reasons"]
    assert (v["reference"]["cell"], v["reference"]["pin"], v["replay"]["cell"]) == (ARMS[arm][0], ARMS[arm][1],
                                                                                      w["replay"])
    assert ("canon" in v["replay"]["files"]) is (arm == "CANON") and ("canon" in v["reference"]["files"]) is (
        arm == "CANON")
    assert v["reference"]["backend"].startswith("sdpa")


def test_a_plan_of_the_four_replays_serves_each_check(tmp_path):
    names = [f"exp36_{a}_{C4}" for a in ("YAW", "P1", "BF", "CANON")]
    for arm in ("P1", "CANON"):
        w = make(tmp_path, arm, replays=names)
        assert check(w)[0] == "equal"
        w["plan"] = _plan(w["new"], names[1:] + ["exp25_YAW_C4_S40000_K1_s43_th90"])   # a non-replay planned
        status, v = check(w)
        assert status == "refused" and any("replay run plan" in x for x in v["reasons"]), v["reasons"]
        w["plan"] = _plan(w["new"], [n for n in names if arm not in n])                  # this replay not planned
        assert check(w)[0] == "refused"


def test_the_raw_canon_sidecar_must_be_the_archived_one(tmp_path):
    """The dump manifest binds the canon sidecar's bytes, so the raw comparison is exercised through the pin."""
    w = make(tmp_path, "CANON")
    assert check(w)[0] == "equal"
    w["pins"]["canon_sha256"] = "0" * 64
    status, v = check(w)
    assert status == "refused" and any("canon" in x for x in v["reasons"]), v["reasons"]   # the reference pin
    w = make(tmp_path / "again", "CANON")
    status, v = C.exp36_replay_check(**dict(_kw(w), reference=dict(w["pins"])))
    assert status == "equal"


def _kw(w):
    return dict(replay=w["replay"], pin=H36, plan_json=w["plan"], manifest=MANIFEST, base_dir=str(w["new"]),
                ref_base_dir=str(w["ref"]), ref_dump_root=str(w["ref"] / "dumps"), canon_index=T.canon_index())


@pytest.mark.parametrize("arm", ["P1", "BF", "CANON"])
def test_each_replay_must_reproduce_its_references_waveforms(tmp_path, arm):
    """The CTL-style dump checks: other waveforms (a consistent replay whose dump differs) are a mismatch."""
    w = make(tmp_path, arm)
    r = _row(w["replay"])
    T.write_cell(w["new"], r, pin=H36, dump=True, dump_root=str(w["new"] / "dumps2"), waves=T.waveforms_for(r) * 0.25)
    w["plan"] = _plan2(w["new"], [w["replay"]], str(w["new"] / "dumps2"))
    status, v = check(w)
    assert status == "mismatch" and any("dump waveform sha256" in x for x in v["reasons"]), v["reasons"]
    assert any("COMPLETE marker" in x for x in v["reasons"]), v["reasons"]


def _plan2(new, replays, root):
    entries = C.plan_cells(MANIFEST, [_row(n) for n in replays], farm_root=T.FARM, dump_root=root,
                           canon_index=T.canon_index(), no_dump_blocks=[])
    path = new / "plan2.json"
    path.write_text(json.dumps({"schema_version": 1, "campaign_digest": T.DIGEST, "farm_root": T.FARM,
                                "dump_root": root, "n_selected": len(entries), "cells": entries}))
    return str(path)


def test_the_backend_proof_of_each_reference_kind(tmp_path):
    w = make(tmp_path, "BF")                                         # the launcher's sentinel, re-validated
    text = (w["ref"] / "logs" / "x.log").read_text()
    (w["ref"] / "logs" / "x.log").write_text(text.replace("EXP35_NO_FLASH_LAUNCHER", "SOMETHING_ELSE"))
    w["pins"]["log_sha256"] = _sha(w["ref"] / "logs" / "x.log")
    status, v = check(w)
    assert status == "refused" and any("sentinel" in x for x in v["reasons"]), v["reasons"]
    w = make(tmp_path / "p1", "P1")                                   # exp_25: the fallback pair
    (w["ref"] / "logs" / "x.log").write_text(FALLBACK + FALLBACK)
    w["pins"]["log_sha256"] = _sha(w["ref"] / "logs" / "x.log")
    status, v = check(w)
    assert status == "refused" and any("fallback" in x for x in v["reasons"]), v["reasons"]


@pytest.mark.parametrize("arm", ["YAW", "P1", "BF", "CANON"])
def test_the_real_archived_references_verify(arm):
    """Read-only: each archived reference hashes to the amendment's digests, proves its backend and verifies offline
    at its pin (CANON with the tracked launcher checkpoint manifest)."""
    e = C.EXP36_REPLAYS[f"exp36_{arm}_{C4}"]
    real = C.load_cells(C.CELLS_JSON)
    p = C.cell_paths(next(c for c in real["cells"] if c["cell"] == e["reference"]),
                     os.path.join(real["farm_root"], arm, "epoch=8-step=40000.ckpt"), real["dump_root"])
    if not os.path.isfile(os.path.join(e["worktree"], p.receipt)):
        pytest.skip(f"the archived {arm} reference is not on this machine")
    index = C.load_canon_index(C.find_canon_manifest(), real["arms"]["CANON"]["ckpt"]["run_ns"])
    files, reasons = C.exp36_reference(real, replay=f"exp36_{arm}_{C4}", canon_index=index)
    assert reasons == [], reasons
    assert (files["metrics"]["sha256"], files["receipt"]["sha256"], files["stream"]["sha256"]) == (
        REAL[arm][0], REAL[arm][1], STREAM)
    assert files.get("canon", {}).get("sha256") == REAL[arm][2]
    assert files["dump"] == {"manifest_sha256": DUMPS[arm][0], "waveform_sha256": DUMPS[arm][1]}
