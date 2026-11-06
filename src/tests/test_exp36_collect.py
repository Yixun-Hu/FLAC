"""exp_36 yaw_c4_spread, cycle 2: the collector's no-flash decision and the exp_36 names (plan_yaw_c4_spread.md r0
§2 item 2 as amended by the Codex plan review B1).

``uses_no_flash(row)`` (EXP35_BLOCKS + EXP36_BLOCKS) drives the plan entry's ``no_flash``, receipt schema 2 and its
backend fields, the backend-log re-validation, ``build_receipt`` and ``--count-no-flash``; ``is_exp35()`` and
``EXP35_BLOCKS`` keep the exp_35 identity and collection. An explicit validator admits exactly the 24
``exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270}`` aliases and the one replay ``exp36_YAW_C4_S40000_K8_s42_th90``
in block YAW-C4S, and those names nowhere else.

Synthetic cells are written with the REAL evaluator functions through test_depth_canon_collect's fixtures (N = 8).

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import copy
import json
import os

import pytest

import test_depth_canon_collect as T

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
EXP35 = ("CTL", "BF-pin", "BF-dup", "OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45", "E2-FAIL")
REPLAY = "exp36_YAW_C4_S40000_K8_s42_th90"
CELLS24 = [f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)]
YAW_C4S = [REPLAY] + CELLS24


def _row(name, manifest=MANIFEST):
    (r,) = [c for c in manifest["cells"] if c["cell"] == name]
    return r


def _stem(r):
    return f"epoch=8-step=40000_metrics_{r['metrics_suffix']}"


def test_the_no_flash_decision_is_separate_from_the_exp35_identity():
    assert C.EXP35_BLOCKS == EXP35 and C.EXP36_BLOCKS == ("YAW-C4S",) and C.EXPLICIT_BLOCKS == ("E2-FAIL",)
    for r in MANIFEST["cells"]:
        assert C.uses_no_flash(r) is (r["block"] in EXP35 + ("YAW-C4S",)), r["cell"]
        assert C.is_exp35(r) is (r["block"] in EXP35), r["cell"]
    assert [c["cell"] for c in MANIFEST["cells"] if c["block"] == "YAW-C4S"] == YAW_C4S


def test_plan_entries_of_the_yaw_c4s_cells_are_no_flash_and_dump_free():
    plan = C.plan_cells(MANIFEST, C.select_cells(MANIFEST, blocks=["YAW-C4S"]), farm_root="farm", dump_root="/d",
                        canon_index=None, no_dump_blocks=["YAW-C4S"])
    ck = MANIFEST["arms"]["YAW"]["ckpt"]
    assert [e["cell"] for e in plan] == YAW_C4S
    for e in plan:
        r = _row(e["cell"])
        assert (e["no_flash"], e["dump"], list(e)[-1]) == (True, False, "no_flash"), e["cell"]
        assert (e["target"], e["sha256"], e["link"]) == (ck["path"], ck["sha256"], "farm/YAW/epoch=8-step=40000.ckpt")
        assert e["argv"] == r["eval_args"] + ["--ckpt-path", "farm/YAW/epoch=8-step=40000.ckpt"]
        assert e["stem"] == _stem(r) and e["canon"] is None


def test_exp25_and_exp35_plan_entries_keep_their_no_flash_flag():
    kw = dict(farm_root="farm", dump_root="/d", canon_index=T.canon_index(), no_dump_blocks=[])
    for e in C.plan_cells(MANIFEST, [c for c in MANIFEST["cells"] if c["block"] not in ("YAW-C4S",)], **kw):
        assert ("no_flash" in e) is (_row(e["cell"])["block"] in EXP35), e["cell"]


@pytest.mark.parametrize("args, want", [(["--blocks", "YAW-C4S"], 25), (["--only", REPLAY], 1),
                                        (["--only", ",".join(CELLS24)], 24), ([], 100),
                                        (["--blocks", "YAW-pin"], 0), (["--blocks", "YAW-C4S,CTL"], 27)])
def test_cli_count_no_flash_counts_the_yaw_c4s_cells(tmp_path, capsys, args, want):
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    assert C.main(["--count-no-flash", "--cells", str(cells)] + args) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(want)


def _verify(tmp, r, **kw):
    return C.verify_cell(_stem(r), manifest=kw.pop("manifest", MANIFEST), farm_root=T.FARM,
                         dump_root=os.path.join(str(tmp), "dumps"), pin=PIN, base_dir=str(tmp), check_link=False,
                         **kw)


@pytest.mark.parametrize("name", [REPLAY, CELLS24[0], CELLS24[-1]])
def test_a_dump_free_yaw_c4s_cell_verifies_with_receipt_2_and_its_backend_log(tmp_path, name):
    r = _row(name)
    T.write_cell(tmp_path, r, dump=False)
    rec = json.load(open(tmp_path / (T.FARM + "/YAW/" + _stem(r) + ".cell.json")))
    assert rec["schema_version"] == 2 and list(rec) == list(C.RECEIPT_KEYS_V2) and rec["dump"] is False
    assert {k: rec[k] for k in C.RECEIPT_V2_KEYS} == C.backend_fields() and rec["attention_backend"] == "sdpa"
    assert _verify(tmp_path, r, expect_dump=False, expect_digest=T.DIGEST) == []
    os.unlink(tmp_path / "logs" / "x.log")                                          # the log is re-read every time
    assert any("attention backend is unproven" in x for x in _verify(tmp_path, r, expect_dump=False))


def test_a_schema1_receipt_is_refused_for_a_yaw_c4s_cell(tmp_path):
    r = _row(CELLS24[5])
    T.write_cell(tmp_path, r, dump=False, receipt=False)
    entry = C.plan_cells(MANIFEST, [r], farm_root=T.FARM, dump_root=str(tmp_path / "dumps"), canon_index=None,
                         no_dump_blocks=["YAW-C4S"])[0]
    kw = dict(base_dir=str(tmp_path), lane=1, log="logs/x.log", rc=0, pin=PIN, ckpt_sha256=entry["sha256"],
              ckpt_bytes=T.CKPT_BYTES, campaign_digest=T.DIGEST, spawn_digest=T.DIGEST, post_digest=T.DIGEST)
    with pytest.raises(ValueError, match="attention backend"):
        C.build_receipt(entry, r, backend=None, **kw)                               # the runner's claim is checked
    rec = C.build_receipt(entry, r, backend=C.backend_fields(), **kw)
    check = dict(stem=entry["stem"], link=entry["link"], target=entry["target"], ckpt_sha256=entry["sha256"],
                 dump=False, dump_root=entry["dump_root"], pin=PIN)
    assert rec["schema_version"] == 2 and C.receipt_reasons(rec, r, **check) == []
    v1 = dict({k: v for k, v in rec.items() if k not in C.RECEIPT_V2_KEYS}, schema_version=1)
    reasons = C.receipt_reasons(v1, r, **check)
    assert any("receipt keys" in x for x in reasons) and any("schema_version" in x for x in reasons), reasons
    assert any("attention_backend" in x for x in reasons), reasons


def test_the_25_yaw_c4s_names_are_admitted_and_only_there():
    assert all(C.exp36_name_reasons(r) == [] for r in MANIFEST["cells"])
    for block, name in (("YAW-C4S", "exp25_YAW_C4_S40000_K1_s47_th90"), ("YAW-C4S", "exp36_YAW_C4_S40000_K1_s42_th90"),
                        ("YAW-pin", REPLAY), ("YAW-pin", CELLS24[3]), ("CTL", "exp36_x")):
        assert C.exp36_name_reasons(dict(_row(REPLAY), block=block, cell=name)), (block, name)


@pytest.mark.parametrize("old, new, needle", [(CELLS24[2], "exp25_YAW_C4_S40000_K1_s43_th91", "registered exp_36"),
                                              ("exp25_YAW_C4_S40000_K8_s42_th90", REPLAY, "exp_36 name"),
                                              ("exp25_YAW_T_S40000_K1_s43", CELLS24[0], "exp_36 name")])
def test_verify_cell_refuses_a_misnamed_row(old, new, needle):
    m = copy.deepcopy(MANIFEST)
    r = _row(old, m)
    r["cell"] = new                                                     # the suffix still finds the row
    reasons = C.verify_cell(_stem(r), manifest=m, farm_root=T.FARM, dump_root="/d", pin=PIN)
    assert len(reasons) == 1 and needle in reasons[0], reasons
