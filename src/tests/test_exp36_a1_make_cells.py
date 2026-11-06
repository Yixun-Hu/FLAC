"""exp_36 amendment A1 (Yixun 2026-11-06 15:30/15:32, relayed by the Planner): the seed-42 off-orbit run -- two more
blocks of the evaluation cell manifest, both exp_36 rows (explicit names, no-flash, out of the default selection).

* ``RPL-S42`` (3 rows, orders 371-373, priority 19): the replays ``exp36_{P1,BF,CANON}_C4_S40000_K8_s42_th90`` of
  ``exp25_P1_C4_S40000_K8_s42_th90`` (P1-pin), ``exp35_BF_C4_S40000_K8_s42_th90`` (BF-pin) and
  ``exp25_CANON_C4_S40000_K8_s42_th90`` (C4+45) -- each its reference row under another name (the YAW replay is
  YAW-C4S's first row);
* ``OFF-S42`` (12 rows, orders 374-385, priority 20): ``exp35_{P1,BF,CANON}_OFF_S40000_K{1,8}_s42_th315`` (their
  th225 sibling's argv but ``--rotate-deg 315`` / ``--eval-name``) and ``exp25_YAW_C4_S40000_K{1,8}_s42_th{135,225,315}``
  (the YAW-pin C4 s42 argv but ``--rotate-deg`` / ``--eval-name``).

With the 12 H35 rows ``exp35_{P1,BF,CANON}_OFF_S40000_K{1,8}_s42_th{135,225}`` they are the 24 off-orbit production
cells. The H35 render (440eb1be...) and the exp_25 render (1073ef64...) stay exact (test_exp36_make_cells.py).

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + amendment A1.
"""
import collections

import pytest

from test_exp36_make_cells import MC, _by_name, _swap

C4 = "C4_S40000_K8_s42_th90"
REPLAYS = {f"exp36_P1_{C4}": f"exp25_P1_{C4}", f"exp36_BF_{C4}": f"exp35_BF_{C4}",
           f"exp36_CANON_{C4}": f"exp25_CANON_{C4}"}
TH315 = [f"exp35_{a}_OFF_S40000_K{k}_s42_th315" for a in ("P1", "BF", "CANON") for k in (1, 8)]
YAWOFF = [f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" for k in (1, 8) for d in (135, 225, 315)]
H35_OFF = [f"exp35_{a}_OFF_S40000_K{k}_s42_th{d}" for a in ("P1", "BF", "CANON") for k in (1, 8) for d in (135, 225)]
BOOK = ("cell", "order", "priority", "block", "metrics_suffix", "eval_args")


@pytest.fixture(scope="module")
def full():
    return MC.build_manifest(exp37=False)                 # the exp_36 manifest (exp_37 B3 appends after it)


def _rows(m, block):
    return [c for c in m["cells"] if c["block"] == block]


def test_the_a1_blocks_follow_yaw_c4s(full):
    assert len(full["cells"]) == 386 and full["blocks"][-3:] == ["YAW-C4S", "RPL-S42", "OFF-S42"]
    assert (full["block_counts"]["RPL-S42"], full["block_counts"]["OFF-S42"]) == (3, 12)
    for block, orders, prio in (("RPL-S42", range(371, 374), 19), ("OFF-S42", range(374, 386), 20)):
        rows = _rows(full, block)
        assert [c["order"] for c in rows] == list(orders) and {c["priority"] for c in rows} == {prio}
    assert [c["cell"] for c in _rows(full, "RPL-S42")] == list(REPLAYS)
    assert [c["cell"] for c in _rows(full, "OFF-S42")] == TH315 + YAWOFF
    assert "amendment A1" in full["plan"] and MC.EXP36_BLOCKS == ("YAW-C4S", "RPL-S42", "OFF-S42")
    assert MC.EXP37_BLOCKS == ("RPL-37", "BF-ORB")


def test_each_replay_is_its_reference_row_under_another_name(full):
    by = _by_name(full)
    for replay, ref_name in REPLAYS.items():
        new, ref = by[replay], by[ref_name]
        assert new["eval_args"] == _swap(ref["eval_args"], ref_name, replay), replay
        assert {k: v for k, v in new.items() if k not in BOOK} == {k: v for k, v in ref.items() if k not in BOOK}
        assert new["metrics_suffix"] == ref["metrics_suffix"].replace(ref_name, replay)
    assert by[f"exp36_BF_{C4}"]["metrics_suffix"] == f"1_1.0_exp36_BF_{C4}_fa_invariant_a4_rot90"
    assert by[f"exp36_CANON_{C4}"]["metrics_suffix"] == f"1_1.0_exp36_CANON_{C4}_canon_rot90"
    assert by[f"exp36_P1_{C4}"]["metrics_suffix"] == f"1_1.0_exp36_P1_{C4}_rot90"


def _flags_changed(a, b):
    assert len(a) == len(b) and [x for x in a if x.startswith("--")] == [y for y in b if y.startswith("--")]
    return {a[i - 1] for i in range(1, len(a)) if a[i] != b[i]}


def test_the_th315_rows_are_their_th225_siblings_at_315_degrees(full):
    by = _by_name(full)
    token = {"P1": "", "BF": "_fa_invariant_a4", "CANON": "_canon"}
    for name in TH315:
        new, sib = by[name], by[name.replace("_th315", "_th225")]
        assert sib["block"] == f"OFF-{new['arm']}" and new["rotate_deg"] == 315 and new["subblock"] == "OFF"
        assert _flags_changed(new["eval_args"], sib["eval_args"]) == {"--rotate-deg", "--eval-name"}
        assert new["eval_args"][new["eval_args"].index("--rotate-deg") + 1] == "315"
        same = BOOK + ("rotate_deg",)
        assert {k: v for k, v in new.items() if k not in same} == {k: v for k, v in sib.items() if k not in same}
        assert new["metrics_suffix"] == f"1_1.0_{name}{token[new['arm']]}_rot315", name


def test_the_yaw_off_rows_are_the_yaw_pin_c4_s42_row_at_other_angles(full):
    by = _by_name(full)
    for name in YAWOFF:
        new, ref = by[name], by[f"exp25_YAW_C4_S40000_K{by[name]['K']}_s42_th90"]
        assert (ref["block"], new["subblock"], new["arm"], new["seed"]) == ("YAW-pin", "C4", "YAW", 42)
        assert _flags_changed(new["eval_args"], ref["eval_args"]) == {"--rotate-deg", "--eval-name"}
        same = BOOK + ("rotate_deg",)
        assert {k: v for k, v in new.items() if k not in same} == {k: v for k, v in ref.items() if k not in same}
        d = new["rotate_deg"]
        assert new["metrics_suffix"] == f"1_1.0_{name}_rot{d}" and name.endswith(f"_th{d}")


def test_the_24_off_orbit_production_cells(full):
    by = _by_name(full)
    prod = H35_OFF + TH315 + YAWOFF
    assert len(prod) == len(set(prod)) == 24 and all(n in by for n in prod)
    assert {by[n]["block"] for n in H35_OFF} == {"OFF-P1", "OFF-BF", "OFF-CANON"}            # the H35 rows
    assert collections.Counter((by[n]["arm"], by[n]["K"]) for n in prod) == {(a, k): 3 for a in ("P1", "BF", "CANON",
                                                                                                "YAW") for k in (1, 8)}
    assert all(by[n]["seed"] == 42 and by[n]["dump"] is True for n in prod)   # dumps off at run time


def test_names_and_suffixes_stay_injective_over_386_rows(full):
    assert len({c["cell"] for c in full["cells"]}) == 386 == len({c["metrics_suffix"] for c in full["cells"]})
