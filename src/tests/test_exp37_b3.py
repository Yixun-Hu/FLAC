"""exp_37 B3 (inference orbit size on OrbitRIR; the Planner's task after the exp_36 fix round): the synthesis leaf.

Two exp_37 blocks of the evaluation cell manifest, both no-flash, dumping (announcement 08), out of the default
selection, every row named explicitly:
* ``RPL-37`` (1 row, order 386): the replay ``exp37_BF_C4_S40000_K8_s42_th90`` of ``exp35_BF_C4_S40000_K8_s42_th90``
  (the exp_35 phase-(i) B-F cell, read-only in the exp-35 run worktree at H35) -- the same comparator and dump checks;
* ``BF-ORB`` (8 rows, orders 387-394): B-F@40k with the registered frame-average orbits {0} (ORB1) and {0, 180} (ORB2)
  x theta {0, 90} x K {1, 8} x seed 42 -- the BF-pin template's argv (T s42 for theta 0, C4 s42 th90 for theta 90) but
  ``--frame-avg-angles`` and ``--eval-name``; the metrics suffix encodes the orbit (``_fa_invariant_a1`` / ``_a2``).
The C4 orbit is the REUSED exp_35 phase-(i) evidence (BF-pin T theta 0 and C4 theta 90, s42, K1 / K8). The collector
takes a B-F row's orbit from its own argv (C4 for every older row). The H35, exp_25 and exp_36 renders stay exact.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) for exp_37 B3.
"""
import json
import os

import pytest

import test_depth_canon_collect as T
from test_exp36_make_cells import MC, _at, _by_name, _sha

C, MANIFEST = T.C, T.MANIFEST
H36_I2 = "d92aef106de09496715b26ecc42f924ead104090"
H36_I3 = "6be49f46ec31e899f1bee01f72c6a5827e701c6c"   # the launch commit (review III); its cells = H36_I2's, byte-identical
REPLAY = "exp37_BF_C4_S40000_K8_s42_th90"
ORB = [f"exp37_BF_ORB{o}_S40000_K{k}_s42_th{d}" for o in (1, 2) for k in (1, 8) for d in (0, 90)]
ANGLES = {1: "0", 2: "0,180"}
BOOK = ("cell", "order", "priority", "block", "subblock", "metrics_suffix", "eval_args")


@pytest.fixture(scope="module")
def full():
    return MC.build_manifest()


def _flags_changed(a, b):
    assert len(a) == len(b) and [x for x in a if x.startswith("--")] == [y for y in b if y.startswith("--")]
    return {a[i - 1] for i in range(1, len(a)) if a[i] != b[i]}


def test_the_exp36_render_stays_exact_and_exp37_needs_exp36():
    text = MC.render(MC.build_manifest(exp37=False))
    assert _sha(text) == "79b95d2b68fe0ecfc9bf767259de6faccba9f8b42f329169ca69a765ba2a5454"
    assert text == _at(H36_I2) == _at(H36_I3)
    with pytest.raises(ValueError, match="exp36"):
        MC.build_manifest(exp36=False, exp37=True)


def test_the_two_exp37_blocks_follow_the_exp36_blocks(full):
    assert len(full["cells"]) == 395 and full["blocks"][-2:] == ["RPL-37", "BF-ORB"] and MC.EXP37_BLOCKS == (
        "RPL-37", "BF-ORB")
    assert full["cells"][:386] == MC.build_manifest(exp37=False)["cells"]
    assert [(c["cell"], c["order"], c["priority"]) for c in full["cells"][386:]] == (
        [(REPLAY, 386, 21)] + [(n, 387 + i, 22) for i, n in enumerate(ORB)])
    assert (full["block_counts"]["RPL-37"], full["block_counts"]["BF-ORB"]) == (1, 8)
    assert "exp_37 B3" in full["plan"] and MC.render(full) == open(os.path.join(MC.REPO, MC.EXPREL,
                                                                                "depth_canon_cells.json")).read()


def test_the_replay_is_the_bf_pin_c4_cell_under_another_name(full):
    by = _by_name(full)
    new, ref = by[REPLAY], by["exp35_BF_C4_S40000_K8_s42_th90"]
    assert new["eval_args"] == [REPLAY if x == ref["cell"] else x for x in ref["eval_args"]]
    same = ("cell", "order", "priority", "block", "metrics_suffix", "eval_args")
    assert {k: v for k, v in new.items() if k not in same} == {k: v for k, v in ref.items() if k not in same}
    assert new["metrics_suffix"] == f"1_1.0_{REPLAY}_fa_invariant_a4_rot90" and new["dump"] is True


def test_the_orbit_rows_are_the_bf_pin_template_with_their_orbit(full):
    by = _by_name(full)
    for name in ORB:
        c = by[name]
        o, d = int(name.split("ORB")[1][0]), c["rotate_deg"]
        ref = by[f"exp35_BF_T_S40000_K{c['K']}_s42" if d == 0 else f"exp35_BF_C4_S40000_K{c['K']}_s42_th90"]
        assert _flags_changed(c["eval_args"], ref["eval_args"]) == {"--frame-avg-angles", "--eval-name"}, name
        assert c["eval_args"][c["eval_args"].index("--frame-avg-angles") + 1] == ANGLES[o]
        assert {k: v for k, v in c.items() if k not in BOOK} == {k: v for k, v in ref.items() if k not in BOOK}
        assert c["metrics_suffix"] == f"1_1.0_{name}_fa_invariant_a{o}" + (f"_rot{d}" if d else ""), name
        assert (c["block"], c["subblock"], c["arm"], c["seed"], c["dump"]) == ("BF-ORB", "ORB", "BF", 42, True)
    assert len({c["metrics_suffix"] for c in full["cells"]}) == len(full["cells"]) == 395


def _row(name):
    (r,) = [c for c in MANIFEST["cells"] if c["cell"] == name]
    return r


def test_the_collector_takes_a_bf_rows_orbit_from_its_argv():
    assert C.EXP37_BLOCKS == ("RPL-37", "BF-ORB")
    for name in ORB + [REPLAY]:
        r = _row(name)
        assert C.uses_no_flash(r) and not C.is_exp35(r) and C.exp36_name_reasons(r) == [], name
        want = [0.0] if "ORB1" in name else [0.0, 180.0] if "ORB2" in name else [0.0, 90.0, 180.0, 270.0]
        assert C.row_protocol(r) == dict(C.ARM_PROTOCOL["BF"], frame_avg_angles=want), name
    for r in MANIFEST["cells"]:
        if r["block"] not in C.EXP37_BLOCKS:                         # every older row: its arm's protocol, unchanged
            assert C.row_protocol(r) == C.arm_protocol(r["arm"]), r["cell"]
    assert C.select_cells(MANIFEST) == [c for c in C.select_cells(MANIFEST) if c["block"] not in C.EXP37_BLOCKS]
    assert [c["cell"] for c in C.select_cells(MANIFEST, blocks=["BF-ORB"])] == ORB


@pytest.mark.parametrize("name", [ORB[0], ORB[3], ORB[6], REPLAY])
def test_an_orbit_cell_written_by_the_evaluator_verifies_with_its_dump(tmp_path, name):
    r = _row(name)
    T.write_cell(tmp_path, r, dump=True, dump_root=str(tmp_path / "dumps"))
    stem = f"epoch=8-step=40000_metrics_{r['metrics_suffix']}"
    assert C.verify_cell(stem, manifest=MANIFEST, farm_root=T.FARM, dump_root=str(tmp_path / "dumps"), pin=T.PIN,
                         base_dir=str(tmp_path), check_link=False) == []
    rec = json.load(open(tmp_path / T.FARM / "BF" / f"{stem}.json"))
    assert rec["frame_avg_angles"] == C.row_protocol(r)["frame_avg_angles"]
    rec["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0] if "ORB" in name else [0.0]       # another orbit's record
    assert any("frame_avg_angles" in x for x in C.record_reasons(rec, r, pin=T.PIN, link=rec["ckpt_path"]))


def test_the_replay_is_registered_against_the_exp35_bf_cell():
    e, bf = C.EXP36_REPLAYS[REPLAY], C.EXP36_REPLAYS["exp36_BF_C4_S40000_K8_s42_th90"]
    assert e == bf and e["reference"] == "exp35_BF_C4_S40000_K8_s42_th90"
    assert e["worktree"] == os.path.expanduser("~/codespace/exp-35-orbitrir-yaw") and e["pin"] == C.H35_PIN


def test_exp37_names_belong_to_their_blocks():
    assert C.exp36_name_reasons(dict(_row(ORB[0]), block="BF-pin"))                   # an exp37_ name elsewhere
    assert C.exp36_name_reasons(dict(_row(ORB[0]), cell="exp37_BF_ORB3_S40000_K1_s42_th0"))   # not registered
    assert C.exp36_name_reasons(dict(_row(REPLAY), block="BF-ORB"))                    # the replay is RPL-37's
