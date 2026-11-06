"""exp_36 yaw_c4_spread: the YAW-C4S block of the evaluation cell manifest (plan_yaw_c4_spread.md r0 §2 item 1 as
amended by the Codex plan review B2 / NIT 1-2).

``depth_canon_make_cells.py`` now renders 371 rows: the 346 rows of the H35 manifest (exp_25 + exp_35) byte for byte,
then the 25 rows of block ``YAW-C4S`` (orders 346-370, priority 18): FIRST the replay
``exp36_YAW_C4_S40000_K8_s42_th90`` (the exp_25 YAW-pin C4 K8 s42 th90 cell under another name), then the 24
``exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270}`` cells (the YAW-pin C4 rows at seeds 43-46, under the names
and metrics suffixes the paper's figure script reads). Both historical renders are pinned:
``build_manifest(exp36=False)`` is the H35 manifest (sha256 440eb1be...), ``build_manifest(exp35=False)`` the exp_25
manifest (sha256 1073ef64...).

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + its Codex plan review.
"""
import collections
import hashlib
import importlib.util
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")
H35 = "5b065801e9d1c1844461ec84cd0558702c714326"
H35_CELLS_SHA256 = "440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d"     # Codex B2
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
EXP25_CELLS_SHA256 = "1073ef647434318d219dba3c20e3a18b5a909c90aa3ee6e6bd86021c965f0575"
REPLAY, TEMPLATE = "exp36_YAW_C4_S40000_K8_s42_th90", "exp25_YAW_C4_S40000_K8_s42_th90"
CELLS24 = [f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)]
BOOKKEEPING = ("cell", "order", "priority", "block", "metrics_suffix", "eval_args")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load(os.path.join(REPO, EXPREL, "depth_canon_make_cells.py"), "depth_canon_make_cells_exp36")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _at(commit):
    return subprocess.run(["git", "-C", REPO, "show", f"{commit}:{EXPREL}/depth_canon_cells.json"],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def full():
    return MC.build_manifest()


@pytest.fixture(scope="module")
def h35():
    return MC.build_manifest(exp36=False)


def _by_name(manifest):
    return {c["cell"]: c for c in manifest["cells"]}


def _swap(args, old, new, seed=None):
    """``args`` with the cell name ``old`` -> ``new`` and (``seed``) the value after ``--seed``."""
    return [new if x == old else str(seed) if seed is not None and i and args[i - 1] == "--seed" else x
            for i, x in enumerate(args)]


# ---- Codex B2: both historical renders stay exact ---- #
def test_the_h35_render_is_the_committed_h35_manifest(h35):
    text = MC.render(h35)
    assert _sha(text) == H35_CELLS_SHA256 and text == _at(H35) and len(h35["cells"]) == 346


def test_the_exp25_only_render_is_unchanged():
    text = MC.render(MC.build_manifest(exp35=False))
    assert _sha(text) == EXP25_CELLS_SHA256 and text == _at(EXP25_PIN)
    with pytest.raises(ValueError, match="exp35"):
        MC.build_manifest(exp35=False, exp36=True)


def test_the_committed_file_is_the_full_render(full):
    with open(os.path.join(REPO, EXPREL, "depth_canon_cells.json")) as f:
        assert f.read() == MC.render(full)
    assert MC.main(["--check"]) == 0 and len(full["cells"]) == 371


# ---- the YAW-C4S block: 25 rows appended after the H35 rows (NIT 1: the replay belongs to the count) ---- #
def test_the_full_manifest_appends_the_25_yaw_c4s_rows_to_the_h35_rows(full, h35):
    assert full["cells"][:346] == h35["cells"]
    rows = full["cells"][346:]
    assert [c["order"] for c in rows] == list(range(346, 371))
    assert {c["block"] for c in rows} == {"YAW-C4S"} and {c["priority"] for c in rows} == {18}
    assert full["blocks"] == h35["blocks"] + ["YAW-C4S"] and full["blocks"].index("YAW-C4S") == 18
    assert full["block_counts"] == dict(h35["block_counts"], **{"YAW-C4S": 25})
    assert full["plan"] == h35["plan"] + "; exp_36 rows: plan_yaw_c4_spread.md r0 §2 + its Codex plan review (B2)"
    same = [k for k in h35 if k not in ("cells", "blocks", "block_counts", "plan")]
    assert list(full) == list(h35) and {k: full[k] for k in same} == {k: h35[k] for k in same}
    assert all(list(c) == list(h35["cells"][0]) for c in rows) and len(rows[0]) == 26


def test_the_replay_is_first_and_differs_from_its_template_only_by_name(full):
    rows, by = full["cells"][346:], _by_name(full)
    replay, ref = rows[0], by[TEMPLATE]
    assert (replay["cell"], replay["order"], ref["block"], ref["subblock"]) == (REPLAY, 346, "YAW-pin", "C4")
    assert replay["eval_args"] == _swap(ref["eval_args"], TEMPLATE, REPLAY)
    assert {k: v for k, v in replay.items() if k not in BOOKKEEPING} == {k: v for k, v in ref.items()
                                                                         if k not in BOOKKEEPING}
    assert replay["metrics_suffix"] == ref["metrics_suffix"].replace(TEMPLATE, REPLAY) == f"1_1.0_{REPLAY}_rot90"
    assert replay["dump"] is True                  # dumps are deselected at run time (--no-dump-blocks YAW-C4S)


def test_the_24_cells_are_the_yaw_pin_c4_rows_at_seeds_43_to_46(full):
    rows, by = full["cells"][347:], _by_name(full)
    assert [c["cell"] for c in rows] == CELLS24                       # K, then seed, then angle
    for c in rows:
        d = c["rotate_deg"]
        assert c["metrics_suffix"] == f"1_1.0_exp25_YAW_C4_S40000_K{c['K']}_s{c['seed']}_th{d}_rot{d}"
        ref = by[f"exp25_YAW_C4_S40000_K{c['K']}_s42_th{d}"]
        assert (ref["block"], ref["seed"], c["subblock"], c["arm"]) == ("YAW-pin", 42, "C4", "YAW")
        assert c["eval_args"] == _swap(ref["eval_args"], ref["cell"], c["cell"], seed=c["seed"]), c["cell"]
        differ = BOOKKEEPING + ("seed",)
        assert {k: v for k, v in c.items() if k not in differ} == {k: v for k, v in ref.items() if k not in differ}
    assert collections.Counter((c["K"], c["seed"], c["rotate_deg"]) for c in rows) == {
        (k, s, d): 1 for k in (1, 8) for s in (43, 44, 45, 46) for d in (90, 180, 270)}


def test_the_argv_differs_from_the_seed42_template_only_in_seed_angle_and_name(full):
    """Yixun's order: exactly the flags of the exp_25 YAW_C4 seed-42 cells -- per K, the s42 th90 row's argv except
    ``--seed`` / ``--rotate-deg`` / ``--eval-name``."""
    by = _by_name(full)
    for c in full["cells"][346:]:
        a, b = c["eval_args"], by[f"exp25_YAW_C4_S40000_K{c['K']}_s42_th90"]["eval_args"]
        assert len(a) == len(b) and [x for x in a if x.startswith("--")] == [y for y in b if y.startswith("--")]
        changed = {a[i - 1] for i in range(1, len(a)) if a[i] != b[i]}
        assert "--eval-name" in changed and changed <= {"--seed", "--rotate-deg", "--eval-name"}, c["cell"]


def test_names_and_suffixes_stay_injective(full, h35):
    names = [c["cell"] for c in full["cells"]]
    assert len(set(names)) == 371 and len({c["metrics_suffix"] for c in full["cells"]}) == 371
    old = {c["cell"] for c in h35["cells"]}
    assert not old & set(CELLS24 + [REPLAY])
    assert all("exp14_" not in n and "exp11_" not in n for n in names)


def test_only_a_yaw_c4s_row_takes_an_explicit_name():
    args = ("YAW", 40000, "unseen", 8, 42, ("fixed", 90))
    with pytest.raises(ValueError):
        MC._row("YAW-C4S", "C4", *args)
    with pytest.raises(ValueError):
        MC._row("YAW-pin", "C4", *args, cell=REPLAY)
    assert MC._row("YAW-C4S", "C4", *args, cell=REPLAY)["cell"] == REPLAY
