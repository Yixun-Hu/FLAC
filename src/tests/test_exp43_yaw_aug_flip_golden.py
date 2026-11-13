"""exp_43 std_aug -- the flip-path regression golden (plan §3.3, last item).

``fixtures/exp43_yaw_aug_flip_golden.json`` was captured by the committed tool
``exp43_yaw_aug_flip_golden_capture.py`` (on exp_40's harness). It is a pin against DRIFT of
the ``flip: true`` path -- not an equivalence claim (the flip-ABSENT path is pinned by exp_40's
own golden). Checked here: the fixture is bound to the bytes of both tools, the current tree
replays its record exactly, and the record is internally what the plan defines (offsets = the
reference draws, bits = the next ``randint(0, 2)`` of the same generator, applied indices =
the bits, no global RNG consumed).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

import src.training.diffusion as tdiff

_HERE = Path(__file__).resolve().parent
FIXTURE = _HERE / "fixtures" / "exp43_yaw_aug_flip_golden.json"
EXP40_FIXTURE = _HERE / "fixtures" / "exp40_yaw_aug_enabled_golden.json"
TOOL = _HERE / "exp43_yaw_aug_flip_golden_capture.py"
HARNESS = _HERE / "exp40_yaw_aug_golden_capture.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("exp43_yaw_aug_flip_golden_capture", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


@pytest.fixture(autouse=True)
def _no_flash(monkeypatch):
    import src.models.transformer as transformer
    monkeypatch.setattr(transformer, "flash_attn_func", None)


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def record(fixture):
    return fixture["record"]


@pytest.fixture(scope="module")
def exp40_record():
    return json.loads(EXP40_FIXTURE.read_text())["record"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_draws(step, rank, n, img_w):
    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(42, step, rank))
    offsets = torch.randint(0, img_w, (n,), generator=gen, dtype=torch.long).tolist()
    return offsets, torch.randint(0, 2, (n,), generator=gen, dtype=torch.long).tolist()


def test_fixture_is_bound_to_both_tools_and_its_record(fixture):
    meta = fixture["_meta"]
    assert meta["experiment"] == "exp_43"
    assert meta["capture_tool_sha256"] == _sha(TOOL), (
        "the capture tool changed after the fixture was captured: recapture deliberately")
    assert meta["harness_tool_sha256"] == _sha(HARNESS)
    assert meta["harness_tool_sha256"] == json.loads(EXP40_FIXTURE.read_text())["_meta"]["capture_tool_sha256"]
    assert meta["record_sha256"] == tool.h.canonical_sha256(fixture["record"])
    assert meta["flip_block"] == {"flip": True} and meta["yaw_seed"] == 42
    assert meta["rungs"] == {"8x8": [8, 8], "2x32": [2, 32]}
    assert meta["torch"].startswith("2.7.0") and meta["python"]
    assert meta["source"]["kind"] in ("working-tree", "commit")
    assert set(meta["source"]["closure_sha256"]) == set(tool.CLOSURE)


def test_current_tree_replays_the_flip_golden_exactly(fixture, record):
    got = tool.capture_all()
    for case, ref in record["tiny"].items():
        assert got["tiny"][case] == ref, f"tiny flip training_step {case} drifted"
    assert got["real"] == record["real"], "real-width flip draws drifted"
    assert tool.h.canonical_sha256(got) == fixture["_meta"]["record_sha256"]


def test_real_record_is_the_reference_offsets_then_bits(record, exp40_record):
    per_step = record["real"]["per_step"]
    assert set(per_step) == set(exp40_record["real"]["offsets"]) == {"8x8", "2x32"}
    for rung, (world, micro) in tool.h.RUNGS.items():
        assert set(per_step[rung]) == {str(s) for s in tool.h.REAL_STEPS}
        for step, ranks in per_step[rung].items():
            assert len(ranks) == world
            for rank, entry in enumerate(ranks):
                offsets, bits = _reference_draws(int(step), rank, micro, 512)
                assert entry["offsets"] == offsets == exp40_record["real"]["offsets"][rung][step][rank]
                assert entry["flips"] == bits
                assert entry["applied"] == [i for i, b in enumerate(bits) if b == 1]
    totals = record["real"]["totals"]
    assert totals["samples"] == sum(w * m for w, m in tool.h.RUNGS.values()) * len(tool.h.REAL_STEPS)
    assert totals["flipped"] == sum(len(e["applied"]) for r in per_step.values()
                                    for ranks in r.values() for e in ranks)
    assert 0 < totals["flipped"] < totals["samples"]
    assert record["real"]["global_rng_untouched"] is True


def test_tiny_record_is_the_exp40_roll_plus_the_drawn_flips(record, exp40_record):
    n_flipped = 0
    for (step, rank) in tool.h.TINY_CASES:
        case = f"step{step}_rank{rank}"
        got, ref = record["tiny"][case], exp40_record["tiny"][case]
        offsets, bits = _reference_draws(step, rank, tool.h.TINY_N, tool.h.TINY_IMG_W)
        assert got["offsets"] == offsets == ref["offsets"]
        assert got["alpha_hex"] == ref["alpha_hex"]
        assert got["flips"] == bits and got["applied"] == [i for i, b in enumerate(bits) if b == 1]
        # an unflipped sample reaches the conditioner exactly as on the roll-only arm
        for i, b in enumerate(bits):
            assert (got["metadata_into_conditioner"][i] == ref["metadata_into_conditioner"][i]) is (b == 0)
        # the flip consumes no GLOBAL randomness
        assert got["rng_before"] == ref["rng_before"] and got["rng_after"] == ref["rng_after"]
        n_flipped += len(got["applied"])
    assert 0 < n_flipped < tool.h.TINY_N * len(tool.h.TINY_CASES)


@pytest.mark.parametrize("argv,needle", [
    (["capture", "--out", "x.json", "--sha", "a" * 40], "go together"),
    (["capture", "--out", "x.json", "--repo", "."], "go together"),
    (["capture", "--out", "x.json", "--repo", ".", "--sha", "HEAD"], "40-hex"),
])
def test_capture_cli_refuses_ambiguous_provenance(argv, needle, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match=needle):
        tool.main(argv)
    assert not (tmp_path / "x.json").exists()
