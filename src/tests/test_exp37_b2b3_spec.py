"""exp_37 fix round 2 (Codex code review 2, HIGH 6, BLOCKING 4 / 5, D-a / D-c): the B2 / B3 spec (schema 3) is built
deterministically and checked as an EXACT schema before anything is loaded.

``tools/exp37_build_b2b3_spec.py::build_spec`` enumerates every registered run by its registered stem: B2 P1 / BF x
{0.50, 0.75} x seeds 42-44 (D-a: seed 42 is the timing pass with its replay, seeds 43 / 44 accuracy-only, no other
mode), the nested and c2f timing passes WITH their replays (D-c), the seed 43 / 44 c2f accuracy runs, the 15 B3 runs,
the three orbit-latency passes WITH their replays, the synthesis section and the external inputs; it refuses missing
directories unless planning. ``check_spec`` raises SpecError for any missing / extra key, wrong type, empty path, wrong
timing flag, missing replay, gap, duplicate, or a synthesis section that is not exactly pending or complete.
CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2 (replaces the round-1 spec tests).
"""
import copy
import json
import shutil

import pytest

from exp37_world import registered_world
from tools import exp37_aggregate_b2b3 as tool
from tools import exp37_build_b2b3_spec as builder


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return registered_world(tmp_path_factory.mktemp("world"))


def _spec(world, **kw):
    return builder.build_spec(world.inputs, world.runs, synthesis=world.root / "synthesis.json", **kw)


def test_the_builder_enumerates_every_registered_cell(world):
    spec = _spec(world)
    assert spec["schema_version"] == 3 == tool.SPEC_SCHEMA and spec["require_timing_replays"] is True
    assert {(i["arm"], i["grid"], i["seed"]) for i in spec["b2_runs"]} == {
        (a, g, s) for a in ("P1", "BF") for g in ("0.50", "0.75") for s in (42, 43, 44)}
    assert all(i["timing"] is (i["seed"] == 42) and (i["replay"] is not None) is (i["seed"] == 42) for i in spec["b2_runs"])
    assert all(i["replay"].endswith(f"{i['dir'].rsplit('/', 1)[1]}_replay") for i in spec["b2_timing"])
    assert {(i["arm"], i["seed"]) for i in spec["b2_c2f_runs"]} == {(a, s) for a in ("P1", "BF") for s in (43, 44)}
    assert all(i["dir"].endswith(f"b2a_c2f_{i['arm']}_s{i['seed']}") for i in spec["b2_c2f_runs"])
    assert len(spec["b3_runs"]) == 15 and all(r["replay"] for r in spec["b3_latency"]["runs"])
    assert spec["b3_synthesis"]["status"] == "complete" and len(spec["b3_synthesis"]["cells"]) == 12
    tool.check_spec(spec)


def test_d_a_and_d_c_leave_no_alternate_mode():
    import inspect
    assert "seeds_43_44_timing" not in inspect.signature(builder.build_spec).parameters
    assert "require_timing_replays" not in inspect.signature(builder.build_spec).parameters


def test_the_builder_refuses_missing_runs_unless_planning(world, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(world.runs, runs, symlinks=True)
    shutil.rmtree(runs / "b2a_c2f_BF_s44")
    with pytest.raises(FileNotFoundError, match="b2a_c2f_BF_s44"):
        builder.build_spec(world.inputs, runs, synthesis=world.root / "synthesis.json")
    planned = builder.build_spec(world.inputs, runs, synthesis=None, check_exists=False)
    assert planned["b3_synthesis"] == {"status": "pending", "leaf_commit": None, "runbook": None, "metrics_keys": [],
                                       "cells": [], "note": builder.SYNTHESIS_PLACEHOLDER}
    tool.check_spec(planned)


@pytest.mark.parametrize("change, needle", [
    (lambda s: s.update(schema_version=2), "schema_version"),
    (lambda s: s.update(extra=1), "keys"),
    (lambda s: s.pop("b2_c2f_runs"), "keys"),
    (lambda s: s.update(pilot_0p50=""), "pilot_0p50"),
    (lambda s: s.update(require_timing_replays=False), "require_timing_replays"),
    (lambda s: s["b2_runs"].pop(3), "b2_runs"),
    (lambda s: s["b2_runs"].append(dict(s["b2_runs"][0])), "b2_runs"),
    (lambda s: s["b2_runs"][0].pop("dir"), "b2_runs"),
    (lambda s: s["b2_runs"][0].update(dir=""), "dir"),
    (lambda s: s["b2_runs"][0].update(timing="yes"), "timing"),
    (lambda s: s["b2_runs"][0].update(seed="42"), "b2_runs"),
    (lambda s: s["b2_runs"][0].update(replay=None), "replay"),
    (lambda s: next(i for i in s["b2_runs"] if i["seed"] == 43).update(timing=True, replay="/x"), "seed 43"),
    (lambda s: s["b2_timing"][0].update(replay=None), "replay"),
    (lambda s: s["b2_timing"][0].update(extra=1), "b2_timing"),
    (lambda s: s["b2_c2f_runs"].pop(), "b2_c2f_runs"),
    (lambda s: s["b2_c2f_runs"][0].update(seed=42), "b2_c2f_runs"),
    (lambda s: s["b3_runs"].append({"orbit": "c4", "theta": 0, "seed": 42, "dir": "/x"}), "b3_runs"),
    (lambda s: s["b3_runs"][0].update(theta=0.0), "b3_runs"),
    (lambda s: s["b3_latency"]["runs"][0].update(replay=None), "replay"),
    (lambda s: s["b3_latency"].update(extra=1), "b3_latency"),
    (lambda s: s.pop("b3_synthesis"), "keys"),
    (lambda s: s["b3_synthesis"]["cells"].pop(), "b3_synthesis"),
    (lambda s: s["b3_synthesis"]["cells"][0].pop("receipt"), "b3_synthesis"),
    (lambda s: s["b3_synthesis"].update(leaf_commit="abc"), "b3_synthesis"),
    (lambda s: s["b3_synthesis"].update(status="maybe"), "b3_synthesis"),
    (lambda s: s["external"].update(extra=1), "external"),
])
def test_check_spec_is_an_exact_schema(world, change, needle):
    spec = copy.deepcopy(_spec(world))
    change(spec)
    with pytest.raises(tool.SpecError, match=needle):
        tool.check_spec(spec)


def test_the_cli_requires_the_replay_flag_and_writes_once(world, tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(world.inputs))
    out = tmp_path / "spec.json"
    argv = ["--inputs", str(inputs), "--runs-root", str(world.runs), "--synthesis", str(world.root / "synthesis.json"),
            "--output", str(out)]
    assert builder.main(argv) == 2 and not out.exists()                    # D-c: --require-timing-replays is required
    assert builder.main(argv + ["--require-timing-replays"]) == 0 and json.loads(out.read_text())["schema_version"] == 3
    assert builder.main(argv + ["--require-timing-replays"]) == 2         # write-once
