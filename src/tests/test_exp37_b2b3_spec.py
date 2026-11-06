"""exp_37 fix round (Codex BLOCKING 4 / 5): the B2 / B3 aggregation spec is built deterministically and checked for
the EXACT registered coverage before anything is loaded.

``tools/exp37_build_b2b3_spec.py::build_spec`` enumerates every registered run by its registered stem (B2: P1 / BF x
{0.50, 0.75} x seeds 42-44, seed 42 a timing pass with its replay; the nested and c2f passes per arm; B3: the 15
orbit x theta x seed runs -- C4 theta 0 is B2's B-F 0.50 -- and the three orbit-latency passes; the synthesis section;
the external references) and refuses missing directories unless planning. ``exp37_aggregate_b2b3.check_spec`` refuses
any gap, duplicate, extra cell, reported timing pass without a replay, or an unfilled synthesis section.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import copy
import json
import shutil

import pytest

from exp37_world import World
from tools import exp37_aggregate_b2b3 as tool
from tools import exp37_build_b2b3_spec as builder


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("world")).build()
    w.synthesis()
    w.external()
    return w


def test_the_builder_enumerates_every_registered_cell(world):
    spec = builder.build_spec(world.inputs, world.runs, synthesis=world.root / "synthesis.json")
    keys = {(i["arm"], i["grid"], i["seed"]) for i in spec["b2_runs"]}
    assert len(spec["b2_runs"]) == 12 and keys == {(a, g, s) for a in ("P1", "BF") for g in ("0.50", "0.75")
                                                    for s in (42, 43, 44)}
    for item in spec["b2_runs"]:
        assert item["timing"] is (item["seed"] == 42)
        assert (item["replay"] is not None) is (item["seed"] == 42)
        assert item["dir"].endswith(f"b2_{item['grid'].replace('.', 'p')}_{item['arm']}_s{item['seed']}")
    assert {(i["arm"], i["kind"]) for i in spec["b2_timing"]} == {(a, k) for a in ("P1", "BF") for k in ("nested_1m", "c2f")}
    assert len(spec["b3_runs"]) == 15 and ("c4", 0) not in {(i["orbit"], i["theta"]) for i in spec["b3_runs"]}
    assert sorted(r["orbit"] for r in spec["b3_latency"]["runs"]) == ["0", "0_180", "c4"]
    assert next(r for r in spec["b3_latency"]["runs"] if r["orbit"] == "c4")["dir"].endswith("b1_k8_BF_s42")
    assert spec["b3_synthesis"]["status"] == "complete" and len(spec["b3_synthesis"]["cells"]) == 12
    tool.check_spec(spec)                                                           # complete -> no SpecError


def test_the_builder_refuses_missing_runs_unless_planning(world, tmp_path):
    runs = tmp_path / "runs"
    shutil.copytree(world.runs, runs, symlinks=True)
    shutil.rmtree(runs / "b3_c4_t90_s44")
    with pytest.raises(FileNotFoundError, match="b3_c4_t90_s44"):
        builder.build_spec(world.inputs, runs, synthesis=world.root / "synthesis.json")
    planned = builder.build_spec(world.inputs, runs, synthesis=None, check_exists=False)
    assert planned["b3_synthesis"]["status"] == "pending" and planned["b3_synthesis"]["cells"] == []


def test_optional_timing_repeats_and_replays_are_explicit(world):
    spec = builder.build_spec(world.inputs, world.runs, synthesis=world.root / "synthesis.json", check_exists=False,
                              seeds_43_44_timing=True, require_timing_replays=True)
    assert all(i["timing"] and i["replay"] for i in spec["b2_runs"])
    assert all(i["replay"] for i in spec["b2_timing"]) and all(r["replay"] for r in spec["b3_latency"]["runs"])
    assert spec["require_timing_replays"] is True


def _mutated(world, change):
    spec = builder.build_spec(world.inputs, world.runs, synthesis=world.root / "synthesis.json")
    spec = copy.deepcopy(spec)
    change(spec)
    return spec


@pytest.mark.parametrize("change, needle", [
    (lambda s: s["b2_runs"].pop(3), "b2_runs"),
    (lambda s: s["b2_runs"].append(dict(s["b2_runs"][0])), "b2_runs"),
    (lambda s: s["b2_runs"][0].update(timing=False, replay=None), "seed 42"),
    (lambda s: s["b2_runs"][0].update(replay=None), "replay"),
    (lambda s: next(i for i in s["b2_runs"] if i["seed"] == 43).update(replay="/x"), "replay"),
    (lambda s: s["b2_timing"].pop(), "b2_timing"),
    (lambda s: s["b3_runs"].append({"orbit": "c4", "theta": 0, "seed": 42, "dir": "/x"}), "b3_runs"),
    (lambda s: s["b3_runs"].pop(0), "b3_runs"),
    (lambda s: s["b3_latency"]["runs"].append(dict(s["b3_latency"]["runs"][0])), "b3_latency"),
    (lambda s: s.pop("b3_synthesis"), "b3_synthesis"),
    (lambda s: s["b3_synthesis"]["cells"].pop(), "b3_synthesis"),
    (lambda s: s["b3_synthesis"]["cells"][0].update(source="reused_exp35"), "b3_synthesis"),
    (lambda s: s["b3_synthesis"].update(leaf_commit="abc"), "b3_synthesis"),
    (lambda s: s.pop("external"), "external"),
    (lambda s: s.update(require_timing_replays=True), "replay"),
    (lambda s: s.update(schema_version=1), "schema_version"),
])
def test_check_spec_refuses_every_gap(world, change, needle):
    with pytest.raises(tool.SpecError, match=needle):
        tool.check_spec(_mutated(world, change))


def test_a_pending_synthesis_section_is_a_valid_spec_but_says_so(world):
    spec = builder.build_spec(world.inputs, world.runs, synthesis=None)
    tool.check_spec(spec)
    assert spec["b3_synthesis"] == {"status": "pending", "leaf_commit": None, "runbook": None, "metrics_keys": [],
                                    "cells": [], "note": builder.SYNTHESIS_PLACEHOLDER}


def test_the_cli_writes_the_spec_once(world, tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(world.inputs))
    out = tmp_path / "spec.json"
    argv = ["--inputs", str(inputs), "--runs-root", str(world.runs), "--synthesis", str(world.root / "synthesis.json"),
            "--output", str(out)]
    assert builder.main(argv) == 0 and json.loads(out.read_text())["schema_version"] == tool.SPEC_SCHEMA
    assert builder.main(argv) == 2                                                  # write-once
