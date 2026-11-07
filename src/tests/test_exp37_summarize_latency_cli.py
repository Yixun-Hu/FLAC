"""exp_37 round B-b: the 3-arm latency summarizer CLI (``tools/exp37_summarize_latency.py``): arms x K_ctx rows with
the bundle's statistics, peak memory as separate quantities (whole-query peak, selector peak, their MAXIMUM, the
loaded-weight footprint), co-tenancy validity, and the stage split only when it agrees with the headline within
2 %. CPU only, synthetic runs. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-b.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from exp37_fakes import attest_run, attest_selector, example_identity, host_snapshot, selector_binding, \
    write_attestation
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result
from tools import exp37_summarize_latency as summ

REPO = Path(__file__).resolve().parents[2]
COUNTS = {1: 10, 2: 30}
_AUDIT_BODY = {"mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}
AUDIT = {**_AUDIT_BODY, "sha256": canonical_sha256(_AUDIT_BODY)}


def _write_hashed(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


@pytest.fixture
def world(tmp_path):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c, "candidate_indices_sha256": f"h{i}"}
               for i, c in COUNTS.items()]
    selection = {"schema_version": 1, "query_count": 2, "records": records}
    selection = {**selection, "sha256": canonical_sha256(selection)}
    sel_path = _write_hashed(tmp_path / "selection.json", {k: v for k, v in selection.items() if k != "sha256"})
    from exp37_fakes import selector_fields
    fields = selector_fields(selection, 0.5, peak=3000)
    binding = selector_binding(fields, tmp_path / "selector.json")
    selector = _write_hashed(tmp_path / "selector.json", {**fields, "binding": binding})
    attest_selector(selector, binding)
    _write_hashed(tmp_path / "audit.json", _AUDIT_BODY)

    def run(name, arm, k=8, scale=1.0, split=False, cotenant=False):
        expected = ident.registered_identity(arm, selection=selection, audit=AUDIT, seed=42, timing=True,
                                             context_count=k, stage_split=split)
        identity = example_identity(expected, AUDIT)
        run_dir = tmp_path / name
        manifest = initialize_run(run_dir, identity)
        for i, n in COUNTS.items():
            core = (0.1 + n / 100) * scale
            result = {"run_manifest_sha256": manifest["sha256"], "query_index": i, "query_id": f"q{i}.wav",
                      "candidate_count": n, "peak_memory_bytes": 1000 + n, "candidate_indices_sha256": f"h{i}",
                      "latency_protocol": {"name": f"fem_core_aligned_kctx{k}_kgen1", "context_count": k,
                                           **({"stage_split": True} if split else {})},
                      "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": core - 0.1,
                                          "core_forward_total": core}}
            if split:
                result.update(stage_seconds={"context_conditioning": 0.1, "candidate_conditioning": (core - 0.1) / 4,
                                             "dit": (core - 0.1) / 4, "vae_decode": (core - 0.1) / 2},
                              stage_sum_seconds=core, stage_peak_memory={
                                  "dit": {"allocated_bytes": 2000, "reserved_bytes": 2500},
                                  "whole_query": {"allocated_bytes": 2100, "reserved_bytes": 2500}})
            save_query_result(run_dir, result=result, candidates=np.zeros((n, 3)), similarities=np.zeros((n, 1)))
        attest_run(run_dir, co_tenant=cotenant, loaded_extra={"memory_after_flac_load": {"memory_allocated_bytes": 700},
                                                              "memory_after_agree_load": {"memory_allocated_bytes": 1100}})
        return run_dir
    return selection, sel_path, selector, run


def test_rows_carry_the_bundle_statistics_memory_and_validity(world):
    selection, _sel_path, selector, run = world
    out = summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, run("p1", "P1")), ("BF", 1, run("bf1", "BF", k=1))])
    p1 = out["rows"][0]
    assert (p1["arm"], p1["context_count"], p1["timing_valid"]) == ("P1", 8, True)
    assert p1["overall"]["summed_seconds"] == pytest.approx(0.2 + 0.4 + 1.0)
    assert p1["memory"] == {"query_peak_allocated_max_bytes": 1030, "query_peak_allocated_median_bytes": 1020.0,
                            "selector_peak_allocated_max_bytes": 3000, "combined_inference_peak_allocated_bytes": 3000,
                            "weights_footprint": {"memory_after_flac_load": {"memory_allocated_bytes": 700},
                                                  "memory_after_agree_load": {"memory_allocated_bytes": 1100},
                                                  "loaded_record_count": 1}}
    assert out["rows"][1]["context_count"] == 1 and "stage_split" not in p1
    assert out["sha256"] == canonical_sha256({k: v for k, v in out.items() if k != "sha256"})


def test_the_split_is_published_within_two_percent_and_withheld_beyond(world):
    selection, _sel_path, selector, run = world
    head = run("head", "YAW")
    near = summ.build_summary(selection, selector, audit=AUDIT, runs=[("YAW", 8, head)], splits=[("YAW", 8, run("s1", "YAW", scale=1.01, split=True))])
    assert near["rows"][0]["stage_split"]["published"] is True and "breakdown" in near["rows"][0]["stage_split"]
    far = summ.build_summary(selection, selector, audit=AUDIT, runs=[("YAW", 8, head)], splits=[("YAW", 8, run("s2", "YAW", scale=1.05, split=True))])
    withheld = far["rows"][0]["stage_split"]
    assert withheld["published"] is False and "breakdown" not in withheld
    assert withheld["comparison"]["aggregate_relative_difference"] == pytest.approx(0.05)
    assert far["rows"][0]["overall"] == near["rows"][0]["overall"]                  # the headline is never withheld


def test_bad_combinations_are_refused(world):
    selection, _sel_path, selector, run = world
    p1 = run("p1", "P1")
    with pytest.raises(ValueError, match="duplicate"):
        summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1), ("P1", 8, p1)])
    with pytest.raises(ValueError, match="no headline"):
        summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1)], splits=[("BF", 8, p1)])


def test_the_summary_records_the_selector_validity(world, tmp_path):
    selection, _sel_path, selector, run = world
    p1 = run("p1", "P1")
    assert summ.build_summary(selection, selector, audit=AUDIT, runs=[("P1", 8, p1)])["selector_timing_valid"] is True
    body = {k: v for k, v in json.loads(selector.read_text()).items() if k not in ("sha256", "binding")}
    bare = _write_hashed(tmp_path / "bare_selector.json", body)
    out = summ.build_summary(selection, bare, audit=AUDIT, runs=[("P1", 8, p1)])
    assert out["selector_host"]["timing_valid"] is False and out["selector_timing_valid"] is False   # no binding


# --------------------------------------------------------------------------- #
# fix round (Codex BLOCKING 4 / HIGH 3, NIT 2): the registered B1 table, replays, inventory, diagnostic outputs
# --------------------------------------------------------------------------- #
def _registered(run, *, k1_splits=True, busy=None):
    runs = [(arm, k, run(f"h_{arm}_{k}", arm, k=k, cotenant=(arm, k) == busy)) for arm in ("P1", "YAW", "BF")
            for k in (8, 1)]
    splits = [(arm, k, run(f"s_{arm}_{k}", arm, k=k, split=True)) for arm in ("P1", "YAW", "BF")
              for k in ((8, 1) if k1_splits else (8,))]
    return runs, splits


def _replays(tmp_path, selection, runs, splits):
    """D-c: every pass's untimed artifact replay (bitwise, dumped), as --replay / --split-replay arguments."""
    from exp37_world import World
    w = World(tmp_path / "replays")
    out = []
    for flag, items in (("--replay", runs), ("--split-replay", splits)):
        for arm, k, d in items:
            out += [flag, arm, k, w.make(f"{Path(d).name}_replay", arm, selection, AUDIT, kind="replay", of=d,
                                         context_count=k)]
    return out


def _argv(sel_path, selector, tmp_path, runs, splits, out, extra=()):
    argv = ["--selection", sel_path, "--selector", selector, "--geometry-audit", tmp_path / "audit.json",
            "--output-json", out, "--output-md", out.with_suffix(".md")]
    for arm, k, d in runs:
        argv += ["--run", arm, k, d]
    for arm, k, d in splits:
        argv += ["--split", arm, k, d]
    return [str(a) for a in argv + list(extra)]


def _main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["exp37_summarize_latency.py"] + argv)
    return summ.main()


def _complete(world, tmp_path, **kw):
    selection, sel_path, selector, run = world
    runs, splits = _registered(run, **kw)
    return runs, splits, _replays(tmp_path, selection, runs, splits) + ["--require-replays"]


def test_the_registered_table_publishes_once_with_its_inventory(world, tmp_path, monkeypatch):
    selection, sel_path, selector, _run = world
    runs, splits, extra = _complete(world, tmp_path)
    out = tmp_path / "b1" / "summary.json"
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra)) == 0
    summary = json.loads(out.read_text())
    inventory = summary["inventory"]
    assert set(inventory["runs"]) == {str(d) for _a, _k, d in runs + splits}
    assert all(len(e["attestation"]["monitor_head_sha256"]) == 64 and e["replay"]["waveforms"]["mode"] == "replay"
               for e in inventory["runs"].values())
    assert inventory["selector"]["binding_sha256"] and inventory["selection"]["sha256"] == selection["sha256"]
    assert summary["published_files"] == {out.with_suffix(".md").name: hashlib.sha256(
        out.with_suffix(".md").read_bytes()).hexdigest()}
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra)) == 2          # write-once


def test_the_k1_splits_are_all_or_none_and_the_rows_are_exactly_registered(world, tmp_path, monkeypatch):
    selection, sel_path, selector, run = world
    runs, splits, extra = _complete(world, tmp_path, k1_splits=False)       # NIT 2: the K_ctx 1 splits dropped
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, tmp_path / "a" / "s.json", extra)) == 0
    for bad_runs, bad_splits in ((runs[:-1], splits), (runs, splits[:2])):
        out = tmp_path / "bad" / "s.json"
        assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, bad_runs, bad_splits, out, extra)) == 2
        assert not out.parent.exists()


def test_an_unattested_row_writes_only_a_diagnostic(world, tmp_path, monkeypatch):
    selection, sel_path, selector, run = world
    runs, splits, extra = _complete(world, tmp_path, busy=("BF", 1))
    out = tmp_path / "b1" / "summary.json"
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra)) == 3
    assert not out.exists() and not out.with_suffix(".md").exists()
    diagnostic = json.loads((tmp_path / "b1" / "summary.DIAGNOSTIC.json").read_text())
    assert diagnostic["publishable"] is False and any("BF" in r for r in diagnostic["reasons"])
    assert (tmp_path / "b1" / "summary.DIAGNOSTIC.md").read_text().startswith("# DIAGNOSTIC")


def test_the_replays_are_mandatory(world, tmp_path, monkeypatch):
    """D-c: no written exp_37 exemption -- --require-replays and one validated replay per pass, always."""
    selection, sel_path, selector, run = world
    runs, splits, extra = _complete(world, tmp_path)
    out = tmp_path / "r" / "s.json"
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out)) == 2                 # none
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra[:-1])) == 2     # no flag
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra[4:])) == 2      # one missing
    assert not out.parent.exists()
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra)) == 0


def test_publication_is_one_transaction(world, tmp_path, monkeypatch):
    """Review 2 HIGH 7: a failure while rendering or publishing leaves NO final output (and no staging)."""
    selection, sel_path, selector, run = world
    runs, splits, extra = _complete(world, tmp_path)
    out = tmp_path / "t" / "s.json"

    def broken(summary):
        raise RuntimeError("markdown renderer failed")
    monkeypatch.setattr(summ, "render_markdown", broken)
    with pytest.raises(RuntimeError, match="renderer"):
        _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra))
    assert not out.exists() and not out.with_suffix(".md").exists() and not list(out.parent.glob(".publish-*"))


@pytest.mark.parametrize("key", ["pid", "pid_start_ticks", "hostname", "card_uuid"])
def test_an_incomplete_producer_record_refuses_cleanly(world, tmp_path, monkeypatch, key):
    """Review 3 H-05: a hash-valid producer.json missing one field is a timing-invalid row (exit 3, diagnostic only),
    never a KeyError traceback."""
    selection, sel_path, selector, _run = world
    runs, splits, extra = _complete(world, tmp_path)
    path = Path(runs[0][2]) / "producer.json"
    body = {k: v for k, v in json.loads(path.read_text()).items() if k not in ("sha256", key)}
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    out = tmp_path / "b1" / "summary.json"
    assert _main(monkeypatch, _argv(sel_path, selector, tmp_path, runs, splits, out, extra)) == 3
    diagnostic = json.loads((tmp_path / "b1" / "summary.DIAGNOSTIC.json").read_text())
    assert not out.exists() and any(key in r for r in diagnostic["reasons"])
