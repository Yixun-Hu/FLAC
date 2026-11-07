"""exp_37 final fix round (Codex code review 3, H-05): every PRODUCER_KEYS field is validated before it is used.

A hash-valid producer record (``producer.json``) or selector binding whose pid / pid_start_ticks / hostname /
card_uuid is missing, of the wrong type or empty makes the timing INVALID with a reason naming the field; nothing
raises (no KeyError / TypeError traceback) -- ``host_cotenancy``, ``snapshot_reasons``, ``monitor_verdict`` and
``selector_cotenancy`` all refuse cleanly, and so does the B1 summarizer (exit 3, diagnostic only; see
``test_exp37_summarize_latency_cli.py``). pid must be a positive int, pid_start_ticks a nonnegative int (never a
bool), hostname and card_uuid nonempty strings. CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), final
fix round: fixed after review 3, not re-reviewed (cap).
"""
import json

import pytest

from exp37_fakes import attest_run, attest_selector, host_snapshot, selector_binding
from src.localization import exp37_cotenancy as cot
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from test_exp37_latency_summary import _arm_run, _selection, _selector_file

GOOD = {"pid": 42, "pid_start_ticks": 123, "hostname": "host", "card_uuid": "GPU-0a1a"}
MISSING = object()
BAD = [("pid", MISSING), ("pid", None), ("pid", "42"), ("pid", True), ("pid", 0), ("pid", -3), ("pid", 42.0),
       ("pid_start_ticks", MISSING), ("pid_start_ticks", None), ("pid_start_ticks", "123"), ("pid_start_ticks", -1),
       ("pid_start_ticks", False), ("hostname", MISSING), ("hostname", None), ("hostname", ""), ("hostname", "  "),
       ("hostname", 7), ("card_uuid", MISSING), ("card_uuid", None), ("card_uuid", ""), ("card_uuid", 5),
       ("card_uuid", ["GPU-0a1a"])]


def _bad(base, key, value):
    out = dict(base)
    if value is MISSING:
        out.pop(key)
    else:
        out[key] = value
    return out


def test_a_complete_producer_has_no_field_reasons():
    assert cot.producer_field_reasons(GOOD) == []
    assert cot.producer_field_reasons(None) and cot.producer_field_reasons([1, 2])


@pytest.mark.parametrize("key, value", BAD)
def test_every_missing_mistyped_or_empty_field_is_named(key, value):
    reasons = cot.producer_field_reasons(_bad(GOOD, key, value))
    assert reasons and all(key in r for r in reasons)
    assert any(key in r for r in lat.snapshot_reasons(host_snapshot(), _bad(GOOD, key, value), "snap"))
    assert lat.producer_missing(_bad(GOOD, key, value)) is True


@pytest.mark.parametrize("key, value", BAD)
def test_a_hash_valid_incomplete_producer_record_is_timing_invalid_not_a_crash(tmp_path, key, value):
    run = attest_run(_arm_run(tmp_path, "run", "P1", _selection({1: 10})))
    assert lat.host_cotenancy(run)["timing_valid"] is True
    path = run / "producer.json"
    body = _bad({k: v for k, v in json.loads(path.read_text()).items() if k != "sha256"}, key, value)
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))      # a valid envelope hash
    verdict = lat.host_cotenancy(run)
    assert verdict["timing_valid"] is False and any(key in r for r in verdict["reasons"])
    monitor = cot.monitor_verdict(run / "cotenancy", binding_sha256=json.loads((run / "run_manifest.json").read_text())[
        "sha256"], producer={k: body.get(k) for k in cot.PRODUCER_KEYS})
    assert monitor["timing_valid"] is False


@pytest.mark.parametrize("key, value", [("pid", "42"), ("pid_start_ticks", True), ("hostname", ""), ("card_uuid", 5)])
def test_a_selector_binding_with_an_invalid_producer_field_is_refused_cleanly(tmp_path, key, value):
    selection = _selection({1: 10})
    path = _selector_file(tmp_path, selection)
    payload = {k: v for k, v in json.loads(path.read_text()).items() if k != "sha256"}
    binding = {k: v for k, v in selector_binding(payload, path).items() if k != "sha256"}
    binding[key] = value
    binding["sha256"] = canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
    body = {**payload, "binding": binding}
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    attest_selector(path, binding)
    verdict = lat.selector_cotenancy(path)
    assert verdict["timing_valid"] is False and any(key in r for r in verdict["reasons"])
