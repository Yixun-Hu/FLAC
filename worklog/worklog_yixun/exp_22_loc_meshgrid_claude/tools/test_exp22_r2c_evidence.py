"""exp_22 A7 (plan §9 A7 r2 (iv), r3 (a) (c) (g), r4, r5): the backend-evidence rule on synthetic logs; T's
three-layer admission rule (exit 0, the JUnit file, the quiet-form terminal summary + eleven PASSED lines) against
REAL runs of the pinned pytest 9.1.1 on tiny fixture projects (CPU) -- every rejection asserting all three layers;
the exclusive / atomic writers."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import load

E = load("exp22_r2c_evidence")
TARGET = "localize_meshgrid.py"
GOOD = [E.OVERLAY_LINE, E.FALLBACK_LINE, E.sentinel_line(TARGET), "G1 audit re-verified: 16 rooms", "done"]
PYTEST_MAIN = "/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytest/__main__.py"
REPO = os.path.abspath(os.path.join(os.path.dirname(E.__file__), "..", "..", "..", ".."))
EVAL = "src/tests/test_eval_localization.py"
LOCQ = "src/tests/test_loc_meshgrid_queries.py"
IDS = [f"{EVAL}::test_integration_generation_is_reproducible_for_a_fixed_noise_key",
       f"{EVAL}::test_integration_parity_one_query_matches_the_eval_flac_path",
       f"{EVAL}::test_integration_real_mk_batch_split_is_bitwise_irrelevant",
       f"{EVAL}::test_integration_real_mk_parity_is_exact_without_conditioning_autocast",
       f"{EVAL}::test_run_query_matches_a_candidate_major_replay_with_full_conditioning[4]",
       f"{EVAL}::test_run_query_matches_a_candidate_major_replay_with_full_conditioning[64]",
       f"{EVAL}::test_run_query_mk_replay_is_stable_across_the_batch_split",
       f"{LOCQ}::test_our_call_graph_matches_an_eval_flac_faithful_reference",
       f"{LOCQ}::test_real_bounded_slice_materializes_through_the_released_loader",
       f"{LOCQ}::test_release_stack_resolves_and_records_the_agree_checkpoint",
       f"{LOCQ}::test_skipping_the_metric_stack_changes_the_draw"]
T_ARGS = ["-x", "-q", "--color=no", "-o", "xfail_strict=true", "-rA", "-p", "no:cacheprovider", "--junitxml"]


def text(lines):
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# the backend rule (A7 r3 (a))
# --------------------------------------------------------------------------- #
def test_the_pinned_literals():
    assert E.SENTINEL == "EXP35_NO_FLASH_LAUNCHER"
    assert E.sentinel_line("x.py") == (
        "EXP35_NO_FLASH_LAUNCHER attention_backend=sdpa overlay_sha256=0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f6"
        "21181a0f3e7c023 launcher_sha256=bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de target=x.py")
    assert E.OVERLAY_LINE == "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
    assert E.FALLBACK_LINE == "flash_attn not installed, disabling Flash Attention"
    assert E.PIN_FILES == {
        "overlays/no_flash/flash_attn.py": "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023",
        "overlays/no_flash/exp35_no_flash_launch.py": "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de",
        "src/tests/test_exp35_no_flash_launch.py": "d61a3c02a7e1295a48b3edb60e8abccab508cadb6f4c11bdca9872234bd961dd",
        "pyproject.toml": "6422eae5482773db252372760825749f41ac061487be892e350fda997b619a74"}


def test_the_files_of_this_checkout_hash_to_the_pins():
    for rel, sha in E.PIN_FILES.items():
        with open(os.path.join(REPO, rel), "rb") as f:
            assert E.sha256_bytes(f.read()) == sha, rel


def test_a_proper_launch_log_is_admitted():
    assert E.backend_log_reasons(text(GOOD), TARGET) == []
    child = GOOD + [E.OVERLAY_LINE, E.FALLBACK_LINE, "worker ok"]                 # a child interpreter's pair
    assert E.backend_log_reasons(text(child), TARGET) == []


@pytest.mark.parametrize("lines, needle", [
    ([E.OVERLAY_LINE, E.FALLBACK_LINE, E.sentinel_line("eval_FLAC.py")], "expected"),          # another target
    ([E.OVERLAY_LINE, E.FALLBACK_LINE], "0 lines"),                                           # no sentinel
    (GOOD + [E.sentinel_line(TARGET)], "2 lines"),                                            # two sentinels
    ([E.OVERLAY_LINE, E.FALLBACK_LINE, E.sentinel_line(TARGET) + " extra"], "expected"),      # not the whole line
    ([E.OVERLAY_LINE, E.FALLBACK_LINE, E.sentinel_line(TARGET).replace("sdpa", "flash")], "expected"),
    ([E.sentinel_line(TARGET), E.OVERLAY_LINE, E.FALLBACK_LINE], "before the sentinel"),      # sentinel first
    ([E.FALLBACK_LINE, E.OVERLAY_LINE, E.sentinel_line(TARGET)], "before the sentinel"),      # reversed pair
    ([E.OVERLAY_LINE, "x", E.FALLBACK_LINE, E.sentinel_line(TARGET)], "before the sentinel"),  # not back to back
    ([E.OVERLAY_LINE, E.FALLBACK_LINE] * 2 + [E.sentinel_line(TARGET)], "before the sentinel"),
    ([E.FALLBACK_LINE, E.sentinel_line(TARGET)], "before the sentinel"),                      # no overlay line
    (GOOD + [E.OVERLAY_LINE], "not followed"),                                                # orphan after
    (GOOD + [E.FALLBACK_LINE], "not preceded"),
    (GOOD[:3] + [E.FALLBACK_LINE] + GOOD[3:], "not preceded"),                                # right after it
    (GOOD + ["ModuleNotFoundError: No module named 'flash_attn'"], "No module named"),
    (GOOD + ["NO-FLASH LAUNCH REFUSED (nothing was run):"], "REFUSED"),
], ids=["target", "none", "two", "suffix", "backend", "first", "reversed", "apart", "two-pairs", "no-overlay",
        "orphan-overlay", "orphan-fallback", "fallback-after-sentinel", "missing-module", "refused"])
def test_every_departure_from_the_rule_is_refused(lines, needle):
    why = E.backend_log_reasons(text(lines), TARGET)
    assert why and needle in " ".join(why), why


# --------------------------------------------------------------------------- #
# T: the JUnit identities and the terminal summary (A7 r3 (c), r4, r5)
# --------------------------------------------------------------------------- #
def test_junit_identities_follow_pytest_mangling():
    assert E.junit_identity(IDS[4]) == ("src.tests.test_eval_localization",
                                        "test_run_query_matches_a_candidate_major_replay_with_full_conditioning[4]")
    assert E.junit_identity("a/b.py::C::t[x::y]") == ("a.b.C", "t[x::y]")


@pytest.mark.parametrize("line, ok", [
    ("11 passed in 12.34s", True), ("11 passed, 1 warning in 0.50s", True), ("11 passed, 2 warnings in 3.00s", True),
    ("11 passed in 75.20s (0:01:15)", True), ("11 passed, 15 warnings in 3725.10s (1:02:05)", True),
    ("11 passed in 1.2s", False), ("=== 11 passed in 1.23s ===", False), ("11 passed, 1 xpassed in 1.00s", False),
    ("11 passed, 0 warnings in 1.00s", False), ("10 passed in 1.00s", False), ("11 passed in 1.00s ", False),
    ("11 passed, 1 deselected in 1.00s", False), ("1 failed, 11 passed in 1.00s", False),
    ("11 passed, 1 warning, 1 error in 1.00s", False), ("11 passed in 1.00s (a(b))", False),
])
def test_the_quiet_form_summary_regex(line, ok):
    assert bool(E.SUMMARY_RE.fullmatch(line)) is ok


BODY = {"pass": "def {n}({p}):\n    pass\n",
        "fail": "def {n}({p}):\n    assert 1 == 2\n",
        "skip": "def {n}({p}):\n    pytest.skip('an asset is missing')\n",
        "xfail": "@pytest.mark.xfail(reason='known')\ndef {n}({p}):\n    assert 0\n",
        "xpass": "@pytest.mark.xfail(reason='known')\ndef {n}({p}):\n    pass\n",
        "xpass_nonstrict": "@pytest.mark.xfail(reason='known', strict=False)\ndef {n}({p}):\n    pass\n",
        "setup_error": "def {n}(boom_setup{c}{p}):\n    pass\n",
        "teardown_error": "def {n}(boom_teardown{c}{p}):\n    pass\n",
        "warn": "def {n}({p}):\n    warnings.warn('a warning')\n"}
HEAD = ("import warnings\nimport pytest\n\n\n@pytest.fixture\ndef boom_setup():\n    raise RuntimeError('setup')\n\n\n"
        "@pytest.fixture\ndef boom_teardown():\n    yield\n    raise RuntimeError('teardown')\n\n\n")


def fixture_project(root, kinds):
    """A tiny checkout with the eleven registered names; ``kinds`` maps a node id to its behaviour (default pass;
    a kind given for one parameter of the parametrized test applies to both)."""
    shutil.copy(os.path.join(REPO, "pyproject.toml"), root / "pyproject.toml")       # the real rootdir marker
    (root / "src" / "tests").mkdir(parents=True)
    code, seen = {EVAL: HEAD, LOCQ: HEAD}, set()
    for nid in IDS:
        path, name = nid.split("[")[0].split("::")
        if (path, name) in seen:
            continue
        seen.add((path, name))
        kind = next((k for i, k in kinds.items() if i.split("[")[0] == f"{path}::{name}"), "pass")
        param = "[" in nid
        code[path] += (("@pytest.mark.parametrize('n', [4, 64])\n" if param else "")
                       + BODY[kind].format(n=name, p="n" if param else "", c=", " if param else "") + "\n\n")
    for path, body in code.items():
        (root / path).write_text(body)
    return root


def run_t(root, extra=(), args=None):
    junit = root / "T_junit.xml"
    argv = [sys.executable, PYTEST_MAIN, *(args or T_ARGS), str(junit), *extra, *IDS]
    p = subprocess.run(argv, cwd=str(root), capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    raw = junit.read_bytes() if junit.exists() else b""
    log = p.stdout + p.stderr
    summary, info = E.pytest_summary_reasons(log, IDS)
    return p.returncode, E.junit_reasons(raw, IDS), summary, info, log


def three(rc, junit, summary):
    """The admission: all three layers (A7 r4 (1)-(3) with r5's regex)."""
    return rc == 0 and not junit and not summary


@pytest.mark.parametrize("kinds, needle", [({}, "11 passed in "),
                                           ({IDS[0]: "warn"}, "11 passed, 1 warning in "),
                                           ({IDS[0]: "warn", IDS[8]: "warn"}, "11 passed, 2 warnings in ")],
                         ids=["plain", "one-warning", "two-warnings"])
def test_eleven_passes_are_admitted_with_or_without_warnings(tmp_path, kinds, needle):
    rc, junit, summary, info, log = run_t(fixture_project(tmp_path, kinds))
    assert three(rc, junit, summary), (rc, junit, summary, log[-2000:])
    assert info["pytest_summary"].startswith(needle) and len(info["passed_lines"]) == 11
    assert info["passed_lines_sha256"] == E.sha256_bytes(("\n".join(info["passed_lines"]) + "\n").encode())


@pytest.mark.parametrize("kind, target, rc_want, junit_needle, summary_needle", [
    ("fail", 2, 1, "failures='1'", "FAILED"),
    ("setup_error", 3, 1, "errors='1'", "ERROR"),
    ("teardown_error", 9, 1, "errors='1'", "ERROR"),
    ("xfail", 6, 0, "skipped='1'", "XFAIL"),
    ("xpass", 7, 1, "failures='1'", "FAILED"),                      # strict under -o xfail_strict=true
    ("xpass_nonstrict", 10, 0, None, "XPASS"),                     # explicit strict=False: JUnit counts it passed
    ("skip", 8, 0, "skipped='1'", "SKIPPED"),
], ids=["assertion-failure", "setup-error", "teardown-error", "xfail", "xpass-strict", "xpass-strict-false", "skip"])
def test_every_other_outcome_is_refused_by_the_layers_that_see_it(tmp_path, kind, target, rc_want, junit_needle,
                                                                  summary_needle):
    rc, junit, summary, info, log = run_t(fixture_project(tmp_path, {IDS[target]: kind}))
    assert rc == rc_want, log[-2000:]
    if junit_needle is None:
        assert junit == [], junit                                  # why the summary layer is needed (A7 r4)
    else:
        assert junit and junit_needle in " ".join(junit), junit
    assert summary and summary_needle in " ".join(summary), (summary, log[-1500:])
    assert not three(rc, junit, summary)


def test_a_non_strict_xpass_is_a_junit_pass_without_the_override(tmp_path):
    """A7 r4: pinned pytest 9.1.1 reports a non-strict XPASS as an ordinary JUnit pass -- only the terminal layer
    (exit 0 too) refuses it; with -o xfail_strict=true the same marker becomes a FAILED report."""
    root = fixture_project(tmp_path, {IDS[0]: "xpass"})
    no_override = [a for a in T_ARGS if a not in ("-o", "xfail_strict=true")]
    rc, junit, summary, info, log = run_t(root, args=no_override)
    assert rc == 0 and junit == [] and summary and "XPASS" in " ".join(summary), (junit, summary)
    rc, junit, summary, info, log = run_t(root)
    assert rc == 1 and junit and summary and "FAILED" in " ".join(summary)


def test_a_deselection_is_refused(tmp_path):
    root = fixture_project(tmp_path, {})
    rc, junit, summary, info, log = run_t(root, extra=["-k", "not test_skipping_the_metric_stack_changes_the_draw"])
    assert rc == 0 and junit and "tests='10'" in " ".join(junit)
    assert summary and "deselected" in info["pytest_summary"]


def test_eleven_passes_of_other_ids_are_refused():
    xml = ('<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="0" '
           'failures="0" skipped="0" tests="11">' + "".join(
               f'<testcase classname="{c}" name="{n}" time="0.1" />' for c, n in
               [E.junit_identity(i) for i in IDS[:10]] + [("src.tests.test_loc_meshgrid_queries", "test_other")])
           + "</testsuite></testsuites>")
    assert E.junit_reasons(xml.encode(), IDS)
    assert E.junit_reasons(xml.replace("test_other", IDS[10].split("::")[1]).encode(), IDS) == []
    assert E.junit_reasons(b"", IDS) and E.junit_reasons(b"<testsuite/>", IDS)
    log = "\n".join(["=" * 20 + " short test summary info " + "=" * 20] + [f"PASSED {i}" for i in IDS[:10]]
                    + [f"PASSED {IDS[0]}", "11 passed in 1.00s"])
    assert E.pytest_summary_reasons(log, IDS)[0]                     # a duplicate in place of the eleventh


def test_a_log_without_exactly_one_short_summary_is_refused():
    body = [f"PASSED {i}" for i in IDS] + ["11 passed in 1.00s"]
    hdr = "=" * 20 + " short test summary info " + "=" * 20
    assert E.pytest_summary_reasons("\n".join(body), IDS)[0]
    assert E.pytest_summary_reasons("\n".join([hdr] + body + [hdr] + body), IDS)[0]
    assert E.pytest_summary_reasons("\n".join([hdr] + body), IDS)[0] == []


# --------------------------------------------------------------------------- #
# the writers
# --------------------------------------------------------------------------- #
def test_an_exclusive_record_is_never_overwritten(tmp_path):
    p = tmp_path / "dispatch_01_T.json"
    sha = E.write_exclusive(str(p), b'{"a": 1}\n')
    assert sha == E.sha256_bytes(b'{"a": 1}\n') and p.read_bytes() == b'{"a": 1}\n'
    with pytest.raises(FileExistsError):
        E.write_exclusive(str(p), b"other")
    assert p.read_bytes() == b'{"a": 1}\n'
    os.symlink(tmp_path / "elsewhere", tmp_path / "link.json")
    with pytest.raises(FileExistsError):
        E.write_exclusive(str(tmp_path / "link.json"), b"x")
    assert not (tmp_path / "elsewhere").exists()


def test_an_atomic_state_write_replaces_whole_or_not_at_all(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    E.write_json_atomic(str(p), {"phase": "reserved"})
    assert json.loads(p.read_text()) == {"phase": "reserved"}
    monkeypatch.setattr(E.json, "dumps", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        E.write_json_atomic(str(p), {"phase": "supervisor_started"})
    assert json.loads(p.read_text()) == {"phase": "reserved"} and os.listdir(tmp_path) == ["state.json"]
