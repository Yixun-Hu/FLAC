#!/usr/bin/env python3
"""exp_45 L1 regression comparison: the merged tree against BOTH parents (plan_rf_data_curve.md r3 §3.2, §5 L1;
Codex plan r2 finding 6 -- exp43_regression_baseline.py compare takes one baseline and cannot express this).

    exp45_regression_compare.py --merged merged.xml --exp38 parent_3593f34c.xml --exp43 parent_4a7e3b3a.xml [--out r.json]

Inputs: the three junit-XML records of the SAME full-suite command run in a CPU Slurm job on the merged tree, on
3593f34c and on 4a7e3b3a. The registered node policy, exactly:
  1. a node present in a parent must be present in the merged tree (no silent loss);
  2. a merged-tree failure / error is allowed only if (i) the same node failed in at least one parent with the same
     error signature, or (ii) it is one of the three named exp_25 tie-audit tests (they bind the bytes of
     dataset.py / AR_md.py, which exp_38's data side changes; brief §3.1);
  3. the two compat-edited nodes (plan §3.1) and every new exp_45 node must pass (a new exp_45 node may be skipped
     only for its declared environment reason, reported);
  anything else refuses (exit 1). Exit 0 = the policy passes; 2 = an input is unreadable.
A signature is (kind, exception line) with pytest tmp paths, hex addresses and object ids normalised.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

TIE_AUDIT = frozenset({
    "src/tests/test_depth_canon_audit_gate.py::test_the_real_full_audit_passes_the_registered_contract_without_rehash",
    "src/tests/test_depth_canon_audit_gate.py::test_a_report_audited_in_another_checkout_is_accepted",
    "src/tests/test_depth_canon_collect.py::test_the_overlap_derived_from_the_real_validated_audit"})
COMPAT_EDITED = frozenset({
    "src/tests/test_exp40_kit.py::test_prefigure_parse_matches_the_expected_values_type_strict",
    "src/tests/test_exp43_kit.py::test_prefigure_parse_matches_the_expected_values_type_strict"})
RANK = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
NEW_FILE = re.compile(r"^src/tests/(test_exp45_[^/]+|test_gen_model_comparison_exp45)\.py::")


class CompareError(RuntimeError):
    pass


def node_id(case):
    """``src/tests/<file>.py::<name>`` of a junit testcase (pytest's classname is the dotted module path, plus a
    class for methods)."""
    cls, name = case.get("classname", ""), case.get("name", "")
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):                      # the module is the longest prefix ending in test_*
        if parts[i - 1].startswith("test_"):
            return "/".join(parts[:i]) + ".py::" + "::".join(parts[i:] + [name])
    raise CompareError(f"testcase classname {cls!r} names no test module")


def signature(kind, element):
    text = (element.get("message") or element.text or "").strip().split("\n")[0]
    text = re.sub(r"/[^\s'\"]*pytest-of-[^\s'\"]*", "<tmp>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<addr>", text)
    text = re.sub(r"\bpid \d+|\bat \d{5,}\b", "<id>", text)
    return f"{kind}: {text}"


def load(path):
    """{node: (outcome, signature or None)} of one junit XML; outcome in passed / failed / error / skipped."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as err:
        raise CompareError(f"{path}: {err}")
    out = {}
    for case in root.iter("testcase"):
        node = node_id(case)
        kids = {k.tag: k for k in case}
        if "error" in kids:
            res = ("error", signature("error", kids["error"]))
        elif "failure" in kids:
            res = ("failed", signature("failed", kids["failure"]))
        elif "skipped" in kids:
            res = ("skipped", None)
        else:
            res = ("passed", None)
        if node not in out or RANK[res[0]] > RANK[out[node][0]]:   # a teardown error may add a second record
            out[node] = res
    if not out:
        raise CompareError(f"{path}: no testcase")
    return out


def compare(merged, exp38, exp43):
    """(problems, report) of the registered node policy."""
    problems, allowed, new_skips = [], [], []
    for name, parent in (("3593f34c", exp38), ("4a7e3b3a", exp43)):
        lost = sorted(set(parent) - set(merged))
        problems += [f"{n}: present in {name}, missing in the merged tree" for n in lost]
    for node, (outcome, sig) in sorted(merged.items()):
        if node in COMPAT_EDITED or NEW_FILE.match(node):
            if outcome in ("failed", "error") or (node in COMPAT_EDITED and outcome != "passed"):
                problems.append(f"{node}: must pass (compat-edited / new exp_45 node), got {outcome} {sig or ''}")
            elif outcome == "skipped":
                new_skips.append(node)
            continue
        if outcome not in ("failed", "error"):
            continue
        if node in TIE_AUDIT:
            allowed.append(f"{node}: exp_25 tie-audit test (registered exception)")
            continue
        inherited = [name for name, parent in (("3593f34c", exp38), ("4a7e3b3a", exp43)) if parent.get(node) == (outcome, sig)]
        if inherited:
            allowed.append(f"{node}: inherited from {', '.join(inherited)} ({sig})")
        else:
            seen = {name: parent.get(node) for name, parent in (("3593f34c", exp38), ("4a7e3b3a", exp43))}
            problems.append(f"{node}: {outcome} in the merged tree ({sig}); parents {seen}")
    for node in sorted(COMPAT_EDITED - set(merged)):
        problems.append(f"{node}: the compat-edited node is missing")
    report = {"verdict": "PASS" if not problems else "FAIL", "problems": problems, "allowed_failures": allowed,
              "new_exp45_skips": new_skips,
              "counts": {name: {o: sum(1 for v in rec.values() if v[0] == o) for o in ("passed", "failed", "error", "skipped")}
                         for name, rec in (("merged", merged), ("3593f34c", exp38), ("4a7e3b3a", exp43))}}
    return problems, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--merged", required=True)
    p.add_argument("--exp38", required=True, help="junit XML of the suite at 3593f34c")
    p.add_argument("--exp43", required=True, help="junit XML of the suite at 4a7e3b3a")
    p.add_argument("--out")
    a = p.parse_args(argv)
    try:
        problems, report = compare(load(a.merged), load(a.exp38), load(a.exp43))
    except CompareError as err:
        print(f"EXP45 REGRESSION COMPARE: unreadable input: {err}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)
    print(f"REGRESSION POLICY {report['verdict']}: {len(problems)} problem(s), {len(report['allowed_failures'])} "
          f"allowed failure(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
