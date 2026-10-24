"""exp_25 depth_canon rB-F1 B1 (Codex rB BLOCKING + rA B1): ONE tie-audit predicate,
``worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_audit_gate.py``, shared by the
collector (and, next, the launcher). Fail-closed: the report must be exactly what the audit
script writes for a complete PASS over the three registered splits, its census must be the
one the audit's OWN enumeration derives from the split JSONs, and the map bytes on disk must
still hash to the rows.

Fixtures are REAL audits: ``depth_canon_tie_audit.py``'s own pipeline (``read_split_config`` ->
``collect_maps`` -> ``run_rows`` -> ``build_report`` -> ``write_report``) over tiny synthetic
256x512 maps (``test_depth_canon_tie_audit._write_tree``); mutations are re-written with the
audit's ``write_report`` (so the sidecar matches and the CONTENT check must catch them).
CPU only, no dataset.

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort), round rB-F1.
"""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import test_depth_canon_tie_audit as TA
from test_depth_canon import _scalar_depth

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")
GATE_PY = os.path.join(REPO, EXPREL, "depth_canon_audit_gate.py")
REAL_AUDIT = os.path.join(REPO, EXPREL, "depth_canon_tie_audit.json")
NAMES = ("train", "unseen", "seen")
W = 512


def _load_gate():
    spec = importlib.util.spec_from_file_location("depth_canon_audit_gate", GATE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load_gate()


def _n_items(splits, name):
    return sum(len(files) for rooms in splits[name].values() for files in rooms.values())


def _n_maps(splits):
    keys = set()
    for rooms in splits.values():
        for scene, ids in rooms.items():
            for sid, files in ids.items():
                keys |= {(scene, sid, int(f.split("_")[1][1:])) for f in files}
    return len(keys)


def make_audit(tmp, splits=TA.SPLITS, maps=None, names=NAMES, limit=None, out_name="depth_canon_tie_audit.json"):
    """A REAL tie audit of a tiny synthetic tree (the audit script's own functions, in
    main()'s order) + the contract a gate needs for it. Returns ``(path, report, contract)``."""
    cfgs = TA._write_tree(tmp, splits=splits, maps=maps)
    sp = {n: TA.audit.read_split_config(cfgs[n]) for n in names}
    found, ledger = TA.audit.collect_maps(sp)
    module = sp[names[0]]["custom_metadata_module_abs"]
    rows = TA.audit.run_rows(found, module, sp[names[0]]["root_abs"], 1, limit)
    rep = TA.audit.build_report(rows, ledger, list(names), len(found), limit, module)
    out = os.path.join(str(tmp), out_name)
    TA.audit.write_report(rep, out)
    contract = copy.deepcopy(G.REGISTERED)
    contract["splits"] = {n: {"dataset_config": cfgs[n], "n_items": _n_items(splits, n)} for n in NAMES}
    contract["n_maps"] = _n_maps(splits)          # counted here, independently of the audit
    contract["custom_metadata_module"] = TA._ar_md().__file__   # the tree's configs name it absolutely
    return out, json.loads(json.dumps(rep)), contract


def rewrite(out, rep):
    """Re-publish a (mutated) report with the audit's own writer: sidecar recomputed."""
    TA.audit.write_report(rep, out)


def validate(out, contract, **kw):
    return G.validate_tie_audit(out, REPO, contract=contract, **kw)


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gate")
    return make_audit(tmp)


def test_a_genuine_complete_pass_audit_is_accepted(genuine):
    out, rep, contract = genuine
    ok, problems, summary = validate(out, contract)
    assert ok and problems == [], problems
    assert summary["audit_sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert summary["n_maps"] == contract["n_maps"] == 6
    assert summary["per_split"] == rep["per_split"]
    assert summary["split_maps"]["seen"] == ["depth_map/Cafe/Cafe_idx_0/1.npy", "depth_map/Cafe/Cafe_idx_0/3.npy"]
    assert [i["split"] for i in summary["inputs"]] == list(NAMES)
    assert summary["inputs"][1]["json_abs"] == rep["inputs"][1]["json_abs"]


# --------------------------------------------------------------------------- #
# content mutations, re-published with a MATCHING sidecar: the content check must refuse
# --------------------------------------------------------------------------- #
def _survivor(r):
    r["survivors"].append({"path": r["rows"][0]["path"], "candidates": [1, 2], "survivors": [1, 2],
                           "counts": r["rows"][0]["counts"]})


MUTATIONS = {
    "input_cardinality": (lambda r: r["inputs"][1].__setitem__("n_items", r["inputs"][1]["n_items"] - 1), "inputs"),
    "input_json_hash": (lambda r: r["inputs"][0].__setitem__("json_sha256", "0" * 64), "inputs"),
    "input_config_hash": (lambda r: r["inputs"][2].__setitem__("dataset_config_sha256", "0" * 64), "inputs"),
    "tool_code_hash": (lambda r: r["tool"]["code_sha256"].__setitem__("src/data/yaw_rotation.py", "0" * 64),
                       "code_sha256"),
    "tool_script_hash": (lambda r: r["tool"].__setitem__("script_sha256", "1" * 64), "script"),
    "tool_dirty": (lambda r: r["tool"].__setitem__("dirty_code", " M src/data/yaw_rotation.py"), "dirty"),
    "tool_dirty_unknown": (lambda r: r["tool"].__setitem__("dirty_code", None), "dirty"),
    "tool_source_sha": (lambda r: r["tool"].__setitem__("source_sha", "HEAD"), "source_sha"),
    "add_survivor_listing": (_survivor, "survivors"),
    "coverage_incomplete": (lambda r: r["coverage"].__setitem__("complete", False), "coverage"),
    "coverage_limit": (lambda r: r["coverage"].__setitem__("limit", 6), "coverage"),
    "schema_true": (lambda r: r.__setitem__("schema_version", True), "schema_version"),
    "schema_float": (lambda r: r.__setitem__("schema_version", 1.0), "schema_version"),
    "gate_exit_bool": (lambda r: r["gate"].__setitem__("exit_code", False), "gate"),
    "gate_verdict": (lambda r: r["gate"].__setitem__("verdict", "FAIL"), "gate"),
    "harness_tag": (lambda r: r.__setitem__("harness", True), "harness"),

}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_a_content_mutation_with_a_recomputed_sidecar_is_refused(genuine, tmp_path, name):
    out, rep, contract = genuine
    edit, needle = MUTATIONS[name]
    bad = copy.deepcopy(rep)
    edit(bad)
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, bad)
    ok, problems, _ = validate(path, contract)
    assert not ok and any(needle in p for p in problems), (name, problems)


def test_a_tampered_sidecar_is_refused(genuine, tmp_path):
    out, rep, contract = genuine
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, rep)
    assert validate(path, contract)[0]
    open(path + ".sha256", "w").write("0" * 64 + "  depth_canon_tie_audit.json\n")
    ok, problems, _ = validate(path, contract)
    assert not ok and any("sidecar" in p for p in problems), problems
    os.unlink(path + ".sha256")
    assert not validate(path, contract)[0]


def test_an_unreadable_or_non_json_report_is_refused(tmp_path, genuine):
    _, _, contract = genuine
    path = str(tmp_path / "depth_canon_tie_audit.json")
    assert not validate(path, contract)[0]                                 # absent
    open(path, "w").write("{not json")
    open(path + ".sha256", "w").write(hashlib.sha256(b"{not json").hexdigest() + "  depth_canon_tie_audit.json\n")
    ok, problems, _ = validate(path, contract)
    assert not ok and problems


# --------------------------------------------------------------------------- #
# runs that are not a complete PASS, and data that changed after the audit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw, needle", [(dict(limit=4), "coverage"), (dict(names=("unseen", "seen")), "coverage")],
                         ids=["limit", "subset"])
def test_a_limit_or_subset_run_is_refused(tmp_path, kw, needle):
    out, rep, contract = make_audit(tmp_path, **kw)
    assert rep["gate"]["verdict"] == "REFUSED_SUBSET"
    ok, problems, _ = validate(out, contract)
    assert not ok and any(needle in p for p in problems), problems


def test_an_audit_that_found_a_survivor_is_refused(tmp_path):
    periodic = _scalar_depth(256, W, seed=5, period=128, peaks=[(60, 11)]).numpy()
    out, rep, contract = make_audit(tmp_path, maps={TA.PERIODIC_KEY: periodic})
    assert rep["gate"]["verdict"] == "FAIL"
    ok, problems, _ = validate(out, contract)
    assert not ok and any("gate" in p for p in problems) and any("survivor" in p for p in problems), problems


def test_a_split_json_changed_after_the_audit_is_refused(tmp_path):
    out, rep, contract = make_audit(tmp_path)
    split = rep["inputs"][1]["json_abs"]
    data = json.load(open(split))
    data["Office"]["Office_idx_2"].append("S0099_R006_hybrid_IR.wav")    # one more item, same map
    open(split, "w").write(json.dumps(data))
    ok, problems, _ = validate(out, contract)
    assert not ok and any("inputs" in p for p in problems), problems


def test_the_registered_cardinalities_bind_the_census(genuine):
    out, rep, contract = genuine
    for edit in (lambda c: c["splits"]["unseen"].__setitem__("n_items", 3), lambda c: c.__setitem__("n_maps", 7)):
        c = copy.deepcopy(contract)
        edit(c)
        ok, problems, _ = validate(out, c)
        assert not ok and any("registered" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# the registered contract itself, against the real split JSONs (the audit's own code)
# --------------------------------------------------------------------------- #
def test_the_registered_contract_is_the_real_census():
    R = G.REGISTERED
    assert R["audit_script"] == os.path.join(EXPREL, "depth_canon_tie_audit.py").replace(os.sep, "/")
    assert R["n_maps"] == 9650 and R["img_w"] == 512
    assert {n: s["n_items"] for n, s in R["splits"].items()} == {"train": 291210, "unseen": 6337, "seen": 6217}
    sp = {n: TA.audit.read_split_config(R["splits"][n]["dataset_config"]) for n in NAMES}
    found, ledger = TA.audit.collect_maps(sp)
    assert len(found) == 9650 and [r["n_items"] for r in ledger] == [291210, 6337, 6217]
    assert sum(1 for e in found.values() if e["counts"]["unseen"] and e["counts"]["seen"]) == 112


@pytest.mark.skipif(not os.path.exists(REAL_AUDIT), reason="the full rung-3 audit is not in this checkout")
def test_the_real_full_audit_passes_the_registered_contract_without_rehash():
    """The genuine report (sidecar, schema, census, hashes). Its 9,650 map re-hashes run in
    the launcher / collector (minutes over CIFS), not in the unit suite."""
    ok, problems, summary = G.validate_tie_audit(REAL_AUDIT, REPO, rehash=False)
    assert ok, problems
    assert summary["n_maps"] == 9650 and len(summary["split_maps"]["unseen"]) == 644


# --------------------------------------------------------------------------- #
# the CLI: 0 ok / 1 invalid with the reasons; --print-sha for the launcher binding
# --------------------------------------------------------------------------- #
def test_the_cli(genuine, monkeypatch, capsys, tmp_path):
    out, rep, contract = genuine
    monkeypatch.setattr(G, "REGISTERED", contract)
    assert G.main(["--audit", out]) == 0
    assert "TIE AUDIT GATE OK" in capsys.readouterr().out
    assert G.main(["--audit", out, "--print-sha"]) == 0
    assert capsys.readouterr().out == hashlib.sha256(open(out, "rb").read()).hexdigest() + "\n"
    bad = copy.deepcopy(rep)
    bad["gate"]["verdict"] = "FAIL"
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, bad)
    assert G.main(["--audit", path, "--print-sha"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("TIE AUDIT GATE FAILED: ") and "gate.verdict" in cap.err


def test_the_gate_runs_as_a_program(tmp_path):
    proc = subprocess.run([sys.executable, GATE_PY, "--audit", str(tmp_path / "absent.json")],
                          capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 1 and "TIE AUDIT GATE FAILED" in proc.stderr and proc.stdout == ""


def test_the_gate_module_imports_only_the_standard_library():
    """Importing it must stay cheap (the generator loads the collector, which loads the gate);
    the audit's own enumeration code is loaded lazily, inside validate_tie_audit."""
    import ast
    tree = ast.parse(open(GATE_PY).read())
    top = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert top <= set(sys.stdlib_module_names), top - set(sys.stdlib_module_names)


# --------------------------------------------------------------------------- #
# the rows: exactly the expected census, typed, summaries recomputed, bytes re-hashed
# --------------------------------------------------------------------------- #
ROW_MUTATIONS = {   # the census, the per-row contract and the recomputed summaries
    "drop_train_per_split": (lambda r: r["per_split"].pop("train"), "per_split"),
    "per_split_count": (lambda r: r["per_split"]["unseen"]["items"].__setitem__(
        "none", r["per_split"]["unseen"]["items"]["none"] + 1), "per_split"),
    "row_survivor": (lambda r: r["rows"][0].__setitem__("survivor", True), "row"),
    "row_error": (lambda r: r["rows"][1].__setitem__("error", "x"), "row"),
    "duplicate_row": (lambda r: r["rows"].insert(1, copy.deepcopy(r["rows"][0])), "rows"),
    "arbitrary_row_path": (lambda r: r["rows"][0].__setitem__("path", "m0"), "rows"),
    "dropped_row": (lambda r: r["rows"].pop(), "rows"),
    "row_counts": (lambda r: r["rows"][0]["counts"].__setitem__("train", r["rows"][0]["counts"]["train"] + 1),
                   "counts"),
    "row_tie_class": (lambda r: r["rows"][0].__setitem__("tie_class", "stabilized"), "tie_class"),
    "row_img_w": (lambda r: r["rows"][0].__setitem__("img_w", 256), "img_w"),
    "row_dj": (lambda r: r["rows"][0].__setitem__("dj", 600), "dj"),
    "row_sha": (lambda r: r["rows"][0].__setitem__("sha256", "x"), "sha256"),
    "row_extra_key": (lambda r: r["rows"][0].__setitem__("note", 1), "row"),
    "agreement": (lambda r: r["agreement"]["raw_agree_set"].__setitem__("true", 99), "agreement"),
}


@pytest.mark.parametrize("name", list(ROW_MUTATIONS))
def test_a_row_level_mutation_with_a_recomputed_sidecar_is_refused(genuine, tmp_path, name):
    out, rep, contract = genuine
    edit, needle = ROW_MUTATIONS[name]
    bad = copy.deepcopy(rep)
    edit(bad)
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, bad)
    ok, problems, _ = validate(path, contract)
    assert not ok and any(needle in p for p in problems), (name, problems)


def _map_file(rep, key):
    root = rep["inputs"][0]["root_abs"]
    return os.path.join(root, key)


def test_a_depth_map_changed_after_the_audit_is_refused(tmp_path):
    out, rep, contract = make_audit(tmp_path)
    assert validate(out, contract)[0]
    victim = _map_file(rep, rep["rows"][2]["path"])
    arr = np.load(victim)
    arr[0, 0] += 1e-9
    np.save(victim, arr)
    ok, problems, _ = validate(out, contract)
    assert not ok and any(rep["rows"][2]["path"] in p and "sha256" in p for p in problems), problems
    assert validate(out, contract, rehash=False)[0]                       # only the re-hash sees it


def test_a_missing_depth_map_is_refused(tmp_path):
    out, rep, contract = make_audit(tmp_path)
    os.unlink(_map_file(rep, rep["rows"][0]["path"]))
    ok, problems, _ = validate(out, contract)
    assert not ok and any("missing" in p or "unreadable" in p for p in problems), problems



# --------------------------------------------------------------------------- #
# rA A1 (iii)/(iv): checkout-portable ledger; harness mode = a harness contract file, accepted
# ONLY under DEPTH_CANON_HARNESS and ONLY for a harness-tagged report
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(REAL_AUDIT), reason="the full rung-3 audit is not in this checkout")
def test_a_report_audited_in_another_checkout_is_accepted(tmp_path):
    """The *_abs ledger fields are machine paths: they must END with their relative path, while
    the content fields (relative paths, hashes, counts) must be equal -- so the committed audit
    validates in any checkout of the pin (e.g. the guard suite's clone)."""
    rep = json.load(open(REAL_AUDIT))
    for led in rep["inputs"]:
        for key in ("json_abs", "root_abs", "custom_metadata_module_abs"):
            led[key] = led[key].replace(REPO, "/tmp/another_checkout", 1)
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, rep)
    ok, problems, _ = G.validate_tie_audit(path, REPO, rehash=False)
    assert ok, problems
    rep["inputs"][1]["json_abs"] = "/tmp/another_checkout/data/AR/not_the_split.json"
    rewrite(path, rep)
    ok, problems, _ = G.validate_tie_audit(path, REPO, rehash=False)
    assert not ok and any("json_abs" in p for p in problems), problems


def _harness_report(rep):
    tagged = copy.deepcopy(rep)
    tagged["harness"] = True
    return tagged


def _harness_contract(tmp, contract, **over):
    c = {"splits": contract["splits"], "n_maps": contract["n_maps"],
         "custom_metadata_module": contract["custom_metadata_module"], **over}
    path = tmp / "audit_contract.json"
    path.write_text(json.dumps(c))
    return str(path)


def test_harness_mode_accepts_only_a_tagged_report_against_a_harness_contract(genuine, tmp_path):
    out, rep, contract = genuine
    hc = _harness_contract(tmp_path, contract)
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, _harness_report(rep))
    ok, problems, summary = G.validate_tie_audit(path, REPO, harness_contract=hc)
    assert ok and summary["n_maps"] == 6, problems
    prod = G.validate_tie_audit(path, REPO, contract=contract)                       # production: never
    assert not prod[0] and any("harness" in p for p in prod[1])
    rewrite(path, rep)                                                                # untagged in harness mode
    ok, problems, _ = G.validate_tie_audit(path, REPO, harness_contract=hc)
    assert not ok and any("harness" in p for p in problems)


@pytest.mark.parametrize("over", [{"audit_script": "x.py"}, {"img_w": 256}, {"code": []}],
                         ids=["script", "img_w", "code"])
def test_a_harness_contract_may_only_name_the_splits(genuine, tmp_path, over):
    out, rep, contract = genuine
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, _harness_report(rep))
    ok, problems, _ = G.validate_tie_audit(path, REPO, harness_contract=_harness_contract(tmp_path, contract, **over))
    assert not ok and any("harness contract" in p for p in problems), problems


def test_the_cli_honours_a_harness_contract_only_inside_a_harness(genuine, tmp_path, monkeypatch, capsys):
    out, rep, contract = genuine
    h = tmp_path / "h"
    h.mkdir()
    hc = _harness_contract(h, contract)
    path = str(tmp_path / "depth_canon_tie_audit.json")
    rewrite(path, _harness_report(rep))
    monkeypatch.delenv("DEPTH_CANON_HARNESS", raising=False)
    assert G.main(["--audit", path, "--harness-contract", hc]) == 2                 # no harness
    monkeypatch.setenv("DEPTH_CANON_HARNESS", str(h))
    assert G.main(["--audit", path, "--harness-contract", hc]) == 2                 # no marker
    (h / ".depth_canon_harness").write_text("")
    outside = _harness_contract(tmp_path, contract)
    assert G.main(["--audit", path, "--harness-contract", outside]) == 2            # file outside the harness
    capsys.readouterr()
    assert G.main(["--audit", path, "--harness-contract", hc, "--print-sha"]) == 0
    assert capsys.readouterr().out.strip() == hashlib.sha256(open(path, "rb").read()).hexdigest()
