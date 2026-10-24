#!/usr/bin/env python3
"""exp_25 depth_canon -- THE tie-audit predicate (Codex rB BLOCKING B1, rA BLOCKING B1).

One fail-closed contract for the rung-3 audit report ``depth_canon_tie_audit.json``, used by
the collector (and the launcher): the exact sidecar; the exact report the audit script writes
for a complete PASS over the three registered splits (type-strict: ``1`` is not ``1.0`` is
not ``true``; no extra key -- a ``harness`` tag is never production); the inputs ledger, the
per-map census, ``per_split`` and ``agreement`` recomputed with the AUDIT'S OWN code
(``read_split_config`` / ``collect_maps`` / ``summarize``, loaded lazily from the audit
script whose sha256 the report records); the registered cardinalities; current hashes of the
audit script, operator, loader and AR_md with a clean recorded tree; and every depth map's
CURRENT bytes re-hashed against its row (a map changed after the audit fails closed).

API ``validate_tie_audit(audit_path, repo_root) -> (ok, problems, summary)``; CLI exit 0 ok / 1
invalid (stderr ``TIE AUDIT GATE FAILED: ...``) / 2 usage. Stdlib imports only. Written by the
exp_25 Coder seat (Claude Opus 5.5, max effort), round rB-F1.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sys

EXPREL = "worklog/worklog_yixun/exp_25_depth_canon_claude"
SPLIT_NAMES = ("train", "unseen", "seen")
_CFG = "src/configs/dataset_configs/AR"
REGISTERED = {   # plan v10 §4 (10), §7 rung 3: mirrors the launcher's predicate field by field
    "audit_script": f"{EXPREL}/depth_canon_tie_audit.py",
    "splits": {"train": {"dataset_config": f"{_CFG}/train/acousticroom_train.json", "n_items": 291210},
               "unseen": {"dataset_config": f"{_CFG}/eval/acousticroom_unseeneval.json", "n_items": 6337},
               "seen": {"dataset_config": f"{_CFG}/eval/acousticroom_seeneval.json", "n_items": 6217}},
    "n_maps": 9650,
    "custom_metadata_module": "src/configs/dataset_configs/custom_metadata/AR_md.py",
    "code": ("src/data/yaw_rotation.py", "src/data/dataset.py"),
    "img_w": 512,
}
REPORT_KEYS = ("schema_version", "tool", "inputs", "coverage", "gate", "per_split", "agreement",
               "survivors", "errors", "rows")
TOOL_KEYS = ("script_sha256", "code_sha256", "source_sha", "dirty_code")
ROW_KEYS = ("path", "sha256", "counts", "img_w", "j_star", "dj", "tie_class", "survivor", "n_candidates",
            "candidates", "survivors", "max_sep_deg", "raw_tie_class", "raw_survivor", "raw_agree_set",
            "raw_agree_col", "error")
TIE_CLASSES = ("none", "same_column", "cross_column_image")
PASS_GATE = {"verdict": "PASS", "exit_code": 0, "n_survivors": 0, "n_errors": 0}
_HEX64, _HEX40 = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


def strict_diff(got, want, label):
    """The first TYPE-strict difference as text, or None (dict key sets, list order, bool vs int)."""
    if type(got) is not type(want):
        return f"{label}: {got!r} is not {want!r} ({type(got).__name__} vs {type(want).__name__})"
    if isinstance(want, dict):
        if set(got) != set(want):
            return f"{label}: keys {sorted(got)} != {sorted(want)}"
        return next((d for k in want for d in [strict_diff(got[k], want[k], f"{label}.{k}")] if d), None)
    if isinstance(want, list):
        if len(got) != len(want):
            return f"{label}: {len(got)} entries != {len(want)}"
        return next((d for i, (a, b) in enumerate(zip(got, want)) for d in [strict_diff(a, b, f"{label}[{i}]")]
                     if d), None)
    return None if got == want else f"{label}: {got!r} != {want!r}"


def _expect(problems, label, got, want):
    d = strict_diff(got, want, label)
    if d:
        problems.append(d)


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_audit_module(path):
    spec = importlib.util.spec_from_file_location("depth_canon_tie_audit_gate_view", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tool_problems(tool, repo_root, c):
    at = lambda p: os.path.join(repo_root, p)  # noqa: E731 -- an absolute p stays absolute
    if type(tool) is not dict or list(tool) != list(TOOL_KEYS):
        return [f"tool {tool!r:.200} is not {{{', '.join(TOOL_KEYS)}}}"]
    out, script = [], _sha256(at(c["audit_script"]))
    if script is None or tool["script_sha256"] != script:
        out.append(f"tool.script_sha256: written by another audit script than {c['audit_script']}")
    want = {"script": script, **{p: _sha256(at(p)) for p in c["code"]},
            "custom_metadata_module": _sha256(at(c["custom_metadata_module"]))}
    _expect(out, "tool.code_sha256 (current files)", tool["code_sha256"], want)
    if not (isinstance(tool["source_sha"], str) and _HEX40.fullmatch(tool["source_sha"])):
        out.append(f"tool.source_sha {tool['source_sha']!r} is not a 40-hex commit")
    if tool["dirty_code"] != "":
        out.append(f"tool.dirty_code {tool['dirty_code']!r}: the audit's code was not clean (or unknown)")
    return out


def _validate(audit_path, repo_root, c, workers, rehash):
    problems, summary = [], {}
    try:
        with open(audit_path, "rb") as f:
            raw = f.read()
        with open(audit_path + ".sha256") as f:
            side = f.read()
    except OSError as err:
        return [f"{audit_path} (+ .sha256) unreadable ({err})"], summary
    summary["audit_sha256"] = sha = hashlib.sha256(raw).hexdigest()
    if side != f"{sha}  {os.path.basename(audit_path)}\n":
        problems.append("the sha256 sidecar does not match the audit's bytes")
    try:
        rep = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except ValueError as err:
        return problems + [f"the report is not strict JSON ({err})"], summary
    if type(rep) is not dict or list(rep) != list(REPORT_KEYS):
        keys = list(rep) if type(rep) is dict else type(rep).__name__
        return problems + [f"report keys {keys} != {list(REPORT_KEYS)} (a harness tag is never production)"], summary
    _expect(problems, "schema_version", rep["schema_version"], 1)
    _expect(problems, "gate", rep["gate"], PASS_GATE)
    _expect(problems, "survivors", rep["survivors"], [])
    _expect(problems, "errors", rep["errors"], [])
    _expect(problems, "coverage", rep["coverage"], {"splits": list(SPLIT_NAMES), "limit": None, "subset": False,
                                                   "n_maps_union": c["n_maps"], "n_maps_audited": c["n_maps"],
                                                   "complete": True})
    problems += _tool_problems(rep["tool"], repo_root, c)
    audit = _load_audit_module(os.path.join(repo_root, c["audit_script"]))
    with contextlib.redirect_stdout(io.StringIO()):    # the loader's progress prints are not the verdict
        splits = {n: audit.read_split_config(c["splits"][n]["dataset_config"]) for n in SPLIT_NAMES}
        maps, ledger = audit.collect_maps(splits)
    for n, led in zip(SPLIT_NAMES, ledger):
        if led["n_items"] != c["splits"][n]["n_items"]:
            problems.append(f"{n}: the split JSON holds {led['n_items']} items, not the registered "
                            f"{c['splits'][n]['n_items']}")
        if led["custom_metadata_module"] != c["custom_metadata_module"]:
            problems.append(f"{n}: metadata module {led['custom_metadata_module']} is not the registered one")
    if len(maps) != c["n_maps"]:
        problems.append(f"the splits resolve to {len(maps)} maps, not the registered {c['n_maps']}")
    _expect(problems, "inputs (vs the ledger recomputed from the current configs + split JSONs)",
            rep["inputs"], json.loads(json.dumps(ledger)))
    summary.update(n_maps=len(rep["rows"]) if isinstance(rep["rows"], list) else None,
                   per_split=rep["per_split"],
                   inputs=[{k: led[k] for k in ("split", "json_file_path", "json_abs", "n_items", "n_unique_maps")}
                           for led in ledger],
                   split_maps={n: sorted(k for k, e in maps.items() if e["counts"][n]) for n in SPLIT_NAMES})
    return problems, summary


def validate_tie_audit(audit_path, repo_root, *, contract=None, workers=8, rehash=True):
    """``(ok, problems, summary)``. ``contract`` defaults to :data:`REGISTERED` (tests pass a
    synthetic one); any exception is a refusal, never a pass."""
    try:
        problems, summary = _validate(audit_path, os.path.abspath(repo_root),
                                      REGISTERED if contract is None else contract, workers, rehash)
    except Exception as err:  # noqa: BLE001 -- a crash is never a verdict
        problems, summary = [f"validation crashed ({type(err).__name__}: {err})"], {}
    return not problems, problems, summary


def main(argv=None):
    import argparse
    repo = _repo_root(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", default=os.path.join(repo, EXPREL, "depth_canon_tie_audit.json"))
    ap.add_argument("--repo", default=repo)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--print-sha", action="store_true", help="on success print only the report's sha256")
    args = ap.parse_args(argv)
    ok, problems, summary = validate_tie_audit(args.audit, args.repo, workers=args.workers)
    if not ok:
        print("TIE AUDIT GATE FAILED: " + "; ".join(problems) + " - abort", file=sys.stderr)
        return 1
    print(summary["audit_sha256"] if args.print_sha else
          f"TIE AUDIT GATE OK: {args.audit} sha256 {summary['audit_sha256']}: {summary['n_maps']} maps "
          "(the registered census), 0 survivors, 0 errors, hashes current, map bytes re-hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
