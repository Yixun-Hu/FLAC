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
CURRENT bytes re-hashed against its row (a map changed after the audit fails closed) -- in a
KILLABLE child process (this file, ``--rehash-worker``) under a :data:`REHASH_DEADLINE_S` wall-clock
deadline: a hung read fails the gate closed instead of stalling it (rB-F2 F2-4).

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
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
ABS_FIELDS = {"json_abs": "json_file_path", "root_abs": "root", "custom_metadata_module_abs": "custom_metadata_module"}
HARNESS_KEYS = ("splits", "n_maps", "custom_metadata_module")   # all a harness contract may name (rA A1 iv)
_HEX64, _HEX40 = re.compile(r"[0-9a-f]{64}"), re.compile(r"[0-9a-f]{40}")
REHASH_DEADLINE_S = 900   # rB-F2 F2-4 (reviewed): 15 min; the real 9,650-map re-hash over CIFS takes ~2 min 15 s


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


def _ledger_problems(inputs, ledger):
    """The inputs ledger == the one recomputed now, except the *_abs machine paths, which must
    END with their relative paths (the committed audit validates in any checkout of the pin)."""
    view = lambda led: [{k: v for k, v in r.items() if k not in ABS_FIELDS} if isinstance(r, dict) else r  # noqa: E731
                        for r in led] if isinstance(led, list) else led
    out = []
    _expect(out, "inputs (vs the ledger recomputed from the current configs + split JSONs)", view(inputs),
            view(json.loads(json.dumps(ledger))))
    for i, led in enumerate(inputs if isinstance(inputs, list) else []):
        for key, rel in ABS_FIELDS.items():
            v, r = (led.get(key), led.get(rel)) if isinstance(led, dict) else (None, None)
            if not (isinstance(v, str) and isinstance(r, str) and (
                    v == r if os.path.isabs(r) else os.path.normpath(v).endswith(os.sep + os.path.normpath(r)))):
                out.append(f"inputs[{i}].{key} {v!r} is not a path ending with its {rel} {r!r}")
    return out


def _row_problems(rows, maps, audit, w):
    """Per-row type/shape contract, then the census: exactly the expected map set, in the
    audit's sorted order, with the expected per-split counts."""
    if not isinstance(rows, list):
        return [f"rows is {type(rows).__name__}, not a list"]
    bad = []
    for i, r in enumerate(rows):
        why = None
        if type(r) is not dict or list(r) != list(ROW_KEYS):
            why = "row keys are not the audit's"
        elif not (isinstance(r["path"], str) and isinstance(r["sha256"], str) and _HEX64.fullmatch(r["sha256"])):
            why = f"path {r['path']!r} / sha256 {r['sha256']!r} malformed"
        elif not (type(r["counts"]) is dict and list(r["counts"]) == list(SPLIT_NAMES)
                  and all(type(v) is int and v >= 0 for v in r["counts"].values())):
            why = f"counts {r['counts']!r} malformed"
        elif type(r["img_w"]) is not int or r["img_w"] != w:
            why = f"img_w {r['img_w']!r} != {w}"
        elif type(r["j_star"]) is not int or not 0 <= r["j_star"] < w or type(r["dj"]) is not int \
                or r["dj"] != (w - r["j_star"]) % w:
            why = f"j_star {r['j_star']!r} / dj {r['dj']!r} inconsistent"
        elif r["tie_class"] not in TIE_CLASSES or r["survivor"] is not False or r["error"] is not None:
            why = f"tie_class {r['tie_class']!r} / survivor {r['survivor']!r} / error {r['error']!r}"
        elif not (isinstance(r["candidates"], list) and r["candidates"]
                  and all(type(x) is int and 0 <= x < w for x in r["candidates"])
                  and r["candidates"] == sorted(set(r["candidates"])) and r["n_candidates"] == len(r["candidates"])
                  and type(r["n_candidates"]) is int and r["j_star"] in r["candidates"]
                  and r["survivors"] == [r["j_star"]]
                  and (len(r["candidates"]) == 1) == (r["tie_class"] != "cross_column_image")):
            why = "candidates / survivors / class inconsistent"
        elif type(r["max_sep_deg"]) is not float or r["max_sep_deg"] != audit.max_circular_sep_deg(r["candidates"], w):
            why = f"max_sep_deg {r['max_sep_deg']!r} != the candidates' separation"
        elif r["raw_tie_class"] not in TIE_CLASSES + (None,) or (r["raw_tie_class"] is None) != r["raw_survivor"] \
                or not all(type(r[k]) is bool for k in ("raw_survivor", "raw_agree_set", "raw_agree_col")):
            why = "raw_* fields malformed"
        if why:
            bad.append(f"row {i}: {why}")
    out = [f"{len(bad)} malformed row(s): " + "; ".join(bad[:5])] if bad else []
    paths = [r.get("path") if isinstance(r, dict) else None for r in rows]
    if paths != sorted(maps):
        extra, gone = sorted(set(map(str, paths)) - set(maps)), sorted(set(maps) - set(map(str, paths)))
        out.append(f"rows are not the {len(maps)} expected maps in order ({len(paths)} rows, "
                   f"{len(paths) - len(set(map(str, paths)))} duplicate(s), unexpected {extra[:3]}, missing {gone[:3]})")
    elif not bad:
        wrong = [r["path"] for r in rows if r["counts"] != maps[r["path"]]["counts"]]
        if wrong:
            out.append(f"{len(wrong)} row(s) carry counts that are not the split census: {wrong[:3]}")
    return out


def _hash_paths(paths, workers):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:   # hashlib + file reads release the GIL
        return list(pool.map(_sha256, paths))


def _rehash_argv(workers):
    """The hasher child: this gate in ``--rehash-worker`` mode (stdin: JSON paths; stdout: JSON digests)."""
    return [sys.executable, os.path.abspath(__file__), "--rehash-worker", "--workers", str(max(1, workers))]


def _rehash_in_child(paths, workers, deadline_s):
    """``(digests, None)`` or ``(None, why)``. Past the deadline the child's own process group is
    SIGKILLed and the gate fails closed; a child that still does not exit (an uninterruptible read)
    is abandoned after 10 s, never waited on."""
    try:
        proc = subprocess.Popen(_rehash_argv(workers), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError as err:
        return None, f"the map re-hash child could not start ({err})"
    try:
        out, err = proc.communicate(json.dumps(paths), timeout=deadline_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return None, (f"the map re-hash exceeded its {deadline_s:g} s wall-clock deadline (a hung NAS read?); "
                      "the gate fails closed")
    if proc.returncode != 0:
        return None, f"the map re-hash child exited {proc.returncode} ({err.strip()[-300:]!r})"
    try:
        digests = json.loads(out)
    except ValueError:
        return None, "the map re-hash child printed no JSON"
    if not (isinstance(digests, list) and len(digests) == len(paths)
            and all(d is None or (isinstance(d, str) and _HEX64.fullmatch(d)) for d in digests)):
        return None, "the map re-hash child's answer is malformed"
    return digests, None


def _rehash_problems(rows, maps, workers, deadline_s=REHASH_DEADLINE_S):
    now, why = _rehash_in_child([maps[r["path"]]["path"] for r in rows], workers, deadline_s)
    if why:
        return [why]
    missing = [r["path"] for r, h in zip(rows, now) if h is None]
    changed = [r["path"] for r, h in zip(rows, now) if h is not None and h != r["sha256"]]
    return ([f"{len(missing)} depth map(s) missing / unreadable now: {missing[:3]}"] if missing else []) + (
        [f"{len(changed)} depth map(s) changed since the audit (current sha256 != the row): {changed[:3]}"]
        if changed else [])


def _validate(audit_path, repo_root, c, workers, rehash, harness=False, deadline_s=REHASH_DEADLINE_S):
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
    if harness:   # rA A1 (iv): a harness run validates ONLY a harness-tagged synthetic report
        if type(rep) is not dict or set(rep) != set(REPORT_KEYS) | {"harness"} or rep["harness"] is not True:
            return problems + ["a harness run validates only a harness-tagged report (harness: true)"], summary
    elif type(rep) is not dict or list(rep) != list(REPORT_KEYS):
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
    problems += _ledger_problems(rep["inputs"], ledger)
    row_problems = _row_problems(rep["rows"], maps, audit, c["img_w"])
    problems += row_problems
    if not row_problems:
        per, agree = audit.summarize(rep["rows"], list(SPLIT_NAMES), c["img_w"])
        _expect(problems, "per_split (recomputed from the rows)", rep["per_split"], per)
        _expect(problems, "agreement (recomputed from the rows)", rep["agreement"], agree)
        if rehash:
            problems += _rehash_problems(rep["rows"], maps, workers, deadline_s)
    summary.update(n_maps=len(rep["rows"]) if isinstance(rep["rows"], list) else None,
                   per_split=rep["per_split"],
                   inputs=[{k: led[k] for k in ("split", "json_file_path", "json_abs", "n_items", "n_unique_maps")}
                           for led in ledger],
                   split_maps={n: sorted(k for k, e in maps.items() if e["counts"][n]) for n in SPLIT_NAMES})
    return problems, summary


def validate_tie_audit(audit_path, repo_root, *, contract=None, workers=8, rehash=True, harness_contract=None,
                       rehash_deadline_s=REHASH_DEADLINE_S):
    """``(ok, problems, summary)``. ``contract`` defaults to :data:`REGISTERED` (tests pass a
    synthetic one). ``harness_contract``: a JSON file naming ONLY the splits / their counts / the
    map count / the metadata module of a guard-suite tree; the report must then be harness-
    tagged (production never accepts one). Any exception is a refusal, never a pass."""
    try:
        c, harness = (REGISTERED if contract is None else contract), harness_contract is not None
        if harness:
            with open(harness_contract) as f:
                over = json.load(f)
            if not isinstance(over, dict) or not set(over) <= set(HARNESS_KEYS):
                return False, [f"the harness contract may name only {list(HARNESS_KEYS)}, got "
                               f"{sorted(over) if isinstance(over, dict) else over!r}"], {}
            c = dict(c, **over)
        problems, summary = _validate(audit_path, os.path.abspath(repo_root), c, workers, rehash, harness,
                                      rehash_deadline_s)
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
    ap.add_argument("--rehash-worker", action="store_true",
                    help="internal: the hasher child (stdin: a JSON list of paths; stdout: their sha256 or null)")
    ap.add_argument("--harness-contract", default=None,
                    help="guard suite only: honoured only under DEPTH_CANON_HARNESS (a /tmp dir with its marker) "
                         "for a file inside it")
    args = ap.parse_args(argv)
    if args.rehash_worker:
        print(json.dumps(_hash_paths(json.load(sys.stdin), args.workers)))
        return 0
    if args.harness_contract is not None:
        h = os.path.realpath(os.environ.get("DEPTH_CANON_HARNESS") or "/nonexistent")
        if not (h.startswith("/tmp/") and os.path.isfile(os.path.join(h, ".depth_canon_harness"))
                and os.path.realpath(args.harness_contract).startswith(h + os.sep)):
            print("TIE AUDIT GATE FAILED: --harness-contract is honoured only inside a guard harness "
                  "(DEPTH_CANON_HARNESS under /tmp with its marker, the file inside it) - abort", file=sys.stderr)
            return 2
    ok, problems, summary = validate_tie_audit(args.audit, args.repo, workers=args.workers,
                                               harness_contract=args.harness_contract)
    if not ok:
        print("TIE AUDIT GATE FAILED: " + "; ".join(problems) + " - abort", file=sys.stderr)
        return 1
    print(summary["audit_sha256"] if args.print_sha else
          f"TIE AUDIT GATE OK: {args.audit} sha256 {summary['audit_sha256']}: {summary['n_maps']} maps "
          "(the registered census), 0 survivors, 0 errors, hashes current, map bytes re-hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
