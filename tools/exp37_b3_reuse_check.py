#!/usr/bin/env python3
"""exp_37 B3 reuse check (Codex code review 3, H-04): the reused exp_35 phase-(i) C4 cells, by their EXACT digests.

    --stage run    (runbook B3S.reuse.run, before B3S.run): the C4 / K8 theta-90 cell, the leaf RPL-37 reference;
    --stage cells  (runbook B3S.reuse.cells, before B3S.cells): all four reused cells.

Each cell's metrics JSON, schema-2 receipt and dump manifest must hash to the pinned ``exp37_synthesis.REUSED``
digests, and its stream sidecar to the stream digest that (pinned) receipt binds. On success one hashed, write-once
record goes to ``--output`` (exit 0; the runbook binds its sha256 into the step receipt); a missing, unreadable or
wrong-hash artifact exits 2 and writes nothing. Read-only towards the exp_35 run worktree and the NAS dump root.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), final fix round: fixed after review 3, not re-reviewed (cap).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization import exp37_synthesis as syn  # noqa: E402
from src.localization.pilot import canonical_sha256  # noqa: E402

STAGES = {"run": [("c4", 90, 8)], "cells": [("c4", 0, 1), ("c4", 0, 8), ("c4", 90, 1), ("c4", 90, 8)]}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(stage: str) -> tuple[dict, list[str]]:
    cells, reasons = {}, []
    for orbit, theta, k in STAGES[stage]:
        name, _block, source = syn.cell_name(orbit, theta, k)
        pins = syn.REUSED.get(name) if source == "reused_exp35" else None
        if not pins:
            reasons.append(f"{name}: no pinned reused digests")
            continue
        paths = syn.cell_paths(orbit, theta, k)
        entry = {}
        for key, path in paths.items():
            want = pins.get(syn.SHA_KEY[key])
            try:
                got = _sha(Path(path))
            except OSError as error:
                reasons.append(f"{name}: {key} {path} is missing or unreadable ({error.strerror or error})")
                continue
            if got != want:
                reasons.append(f"{name}: {key} {path} hashes to {got}, not the pinned {want}")
            entry[key] = {"path": str(path), "sha256": got}
        stream = Path(paths["metrics_json"]).with_name(f"{syn.cell_stem(orbit, theta, k)}.stream.json")
        try:
            bound = (json.loads(Path(paths["receipt"]).read_text()).get("artifact_sha256") or {}).get("stream")
            got = _sha(stream)
        except (OSError, ValueError, AttributeError) as error:
            reasons.append(f"{name}: the stream sidecar {stream} or its receipt binding is unreadable ({error})")
        else:
            if got != bound:
                reasons.append(f"{name}: the stream sidecar {stream} hashes to {got}, not the receipt's {bound}")
            entry["stream"] = {"path": str(stream), "sha256": got}
        cells[name] = entry
    return cells, reasons


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.output):
        print(f"REFUSED: {args.output} exists (reuse-check records are write-once)", file=sys.stderr)
        return 2
    cells, reasons = check(args.stage)
    if reasons:
        print("REFUSED: the reused exp_35 B3 evidence is not the pinned evidence:\n  " + "\n  ".join(reasons),
              file=sys.stderr)
        return 2
    body = {"schema_version": 1, "record_type": "exp37_b3_reuse_check", "stage": args.stage, "verdict": "ok",
            "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "h35_pin": syn.H35_PIN,
            "cells": cells}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"stage": args.stage, "verdict": "ok", "cells": sorted(cells), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
