#!/usr/bin/env python3
"""exp_37 B3 synthesis section builder (Codex code review 2, BLOCKING 5): after the leaf's chain b3 has completed.

Reads the 12 cells' artifacts at the pinned layout (``exp37_synthesis.cell_paths``: the 8 new cells in the leaf's run
worktree at its launch commit, the 4 reused exp_35 phase-(i) C4 cells in the exp_35 run worktree, every dump manifest
under the exp25 dump root), hashes them, validates the whole section with ``exp37_synthesis.validate_section`` (each
artifact's own identities, the pinned reused digests, unique files) and only then writes ``--output`` (write-once).
Exit 2 and nothing written on a missing or refused artifact. Read-only towards both worktrees and the NAS.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization import exp37_synthesis as syn  # noqa: E402

METRICS = ("T60", "C50", "EDT", "FD")
CHAIN = "worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh"


def build_section() -> dict:
    cells, missing = [], []
    for orbit in ("0", "0_180", "c4"):
        for theta in (0, 90):
            for k in (1, 8):
                name, _block, source = syn.cell_name(orbit, theta, k)
                paths = syn.cell_paths(orbit, theta, k)
                missing += [p for p in paths.values() if not Path(p).is_file()]
                digests = {syn.SHA_KEY[key]: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for key, path in paths.items() if Path(path).is_file()}
                cells.append({"orbit": orbit, "theta": theta, "k_context": k, "seed": 42, "source": source, "cell": name,
                              **{key: paths[key] for key in syn.SHA_KEY}, **digests})
    if missing:
        raise FileNotFoundError("missing synthesis artifacts: " + ", ".join(missing))
    section = {"status": "complete", "leaf_commit": syn.LEAF_COMMIT,
               "runbook": f"exp37_runbook.sh step B3S.run: cd {syn.LEAF_ROOT} && EXPECT_HEAD={syn.LEAF_COMMIT} bash {CHAIN} b3",
               "metrics_keys": list(METRICS), "cells": [{key: c[key] for key in syn.CELL_KEYS} for c in cells]}
    syn.validate_section(section)
    return section


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"REFUSED: {args.output} exists (the synthesis section is written once)", file=sys.stderr)
        return 2
    try:
        section = build_section()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(section, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "cells": len(section["cells"]), "leaf_commit": section["leaf_commit"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
