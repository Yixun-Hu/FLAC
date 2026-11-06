#!/usr/bin/env python3
"""exp_37: compare two geometry audits, ignoring ONLY the host-dependent path fields (top-level ``mesh_root`` and
``context_manifest``, every room's ``mesh_path``) and the payload ``sha256`` derived from them. Exit 0 when nothing
else differs (prints the differing JSON paths otherwise, exit 1). Used to prove that the parameterized audit tool at
0.5 m reproduces the frozen exp_09 audit on this box. Written by the exp_37 Coder (Claude Opus 5.5), round B-c.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IGNORED_TOP = ("mesh_root", "context_manifest", "sha256")
IGNORED_ROOM = ("mesh_path",)


def _walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(child)
            else:
                _walk(a[key], b[key], child, out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and not all(
            isinstance(x, (int, float)) for x in a + b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)


def _strip(audit: dict) -> dict:
    out = {k: v for k, v in audit.items() if k not in IGNORED_TOP}
    out["rooms"] = {room: {k: v for k, v in row.items() if k not in IGNORED_ROOM}
                    for room, row in audit.get("rooms", {}).items()}
    return out


def audit_differences(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    _walk(_strip(a), _strip(b), "", out)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    differences = audit_differences(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    print(json.dumps({"reference": str(args.reference), "candidate": str(args.candidate),
                      "ignored": {"top": list(IGNORED_TOP), "room": list(IGNORED_ROOM)},
                      "difference_count": len(differences), "differences": differences[:50]}, indent=2))
    return 0 if not differences else 1


if __name__ == "__main__":
    sys.exit(main())
