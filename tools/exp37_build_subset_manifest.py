#!/usr/bin/env python3
"""exp_37 nested-1.0 m subset manifest (plan_loc_efficiency.md §4, §10): per pilot query, the integer-coordinate
subset of its FROZEN 0.5 m candidates (reconstructed from the audit and the sha256-checked remapped meshes, every
audit hash re-verified on the way), recorded as its count and the sha256 of its full-grid indices. The manifest is
hashed and bound to the parent pilot, context manifest and audit; it is the selection the 1.0 m AGREE selector
vector is measured on (``tools/exp37_selector_latency.py --selection``). CPU only.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
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

from src.localization.ar_queries import load_context_manifest  # noqa: E402
from src.localization.engine import filter_frozen_query_candidates, reconstruct_room_base_candidates  # noqa: E402
from src.localization.exp37_grid import nested_subset  # noqa: E402
from src.localization.mesh_remap import resolve_mesh_root  # noqa: E402
from src.localization.pilot import canonical_sha256, load_pilot_manifest, resolve_pilot_records  # noqa: E402


def frozen_candidates(context: dict, audit: dict, mesh_root: Path):
    """``record -> its frozen 0.5 m candidates``: room bases rebuilt once per room from the remapped meshes."""
    meshes = resolve_mesh_root(audit, mesh_root)["rooms"]
    records = {int(r["index"]): r for r in context["records"]}
    bases = {}

    def candidates(selected: dict):
        room = selected["room"]
        if room not in bases:
            bases[room] = reconstruct_room_base_candidates(room, audit, mesh_path=meshes[room]["mesh_path"])
        return filter_frozen_query_candidates(records[int(selected["index"])], audit, bases[room])
    return candidates


def build_subset_manifest(pilot: dict, *, context_sha256: str, audit_sha256: str, candidates_for) -> dict:
    records = []
    for selected in pilot["records"]:
        candidates = candidates_for(selected)
        if len(candidates) != int(selected["candidate_count"]):
            raise RuntimeError(f"query {selected['index']}: candidate count {len(candidates)} is not the pilot's")
        index = nested_subset(candidates)
        records.append({"index": int(selected["index"]), "query_id": selected["query_id"], "scene": selected["scene"],
                        "room": selected["room"], "candidate_count": int(len(index)),
                        "full_candidate_count": int(selected["candidate_count"]),
                        "candidate_indices_sha256": hashlib.sha256(index.astype("<u4").tobytes()).hexdigest(),
                        "parent_candidate_indices_sha256": selected["candidate_indices_sha256"]})
    body = {"schema_version": 1, "subset": "nested_1m_integer",
            "definition": "rows of the frozen origin-anchored 0.5 m candidates whose three coordinates are integers; "
                          "indices are positions in the full 0.5 m candidate list (full-grid common random numbers)",
            "parent_pilot_sha256": pilot["sha256"], "context_manifest_sha256": context_sha256,
            "geometry_audit_sha256": audit_sha256, "query_count": len(records),
            "candidate_query_pairs": int(sum(r["candidate_count"] for r in records)), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"{args.output} exists (manifests are written once)")
    pilot = load_pilot_manifest(args.pilot_manifest)
    context = load_context_manifest(args.context_manifest)
    audit = json.loads(args.geometry_audit.read_text())
    if audit.get("sha256") != canonical_sha256({k: v for k, v in audit.items() if k != "sha256"}):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    resolve_pilot_records(pilot, context, audit)                   # the pilot / context / audit bindings
    manifest = build_subset_manifest(pilot, context_sha256=context["sha256"], audit_sha256=audit["sha256"],
                                     candidates_for=frozen_candidates(context, audit, args.mesh_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "sha256": manifest["sha256"], "queries": manifest["query_count"],
                      "candidate_query_pairs": manifest["candidate_query_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
