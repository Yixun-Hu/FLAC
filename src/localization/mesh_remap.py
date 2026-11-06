"""exp_37 ``--mesh-root`` remap (plan_loc_efficiency.md §3, §10 "Provenance").

The frozen exp_09 geometry audit records each room's ABSOLUTE mesh path on the host that built it
(``<audit mesh_root>/<Scene>/<Room>.obj``). ``resolve_mesh_root`` maps every audited room to
``<mesh_root>/<Scene>/<Room>.obj`` on this box and refuses unless each local file hashes to the audit's
``mesh_sha256``. The audit is only read, so its sha256 and every binding derived from it (pilot manifest, run
identity) stay exactly as frozen; ``reconstruct_room_base_candidates(..., mesh_path=...)`` re-checks the bytes it
actually loads, and the audit's ``base_points_sha256`` still guards the reconstructed grid.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mesh_root(audit: dict, mesh_root: Path | str) -> dict:
    """``{"argument", "resolved", "audit_mesh_root", "rooms": {room: {"mesh_path", "mesh_sha256"}}}``."""
    root = Path(mesh_root)
    if not root.is_dir():
        raise RuntimeError(f"--mesh-root {root} is not a directory")
    recorded_root = str(audit["mesh_root"])
    if not recorded_root.startswith("/"):
        raise RuntimeError(f"the audit's mesh_root {recorded_root!r} is not an absolute path")
    rooms = {}
    for room, row in sorted(audit["rooms"].items()):
        relative = f"{row['scene']}/{room}.obj"
        if row["mesh_path"] != f"{recorded_root.rstrip('/')}/{relative}":
            raise RuntimeError(f"room {room}: the audit's mesh_path {row['mesh_path']!r} does not follow the "
                               f"<mesh_root>/<Scene>/<Room>.obj layout under {recorded_root!r}")
        local = root / row["scene"] / f"{room}.obj"
        if not local.is_file():
            raise RuntimeError(f"room {room}: no mesh at {local}")
        digest = _sha256(local)
        if digest != row["mesh_sha256"]:
            raise RuntimeError(f"room {room}: {local} SHA-256 {digest} differs from the audit's {row['mesh_sha256']}")
        rooms[room] = {"mesh_path": str(local.resolve()), "mesh_sha256": digest}
    return {"argument": str(mesh_root), "resolved": str(root.resolve()), "audit_mesh_root": recorded_root,
            "rooms": rooms}
