"""exp_37 transactional publication (Codex code review 2, HIGH 7).

``publish(finals, render)``: every final output path must be new; ``render(staging)`` writes all of them (by basename)
into a fresh, uniquely owned staging directory next to the first output; each staged file must exist and be nonempty
and is fsync'd; the files are then hard-linked into their final names IN ORDER (no-replace: an existing name fails)
-- the LAST one is the publication record (the summary JSON) -- and the directories are fsync'd; a directory fsync
that fails or is unsupported is fatal (``exp37_waveforms.DurabilityError``; review 3, B-01). Any failure while
rendering, fsyncing or linking unlinks the finals already linked (rollback) and leaves no final output path; the
staging directory is always removed. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2; the final
fix round (fixed after review 3, not re-reviewed (cap)).
"""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

from src.localization.exp37_waveforms import fsync_dir


def staged_sha256(staging: Path, finals) -> dict:
    """The sha256 of already-staged outputs (for the publication record)."""
    return {Path(p).name: hashlib.sha256((Path(staging) / Path(p).name).read_bytes()).hexdigest() for p in finals}


def publish(finals, render) -> None:
    finals = [Path(p) for p in finals]
    if len({p.name for p in finals}) != len(finals):
        raise ValueError("the outputs of one publication need distinct basenames")
    existing = [p for p in finals if os.path.lexists(p)]
    if existing:
        raise FileExistsError(f"{existing[0]} exists (publications are write-once)")
    finals[0].parent.mkdir(parents=True, exist_ok=True)
    staging = finals[0].parent / f".publish-{os.getpid()}-{secrets.token_hex(6)}"
    os.mkdir(staging)
    linked = []
    try:
        render(staging)
        for final in finals:
            staged = staging / final.name
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError(f"the rendered output {final.name} is missing or empty")
            fd = os.open(staged, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fsync_dir(staging)
        try:
            for final in finals:
                final.parent.mkdir(parents=True, exist_ok=True)
                os.link(staging / final.name, final)
                linked.append(final)
            for folder in {final.parent for final in finals}:
                fsync_dir(folder)
        except BaseException:
            for final in linked:                          # rollback: no final output path survives a failure
                os.unlink(final)
            raise
    finally:
        for entry in staging.iterdir():
            entry.unlink()
        staging.rmdir()
