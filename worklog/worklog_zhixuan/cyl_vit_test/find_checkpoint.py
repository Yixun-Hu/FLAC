#!/usr/bin/env python3
"""Resolve exactly one Lightning checkpoint by optimizer step."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:^|-)step=(\d+)(?:\.ckpt)?$")


def checkpoint_step(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    return int(match.group(1)) if match else None


def find_checkpoint(root: Path, step: int | None) -> Path:
    candidates = [path for path in root.rglob("*.ckpt") if checkpoint_step(path) is not None]
    if step is not None:
        candidates = [path for path in candidates if checkpoint_step(path) == step]
    if not candidates:
        label = f"step {step}" if step is not None else "any step"
        raise FileNotFoundError(f"no checkpoint for {label} under {root}")
    if step is None:
        max_step = max(checkpoint_step(path) for path in candidates)
        candidates = [path for path in candidates if checkpoint_step(path) == max_step]
    if len(candidates) != 1:
        rendered = "\n".join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            "checkpoint selection is ambiguous; set CKPT_PATH explicitly. Candidates:\n" + rendered
        )
    return candidates[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    print(find_checkpoint(args.root, args.step))


if __name__ == "__main__":
    main()
