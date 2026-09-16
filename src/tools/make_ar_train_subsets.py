"""Build the nested per-room training subsets of the AcousticRooms split (exp_14 data curve).

Copy-only tool: it *reads* ``data/AR/train.json`` (dict ``scene -> room -> [basenames]``) and
*writes* new split files next to it; the input is never modified.

Frozen algorithm (``plan_data_curve.md`` §3 — any change invalidates the runs pinned to the
emitted files):

1. Iterate scenes in sorted order, rooms in sorted order; within a room sort the file list.
2. One PRNG for the whole build, ``random.Random(seed)``. Per room exactly one draw,
   ``rng.sample(sorted_files, n)`` — the only RNG consumption in this module.
3. Raw prefix for fraction ``f``: the first ``max(1, round(f * n))`` entries of that
   permutation (Python's built-in ``round`` — banker's rounding, ties-to-even).
4. Top-up (deterministic, no RNG): a retained target whose receiver holds no *other* retained
   source would have an empty acoustic-context pool, so the earliest permutation entry with
   the same receiver and a different source is added.
5. Nesting: fractions are processed in ascending order and
   ``S_f = raw_prefix(perm, f) ∪ S_{previous f}``, with the top-up applied to that union, so
   ``S_25 ⊆ S_50 ⊆ S_75``.

Node identity is the raw ``S…`` / ``R…`` token string ("S001" and "S0012" are different
sources); tokens are never int-parsed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
from collections import defaultdict

TARGET_DRAWS_AT_40K_X64 = 40_000 * 64  # optimizer steps x effective batch, the exp_14 budget
COUNT_KEYS = ("raw_prefix", "new_topup", "inherited_topup", "final")
# AR RIR basename grammar: S<digits>_R<digits>_<non-empty tail>, e.g. S0012_R0077_hybrid_IR.wav
_BASENAME_RE = re.compile(r"(S[0-9]+)_(R[0-9]+)_(.+)", re.DOTALL)


def parse_nodes(fname: str) -> tuple[str, str]:
    """Return ``(source_token, receiver_token)`` for an AR RIR basename.

    ``"S0012_R0077_hybrid_IR.wav" -> ("S0012", "R0077")``. The grammar is
    ``S<digits>_R<digits>_<non-empty tail>`` (ASCII digits only, at least one); every real AR
    basename satisfies it. The tokens are returned **verbatim** as strings and never
    int-parsed, so "S001" and "S0012" stay different sources. Anything else raises
    ``ValueError`` — a malformed name must fail loudly rather than silently collapse two
    nodes into one (e.g. ``"Sabc_Rxyz_…"`` must not become the node pair ``("Sabc", "Rxyz")``).
    """
    if not isinstance(fname, str):
        raise ValueError(f"expected a filename string, got {type(fname).__name__}")
    match = _BASENAME_RE.fullmatch(fname)
    if match is None:
        raise ValueError(
            f"malformed RIR basename (expected S<digits>_R<digits>_<tail>): {fname!r}"
        )
    return match.group(1), match.group(2)


def room_permutation(sorted_files: list[str], rng) -> list[str]:
    """Return ``rng.sample(sorted_files, n)`` — the ONLY RNG consumption of this module.

    One draw per room, taken in scene-sorted / room-sorted order, is what makes the whole
    build bit-reproducible from ``seed`` alone.
    """
    return rng.sample(sorted_files, len(sorted_files))


def raw_prefix(perm: list[str], frac: float) -> list[str]:
    """The first ``max(1, round(frac * len(perm)))`` entries of ``perm``.

    ``round`` is Python's built-in banker's rounding (ties-to-even), e.g. ``0.25 * 6 -> 2``
    but ``0.25 * 10 -> 2`` and ``0.75 * 6 -> 4``. Because it is a prefix of one fixed
    permutation, ascending fractions are automatically nested.
    """
    return list(perm[: max(1, round(frac * len(perm)))])


def topup_zero_context(selected: set[str], perm: list[str]) -> tuple[set[str], list[str]]:
    """Add the entries needed so that every retained target has a non-empty context pool.

    A retained target is *starved* when its receiver holds no other retained entry with a
    **different source**: FLAC draws the K acoustic-context RIRs from the other sources at the
    target's receiver, so such a target would be untrainable under the restricted sampler.
    The repair walks ``perm`` in permutation order (restricted to the selection, which grows
    as entries are added) and, for each starved target, adds the **earliest** ``perm`` entry
    that shares the receiver and carries a different source. Deterministic: no RNG.

    One pass suffices: the addition gives that receiver two sources, so the added entry and
    every other selected entry at that receiver are non-starved afterwards.

    Returns ``(new_selection, added_in_order)``; the caller's set is never mutated. Raises
    ``ValueError`` if the full room offers no other source at a starved receiver (impossible
    on AR — the contract must fail loudly rather than emit an untrainable target).
    """
    nodes = {f: parse_nodes(f) for f in perm}
    unknown = set(selected) - nodes.keys()
    if unknown:
        raise ValueError(f"selected entries absent from the permutation: {sorted(unknown)!r}")

    by_receiver = defaultdict(list)  # receiver -> entries in permutation order
    for f in perm:
        by_receiver[nodes[f][1]].append(f)

    current = set(selected)
    sources_at = defaultdict(set)  # receiver -> sources currently retained there
    for f in current:
        src, rec = nodes[f]
        sources_at[rec].add(src)

    added: list[str] = []
    for f in perm:
        if f not in current:
            continue
        src, rec = nodes[f]
        if sources_at[rec] - {src}:
            continue  # already has another source at this receiver
        for cand in by_receiver[rec]:
            if nodes[cand][0] != src:
                break
        else:
            raise ValueError(
                f"receiver {rec!r} has only source {src!r} in the full room list; "
                "cannot build a context pool for it"
            )
        current.add(cand)  # cand cannot already be selected, else f would not be starved
        added.append(cand)
        sources_at[rec].add(nodes[cand][0])
    return current, added


def context_histogram(room_subset: set[str]) -> dict:
    """Bin the retained entries of one room by their number of *other* retained sources.

    For each retained target, count the distinct sources retained at its receiver other than
    its own (that is exactly FLAC's eligible acoustic-context pool under the restricted
    sampler). Bins: ``"0"`` (starved — must be 0 after the top-up), ``"1-7"`` (the sampler
    draws K=8 references WITH replacement) and ``">=8"``.
    """
    sources_at = defaultdict(set)
    nodes = {}
    for f in room_subset:
        src, rec = parse_nodes(f)
        nodes[f] = (src, rec)
        sources_at[rec].add(src)

    hist = {"0": 0, "1-7": 0, ">=8": 0}
    for src, rec in nodes.values():
        n_other = len(sources_at[rec] - {src})
        if n_other == 0:
            hist["0"] += 1
        elif n_other < 8:
            hist["1-7"] += 1
        else:
            hist[">=8"] += 1
    return hist


def _validated_fractions(fractions) -> list[float]:
    """Sorted-ascending, de-duplicated fractions in (0, 1]; the nesting needs that order."""
    fracs = [float(f) for f in fractions]
    if not fracs:
        raise ValueError("no fractions given")
    if len(set(fracs)) != len(fracs):
        raise ValueError(f"duplicate fractions: {fracs!r}")
    for f in fracs:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"fraction out of range (0, 1]: {f!r}")
    return sorted(fracs)


def build_subsets(split: dict, fractions: list[float], seed: int) -> tuple[dict, dict]:
    """Build the nested per-room subsets of ``split`` and the manifest describing them.

    ``split`` is the AR ``scene -> room -> [basenames]`` mapping. The manifest reports, per
    fraction, the four counts + effective epochs (a global quantity) + the eligible-context
    histogram, and repeats the four counts **and that histogram** per room, so every emitted
    room can be audited on its own. Returns
    ``({fraction: split_like}, manifest)``; the emitted split-likes carry the same scene/room
    keys (in sorted order) with the retained files sorted inside each room. The input is
    never mutated.
    """
    fracs = _validated_fractions(fractions)
    rng = random.Random(seed)

    subsets = {f: {} for f in fracs}
    per_room = {f: {} for f in fracs}
    histogram = {f: {"0": 0, "1-7": 0, ">=8": 0} for f in fracs}
    totals = {f: dict.fromkeys(COUNT_KEYS, 0) for f in fracs}
    seen_rooms = set()

    for scene in sorted(split):
        for room in sorted(split[scene]):
            files = split[scene][room]
            if not files:
                raise ValueError(f"empty room {scene}/{room}")
            if len(set(files)) != len(files):
                raise ValueError(f"duplicate basenames in room {scene}/{room}")
            if room in seen_rooms:
                raise ValueError(f"duplicate room name {room!r} (manifest keys room-wise)")
            seen_rooms.add(room)

            perm = room_permutation(sorted(files), rng)
            inherited_selection: set[str] = set()
            for frac in fracs:  # ascending: each fraction inherits the smaller one
                prefix = raw_prefix(perm, frac)
                selection, added = topup_zero_context(set(prefix) | inherited_selection, perm)
                counts = {
                    "raw_prefix": len(prefix),
                    "new_topup": len(added),
                    "inherited_topup": len(selection - set(prefix) - set(added)),
                    "final": len(selection),
                }
                room_histogram = context_histogram(selection)
                subsets[frac].setdefault(scene, {})[room] = sorted(selection)
                per_room[frac][room] = {**counts, "context_histogram": room_histogram}
                for key in COUNT_KEYS:
                    totals[frac][key] += counts[key]
                for bin_name, n in room_histogram.items():
                    histogram[frac][bin_name] += n
                inherited_selection = selection

    manifest = {
        "seed": seed,
        "python_version": platform.python_version(),
        "source_train_json_sha256": None,  # filled in by write_outputs
        "fractions": {
            str(frac): {
                **totals[frac],
                "effective_epochs_at_40k_x64": TARGET_DRAWS_AT_40K_X64 / totals[frac]["final"],
                "context_histogram": histogram[frac],
                "per_room": per_room[frac],
            }
            for frac in fracs
        },
        "files": {},  # filled in by write_outputs
    }
    return subsets, manifest


def _frac_tag(frac: float) -> str:
    """Filename tag for a fraction: three digits of percent (0.25 -> "025", 0.5 -> "050")."""
    percent = frac * 100.0
    tag = round(percent)
    if abs(percent - tag) > 1e-9:
        raise ValueError(f"fraction {frac!r} is not a whole percentage; filename would be lossy")
    if not 0 < tag < 1000:
        raise ValueError(f"fraction {frac!r} does not fit the three-digit percent tag")
    return f"{tag:03d}"


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _dumps(obj) -> bytes:
    """Deterministic JSON bytes: indent=1, insertion (= sorted) key order, trailing newline."""
    return (json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=True) + "\n").encode("utf-8")


def write_outputs(out_dir: str, subsets: dict, manifest: dict, seed: int, train_json_path=None) -> dict:
    """Write the split files, the manifest and a detached ``sha256sum -c`` checksum file.

    Emits ``train_frac<PPP>_s<seed>.json`` per fraction, ``train_frac_manifest_s<seed>.json``
    (which records the sha256 of every split file it emitted **and of the source train.json**,
    but never of itself) and ``train_frac_manifest_s<seed>.sha256``, which covers the three
    splits *and* the manifest. Byte-deterministic: re-running writes identical bytes. The
    caller's ``manifest`` is not mutated. Returns the written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = dict(manifest)
    manifest["source_train_json_sha256"] = (
        _sha256_file(train_json_path) if train_json_path else None
    )

    digests: dict[str, str] = {}
    split_paths: dict[float, str] = {}
    for frac in sorted(subsets):
        name = f"train_frac{_frac_tag(frac)}_s{seed}.json"
        path = os.path.join(out_dir, name)
        payload = _dumps(subsets[frac])
        with open(path, "wb") as fh:
            fh.write(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
        split_paths[frac] = path
    manifest["files"] = digests

    manifest_name = f"train_frac_manifest_s{seed}.json"
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest_payload = _dumps(manifest)
    with open(manifest_path, "wb") as fh:
        fh.write(manifest_payload)

    checksums_name = f"train_frac_manifest_s{seed}.sha256"
    checksums_path = os.path.join(out_dir, checksums_name)
    lines = [f"{digests[name]}  {name}" for name in digests]
    lines.append(f"{hashlib.sha256(manifest_payload).hexdigest()}  {manifest_name}")
    with open(checksums_path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("utf-8"))

    return {"splits": split_paths, "manifest": manifest_path, "checksums": checksums_path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-json", required=True, help="source split (read-only, never modified)")
    ap.add_argument("--out-dir", required=True, help="directory the subsets are written to")
    ap.add_argument("--fractions", default="0.25,0.5,0.75", help="comma-separated fractions")
    ap.add_argument("--seed", type=int, default=2026, help="PRNG seed (part of every filename)")
    args = ap.parse_args(argv)

    with open(args.train_json, "rb") as fh:
        split = json.loads(fh.read())
    fractions = [float(tok) for tok in args.fractions.split(",") if tok.strip()]

    subsets, manifest = build_subsets(split, fractions, args.seed)
    paths = write_outputs(args.out_dir, subsets, manifest, args.seed, args.train_json)

    for frac in sorted(subsets):
        entry = manifest["fractions"][str(frac)]
        hist = entry["context_histogram"]
        print(
            f"frac={frac:<5} raw_prefix={entry['raw_prefix']} new_topup={entry['new_topup']} "
            f"inherited_topup={entry['inherited_topup']} final={entry['final']} "
            f"ctx[0/1-7/>=8]={hist['0']}/{hist['1-7']}/{hist['>=8']} "
            f"eff_epochs@40kx64={entry['effective_epochs_at_40k_x64']:.2f} "
            f"-> {os.path.basename(paths['splits'][frac])}"
        )
    print(f"manifest: {paths['manifest']}")
    print(f"checksums: {paths['checksums']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
