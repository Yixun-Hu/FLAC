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

from collections import defaultdict


def parse_nodes(fname: str) -> tuple[str, str]:
    """Return ``(source_token, receiver_token)`` for an AR RIR basename.

    ``"S0012_R0077_hybrid_IR.wav" -> ("S0012", "R0077")``. The tokens are returned verbatim
    (raw strings, never int-parsed). Anything that is not ``S<id>_R<id>_<tail>`` with all
    three parts non-empty raises ``ValueError`` — malformed names must fail loudly rather
    than silently collapse two nodes into one.
    """
    if not isinstance(fname, str):
        raise ValueError(f"expected a filename string, got {type(fname).__name__}")
    parts = fname.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"malformed RIR basename (expected S<id>_R<id>_<tail>): {fname!r}")
    src, rec, tail = parts
    if len(src) < 2 or src[0] != "S":
        raise ValueError(f"malformed source token {src!r} in {fname!r}")
    if len(rec) < 2 or rec[0] != "R":
        raise ValueError(f"malformed receiver token {rec!r} in {fname!r}")
    if not tail:
        raise ValueError(f"malformed RIR basename (empty tail): {fname!r}")
    return src, rec


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
