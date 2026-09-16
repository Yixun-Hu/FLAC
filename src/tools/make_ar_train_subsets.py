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
