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
4. Top-up (deterministic, no RNG): a retained target none of whose *sampler-reachable*
   contexts is retained would have an empty acoustic-context pool, so the earliest
   permutation entry that is a sampler candidate of it is added.
5. Nesting: fractions are processed in ascending order and
   ``S_f = raw_prefix(perm, f) ∪ S_{previous f}``, with the top-up applied to that union, so
   ``S_25 ⊆ S_50 ⊆ S_75``.

**Eligibility is sampler-faithful** (``ELIGIBILITY_RULE``, plan §3 "Amendment 1"): whether one
entry can serve as another's context is decided by :func:`sampler_candidates`, which rebuilds
candidate names exactly the way the pinned ``AR_md.py`` sampler does. It is emphatically *not*
"same receiver token, different source token" — see that function for why the two differ on
111 AR rooms.

Node identity for *selection* is still the raw ``S…`` / ``R…`` token string ("S001" and
"S0012" are different files); only the eligibility question goes through the sampler's
integer reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re

TARGET_DRAWS_AT_40K_X64 = 40_000 * 64  # optimizer steps x effective batch, the exp_14 budget
COUNT_KEYS = ("raw_prefix", "new_topup", "inherited_topup", "final")
#: The rule by which this tool decides that one entry can serve as another's acoustic
#: context. Recorded in the manifest so an emitted split always names the rule it was built
#: under (round E; the superseded rule was raw-token equality).
ELIGIBILITY_RULE = "sampler_faithful_S00int_v1"
#: The two literals the pinned sampler hard-codes when it rebuilds a candidate file name:
#: ``f"S00{node}_{receiver token}_hybrid_IR.wav"`` (``AR_md.py`` @ ``7bbd8aa``, unchanged).
SAMPLER_SOURCE_PREFIX = "S00"
SAMPLER_TAIL = "hybrid_IR.wav"
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


def sampler_node(source_token: str) -> int:
    """The INTEGER the pinned sampler reduces a source token to (``"S010" -> 10``)."""
    return int(source_token[1:])


def _room_index(room_files) -> tuple[set, list]:
    """``(set of basenames, ascending integer source nodes)`` — built once per room.

    The node universe is exactly the sampler's ``all_src_node``: every source token in the
    room, reduced to an integer, de-duplicated.
    """
    present = set(room_files)
    return present, sorted({sampler_node(parse_nodes(f)[0]) for f in present})


def _candidates_from_index(target_basename: str, present: set, int_nodes: list) -> list[str]:
    """``sampler_candidates`` against a pre-built room index (the hot path)."""
    src_tok, rec_tok = parse_nodes(target_basename)
    me = sampler_node(src_tok)
    out = []
    for node in int_nodes:
        if node == me:
            continue
        name = f"{SAMPLER_SOURCE_PREFIX}{node}_{rec_tok}_{SAMPLER_TAIL}"
        if name in present:
            out.append(name)
    return out


def sampler_candidates(target_basename: str, room_files) -> list[str]:
    """The acoustic-context pool the PINNED sampler would offer for ``target_basename``.

    This mirrors ``AR_md.get_ir_and_location_for_other_sources`` (``7bbd8aa``) exactly, and
    it is the single definition of "eligible context" for this experiment (plan §3
    "Amendment 1"). The sampler does **not** compare tokens: it parses every source token in
    the room to an integer (``all_src_node``), drops the target's own node, rebuilds each
    remaining one as ``f"S00{node}_{receiver token}_hybrid_IR.wav"`` and silently discards a
    rebuilt name that does not exist. Two consequences follow, and both are load-bearing:

    * in the 111 AR training rooms whose ten sources are spelled ``S001…S010``, node 10 is
      rebuilt as ``"S0010"``, which does not exist — so ``S010`` is **never reachable as a
      context** there. It remains a perfectly good *target* (it can use the other nine).
      This is an upstream quirk that the 100 % anchors trained under; exp_14 reproduces it
      rather than fixing it, so that every point of the curve shares one sampler;
    * the tail is hard-coded, so an entry whose basename does not end in ``_hybrid_IR.wav``
      can never be drawn either.

    ``room_files`` is the room's full basename list — on AR, ``train.json``'s room list.
    (The sampler's own universe is ``os.listdir`` of the room directory, a superset; the two
    agree here because any rebuilt name that is in ``room_files`` necessarily contributes its
    own node to ``room_files``'s node set, and a name outside ``room_files`` is outside the
    split and therefore outside every subset anyway.)

    Returns the candidates in ascending node order; a malformed basename raises
    ``ValueError`` via :func:`parse_nodes`.
    """
    present, int_nodes = _room_index(room_files)
    return _candidates_from_index(target_basename, present, int_nodes)


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


def topup_zero_context(
    selected: set[str], perm: list[str], room_files=None
) -> tuple[set[str], list[str]]:
    """Add the entries needed so that every retained target has a non-empty context pool.

    A retained target is *starved* when none of its :func:`sampler_candidates` is retained:
    FLAC draws the K acoustic-context RIRs from that pool, so such a target would be
    untrainable under the restricted sampler. The repair walks ``perm`` in permutation order
    (restricted to the selection, which grows as entries are added) and, for each starved
    target, adds the **earliest ``perm`` entry that is a sampler candidate of it**.
    Deterministic: no RNG.

    ``room_files`` is the room's full basename list, i.e. the node universe eligibility is
    judged against; it defaults to ``perm`` and must be a permutation of it (a caller may
    still pass it explicitly to state the universe it means).

    Unlike the superseded token rule, **one pass does not suffice**: an addition need not be
    reachable *from* the target it repairs (``S010`` can use ``S001`` while ``S001`` cannot
    use ``S010``), and it can sit at an index the walk has already passed. The walk therefore
    repeats until a pass adds nothing. It terminates because every pass strictly grows a
    subset of the finite room, and it is deterministic because each pass visits ``perm`` in
    order. Repeated passes never *un*-starve-then-re-starve a target: ``current`` only grows.

    Returns ``(new_selection, added_in_order)``; the caller's set is never mutated. Raises
    ``ValueError`` if a starved target has no sampler candidate anywhere in the full room
    list — a *dead* target, impossible on AR (``dead_targets_full == 0``), and a contract
    violation that must fail loudly rather than emit an untrainable target.
    """
    room_files = list(perm) if room_files is None else list(room_files)
    present, int_nodes = _room_index(room_files)
    if len(perm) != len(present) or present != set(perm):
        raise ValueError(
            f"perm ({len(perm)} entries) is not a permutation of room_files "
            f"({len(present)} distinct entries)"
        )
    unknown = set(selected) - present
    if unknown:
        raise ValueError(f"selected entries absent from the permutation: {sorted(unknown)!r}")

    order = {f: i for i, f in enumerate(perm)}
    pools: dict[str, list[str]] = {}
    current = set(selected)
    added: list[str] = []
    while True:
        grew = False
        for f in perm:
            if f not in current:
                continue
            pool = pools.get(f)
            if pool is None:
                pool = pools[f] = _candidates_from_index(f, present, int_nodes)
            if any(cand in current for cand in pool):
                continue  # already has a reachable context
            if not pool:
                raise ValueError(
                    f"{f!r} has no sampler-reachable context anywhere in the full room "
                    "list; cannot build a context pool for it"
                )
            cand = min(pool, key=order.__getitem__)  # earliest in permutation order
            current.add(cand)  # cand cannot already be selected, else f would not be starved
            added.append(cand)
            grew = True
        if not grew:
            return current, added


def context_histogram(room_subset: set[str]) -> dict:
    """Bin the retained entries of one room by their number of retained *reachable* contexts.

    For each retained target, count its :func:`sampler_candidates` that are themselves
    retained — that is exactly FLAC's eligible acoustic-context pool under the restricted
    sampler. Bins: ``"0"`` (starved — must be 0 after the top-up), ``"1-7"`` (the sampler
    draws K=8 references WITH replacement) and ``">=8"``.

    Judging the pool against the retained room alone is exact: a candidate must be retained
    to count, and a retained candidate necessarily contributes its own node to the retained
    room's node universe, so nothing is lost by not passing the full room list.
    """
    present, int_nodes = _room_index(room_subset)
    hist = {"0": 0, "1-7": 0, ">=8": 0}
    for f in present:
        n_other = len(_candidates_from_index(f, present, int_nodes))
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


def build_subsets(
    split: dict, fractions: list[float], seed: int, train_json_path=None
) -> tuple[dict, dict]:
    """Build the nested per-room subsets of ``split`` and the manifest describing them.

    ``split`` is the AR ``scene -> room -> [basenames]`` mapping. The manifest reports, per
    fraction, the four counts + effective epochs (a global quantity) + the eligible-context
    histogram, and repeats the four counts **and that histogram** per room, so every emitted
    room can be audited on its own. At the top level it names the eligibility rule the build
    used and the full split's own facts under it (``dead_targets_full``, which must be 0, and
    ``full_split_context_histogram``, whose ``"1-7"`` bin is the replacement-sampling
    baseline the fractions are compared against). Returns
    ``({fraction: split_like}, manifest)``; the emitted split-likes carry the same scene/room
    keys (in sorted order) with the retained files sorted inside each room. The input is
    never mutated. ``train_json_path`` (the file ``split`` was read from) is hashed into the
    manifest as ``source_train_json_sha256`` — the provenance ``write_outputs`` then requires.
    """
    fracs = _validated_fractions(fractions)
    rng = random.Random(seed)

    subsets = {f: {} for f in fracs}
    per_room = {f: {} for f in fracs}
    histogram = {f: {"0": 0, "1-7": 0, ">=8": 0} for f in fracs}
    totals = {f: dict.fromkeys(COUNT_KEYS, 0) for f in fracs}
    full_histogram = {"0": 0, "1-7": 0, ">=8": 0}
    seen_rooms = set()

    for scene in sorted(split):
        if not split[scene]:
            raise ValueError(f"scene {scene!r} has no rooms")
        for room in sorted(split[scene]):
            files = split[scene][room]
            if not files:
                raise ValueError(f"empty room {scene}/{room}")
            if len(set(files)) != len(files):
                raise ValueError(f"duplicate basenames in room {scene}/{room}")
            if room in seen_rooms:
                raise ValueError(f"duplicate room name {room!r} (manifest keys room-wise)")
            seen_rooms.add(room)

            # The full room's own eligibility, before any subsetting: a target with no
            # sampler-reachable context anywhere cannot be repaired by ANY subset, so it is
            # a property of train.json itself and must stop the build (AR has none).
            full_room_hist = context_histogram(set(files))
            if full_room_hist["0"]:
                dead = sorted(f for f in files if not sampler_candidates(f, files))
                raise ValueError(
                    f"room {scene}/{room} has {len(dead)} dead target(s) — no sampler-"
                    f"reachable context in the FULL split: {dead[:5]!r}"
                )
            for bin_name, n in full_room_hist.items():
                full_histogram[bin_name] += n

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
                if room_histogram["0"]:
                    raise ValueError(  # the top-up's post-condition, checked not assumed
                        f"{room_histogram['0']} starved target(s) survived the top-up in "
                        f"{scene}/{room} at fraction {frac}"
                    )
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
        "eligibility": ELIGIBILITY_RULE,
        "dead_targets_full": full_histogram["0"],
        "full_split_context_histogram": full_histogram,
        "source_train_json_sha256": _sha256_file(train_json_path) if train_json_path else None,
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


def write_outputs(out_dir: str, subsets: dict, manifest: dict, seed=None) -> dict:
    """Write the split files, the manifest and a detached ``sha256sum -c`` checksum file.

    Emits ``train_frac<PPP>_s<seed>.json`` per fraction, ``train_frac_manifest_s<seed>.json``
    (which records the sha256 of every split file it emitted **and of the source train.json**,
    but never of itself) and ``train_frac_manifest_s<seed>.sha256``, which covers the three
    splits *and* the manifest. Byte-deterministic: re-running writes identical bytes. The
    caller's ``manifest`` is not mutated. Returns the written paths.

    The seed tag in every filename is taken from ``manifest["seed"]``, so a file can never be
    labelled with a seed the manifest does not describe; the optional ``seed`` argument is a
    cross-check only and must agree with it. ``manifest["source_train_json_sha256"]`` must
    already be set (``build_subsets(..., train_json_path=...)``): an emitted split that cannot
    name the ``train.json`` it came from is unusable as a run pin.
    """
    manifest_seed = manifest["seed"]
    if seed is not None and seed != manifest_seed:
        raise ValueError(
            f"seed {seed!r} disagrees with the manifest's seed {manifest_seed!r}"
        )
    if not manifest.get("source_train_json_sha256"):
        raise ValueError(
            "manifest['source_train_json_sha256'] is missing; build the subsets with "
            "build_subsets(..., train_json_path=<train.json>) so the outputs carry provenance"
        )
    os.makedirs(out_dir, exist_ok=True)
    manifest = dict(manifest)

    digests: dict[str, str] = {}
    split_paths: dict[float, str] = {}
    for frac in sorted(subsets):
        name = f"train_frac{_frac_tag(frac)}_s{manifest_seed}.json"
        path = os.path.join(out_dir, name)
        payload = _dumps(subsets[frac])
        with open(path, "wb") as fh:
            fh.write(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
        split_paths[frac] = path
    manifest["files"] = digests

    manifest_name = f"train_frac_manifest_s{manifest_seed}.json"
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest_payload = _dumps(manifest)
    with open(manifest_path, "wb") as fh:
        fh.write(manifest_payload)

    checksums_name = f"train_frac_manifest_s{manifest_seed}.sha256"
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

    subsets, manifest = build_subsets(split, fractions, args.seed, args.train_json)
    paths = write_outputs(args.out_dir, subsets, manifest, args.seed)

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
