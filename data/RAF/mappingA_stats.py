"""Placement-level statistics for the Mapping-A cross-arm comparison.

exp_21, contract section E / plan section 6. The clustering unit is the PLACEMENT,
not the item: the 36 items of a placement share one room position, one array, one
target source and largely overlapping context, so treating them as 1,152
independent observations would understate every interval by roughly the square root
of that dependence. So: aggregate within placement first, then bootstrap over
placements (room-stratified), and contrast arms by a PAIRED placement-level
randomization -- the arms saw identical items, and throwing that pairing away would
be a strictly weaker test.

No item-i.i.d. intervals, and no generalization claims beyond these two rooms.
"""
import hashlib
import itertools
import json
import math

import numpy as np

EXACT_RANDOMIZATION_LIMIT = 20   # 2**20 sign assignments is still cheap and EXACT

# The registered design (r2 N7). A run that does not have this shape is not the
# experiment these statistics describe, so the shape is checked rather than
# inferred from whatever arrived.
REGISTERED_ROOMS = ("EmptyRoom", "FurnishedRoom")
REGISTERED_PLACEMENTS_PER_ROOM = 16
REGISTERED_SLOTS_PER_PLACEMENT = 36
REGISTERED_N_ITEMS = (len(REGISTERED_ROOMS) * REGISTERED_PLACEMENTS_PER_ROOM
                      * REGISTERED_SLOTS_PER_PLACEMENT)
PER_ITEM_SCHEMA_VERSION = 1

# P3: an arm is a CELL, and a cell is identified by its provenance -- not by which
# files an operator happened to group together. Everything here must be constant
# within an arm; the shared subset must match ACROSS arms, because that is what
# makes the two arms comparable at all.
ARM_IDENTITY_FIELDS = (
    # what the experiment VARIES: constant within an arm, expected to differ
    # across arms (the checkpoint and the conditioning protocol it was trained for)
    "ckpt_sha256", "cond_method", "frame_avg_angles", "frame_avg_fwd_cap",
    "orbit_execution", "model_config_sha256",
    # what it HOLDS FIXED -- see SHARED_IDENTITY_FIELDS
    "steps", "cfg_scale", "are_lambda", "cond_autocast", "batch_size",
    "rotate_mode", "rotate_deg", "rotate_seed", "source_sha",
    "dataset_config_sha256", "publication_prepare_generation",
    "publication_depth_generation", "stream_input_hash",
)
# r4 Q2: everything the comparison holds fixed must MATCH across arms. Sampler
# budget (steps/cfg_scale/are_lambda), numerical protocol (cond_autocast,
# batch_size -- the noise draw is [B, ...], so batching changes the sample),
# rotation protocol, the corpus (BOTH publication generations), the config it was
# read under, the evaluated item stream, and the code that produced them.
SHARED_IDENTITY_FIELDS = (
    "steps", "cfg_scale", "are_lambda", "cond_autocast", "batch_size",
    "rotate_mode", "rotate_deg", "rotate_seed", "source_sha",
    "dataset_config_sha256", "publication_prepare_generation",
    "publication_depth_generation", "stream_input_hash",
)
# The registered Monte-Carlo draws (plan section 6): five seeds, exactly these.
REGISTERED_SEEDS = (42, 43, 44, 45, 46)

# Amendment 4: the Mapping-A corpus is published at x2.0 over its COMPLETE union
# (its clip clamp binds at 2.0401), while Mapping H stays at x3.0. Every results
# artifact carries the disclosure, because the difference licenses some
# comparisons and not others.
MAPPINGA_AMPLITUDE_SCALAR = 2.0
MAPPINGH_AMPLITUDE_SCALAR = 3.0
CROSS_MAPPING_SCALE_DISCLOSURE = (
    f"Mapping-A audio is written at x{MAPPINGA_AMPLITUDE_SCALAR} over its complete "
    f"union; Mapping H is at x{MAPPINGH_AMPLITUDE_SCALAR}. No audio file is shared "
    "between the two publications. Cross-mapping ABSOLUTE level-dependent "
    "comparisons (multi-resolution L1, Env) are therefore unlicensed; the contrasts "
    "reported here are WITHIN Mapping A and unaffected, and T60/C50/EDT are "
    "level-independent.")

# r4 Q3: the arm NAMES this campaign uses are claims about which weights were
# evaluated, so they are pinned to the checkpoints themselves. The four AR 40k
# endpoints come from ar_40k_endpoints/MANIFEST.sha256 (archived 2026-08-21,
# sha-verified against the NAS copy); "finetuned" is exp_19's RAF finetune at
# step 1000 (rcal_weights_sha256.txt). A run labelled P1 that did not evaluate
# P1's weights is a mislabelled result, and mislabelled results are how a
# comparison silently swaps its arms.
REGISTERED_ARM_CHECKPOINTS = {
    "P1": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
    "BF": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
    "YAW": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
    "BV": "ace9f73507070dd331aa0b43a3a00d9a3c69b8059105c11db87d9ddc96187863",
    "finetuned": "6dfc2b2ebdc7deff4903229afa8722120cfcb4178af367415978028c79f4f055",
}


def registered_label_for(ckpt_sha256):
    """The registered arm name for a checkpoint, or None if it is not one."""
    for label, digest in sorted(REGISTERED_ARM_CHECKPOINTS.items()):
        if digest == ckpt_sha256:
            return label
    return None


def assert_registered_label(label, ckpt_sha256):
    """A registered arm NAME must be the weights it names (r4 Q3).

    Unregistered names are free -- an exploratory arm may be called anything --
    but "P1" is an assertion, and this refuses it when the checkpoint says
    otherwise. The checkpoint's own registered name is returned either way, so a
    report can show what was really evaluated.
    """
    known = registered_label_for(ckpt_sha256)
    expected = REGISTERED_ARM_CHECKPOINTS.get(label)
    if expected is not None and ckpt_sha256 != expected:
        raise ValueError(
            f"arm labelled {label!r} evaluated checkpoint {ckpt_sha256}, but the "
            f"registered {label} checkpoint is {expected}"
            + (f" (these weights are the registered {known})" if known else
               " (these weights are not a registered arm checkpoint)")
            + ". A label is a claim about which weights were run.")
    return known


def load_per_item_sidecar(path):
    """Read one ``<metrics>.per_item.json`` written by eval_FLAC --record-per-item.

    Fail-closed on the schema: these rows are the input to a paired comparison, and
    a row that cannot be identified would be paired by position -- the exact
    failure item substitution produces.
    """
    with open(path) as f:
        payload = json.load(f)
    version = payload.get("schema_version")
    if version != PER_ITEM_SCHEMA_VERSION:
        raise ValueError(f"{path}: per-item schema version {version!r}, expected "
                         f"{PER_ITEM_SCHEMA_VERSION}")
    rows = payload.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: carries no per-item rows")
    seen = set()
    for row in rows:
        for key in ("item_id", "room", "placement_id", "mic_slot", "metrics"):
            if key not in row:
                raise ValueError(f"{path}: a row is missing {key}")
        if not isinstance(row["metrics"], dict) or not row["metrics"]:
            raise ValueError(f"{path}: row {row['item_id']} carries no metrics")
        if row["item_id"] in seen:
            raise ValueError(f"{path}: item {row['item_id']} appears twice")
        seen.add(row["item_id"])
    declared = int(payload.get("n_items", len(rows)))
    if declared != len(rows):
        raise ValueError(f"{path}: declares {declared} items, holds {len(rows)}")
    return {"path": path, "provenance": payload.get("provenance") or {},
            "metrics": payload.get("metrics") or [], "rows": rows}


def assert_registered_design(rows, rooms=REGISTERED_ROOMS,
                            placements_per_room=REGISTERED_PLACEMENTS_PER_ROOM,
                            slots=REGISTERED_SLOTS_PER_PLACEMENT):
    """The rows must be the registered 2 rooms x 16 placements x 36 slots.

    Every interval below is a statement about that design: the clustering unit is
    the placement and the macro is equal-room. A short or lopsided run would still
    produce numbers, and they would silently answer a different question.
    """
    by_placement = {}
    for row in rows:
        by_placement.setdefault((row["room"], row["placement_id"]), []).append(row)
    observed_rooms = sorted({room for room, _ in by_placement})
    problems = []
    if observed_rooms != sorted(rooms):
        problems.append(f"rooms {observed_rooms} != registered {sorted(rooms)}")
    for room in sorted(rooms):
        n = sum(1 for r, _ in by_placement if r == room)
        if n != placements_per_room:
            problems.append(f"{room} holds {n} placements, expected "
                            f"{placements_per_room}")
    for key in sorted(by_placement):
        members = by_placement[key]
        if len(members) != slots:
            problems.append(f"{key[0]}/{key[1]} holds {len(members)} items, expected "
                            f"{slots}")
        elif sorted(r["mic_slot"] for r in members) != list(range(slots)):
            problems.append(f"{key[0]}/{key[1]} does not cover slots 0..{slots - 1}")
    if len(rows) != len(rooms) * placements_per_room * slots:
        problems.append(f"{len(rows)} items, expected "
                        f"{len(rooms) * placements_per_room * slots}")
    if problems:
        raise ValueError("the per-item rows are not the registered Mapping-A "
                         "design: " + "; ".join(problems))
    return {"n_items": len(rows), "rooms": observed_rooms,
            "n_placements": len(by_placement), "slots_per_placement": slots}


def arm_identity(provenance, path=""):
    """The identity fields of one sidecar, all of them required (P3).

    A missing field is not a default: it means the producer did not record what
    cell this was, and an arm assembled from such files is an assumption.
    """
    missing = [field for field in ARM_IDENTITY_FIELDS if field not in provenance]
    if missing:
        raise ValueError(
            f"{path}: the sidecar provenance is missing {missing}, so the cell it "
            "came from cannot be identified. Re-run eval_FLAC --record-per-item "
            "with the current producer.")
    return {field: provenance[field] for field in ARM_IDENTITY_FIELDS}


def identity_digest(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def arm_from_sidecars(paths, metric, label=None, enforce_design=True,
                      registered=True):
    """One arm: its per-seed sidecars, indexed by item id.

    Seeds are read from each sidecar's provenance, never from the filename, and a
    repeated seed is an error -- two runs of the same cell are not two seeds.

    P3: the arm's identity is DERIVED from that provenance and must be constant
    across its sidecars. Before this, an operator's file list was the only thing
    saying these runs were one arm, so two checkpoints or two conditioning methods
    could be pooled into a single "arm" and averaged. ``label`` is a display name
    now, not the identity. In ``registered`` mode the seed set must be exactly the
    registered draws -- a one-seed or wrong-seed experiment is not the registered
    experiment and must not receive its report.
    """
    seeds, rows_by_seed, item_ids = {}, {}, None
    identity, identity_path = None, None
    for path in paths:
        sidecar = load_per_item_sidecar(path)
        if enforce_design:
            assert_registered_design(sidecar["rows"])
        seed = sidecar["provenance"].get("seed")
        if seed is None:
            raise ValueError(f"{path}: the sidecar records no seed, so its rows "
                             "cannot be attributed to a Monte-Carlo draw")
        seed = int(seed)
        if seed in seeds:
            raise ValueError(f"seed {seed} appears in both {seeds[seed]} and {path}")
        seeds[seed] = path

        this_identity = arm_identity(sidecar["provenance"], path)
        if label:
            assert_registered_label(label, this_identity["ckpt_sha256"])
        if identity is None:
            identity, identity_path = this_identity, path
        else:
            differing = sorted(field for field in ARM_IDENTITY_FIELDS
                               if this_identity[field] != identity[field])
            if differing:
                raise ValueError(
                    f"{path} is not the same cell as {identity_path}: "
                    + "; ".join(f"{field} {this_identity[field]!r} != "
                                f"{identity[field]!r}" for field in differing)
                    + ". Only the seed may vary within an arm.")
        values = {}
        for row in sidecar["rows"]:
            if metric not in row["metrics"]:
                raise ValueError(f"{path}: item {row['item_id']} carries no {metric}")
            values[row["item_id"]] = {
                "value": row["metrics"][metric], "room": row["room"],
                "placement_id": row["placement_id"], "mic_slot": row["mic_slot"]}
        if item_ids is None:
            item_ids = set(values)
        elif set(values) != item_ids:
            raise ValueError(
                f"{path}: its item set differs from the arm's other seeds "
                f"({sorted(set(values) ^ item_ids)[:4]} ...): the seeds of one arm "
                "must have evaluated identical items")
        rows_by_seed[seed] = values
    if not rows_by_seed:
        raise ValueError(f"arm {label}: no sidecars")
    if registered and tuple(sorted(rows_by_seed)) != REGISTERED_SEEDS:
        raise ValueError(
            f"arm {label or ''} holds seeds {sorted(rows_by_seed)}, but the "
            f"registered experiment is exactly {list(REGISTERED_SEEDS)}: a run over "
            "other draws is a different experiment and does not get the registered "
            "report. Pass registered=False for an exploratory arm.")
    return {"label": label or identity_digest(identity)[:12], "metric": metric,
            "seeds": sorted(rows_by_seed), "by_seed": rows_by_seed,
            "item_ids": sorted(item_ids),
            "identity": identity, "identity_sha256": identity_digest(identity),
            "registered_label": registered_label_for(identity["ckpt_sha256"]),
            "registered": bool(registered),
            "paths": {int(seed): path for seed, path in seeds.items()}}


def assert_paired(arm_a, arm_b):
    """Exact item x seed pairing, or no paired statistic at all.

    The pairing is the design's strength: the arms saw identical items under
    identical conditioning, so their difference is measured item by item. Comparing
    arms that evaluated different items -- or different numbers of seeds -- would be
    an unpaired comparison wearing a paired test's name.
    """
    problems = []
    if arm_a["metric"] != arm_b["metric"]:
        problems.append(f"metrics differ: {arm_a['metric']} vs {arm_b['metric']}")
    # r4 Q3: two arms under one label collapse into one another wherever a report
    # keys anything by name.
    if arm_a["label"] == arm_b["label"]:
        problems.append(
            f"both arms are labelled {arm_a['label']!r}: a contrast between two "
            "arms of the same name cannot be read, and every per-label entry would "
            "overwrite the other's")
    # P3: the arms must differ in what the experiment varies and agree on
    # everything the comparison holds fixed -- the corpus generation, the config
    # they were read under, and the very stream of items that was evaluated.
    for field in SHARED_IDENTITY_FIELDS:
        a_value = (arm_a.get("identity") or {}).get(field)
        b_value = (arm_b.get("identity") or {}).get(field)
        if a_value != b_value:
            problems.append(f"{field} differs: {a_value!r} vs {b_value!r}")
    if (arm_a.get("identity_sha256") is not None
            and arm_a.get("identity_sha256") == arm_b.get("identity_sha256")):
        problems.append(
            "both arms have identity "
            f"{arm_a['identity_sha256'][:12]}: this is one cell compared with "
            "itself, not two arms")
    missing = sorted(set(arm_a["item_ids"]) ^ set(arm_b["item_ids"]))
    if missing:
        problems.append(f"{len(missing)} items are not in both arms "
                        f"(e.g. {missing[:4]})")
    if arm_a["seeds"] != arm_b["seeds"]:
        problems.append(f"seeds differ: {arm_a['seeds']} vs {arm_b['seeds']}")
    if problems:
        raise ValueError(f"arms {arm_a['label']} and {arm_b['label']} are not paired: "
                         + "; ".join(problems))
    return {"n_items": len(arm_a["item_ids"]), "seeds": list(arm_a["seeds"])}


def _records_for_seed(arm, seed):
    return [{"room": entry["room"], "placement_id": entry["placement_id"],
             "item_id": item_id, "metrics": {arm["metric"]: entry["value"]}}
            for item_id, entry in sorted(arm["by_seed"][seed].items())]


def arm_placement_means(arm):
    """{seed: {(room, placement): mean over that placement's items}}."""
    return {seed: aggregate_within_placement(_records_for_seed(arm, seed),
                                             arm["metric"])[0]
            for seed in arm["seeds"]}


def arm_macros(arm):
    """{seed: equal-room macro}, plus the seed-variability summary."""
    per_seed = {seed: macro_two_room(means)["macro"]
                for seed, means in arm_placement_means(arm).items()}
    return {"by_seed": per_seed, "seed_variability": seed_variability(per_seed)}


def paired_placement_differences(arm_a, arm_b):
    """{(room, placement): mean over items and seeds of (a - b)}.

    The difference is taken ITEM BY ITEM and SEED BY SEED before any averaging, so
    the paired structure survives into the placement-level statistic; differencing
    two independently averaged arms would discard it.
    """
    assert_paired(arm_a, arm_b)
    sums, counts = {}, {}
    for seed in arm_a["seeds"]:
        a_rows, b_rows = arm_a["by_seed"][seed], arm_b["by_seed"][seed]
        for item_id in arm_a["item_ids"]:
            a, b = a_rows[item_id], b_rows[item_id]
            if (a["room"], a["placement_id"]) != (b["room"], b["placement_id"]):
                raise ValueError(
                    f"item {item_id} is filed under {a['room']}/{a['placement_id']} in "
                    f"{arm_a['label']} and {b['room']}/{b['placement_id']} in "
                    f"{arm_b['label']}")
            for value, who in ((a["value"], arm_a["label"]), (b["value"], arm_b["label"])):
                if value is None or not np.isfinite(value):
                    raise ValueError(
                        f"{who} item {item_id}: metric is {value!r}; a difference over "
                        "silently dropped items is a different estimand")
            key = (a["room"], a["placement_id"])
            sums[key] = sums.get(key, 0.0) + (float(a["value"]) - float(b["value"]))
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}, counts


def paired_cluster_bootstrap(differences, n_resamples=10000, alpha=0.05,
                             label="mappingA-contrast"):
    """Room-stratified cluster bootstrap over the PAIRED placement differences.

    The registered interval for a contrast: it resamples placements (the clustering
    unit) within room and reports the equal-room macro difference, so it inherits
    both the pairing and the dependence structure. An interval built from two
    independent per-arm bootstraps would be neither.
    """
    return cluster_bootstrap(differences, n_resamples=n_resamples, alpha=alpha,
                             label=label)


def contrast_report(arm_a, arm_b, n_resamples=10000, alpha=0.05,
                    enforce_design=True, require_registered=True):
    """The registered cross-arm report for one metric.

    Everything it claims is paired at the item level, aggregated at the placement
    level, and macro-averaged over the two rooms; seed variability is reported
    BESIDE the interval, never inside it, because it is a fact about the sampler
    and not about the estimand.
    """
    if require_registered and not (arm_a.get("registered")
                                   and arm_b.get("registered")):
        raise ValueError(
            "the registered contrast report is for registered arms (exactly seeds "
            f"{list(REGISTERED_SEEDS)}); pass require_registered=False to report an "
            "exploratory contrast, which must then be labelled as one.")
    pairing = assert_paired(arm_a, arm_b)
    differences, counts = paired_placement_differences(arm_a, arm_b)
    if enforce_design:
        expected = len(REGISTERED_ROOMS) * REGISTERED_PLACEMENTS_PER_ROOM
        if len(differences) != expected:
            raise ValueError(
                f"the contrast covers {len(differences)} placements, expected "
                f"{expected} ({len(REGISTERED_ROOMS)} rooms x "
                f"{REGISTERED_PLACEMENTS_PER_ROOM})")
    macro = macro_two_room(differences)
    interval = paired_cluster_bootstrap(differences, n_resamples=n_resamples,
                                        alpha=alpha,
                                        label=f"{arm_a['label']}-vs-{arm_b['label']}")
    keys = sorted(differences)
    randomization = sign_flip_test(
        [differences[k] for k in keys], keys, n_resamples=n_resamples,
        label=f"{arm_a['label']}-vs-{arm_b['label']}")
    return {
        "metric": arm_a["metric"],
        "arms": [arm_a["label"], arm_b["label"]],
        # POSITIONAL (r4 Q3): keyed by label, one arm's entry would overwrite the
        # other's the moment two arms shared a name.
        "arm_identities": [
            {"label": arm["label"], "identity_sha256": arm.get("identity_sha256"),
             "registered_label": arm.get("registered_label"),
             "ckpt_sha256": (arm.get("identity") or {}).get("ckpt_sha256"),
             "registered": bool(arm.get("registered")),
             "seed_variability": arm_macros(arm)["seed_variability"]}
            for arm in (arm_a, arm_b)],
        "registered": bool(arm_a.get("registered") and arm_b.get("registered")),
        "seeds": list(arm_a["seeds"]),
        "pairing": pairing,
        "unit": "placement",
        "n_items_per_placement": {f"{room}/{placement}": n
                                  for (room, placement), n in sorted(counts.items())},
        "difference": macro,
        "interval": interval,
        "randomization": randomization,
        "scale_disclosure": CROSS_MAPPING_SCALE_DISCLOSURE,
        "note": ("paired at the item level, clustered at the placement level, "
                 "equal-room macro; seed variability is reported separately and is "
                 "not part of the interval"),
    }


def aggregate_within_placement(records, metric):
    """{(room, placement): mean of ``metric`` over that placement's items}."""
    sums, counts = {}, {}
    for record in records:
        key = (record["room"], record["placement_id"])
        value = record["metrics"][metric]
        if value is None or not np.isfinite(value):
            raise ValueError(
                f"{key} item {record.get('item_id')}: metric {metric} is {value!r}; "
                "an aggregate over silently dropped items is a different estimand")
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}, counts


def macro_two_room(placement_means):
    """Mean over placements within each room, then the equal-room macro.

    Equal-room on purpose: the two rooms are the population, and weighting by
    placement count would let whichever room contributed more placements set the
    headline.
    """
    by_room = {}
    for (room, _placement), value in placement_means.items():
        by_room.setdefault(room, []).append(value)
    room_means = {room: float(np.mean(values)) for room, values in sorted(by_room.items())}
    return {
        "rooms": room_means,
        "macro": float(np.mean([room_means[r] for r in sorted(room_means)])),
        "n_placements": {room: len(values) for room, values in sorted(by_room.items())},
    }


def _stable_seed(label):
    return int(hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)


def cluster_bootstrap(placement_means, n_resamples=10000, alpha=0.05, label="mappingA"):
    """Room-stratified cluster bootstrap over PLACEMENTS.

    Resampling placements (with replacement, within room) is what makes the
    interval reflect the dependence the design actually has; resampling items would
    treat 36 recordings of one array position as 36 independent facts.
    """
    by_room = {}
    for (room, _placement), value in placement_means.items():
        by_room.setdefault(room, []).append(value)
    rooms = sorted(by_room)
    arrays = {room: np.asarray(by_room[room], dtype=np.float64) for room in rooms}
    rng = np.random.default_rng(_stable_seed(label))

    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        room_means = []
        for room in rooms:
            values = arrays[room]
            picks = rng.integers(0, len(values), size=len(values))
            room_means.append(values[picks].mean())
        draws[i] = float(np.mean(room_means))

    point = macro_two_room(placement_means)["macro"]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": float(alpha),
        "n_resamples": int(n_resamples),
        "unit": "placement",
        "stratified_by": "room",
        "n_placements": {room: len(arrays[room]) for room in rooms},
    }


def paired_randomization(arm_a, arm_b, n_resamples=10000, label="mappingA"):
    """Paired placement-level randomization test for an arm contrast.

    The arms evaluated identical items under identical conditioning, so the pairing
    is real and the null is a per-placement sign flip. Exact enumeration for up to
    ``EXACT_RANDOMIZATION_LIMIT`` placements -- with 32 placements the sampled
    p-value is used and reported as such.
    """
    keys = sorted(set(arm_a) & set(arm_b))
    missing = sorted(set(arm_a) ^ set(arm_b))
    if missing:
        raise ValueError(
            f"the two arms do not cover the same placements: {missing}. A paired "
            "test over different placements is not a paired test.")
    if not keys:
        raise ValueError("no placements to compare")

    diffs = np.array([arm_a[k] - arm_b[k] for k in keys], dtype=np.float64)
    return sign_flip_test(diffs, keys, n_resamples=n_resamples, label=label)


def sign_flip_test(diffs, keys, n_resamples=10000, label="mappingA"):
    """The randomization core, over ALREADY PAIRED differences.

    Exposed so a contrast built from item-level differences can use the same test
    without inventing a second arm of zeros to subtract.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    observed = float(np.mean(diffs))
    n = len(diffs)

    if n <= EXACT_RANDOMIZATION_LIMIT:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = (signs * diffs).mean(axis=1)
        exact = True
        n_used = len(stats)
    else:
        rng = np.random.default_rng(_stable_seed(label))
        signs = rng.choice([1.0, -1.0], size=(n_resamples, n))
        stats = (signs * diffs).mean(axis=1)
        exact = False
        n_used = n_resamples

    p_value = float(np.mean(np.abs(stats) >= abs(observed) - 1e-15))
    return {
        "observed_difference": observed,
        "p_value": p_value,
        "exact": exact,
        "n_assignments": int(n_used),
        "n_placements": n,
        "unit": "placement",
        "placements": list(keys),
    }


def seed_variability(per_seed_macros):
    """Monte-Carlo variability of the diffusion seed, reported SEPARATELY.

    It is not a confidence interval about the arm: it says how much one arm's
    number moves when only the sampler's noise changes.
    """
    values = np.asarray(list(per_seed_macros.values()), dtype=np.float64)
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "note": "Monte-Carlo seed variability, not an interval about the estimand",
    }
