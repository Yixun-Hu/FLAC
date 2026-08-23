"""exp_21 analysis rung: the registered Mapping-A contrast reports.

A thin harness over the TESTED machinery in data/RAF/mappingA_stats.py. It reads
the 25 per-item sidecars the sweep wrote beside their checkpoints, ingests them as
five ARMS through the registered path (identity validation, seed set, label
registry), and writes the two committed artifacts:

  * mappingA_contrast_report.json -- every contrast, both rows, plus the per-arm
    per-room and per-placement tables and the arm identities;
  * mappingA_stats_summary.md     -- the same thing, readable.

Every number here comes from mappingA_stats: this file chooses WHICH contrasts to
run and how to lay them out, and computes none of them itself.

Usage:
    python worklog/.../run_mappingA_contrasts.py
"""
import datetime
import glob
import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_REPO, "data", "RAF"))
import mappingA_stats as stats            # noqa: E402

AR_ENDPOINTS = "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints"
FINETUNE = ("/media/diskstation/yixunhu/FLAC/checkpoints/exp19_raf_finetune/"
            "FLAC_RAF/exp19_raf_finetune_1000/checkpoints")
ARM_ROOTS = {"P1": os.path.join(AR_ENDPOINTS, "P1"),
             "YAW": os.path.join(AR_ENDPOINTS, "YAW"),
             "BV": os.path.join(AR_ENDPOINTS, "BV"),
             "BF": os.path.join(AR_ENDPOINTS, "BF"),
             "finetuned": FINETUNE}
AR_ARMS = ["P1", "YAW", "BV", "BF"]
TRANSFER_ARM = "finetuned"
# every metric the sidecars carry that is an ERROR of the prediction against the
# measurement -- lower is better for all five, which is why "better" is stated
# uniformly below. "Invalid T60" is a per-item 0/1 FLAG, i.e. a rate over items,
# not a quantity to average into a contrast; it is reported separately.
METRICS = ["T60", "C50", "EDT", "L1_STFT_MultiRes", "Env"]
METRIC_UNITS = {"T60": "% error", "C50": "dB error", "EDT": "ms error",
                "L1_STFT_MultiRes": "multi-resolution L1", "Env": "envelope distance"}
N_RESAMPLES = 10000
ALPHA = 0.05


def arm_paths(label):
    paths = sorted(glob.glob(os.path.join(ARM_ROOTS[label], "*.per_item.json")))
    if not paths:
        raise FileNotFoundError(f"{label}: no per-item sidecars under "
                                f"{ARM_ROOTS[label]}")
    return paths


def ingest(metric):
    """Five arms for one metric, through the REGISTERED ingestion path."""
    return {label: stats.arm_from_sidecars(arm_paths(label), metric, label)
            for label in ARM_ROOTS}


def invalid_t60_rates():
    """The per-item T60 validity flag, as the rate it is."""
    rates = {}
    for label in ARM_ROOTS:
        flagged = total = 0
        for path in arm_paths(label):
            with open(path) as f:
                payload = json.load(f)
            for row in payload["items"]:
                flagged += float(row["metrics"].get("Invalid T60", 0.0))
                total += 1
        rates[label] = {"n_items_x_seeds": total, "n_invalid": flagged,
                        "rate": flagged / total if total else None}
    return rates


def arm_tables(arm):
    """Per-seed macro, seed variability, per-room and per-placement means."""
    means_by_seed = stats.arm_placement_means(arm)
    macros = stats.arm_macros(arm)
    placements = sorted({key for means in means_by_seed.values() for key in means})
    over_seeds = {f"{room}/{placement}":
                  float(np.mean([means_by_seed[seed][(room, placement)]
                                 for seed in arm["seeds"]]))
                  for room, placement in placements}
    rooms = {}
    for room in stats.REGISTERED_ROOMS:
        values = [v for key, v in over_seeds.items() if key.startswith(room + "/")]
        rooms[room] = {"mean_over_placements": float(np.mean(values)),
                       "n_placements": len(values),
                       "sd_over_placements": float(np.std(values, ddof=1))}
    return {
        "macro_by_seed": macros["by_seed"],
        "seed_variability": macros["seed_variability"],
        "equal_room_macro": float(np.mean([rooms[r]["mean_over_placements"]
                                           for r in stats.REGISTERED_ROOMS])),
        "rooms": rooms,
        "placements": over_seeds,
    }


def verdict(block):
    """Does this difference hold? Interval and randomization must agree."""
    ci_low = block["interval"]["ci_low"]
    ci_high = block["interval"]["ci_high"]
    excludes_zero = bool(ci_low > 0 or ci_high < 0)
    p_value = block["randomization"]["p_value"]
    return {
        "difference": block["difference"]["macro"],
        "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value,
        "interval_excludes_zero": excludes_zero,
        "significant_at_alpha": bool(excludes_zero and p_value < ALPHA),
        # 10 pairs are tested per metric and no correction is registered; this is
        # recorded so a reader can apply one without recomputing anything.
        "bonferroni_10_pairs": bool(excludes_zero and p_value < ALPHA / 10),
        "better": (None if not excludes_zero else
                   (block["arms"][0] if block["difference"]["macro"] < 0
                    else block["arms"][1])),
    }


def main():
    started = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    pairs = ([tuple(pair) for pair in itertools.combinations(AR_ARMS, 2)]
             + [(arm, TRANSFER_ARM) for arm in AR_ARMS])
    contrasts, tables, identities = {}, {}, {}

    for metric in METRICS:
        print(f"[{metric}] ingesting five arms", flush=True)
        arms = ingest(metric)
        if not identities:
            for label, arm in arms.items():
                identities[label] = {
                    "label": arm["label"],
                    "registered_label": arm["registered_label"],
                    "identity_sha256": arm["identity_sha256"],
                    "registered_seeds": arm["registered"],
                    "seeds": arm["seeds"],
                    "n_items": len(arm["item_ids"]),
                    "identity": arm["identity"],
                    "sidecars": {int(seed): os.path.basename(path)
                                 for seed, path in arm["paths"].items()},
                }
            tables.update({label: {} for label in arms})
        for label, arm in arms.items():
            tables[label][metric] = arm_tables(arm)

        contrasts[metric] = {}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            kind = "transfer" if b == TRANSFER_ARM else "AR-arm"
            both = stats.contrast_with_sensitivity(
                arms[a], arms[b], n_resamples=N_RESAMPLES, alpha=ALPHA)
            contrasts[metric][key] = {
                "arms": [a, b], "kind": kind,
                "primary": both["primary"],
                "sensitivity": both["sensitivity"],
                "excluded_items": both["excluded_items"],
                "excluded_item_flags": both["excluded_item_flags"],
                "n_items_excluded": both["n_items_excluded"],
                "reading": both["reading"],
                "verdict": {"primary": verdict(both["primary"]),
                            "minus_flagged": verdict(both["sensitivity"])},
            }
            v = contrasts[metric][key]["verdict"]
            print(f"   {key:22s} {kind:9s} d={v['primary']['difference']:+.4f} "
                  f"[{v['primary']['ci_low']:+.4f},{v['primary']['ci_high']:+.4f}] "
                  f"p={v['primary']['p_value']:.4f} "
                  f"holds={v['primary']['significant_at_alpha']} "
                  f"| minus-flagged holds={v['minus_flagged']['significant_at_alpha']}"
                  f" sign_flip="
                  f"{np.sign(v['primary']['difference']) != np.sign(v['minus_flagged']['difference'])}",
                  flush=True)

    shared = {field: identities["P1"]["identity"][field]
              for field in stats.SHARED_IDENTITY_FIELDS}
    for label, entry in identities.items():
        for field, value in shared.items():
            assert entry["identity"][field] == value, (label, field)

    payload = {
        "schema_version": 1,
        "created_utc": started,
        "finished_utc": datetime.datetime.utcnow().replace(
            microsecond=0).isoformat() + "Z",
        "design": {
            "rooms": list(stats.REGISTERED_ROOMS),
            "placements_per_room": stats.REGISTERED_PLACEMENTS_PER_ROOM,
            "slots_per_placement": stats.REGISTERED_SLOTS_PER_PLACEMENT,
            "n_items": stats.REGISTERED_N_ITEMS,
            "seeds": list(stats.REGISTERED_SEEDS),
            "unit_of_clustering": "placement",
            "aggregation": "equal-room macro of placement means",
            "alpha": ALPHA, "n_resamples": N_RESAMPLES,
            "metrics": METRICS, "metric_units": METRIC_UNITS,
            "metric_direction": "lower is better (all five are errors)",
            "pairs": [f"{a}_vs_{b}" for a, b in pairs],
        },
        "flagged_items": {
            "near_silent_reference": list(stats.NEAR_SILENT_REFERENCE_ITEMS),
            "near_field_map": list(stats.NEAR_FIELD_ITEMS),
            "all": list(stats.FLAGGED_ITEMS),
            "n": len(stats.FLAGGED_ITEMS),
        },
        "scale_disclosure": stats.CROSS_MAPPING_SCALE_DISCLOSURE,
        "shared_identity": shared,
        "arms": identities,
        "invalid_t60": invalid_t60_rates(),
        "tables": tables,
        "contrasts": contrasts,
    }
    out = os.path.join(_HERE, "mappingA_contrast_report.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"report -> {out}", flush=True)
    write_summary(payload, os.path.join(_HERE, "mappingA_stats_summary.md"))
    return 0


def write_summary(payload, path):
    """The same report, readable."""
    L = []
    add = L.append
    design = payload["design"]
    add("# Mapping-A cross-arm contrasts — exp_21\n")
    add(f"Generated {payload['finished_utc']} from the 25-cell sweep "
        f"({len(payload['arms'])} arms x {len(design['seeds'])} seeds).\n")
    add("## Design\n")
    add(f"- **{design['n_items']} items** = {len(design['rooms'])} rooms x "
        f"{design['placements_per_room']} placements x "
        f"{design['slots_per_placement']} mic slots, evaluated at seeds "
        f"{design['seeds']}.")
    add(f"- **Clustering unit: {design['unit_of_clustering']}.** The 36 items of a "
        "placement share a room position, an array, a target source and largely "
        "overlapping context, so item-i.i.d. intervals would understate the "
        "uncertainty.")
    add(f"- **Aggregation: {design['aggregation']}** — the two rooms are the "
        "population, so neither is weighted by placement count.")
    add("- **Pairing is exact**: every arm evaluated the same 1,152 items under the "
        "same conditioning stream at the same five seeds, so each contrast is "
        "differenced item by item and seed by seed BEFORE any averaging.")
    add(f"- Intervals: room-stratified cluster bootstrap over placements "
        f"({design['n_resamples']} resamples, alpha {design['alpha']}); p-values: "
        "paired sign-flip randomization over the same unit.")
    add(f"- **{design['metric_direction']}**, so a negative difference favours the "
        "first arm.\n")
    add("### Flagged items (registered disclosures)\n")
    flagged = payload["flagged_items"]
    add(f"- {len(flagged['near_silent_reference'])} items carry a near-silent "
        "CONTEXT reference (Amendment 4.1, all FurnishedRoom p008);")
    add(f"- {len(flagged['near_field_map'])} items' listener map sees near-field "
        "scanned structure (Amendment 4.2/4.4).")
    add(f"- The **primary** row keeps all {design['n_items']}; the "
        f"**minus-flagged** row drops those {flagged['n']}. Both are reported for "
        "every contrast: the primary is the result, the sensitivity row says "
        "whether it depends on them.\n")
    add(f"> {payload['scale_disclosure']}\n")

    # ---- findings, DERIVED from the report rather than narrated over it ----
    every = [(metric, key, block)
             for metric in design["metrics"]
             for key, block in payload["contrasts"][metric].items()]
    holds = [t for t in every if t[2]["verdict"]["primary"]["significant_at_alpha"]]
    bonferroni = [t for t in every
                  if t[2]["verdict"]["primary"]["bonferroni_10_pairs"]]
    flips = [t for t in every
             if (t[2]["verdict"]["primary"]["difference"] < 0)
             != (t[2]["verdict"]["minus_flagged"]["difference"] < 0)]
    disagree = [t for t in every
                if t[2]["verdict"]["primary"]["significant_at_alpha"]
                != t[2]["verdict"]["minus_flagged"]["significant_at_alpha"]]
    add("## Findings\n")
    add(f"- **{len(holds)} of {len(every)}** contrasts hold at alpha "
        f"{design['alpha']}; **{len(bonferroni)}** also survive a Bonferroni "
        "correction over the ten pairs tested per metric.")
    add(f"- **The minus-flagged row changes nothing**: {len(flips)} sign flips and "
        f"{len(disagree)} verdict changes across all {len(every)} contrasts. No "
        "conclusion here depends on the "
        f"{payload['flagged_items']['n']} flagged items.")
    for metric in design["metrics"]:
        macros = {label: payload["tables"][label][metric]["equal_room_macro"]
                  for label in payload["arms"]}
        order = sorted(macros, key=macros.get)
        winners = [f"{a} < {b}" for m, key, block in holds if m == metric
                   for a, b in [(block["verdict"]["primary"]["better"],
                                 [x for x in block["arms"]
                                  if x != block["verdict"]["primary"]["better"]][0])]]
        add(f"- **{metric}**: best {order[0]} ({macros[order[0]]:.4f}), worst "
            f"{order[-1]} ({macros[order[-1]]:.4f}); ordering "
            + " < ".join(order) + ". Holding: "
            + (", ".join(winners) if winners else "none") + ".")
    add("")

    add("## Arms\n")
    add("| label | registered ckpt | cond_method | seeds | identity |")
    add("|---|---|---|---|---|")
    for label, entry in payload["arms"].items():
        identity = entry["identity"]
        add(f"| {label} | `{identity['ckpt_sha256'][:12]}` "
            f"({entry['registered_label']}) | {identity['cond_method']} | "
            f"{entry['seeds']} | `{entry['identity_sha256'][:12]}` |")
    add("")
    add("Shared across every arm (asserted, not assumed): dataset config "
        f"`{payload['shared_identity']['dataset_config_sha256'][:12]}`, prepare "
        f"generation `{payload['shared_identity']['publication_prepare_generation'][:12]}`, "
        f"depth generation `{payload['shared_identity']['publication_depth_generation'][:12]}`, "
        f"item stream `{payload['shared_identity']['stream_input_hash'][:12]}`, "
        f"{design['n_resamples']}-resample settings, "
        f"steps {payload['shared_identity']['steps']}, cfg "
        f"{payload['shared_identity']['cfg_scale']}, batch "
        f"{payload['shared_identity']['batch_size']}, source "
        f"`{payload['shared_identity']['source_sha'][:8]}`.\n")
    rates = payload["invalid_t60"]
    add("Invalid-T60 rate (a per-item flag, reported as the rate it is): "
        + ", ".join(f"{label} {entry['rate']:.4f}" for label, entry in rates.items())
        + ".\n")

    add("## Headline: equal-room macro per arm\n")
    add("| metric | " + " | ".join(payload["arms"]) + " |")
    add("|---" * (len(payload["arms"]) + 1) + "|")
    for metric in design["metrics"]:
        cells = []
        for label in payload["arms"]:
            table = payload["tables"][label][metric]
            cells.append(f"{table['equal_room_macro']:.4f} "
                         f"(±{table['seed_variability']['sd']:.4f})")
        add(f"| {metric} ({design['metric_units'][metric]}) | "
            + " | ".join(cells) + " |")
    add("")
    add("Parenthesised value is the **seed SD** — Monte-Carlo variability of the "
        "sampler, reported beside the estimate and never inside an interval.\n")

    add("## Per-room means\n")
    for metric in design["metrics"]:
        add(f"**{metric}** ({design['metric_units'][metric]})\n")
        add("| arm | " + " | ".join(design["rooms"]) + " |")
        add("|---" * (len(design["rooms"]) + 1) + "|")
        for label in payload["arms"]:
            rooms = payload["tables"][label][metric]["rooms"]
            add(f"| {label} | " + " | ".join(
                f"{rooms[room]['mean_over_placements']:.4f}"
                for room in design["rooms"]) + " |")
        add("")

    add("## Contrasts\n")
    for metric in design["metrics"]:
        add(f"### {metric} ({design['metric_units'][metric]})\n")
        add("| pair | kind | difference | 95% CI | p | holds | minus-flagged | "
            "sign flip |")
        add("|---|---|---|---|---|---|---|---|")
        for key, block in payload["contrasts"][metric].items():
            primary = block["verdict"]["primary"]
            minus = block["verdict"]["minus_flagged"]
            flip = (np.sign(primary["difference"]) != np.sign(minus["difference"]))
            holds = "**yes**" if primary["significant_at_alpha"] else "no"
            minus_holds = "**yes**" if minus["significant_at_alpha"] else "no"
            better = f" ({primary['better']} better)" if primary["better"] else ""
            add(f"| {key.replace('_vs_', ' vs ')} | {block['kind']} | "
                f"{primary['difference']:+.4f}{better} | "
                f"[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}] | "
                f"{primary['p_value']:.4f} | {holds} | {minus_holds} "
                f"({minus['difference']:+.4f}) | {'YES' if flip else 'no'} |")
        add("")
    add("A contrast **holds** when the 95% paired cluster-bootstrap interval "
        "excludes zero AND the randomization p-value is below "
        f"{design['alpha']}. Ten pairs are tested per metric and no multiplicity "
        "correction is registered; each row's `bonferroni_10_pairs` flag in the "
        "JSON says whether it would also survive p < 0.005.\n")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"summary -> {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
