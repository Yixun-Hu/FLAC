#!/usr/bin/env python
"""exp_18 loc_invert -- localization heatmaps and the HTML data extracts.

plan_loc_invert §2.7 asks for per-query maps of representative SHARP-SUCCESS,
AMBIGUOUS and FAILURE cases. "Representative" is the trap: hand-picking three
pretty maps is cherry-picking, so the cases are chosen by a PRE-REGISTERED rule
over the published rows, evaluated the same way for every run:

  sharp success -- top-1 correct, LARGEST top-2 margin of the displayed map
  ambiguous     -- SMALLEST top-2 margin, whatever the outcome
  failure       -- LARGEST localization error e_loc

three of each: the LITERAL extrema of each category, ties broken by query id.
No diversity policy is applied -- an earlier version preferred distinct rooms and
excluded queries already shown, which changed which cases were displayed and was
never registered (r8 review finding 1). A query that is extremal in two
categories is legitimately shown in both. The rule is a pure function of the
rows: rerunning it on the same file reproduces the same nine cases.

The displayed map is the plan's own visualization transform -- softmax(S / T)
over the candidate scores with T = the run's registered tau -- and it is used
for display and for the margin only; nothing here re-scores or re-aggregates
anything, and no number in the extracts is computed here (they are read from
published artifacts, with the source path recorded in every file).

Usage (both registered runs, from the repository root):

    python worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_heatmaps.py \\
        --rows outputs_loc/exp18/<R2 seed42 rows>.jsonl --run-label "R2 K_ctx=8 seed 42" \\
        --out-dir worklog/worklog_yixun/exp_18_loc_invert_claude/loc_invert_results_assets

    ... --extracts   # additionally writes the HTML data extracts
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:                       # the decoders live in the repo
    sys.path.insert(0, _REPO_ROOT)

#: the three pre-registered case kinds, in the order they are selected.
CASE_KINDS = ("sharp_success", "ambiguous", "failure")
CASES_PER_KIND = 3

#: the plan's §2.7 display transform: softmax(S / T) with T = the run's tau.
#: It is a VISUALIZATION normalization, never a calibrated posterior.
DISPLAY_TEMPERATURE_SOURCE = "the run's registered tau (rows carry it per query)"

SELECTION_RULE = (
    "sharp_success = correct top-1 with the largest top-2 margin of softmax(S/tau); "
    "ambiguous = smallest top-2 margin; failure = largest e_loc; three of each, the literal "
    "category extrema, ties by query_id ascending; no diversity or exclusion policy"
)

#: verbatim disclosure the HTML must carry beside any sharp-success map.
SATURATION_CAVEAT = (
    "at T=0.02 sharp-success margins saturate within 1e-16-1e-13 of 1; ordering among them "
    "reflects tiny numerical differences; they represent a large class (1,150 queries > 0.9999)"
)
#: a margin above this is reported as saturated rather than as a ranking.
SATURATION_THRESHOLD = 0.9999

#: the data extracts the HTML consumes (no plotting, no new statistics).
EXTRACT_FILES = (
    "extract_two_regime.json",
    "extract_delta.json",
    "extract_families.json",
    "extract_per_room.json",
    "extract_conclusions.json",
    "extract_timeline.json",
)


# --------------------------------------------------------------------------- #
# the display transform and the margin the rule is defined on
# --------------------------------------------------------------------------- #
def display_scores(scores, temperature):
    """``softmax(S / T)`` over the candidates -- the plan's display map."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"the display temperature must be > 0, got {temperature}")
    shifted = scores / temperature
    shifted = shifted - shifted.max()                # overflow-safe, same softmax
    weights = np.exp(shifted)
    return weights / weights.sum()


def top2_margin(probabilities):
    """Gap between the largest and second-largest displayed probabilities."""
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64).reshape(-1))[::-1]
    if ordered.size == 1:
        return float(ordered[0])
    return float(ordered[0] - ordered[1])


def is_saturated(margin):
    """Whether the displayed margin is in the saturated band (r8 review F3)."""
    return float(margin) > SATURATION_THRESHOLD


def format_margin(margin):
    """Print a saturated margin as its distance from 1, never as "1"."""
    margin = float(margin)
    if is_saturated(margin):
        gap = 1.0 - margin
        return "1" if gap <= 0.0 else f"1 - {gap:.1e}"
    return f"{margin:.4g}"


def case_caption(record, run_label=""):
    """The caption the HTML shows under a map -- with the caveat where it applies."""
    kind = str(record.get("kind") or "").replace("_", " ")
    outcome = "correct" if record["correct"] else "wrong"
    caption = (f"{kind} -- {record['room_id']}, {run_label}: top-2 margin "
               f"{format_margin(record['margin'])}, e_loc {record['e_loc']:.2f} m "
               f"({outcome} top-1)").strip()
    if is_saturated(record["margin"]):
        caption += f". Caveat: {SATURATION_CAVEAT}"
    return caption


def _receiver_world(row):
    """The receiver position in world coordinates, or ``None``.

    The rows carry each candidate in both frames and the camera frame is the
    receiver's, so ``world - cam`` is the receiver -- but only if that offset is
    genuinely constant across the candidates (a pure translation). If it is not,
    the marker is dropped rather than drawn somewhere invented.
    """
    world = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
    cam = np.asarray(row.get("candidate_xyz_cam") or [], dtype=np.float64)
    if cam.shape != world.shape or not cam.size:
        return None
    offsets = world - cam
    if float(np.abs(offsets - offsets[0]).max()) > 1e-6:
        return None
    return [float(v) for v in offsets[0]]


def case_record(row):
    """Everything one map needs, plus the two rule values, from one row."""
    from src.localization.reaggregate import decode_sims

    scores = decode_sims([list(row["scores_hex"])]).numpy()[0].astype(np.float64)
    temperature = float(row.get("tau") or 0.02)
    probabilities = display_scores(scores, temperature)
    return {
        "query_id": str(row["query_id"]), "room_id": str(row["room_id"]),
        "relpath": row.get("relpath"),
        "margin": top2_margin(probabilities),
        "e_loc": float(row["e_loc"]),
        "correct": bool(int(row["pred_index"]) == int(row["gt_index"])),
        "pred_index": int(row["pred_index"]), "gt_index": int(row["gt_index"]),
        "n_candidates": int(len(scores)), "temperature": temperature,
        "scores": [float(v) for v in scores],
        "probabilities": [float(v) for v in probabilities],
        "candidate_xyz_world": [[float(v) for v in xyz]
                                for xyz in row["candidate_xyz_world"]],
        "gt_xyz_world": [float(v) for v in row["gt_xyz_world"]],
        "pred_xyz_world": [float(v) for v in (row.get("pred_xyz_world")
                                              or row["candidate_xyz_world"][
                                                  int(row["pred_index"])])],
        "receiver_xyz_world": _receiver_world(row),
        "receiver_node": row.get("receiver_node"),
        "context_member": [bool(v) for v in (row.get("context_member") or [])],
    }


# --------------------------------------------------------------------------- #
# the pre-registered selection
# --------------------------------------------------------------------------- #
#: (kind -> (filter, sort key)); every key ends in query_id, so ties are broken
#: deterministically and the rule cannot depend on the order rows arrive in.
_ORDERINGS = {
    "sharp_success": (lambda r: r["correct"], lambda r: (-r["margin"], r["query_id"])),
    "ambiguous": (lambda r: True, lambda r: (r["margin"], r["query_id"])),
    "failure": (lambda r: True, lambda r: (-r["e_loc"], r["query_id"])),
}


def select_cases(records, per_kind=CASES_PER_KIND):
    """The nine registered cases: the LITERAL extrema of each category.

    Each category is sorted by its registered key and the first ``per_kind``
    records are taken -- nothing else. The categories are independent, so a query
    that is both the narrowest margin and the worst error appears in both.
    """
    records = list(records)
    cases = {}
    for kind in CASE_KINDS:
        keep, key = _ORDERINGS[kind]
        ordered = sorted((r for r in records if keep(r)), key=key)
        chosen = ordered[:per_kind]
        if len(chosen) < per_kind:
            raise ValueError(
                f"only {len(chosen)} {kind!r} cases are available; the rule needs {per_kind} "
                f"(the run has {len(records)} queries)")
        cases[kind] = [dict(record, kind=kind, rank=rank)
                       for rank, record in enumerate(chosen, start=1)]
    return cases


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _depth_silhouette(record):
    """The room's depth-map silhouette, when it is trivially loadable.

    exp_18's rows do not carry a room outline, and the depth maps are per-view
    tensors in the dataset rather than a floor plan -- reconstructing one would
    be a projection step of its own, well outside a plotting script. The maps
    therefore label the CANDIDATE EXTENT instead, which is the region the
    protocol actually searched, and this hook documents the decision.
    """
    return None


def render_case(record, out_path, kind=None, run_label="", dpi=150):
    """One top-down map: candidates by displayed probability, GT, prediction."""
    import matplotlib
    matplotlib.use("Agg")                       # headless by construction
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    world = np.asarray(record["candidate_xyz_world"], dtype=np.float64)
    probabilities = np.asarray(record["probabilities"], dtype=np.float64)
    gt = np.asarray(record["gt_xyz_world"], dtype=np.float64)
    pred = np.asarray(record["pred_xyz_world"], dtype=np.float64)
    receiver = record.get("receiver_xyz_world")
    kind = kind or record.get("kind") or ""

    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    silhouette = _depth_silhouette(record)
    if silhouette is not None:                  # documented: not available here
        axis.plot(silhouette[:, 0], silhouette[:, 1], color="0.75", linewidth=1.0)

    # the region the protocol searched, labelled as such (never a room outline)
    low, high = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
    pad = 0.08 * max(float((high - low).max()), 1.0)
    # the drawn window must also contain the receiver, which often sits outside
    # the candidate extent -- clipping it would silently drop the marker
    window = np.vstack([world[:, :2]] + ([np.asarray(receiver[:2])[None, :]]
                                         if receiver is not None else []))
    window_low, window_high = window.min(axis=0), window.max(axis=0)
    axis.add_patch(plt.Rectangle(low - pad, *(high - low + 2 * pad), fill=False,
                                 linestyle=(0, (4, 3)), edgecolor="0.6", linewidth=1.0))
    axis.annotate("candidate extent", xy=(low[0] - pad, high[1] + pad), xytext=(0, 4),
                  textcoords="offset points", color="0.45", fontsize=8)

    scatter = axis.scatter(world[:, 0], world[:, 1], c=probabilities, cmap="viridis",
                           s=170, vmin=0.0, vmax=max(float(probabilities.max()), 1e-6),
                           edgecolors="0.25", linewidths=0.6, zorder=3)
    for index, (x, y) in enumerate(world[:, :2]):     # z is annotated, not projected
        axis.annotate(f"z={world[index, 2]:.2f}", xy=(x, y), xytext=(0, -13),
                      textcoords="offset points", ha="center", fontsize=6.5, color="0.3")

    axis.scatter([gt[0]], [gt[1]], marker="*", s=340, facecolor="none",
                 edgecolors="crimson", linewidths=1.6, zorder=5)
    axis.scatter([pred[0]], [pred[1]], marker="o", s=430, facecolor="none",
                 edgecolors="darkorange", linewidths=2.0, zorder=4)
    if receiver is not None:
        axis.scatter([receiver[0]], [receiver[1]], marker="^", s=150, color="#1f77b4",
                     edgecolors="0.15", linewidths=0.6, zorder=5)

    # equal aspect with a fixed data window: the axes BOX shrinks instead of the
    # data range growing, so the map is not padded with empty floor
    axis.set_xlim(window_low[0] - 2 * pad, window_high[0] + 2 * pad)
    axis.set_ylim(window_low[1] - 2 * pad, window_high[1] + 2 * pad)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_title(f"{kind.replace('_', ' ')} -- {record['room_id']}\n"
                   f"{run_label}   margin={format_margin(record['margin'])}   "
                   f"e_loc={record['e_loc']:.2f} m", fontsize=10)
    bar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label(f"softmax(S / T), T = {record['temperature']:g} (display only)",
                  fontsize=8)

    handles = [Line2D([], [], linestyle="", marker="*", markerfacecolor="none",
                      markeredgecolor="crimson", markersize=14, label="ground truth"),
               Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                      markeredgecolor="darkorange", markersize=13, label="prediction")]
    if receiver is not None:
        handles.append(Line2D([], [], linestyle="", marker="^", color="#1f77b4",
                              markersize=10, label="receiver"))
    # the legend lives OUTSIDE the axes: inside it covered the GT star and the
    # receiver whenever they sat in the corner it picked
    saturated = is_saturated(record["margin"])
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
                  frameon=False, bbox_to_anchor=(0.5, 0.075 if saturated else 0.0))

    if saturated:
        import textwrap
        figure.text(0.5, 0.012, "\n".join(textwrap.wrap(SATURATION_CAVEAT, 96)),
                    ha="center", va="bottom", fontsize=6, color="0.35")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    figure.tight_layout(rect=(0, 0.16 if saturated else 0.06, 1, 1))
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


def _file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def gallery_manifest(cases, run_label, rows_path, out_dir, rows_sha256=None, pngs=None,
                     records=None):
    """What was drawn, from which rows, under which rule -- and with what caveat."""
    from datetime import datetime, timezone

    pngs = pngs or {}
    entries = {}
    for kind, chosen in cases.items():
        entries[kind] = [{
            "rank": record.get("rank"), "query_id": record["query_id"],
            "room_id": record["room_id"], "relpath": record.get("relpath"),
            "margin": record["margin"], "margin_display": format_margin(record["margin"]),
            "saturated": is_saturated(record["margin"]),
            "e_loc": record["e_loc"],
            "correct": record["correct"], "n_candidates": record["n_candidates"],
            "pred_index": record["pred_index"], "gt_index": record["gt_index"],
            "temperature": record["temperature"],
            "caption": case_caption(record, run_label),
            "png": pngs.get(record["query_id"], _png_name(kind, record)),
        } for record in chosen]
    saturated = (sum(1 for r in records if is_saturated(r["margin"]))
                 if records is not None else None)
    return {
        "run_label": run_label, "rows_path": str(rows_path),
        "rows_sha256": rows_sha256,
        "selection_rule": SELECTION_RULE,
        "saturation_caveat": SATURATION_CAVEAT,
        "saturation_threshold": SATURATION_THRESHOLD,
        "n_queries_above_saturation_threshold": saturated,
        "n_queries": None if records is None else len(records),
        "display_temperature": DISPLAY_TEMPERATURE_SOURCE,
        "depth_silhouette": ("not drawn: the dataset ships per-view depth tensors, not a "
                             "floor plan; the maps label the candidate extent instead"),
        "cases_per_kind": CASES_PER_KIND, "kinds": list(CASE_KINDS),
        "out_dir": str(out_dir), "cases": entries,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _png_name(kind, record):
    stem = record["room_id"].replace("/", "-")
    return f"{kind}_{record.get('rank', 0)}_{stem}.png"


# --------------------------------------------------------------------------- #
# the HTML data extracts -- read from published artifacts, never recomputed
# --------------------------------------------------------------------------- #
def _summary_of(payload):
    return payload.get("summary", payload)


def extract_two_regime(k8_summary, k1_summary, k8_source="", k1_source=""):
    """(a) {K_ctx=8, K_ctx=1} x {FLAC+AGREE, chance, retrieval control}.

    Straight out of each run's published summary: pooled and equal-room macro
    top-1 for the three arms, plus the clustered CI on the campaign primary and
    the paired test against the retrieval control that the summary already
    carries. No statistic is computed here.
    """
    regimes = {}
    for label, payload, source in (("K8", k8_summary, k8_source),
                                   ("K1", k1_summary, k1_source)):
        summary = _summary_of(payload)
        control = summary["controls"]["nearest_context_masked"]
        chance = summary["baselines"]["context_conditioned"]
        regimes[label] = {
            "source": source,
            "n_queries": summary["n_queries"], "n_rooms": summary["n_rooms"],
            "context_member_prediction_rate": summary.get("context_member_prediction_rate"),
            "arms": {
                "flac": {"pooled_top1": summary["flac"]["pooled"]["top1"],
                         "macro_top1": summary["flac"]["macro"]["top1"],
                         "pooled_median_e_loc": summary["flac"]["pooled"]["median_e_loc"],
                         "macro_mean_e_loc": summary["flac"]["macro"].get(
                             "mean_of_room_means")},
                "chance": {"pooled_top1": chance["pooled"]["top1"],
                           "macro_top1": chance["macro"]["top1"]},
                "retrieval": {"pooled_top1": control["pooled"]["top1"],
                              "macro_top1": control["macro"]["top1"]},
            },
            "clustered_ci_median_e_loc": summary["statistics"]["clustered_ci"],
            "paired_vs_retrieval": summary["statistics"]["paired_vs_nearest_context_masked"],
            "paired_vs_chance": summary["statistics"]["paired_vs_context_conditioned"],
        }
    return {
        "regimes": regimes,
        "convention": ("pooled = over all queries; macro = equal-room mean, the convention "
                       "the campaign's headline top-1 and the 0.689 reference use"),
        "note": ("top-1 confidence intervals are not published by the runs; the CI shown is "
                 "the room-clustered interval on the campaign primary (pooled median e_loc)"),
        "source": "the registered R2 / R2b summary JSONs (seed 42)",
    }


def extract_delta(calibration, report, calibration_source="", report_source=""):
    """(b) the seen Delta_max grid and the unseen Delta = 0 collapse."""
    grid = [{"delta_max": int(point["delta_max"]), "seen_dev_top1": float(point["top1"])}
            for point in calibration["delta_max"]["grid"]]
    table = report["seed_table"]
    collapse = {}
    for family, zero in (("m1", "m1_delta0"), ("m5", "m5_delta0")):
        if family in table and zero in table:
            collapse[family] = {
                "registered_macro_top1": table[family]["macro_top1"]["mean"],
                "registered_pooled_top1": table[family]["top1"]["mean"],
                "delta0_macro_top1": table[zero]["macro_top1"]["mean"],
                "delta0_pooled_top1": table[zero]["top1"]["mean"],
                "macro_drop": (table[family]["macro_top1"]["mean"]
                               - table[zero]["macro_top1"]["mean"]),
            }
    return {
        "seen_grid": grid,
        "registered_delta_max": int(calibration["delta_max"]["selected"]),
        "grid_note": ("selected on the R1 SEEN prefix by dev top-1, ties to the smallest, "
                      "and frozen before any unseen pass"),
        "unseen_collapse": collapse,
        "collapse_note": ("Delta = 0 is the mandated alignment-sensitivity row: the same "
                          "family with the alignment search switched off, on unseen data"),
        "source": {"seen_grid": calibration_source, "collapse": report_source},
    }


#: how the promoted report labels a family; mirrored here so the HTML can group.
def extract_families(report, report_source=""):
    """(c) the R4 family table: primaries, secondaries, oracle ceilings."""
    table = report["seed_table"]
    families = {}
    for family, columns in table.items():
        families[family] = {
            "kind": columns["label"],
            "pooled_top1": columns["top1"]["mean"], "sd": columns["top1"]["sd"],
            "per_seed_top1": columns["top1"]["per_seed"],
            "macro_top1": columns["macro_top1"]["mean"],
            "median_e_loc": columns["median_e_loc"]["mean"],
            "mean_e_loc": columns["mean_e_loc"]["mean"],
            "success_0.5": columns["success_0.5"]["mean"],
            "success_1.0": columns["success_1.0"]["mean"],
            "mrr": columns["mrr"]["mean"],
            "matched_control_pooled_top1": columns["retrieval_masked_top1"]["mean"],
            "matched_control_macro_top1": columns["retrieval_masked_macro_top1"]["mean"],
            "oracle_top1": columns["oracle_top1"]["mean"],
            "oracle_macro_top1": columns["oracle_macro_top1"]["mean"],
            "context_member_rate": columns["context_member_rate"]["mean"],
            "power_mean": columns["power_mean"]["mean"],
        }
    # The paired tests are PER SEED. Naming the seed the shown block came from is
    # not optional -- the adjusted p-values differ across seeds (r8 review F2) --
    # so the inference seed is labelled and every seed's block travels with it.
    seed_blocks = report.get("seeds") or [{}]
    first = seed_blocks[0]
    per_seed_holm, adjusted = {}, {}
    for block in seed_blocks:
        seed = str(block.get("seed"))
        holm = (block.get("holm") or {}).get("top1")
        if holm is None:
            continue
        per_seed_holm[seed] = holm
        for test in holm.get("tests", []):
            adjusted.setdefault(test["label"], {})[seed] = test["p_adjusted"]
    return {
        "families": families,
        "groups": report.get("families", {}),
        "primary_tests": {
            "seed": first.get("seed"),
            "note": "one seed's paired tests; every seed's Holm block is in holm_per_seed",
            "tests": [{"label": c["label"], "top1_delta": c["top1_delta"],
                       "p_top1": c["top1"]["p_value"],
                       "median_e_loc_delta": c["e_loc"]["point"]}
                      for c in first.get("primary_comparisons", [])]},
        "holm": dict((first.get("holm") or {}).get("top1") or {}, seed=first.get("seed")),
        "holm_per_seed": per_seed_holm,
        "adjusted_p_per_seed": adjusted,
        "seeds": report["provenance"]["seeds"],
        "references": report["provenance"]["references"],
        "status": report["provenance"]["status"],
        "source": report_source or "the promoted R4 metrics report JSON",
    }


def extract_per_room(k8_summary, k1_summary, report, family="m2", sources=None):
    """(d) per-room top-1 for AGREE K_ctx=8, AGREE K_ctx=1 and one R4 family."""
    k8 = _summary_of(k8_summary)["flac"]["per_room"]
    k1 = _summary_of(k1_summary)["flac"]["per_room"]
    seed_block = (report.get("seeds") or [{}])[0]
    metric = ((seed_block.get("families") or {}).get(family) or {}).get("primary", {}) \
        .get("per_room", {})
    rooms = {}
    for room in sorted(set(k8) | set(k1) | set(metric)):
        rooms[room] = {
            "agree_k8": (k8.get(room) or {}).get("top1"),
            "agree_k1": (k1.get(room) or {}).get("top1"),
            f"{family}_k8": (metric.get(room) or {}).get("top1"),
            "n_queries": ((k8.get(room) or {}).get("n_queries")
                          or (k1.get(room) or {}).get("n_queries")),
        }
    return {
        "rooms": rooms,
        "arms": {"agree_k8": "FLAC + AGREE, K_ctx = 8 (R2 seed 42)",
                 "agree_k1": "FLAC + AGREE, K_ctx = 1 (R2b seed 42)",
                 f"{family}_k8": f"R4 {family}, K_ctx = 8 (seed 42)"},
        "family": family,
        "source": sources or "R2 / R2b summaries and the promoted R4 report (seed 42)",
    }


def extract_conclusions(report, report_source=""):
    """(e) the six conclusion answers, verbatim from the promoted report."""
    return {
        "questions": report["conclusions"],
        "verbatim": True,
        "status": report["provenance"]["status"],
        "seeds": report["provenance"]["seeds"],
        "source": report_source or "the promoted R4 metrics report JSON (conclusions block)",
    }


def campaign_timeline():
    """(f) the campaign's facts, hardcoded from the COMMITTED record.

    Every figure below is quoted from a committed file; the citation is the file
    and the line that carries it, so a reader can check the number without
    trusting this script:

      * loc_invert_results.md:4       R-1a readback gate, 6,337 / 17 rooms
      * loc_invert_results.md:6-7     R-1b oracle + baselines, identity gate
      * loc_invert_results.md:21-22   R0 probe + scorer noise
      * loc_invert_results.md:24-28   R1 dev-tune, tau sweep, R2 registration
      * loc_invert_results.md:30-48   R2 seeds 42/43/44 headline
      * loc_invert_results.md:50-64   R2b seeds 42/43/44 (K_ctx = 1)
      * loc_invert_command.md         the exact launch commands of every run
      * commits_loc_invert.md         the per-round commit ledger (r1 ... r4m6)
    """
    return {
        "runs": [
            {"label": "R-1a readback gate", "date": "2026-08-19",
             "detail": "6,337 unseen queries / 17 rooms; split digests 3/3 PASS; 169 wavs "
                       "decoded; 1 registered warning (LRH_idx_30 S10 metadata-only)"},
            {"label": "R-1b identity oracle + baselines", "date": "2026-08-19",
             "detail": "identity oracle 1.000; context-conditioned chance macro 0.490; "
                       "masked nearest-context control macro 0.689 (pooled 0.6317)"},
            {"label": "R0 probe + scorer noise", "date": "2026-08-19",
             "detail": "1.32 s/query; mean readout removes ~7e-5 cosine noise"},
            {"label": "R1 dev-tune + tau registration", "date": "2026-08-20",
             "detail": "1,194-query seen prefix; tau = 0.02 registered from a flat 28-config "
                       "sweep; R2 registration manifest committed"},
            {"label": "R2 registered unseen headline (seeds 42/43/44)", "date": "2026-08-20",
             "detail": "macro top-1 0.5007 +- 0.0008 over 6,337 queries; pooled 0.5618; "
                       "context-member prediction rate 0.376"},
            {"label": "R2b K_ctx = 1 (seeds 42/43/44)", "date": "2026-08-20",
             "detail": "FLAC 0.5029 +- 0.0032 vs retrieval 0.1079 +- 0.0017 -- the "
                       "sparse-context reversal"},
            {"label": "R3 constant-source wiring control", "date": "2026-08-21",
             "detail": "conditioning proven load-bearing"},
            {"label": "R4 seen calibration + registration", "date": "2026-08-20",
             "detail": "1,194-query replay; delta_max = 8 selected on the seen grid; m4 "
                       "mu/sigma frozen; metric manifest committed before any unseen pass"},
            {"label": "R4 unseen replay + metrics (seeds 42/43/44)", "date": "2026-08-20",
             "detail": "6,337 queries per seed, all five families plus the declared "
                       "secondaries and the Delta = 0 rows"},
            {"label": "R4 oracle control (seeds 42/43/44)", "date": "2026-08-21",
             "detail": "measured-candidate ceiling, each pass bound to its paired replay's "
                       "context-stream digest"},
        ],
        "gates": [
            "identity gate: every query scored in the registered order, 6,337 per seed",
            "split digest pinning: file, query count, room-node map",
            "registration gate: full-hex sha, in-repo manifest, byte-identical, ancestor of HEAD",
            "metric registration: constants, source blob and R2 manifest digests verified",
            "publication: nothing gets a final name until every end gate passes",
            "context binding: the unseen control refuses without its paired replay digest",
        ],
        # r8 review finding 4: a hardcoded count is only sourced if the record it
        # cites actually contains it. This one is quoted from the committed
        # ledger line written in the same round, and a test greps for it.
        "tests": {
            "suite_total": 2688, "suite_skipped": 10, "localization_files": 7,
            "source_file": ("worklog/worklog_yixun/exp_18_loc_invert_claude/"
                            "commits_loc_invert.md"),
            "source_quote": "2688 passed, 10 skipped, 1 pre-existing unrelated failure",
            "note": ("full repository suite at the end of r8b; the one failure is exp_11 "
                     "registry drift owned by exp_15, not exp_18"),
        },
        "rounds": ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "R4-r1", "R4-r2",
                   "r4m3", "r4m4", "r4m5", "r4m6", "r8", "r8b"],
        "source": "committed record: loc_invert_results.md, loc_invert_command.md, "
                  "commits_loc_invert.md (line citations in the function's docstring)",
    }


def write_extracts(out_dir, k8_summary, k1_summary, calibration, report, sources=None):
    """Write the six HTML extracts; returns the paths written."""
    sources = sources or {}
    os.makedirs(out_dir, exist_ok=True)
    payloads = {
        "extract_two_regime.json": extract_two_regime(
            k8_summary, k1_summary, sources.get("k8", ""), sources.get("k1", "")),
        "extract_delta.json": extract_delta(
            calibration, report, sources.get("calibration", ""), sources.get("report", "")),
        "extract_families.json": extract_families(report, sources.get("report", "")),
        "extract_per_room.json": extract_per_room(k8_summary, k1_summary, report),
        "extract_conclusions.json": extract_conclusions(report, sources.get("report", "")),
        "extract_timeline.json": campaign_timeline(),
    }
    written = []
    for name, payload in payloads.items():
        path = os.path.join(out_dir, name)
        with open(path + ".partial", "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(path + ".partial", path)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def render_run(rows_path, run_label, out_dir, slug, dpi=150):
    """Select, draw and manifest one run's nine cases."""
    records = []
    with open(rows_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(case_record(json.loads(line)))
    cases = select_cases(records)
    os.makedirs(out_dir, exist_ok=True)
    pngs = {}
    for kind, chosen in cases.items():
        for record in chosen:
            name = f"{slug}_{_png_name(kind, record)}"
            render_case(record, os.path.join(out_dir, name), kind=kind, run_label=run_label,
                        dpi=dpi)
            pngs[record["query_id"]] = name
    manifest = gallery_manifest(cases, run_label, rows_path, out_dir,
                                rows_sha256=_file_sha256(rows_path), pngs=pngs,
                                records=records)
    manifest_path = os.path.join(out_dir, f"{slug}_gallery.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[heatmaps] {run_label}: {len(records)} queries -> 9 cases, {manifest_path}")
    for kind in CASE_KINDS:
        for entry in manifest["cases"][kind]:
            print(f"    {kind:14s} #{entry['rank']} {entry['query_id']}  "
                  f"margin={entry['margin_display']}  e_loc={entry['e_loc']:.3f}  "
                  f"correct={entry['correct']}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", help="a published rows JSONL to draw")
    parser.add_argument("--run-label", default="", help="how the run is named on the maps")
    parser.add_argument("--slug", default=None, help="PNG filename prefix")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--extracts", action="store_true",
                        help="also write the HTML data extracts")
    parser.add_argument("--k8-summary", default=None)
    parser.add_argument("--k1-summary", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    if args.rows:
        slug = args.slug or os.path.basename(args.rows).split("_")[1]
        render_run(args.rows, args.run_label or slug, args.out_dir, slug, dpi=args.dpi)

    if args.extracts:
        for flag in ("k8_summary", "k1_summary", "calibration", "report"):
            if not getattr(args, flag):
                parser.error(f"--extracts needs --{flag.replace('_', '-')}")
        written = write_extracts(
            args.out_dir,
            k8_summary=json.load(open(args.k8_summary)),
            k1_summary=json.load(open(args.k1_summary)),
            calibration=json.load(open(args.calibration)),
            report=json.load(open(args.report)),
            sources={"k8": args.k8_summary, "k1": args.k1_summary,
                     "calibration": args.calibration, "report": args.report})
        for path in written:
            print(f"[extract] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
