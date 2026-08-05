#!/usr/bin/env python
"""exp_11 P0 profiling collector — manifest-bound ``P0RESULT`` -> ``p0_report_<runid>.md``.

One 30-step job per (cell, workers). ``p0_runner``'s callback timestamps completed
optimizer steps 10 and 30 inside that single fit, so

    steps/s = (30 - 10) / (t30_mono - t10_mono)

contains no startup, rendezvous or teardown time.

Provenance (round-2 review B4 + re-review B2/B3): a row is admitted only if its
run id, commit sha, job id, cell, config sha256 AND the exact execution shape
recorded by the submitter — ``maxsteps``, ``mb``, ``ngpu``, ``workers`` — all
match the manifest row; the poller CSV must exist, hash-match, and show complete
in-window ticks with finite utilisation and power for every allocated UUID.
Anything else makes that cell INVALID. Every expected row is reported
(PENDING/MISSING/MALFORMED/INVALID/OOM/FAILED/OK) and a short-of-all-OK run
withholds the derived tables and exits nonzero.

Attribution (re-review B6): the per-orbit-pass fit is over the EXACT
{FA1, C4L, C8} set. FA1 is ``fa_invariant`` with ``frame_avg_angles=[0.0]`` — the
same cylindrical pose path and dispatch as C4L/C8 but exactly ONE ViT pass — so
the slope is the cost of an ADDITIONAL orbit pass and the intercept is the fa
base step cost (still reported as an *unattributed residual*, not a bottleneck
diagnosis). Canonical VAN is reported as a separate vanilla-vs-FA1 contrast: it
uses the ordinary Cartesian pose path, so it must never sit inside the fit.

The manifest declares a MODE: only ``matrix`` promises an attribution fit;
``spot`` and ``workers`` collections succeed on their own rows (B1). Usage:

    python p0_collect.py --manifest p0_manifest_<runid>.txt [--dir .] [--out ...]
"""
import argparse
import glob
import hashlib
import math
import os
import sys

WINDOW_LO, WINDOW_HI = 10, 30
MATRIX_WORKERS = 6          # the worker count every matrix/spot cell must run with

# ViT forward passes per training step (orbit size).
N_ORBIT_PASSES = {"FA1": 1, "C4L": 4, "C8": 8, "C16": 16, "C32": 32, "CKPT4": 4, "VAN": 1}
# The fit needs exactly these three families at a rung — no subsets, and never
# VAN (different pose path => structurally confounded, re-review B6).
FIT_FAMILIES = ("FA1", "C4L", "C8")
FAMILY_ORDER = ("VAN", "FA1", "C4L", "C8", "C16", "C32", "CKPT4")
MODES = ("matrix", "spot", "workers")

_INT_FIELDS = ("jobid", "maxsteps", "ngpu", "mb", "workers", "rc", "peak_overall_mib", "valid")
_FLOAT_FIELDS = ("wall_fit", "t10", "t30", "t10_mono", "t30_mono")
_STR_FIELDS = ("runid", "sha", "cell", "config_sha", "peak_per_uuid", "vram_csv",
               "pollcsv_sha256")
_REQUIRED = _INT_FIELDS + _FLOAT_FIELDS + _STR_FIELDS

_OK_STATUSES = ("OK",)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_peak_per_uuid(text):
    """``'GPU-a:123,GPU-b:456'`` -> ``{'GPU-a': 123, 'GPU-b': 456}``."""
    if text in ("", "-", "none"):
        return {}
    out = {}
    for item in text.split(","):
        uuid, _, mib = item.partition(":")
        if not uuid or not mib.isdigit():
            raise ValueError(f"malformed peak_per_uuid entry {item!r}")
        out[uuid] = int(mib)
    return out


def parse_p0_line(line):
    """Parse one ``P0RESULT`` line into a typed row dict.

    ``None`` for a non-P0RESULT line; ``ValueError`` naming the offending key for
    a missing field, a non-numeric or NON-FINITE number, a negative peak, a
    ``valid`` outside {0, 1} or a non-positive ``wall_fit`` (a job that ran
    cannot take zero wall time)."""
    fields = line.strip().split()
    if not fields or fields[0] != "P0RESULT":
        return None
    row = {}
    for token in fields[1:]:
        key, sep, value = token.partition("=")
        if sep:
            row[key] = value
    missing = [k for k in _REQUIRED if k not in row]
    if missing:
        raise ValueError(f"P0RESULT line missing field(s) {missing}: {line.strip()!r}")
    for key in _INT_FIELDS:
        try:
            row[key] = int(row[key])
        except ValueError:
            raise ValueError(f"P0RESULT {key}={row[key]!r} is not an integer")
    for key in _FLOAT_FIELDS:
        try:
            row[key] = float(row[key])
        except ValueError:
            raise ValueError(f"P0RESULT {key}={row[key]!r} is not a number")
        if not math.isfinite(row[key]):
            raise ValueError(f"P0RESULT {key} is non-finite ({row[key]})")
    if row["peak_overall_mib"] < 0:
        raise ValueError(f"P0RESULT peak_overall_mib={row['peak_overall_mib']} is negative")
    if row["valid"] not in (0, 1):
        raise ValueError(f"P0RESULT valid={row['valid']} must be 0 or 1")
    if row["wall_fit"] <= 0:
        raise ValueError(f"P0RESULT wall_fit={row['wall_fit']} must be positive")
    row["peak_per_uuid"] = parse_peak_per_uuid(row["peak_per_uuid"])
    row["key"] = (row["cell"], row["workers"])
    return row


def parse_manifest(text):
    """Parse the submitter's manifest.

    Returns ``{runid, sha, mode, expected: [row, ...]}`` where each expected row
    is keyed ``(cell, workers)`` and carries the FULL execution shape the job was
    submitted with (maxsteps, mb, ngpu, workers, time limit) — the collector
    compares against these, never against something reconstructed from a label."""
    runid = sha = mode = None
    expected, seen = [], set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "runid" and len(parts) == 2:
            runid = parts[1]
        elif parts[0] == "sha" and len(parts) == 2:
            sha = parts[1]
        elif parts[0] == "mode" and len(parts) == 2:
            mode = parts[1]
        elif parts[0] == "cell" and len(parts) == 9:
            cell, maxsteps, jobid, config_sha, mb, ngpu, workers, timelimit = parts[1:]
            key = (cell, int(workers))
            if key in seen:
                raise ValueError(f"manifest lists {key} more than once")
            seen.add(key)
            expected.append({
                "cell": cell, "maxsteps": int(maxsteps),
                "jobid": int(jobid) if jobid.isdigit() else jobid,
                "config_sha": config_sha, "mb": int(mb), "ngpu": int(ngpu),
                "workers": int(workers), "timelimit": timelimit, "key": key,
            })
        elif parts[0] == "cell":
            raise ValueError(f"manifest cell row has {len(parts)} fields, expected 9: {line!r}")
    if not runid or not sha:
        raise ValueError("manifest must declare both 'runid' and 'sha'")
    if mode not in MODES:
        raise ValueError(f"manifest mode {mode!r} must be one of {MODES}")
    if not expected:
        raise ValueError("manifest declares no expected cells")
    return {"runid": runid, "sha": sha, "mode": mode, "expected": expected}


def load_manifest(path):
    with open(path, "r") as fh:
        return parse_manifest(fh.read())


def family_of(cell):
    """``'C4L_32x2'`` -> ``'C4L'``."""
    return cell.split("_")[0]


def rung_of(cell):
    """``'C4L_32x2'`` -> ``('32x2', 32, 2)``; ``(None, None, None)`` if unparsable."""
    _, _, rung = cell.partition("_")
    mb, _, ngpu = rung.partition("x")
    if mb.isdigit() and ngpu.isdigit():
        return rung, int(mb), int(ngpu)
    return None, None, None


def _cell_key(cell):
    fam = family_of(cell)
    rank = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
    _, _, ngpu = rung_of(cell)
    return (rank, fam, ngpu if ngpu is not None else 0, cell)


def _sort_key(summary):
    return (_cell_key(summary["cell"]), -summary.get("workers", 0))


# --------------------------------------------------------------------------- #
# manifest binding
# --------------------------------------------------------------------------- #
def admit_rows(rows, manifest):
    """Keep only rows matching the manifest exactly (identity AND shape).

    Returns ``(admitted: {(cell, workers): row}, rejected: [(cell, reason)])``. A
    key with two candidates is dropped entirely — preferring one would mix runs."""
    by_key = {e["key"]: e for e in manifest["expected"]}
    jobids = {e["jobid"] for e in manifest["expected"]}
    admitted, rejected, dupes = {}, [], set()
    for row in rows:
        cell, key = row["cell"], row["key"]
        if row["runid"] != manifest["runid"]:
            rejected.append((cell, f"runid {row['runid']} is not this run ({manifest['runid']})"))
            continue
        if row["sha"] != manifest["sha"]:
            rejected.append((cell, f"sha {row['sha'][:12]} != manifest sha {manifest['sha'][:12]}"))
            continue
        if row["jobid"] not in jobids:
            rejected.append((cell, f"jobid {row['jobid']} is not in the manifest"))
            continue
        expect = by_key.get(key)
        if expect is None:
            other = [e for e in manifest["expected"] if e["cell"] == cell]
            reason = (f"workers {row['workers']} is not an expected variant of {cell}"
                      if other else f"cell {cell} is not expected by this manifest")
            rejected.append((cell, reason))
            continue
        if expect["jobid"] != row["jobid"]:
            rejected.append((cell, f"jobid {row['jobid']} does not match {key}"))
            continue
        if expect["config_sha"] != row["config_sha"]:
            rejected.append((cell, f"config_sha {row['config_sha'][:12]} != expected "
                                   f"{expect['config_sha'][:12]}"))
            continue
        shape = [f for f in ("maxsteps", "mb", "ngpu", "workers") if expect[f] != row[f]]
        if shape:
            rejected.append((cell, "execution shape differs from the manifest: " + ", ".join(
                f"{f} {row[f]} != {expect[f]}" for f in shape)))
            continue
        if key in admitted or key in dupes:
            admitted.pop(key, None)
            dupes.add(key)
            rejected.append((cell, f"duplicate result rows for {key}"))
            continue
        admitted[key] = row
    return admitted, rejected


# --------------------------------------------------------------------------- #
# rate and per-cell summary
# --------------------------------------------------------------------------- #
def steps_per_second(t_lo, t_hi, steps_lo=WINDOW_LO, steps_hi=WINDOW_HI):
    """In-fit steady-state rate; fail-closed on non-finite or non-positive deltas."""
    if not (math.isfinite(t_lo) and math.isfinite(t_hi)):
        raise ValueError(f"non-finite window marks ({t_lo}, {t_hi})")
    delta = t_hi - t_lo
    if delta <= 0:
        raise ValueError(f"non-positive window delta ({t_hi} - {t_lo} = {delta})")
    return (steps_hi - steps_lo) / delta


def _blank(expect, status, note):
    cell = expect["cell"]
    return {"cell": cell, "family": family_of(cell), "rung": rung_of(cell)[0],
            "mb": expect["mb"], "ngpu": expect["ngpu"], "workers": expect["workers"],
            "key": expect["key"], "steps_s": None, "peak_overall_mib": None,
            "peak_per_gpu_max_mib": None, "status": status, "note": note,
            "t10": None, "t30": None, "vram_csv": None, "pollcsv_sha256": None,
            "jobid": expect["jobid"]}


def summarize(manifest, admitted, malformed=()):
    """One row per EXPECTED (cell, workers), in canonical order. Nothing is dropped."""
    malformed_by_cell = {}
    for cell, message in malformed:
        malformed_by_cell.setdefault(cell, message)

    out = []
    for expect in manifest["expected"]:
        cell, key = expect["cell"], expect["key"]
        row = admitted.get(key)
        if row is None:
            if cell in malformed_by_cell:
                out.append(_blank(expect, "MALFORMED", malformed_by_cell[cell]))
            elif not isinstance(expect["jobid"], int):
                out.append(_blank(expect, "MISSING", f"never submitted ({expect['jobid']})"))
            else:
                out.append(_blank(expect, "PENDING", f"no P0RESULT for job {expect['jobid']}"))
            continue

        per_gpu = list(row["peak_per_uuid"].values())
        steps_s, note = None, ""
        if row["rc"] == 3:
            status = "OOM"
        elif row["rc"] == 5 or row["valid"] == 0:
            status = "INVALID"
            note = f"measurement invalid (rc={row['rc']}, valid={row['valid']})"
        elif row["rc"] != 0:
            status = "FAILED"
            note = f"rc={row['rc']}"
        else:
            try:
                steps_s = steps_per_second(row["t10_mono"], row["t30_mono"])
                status = "OK"
            except ValueError as exc:
                status, note = "INVALID", str(exc)
        out.append({
            "cell": cell, "family": family_of(cell), "rung": rung_of(cell)[0],
            "mb": expect["mb"], "ngpu": expect["ngpu"], "workers": expect["workers"],
            "key": key, "steps_s": steps_s,
            "peak_overall_mib": row["peak_overall_mib"],
            "peak_per_gpu_max_mib": max(per_gpu) if per_gpu else row["peak_overall_mib"],
            "status": status, "note": note, "t10": row["t10"], "t30": row["t30"],
            "vram_csv": row["vram_csv"], "pollcsv_sha256": row["pollcsv_sha256"],
            "peak_per_uuid": row["peak_per_uuid"], "jobid": row["jobid"],
        })
    out.sort(key=_sort_key)
    return out


def all_ok(summaries):
    """True only when every expected row produced a usable measurement."""
    return bool(summaries) and all(s["status"] in _OK_STATUSES for s in summaries)


# --------------------------------------------------------------------------- #
# poller evidence — MANDATORY provenance (re-review B2)
# --------------------------------------------------------------------------- #
def parse_poll_tick(line):
    """``tick=.. uuid=.. mem=.. util=.. power=.. ts=..`` -> dict (or ``None``)."""
    fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
    if not {"tick", "uuid", "mem", "ts"} <= set(fields):
        return None
    try:
        out = {"tick": int(fields["tick"]), "uuid": fields["uuid"],
               "mem": int(fields["mem"]), "ts": float(fields["ts"])}
    except ValueError:
        return None
    for key in ("util", "power"):
        try:
            out[key] = float(fields.get(key, "nan"))
        except ValueError:
            out[key] = float("nan")     # e.g. '[N/A]'
    return out


def summarize_poller(path, t10, t30, expected_uuids):
    """Per-UUID mem/util/power over the measured step-10 -> step-30 window.

    A UUID with zero in-window ticks keeps ``ticks == 0`` — that is missing
    rank-placement evidence, not something to average away."""
    per = {u: {"ticks": 0, "mem_max_mib": None, "util_mean": None, "power_mean_w": None}
           for u in expected_uuids}
    acc = {u: {"mem": [], "util": [], "power": []} for u in expected_uuids}
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return per
    for line in lines:
        tick = parse_poll_tick(line)
        if tick is None or tick["uuid"] not in acc or not (t10 <= tick["ts"] <= t30):
            continue
        acc[tick["uuid"]]["mem"].append(tick["mem"])
        for key in ("util", "power"):
            if math.isfinite(tick[key]):
                acc[tick["uuid"]][key].append(tick[key])
    for uuid, vals in acc.items():
        per[uuid]["ticks"] = len(vals["mem"])
        if vals["mem"]:
            per[uuid]["mem_max_mib"] = max(vals["mem"])
        if vals["util"]:
            per[uuid]["util_mean"] = sum(vals["util"]) / len(vals["util"])
        if vals["power"]:
            per[uuid]["power_mean_w"] = sum(vals["power"]) / len(vals["power"])
    return per


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_poller_evidence(summaries, dirpath):
    """Verify each OK row's poller artifact; downgrade to INVALID when it fails.

    Required: the CSV exists next to the logs, its sha256 matches the one the job
    emitted, and every allocated UUID has at least one in-window tick with finite
    utilisation AND power. Returns ``(summaries, {key: per_uuid_summary})``."""
    out, poller = [], {}
    for s in summaries:
        if s["status"] != "OK":
            out.append(s)
            continue
        s = dict(s)
        name = s.get("vram_csv")
        path = os.path.join(dirpath, os.path.basename(name)) if name else None
        uuids = sorted(s.get("peak_per_uuid") or {})
        if not name or path is None or not os.path.exists(path):
            s["status"], s["note"] = "INVALID", f"poller CSV missing ({name})"
        elif _sha256_file(path) != s.get("pollcsv_sha256"):
            s["status"], s["note"] = "INVALID", "poller CSV hash mismatch (artifact altered)"
        elif len(uuids) != s["ngpu"]:
            s["status"], s["note"] = "INVALID", (
                f"peak_per_uuid lists {len(uuids)} GPU(s), expected {s['ngpu']}")
        else:
            per = summarize_poller(path, s["t10"], s["t30"], uuids)
            starved = [u for u, v in per.items() if v["ticks"] < 1]
            no_util = [u for u, v in per.items() if v["util_mean"] is None]
            no_power = [u for u, v in per.items() if v["power_mean_w"] is None]
            if starved:
                s["status"], s["note"] = "INVALID", (
                    f"no poller ticks inside the measured window for {starved}")
            elif no_util or no_power:
                s["status"], s["note"] = "INVALID", (
                    f"no finite util/power evidence for {sorted(set(no_util + no_power))}")
            else:
                poller[s["key"]] = per
        out.append(s)
    return out, poller


# --------------------------------------------------------------------------- #
# derived attribution (pure functions over summaries)
# --------------------------------------------------------------------------- #
def _usable(summaries, workers=MATRIX_WORKERS):
    """OK rows with a rate. Restricted to the standard worker count: the
    0-worker half is a dataloader diagnostic, not a matrix measurement."""
    return [s for s in summaries
            if s["status"] == "OK" and s["steps_s"]
            and (workers is None or s.get("workers", MATRIX_WORKERS) == workers)]


def orbit_pass_fit(summaries):
    """Per rung: least-squares fit of step time against ViT orbit passes.

    ``step_time = unattributed_residual + slope * n_passes`` over the EXACT
    {FA1, C4L, C8} set. FA1 shares C4L/C8's fa dispatch and cylindrical pose
    path, so the slope is the marginal cost of one ADDITIONAL orbit pass; VAN
    (Cartesian pose path) is structurally different and is never admitted here.
    Two points would fit any line perfectly, so incomplete rungs are omitted;
    ``ambiguous`` flags a physically impossible fit."""
    by_rung = {}
    for s in _usable(summaries):
        if s["family"] in FIT_FAMILIES and s["rung"]:
            by_rung.setdefault(s["rung"], {})[s["family"]] = s
    fits = {}
    for rung, cells in by_rung.items():
        if set(cells) != set(FIT_FAMILIES):
            continue
        pts = sorted((N_ORBIT_PASSES[f], 1.0 / cells[f]["steps_s"]) for f in cells)
        n_mean = sum(p[0] for p in pts) / len(pts)
        t_mean = sum(p[1] for p in pts) / len(pts)
        denom = sum((p[0] - n_mean) ** 2 for p in pts)
        if denom == 0:
            continue
        slope = sum((p[0] - n_mean) * (p[1] - t_mean) for p in pts) / denom
        residual = t_mean - slope * n_mean
        ss_tot = sum((p[1] - t_mean) ** 2 for p in pts)
        ss_res = sum((p[1] - (residual + slope * p[0])) ** 2 for p in pts)
        fits[rung] = {
            "slope_s_per_pass": slope,
            "unattributed_residual_s": residual,
            "n_points": len(pts),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "families": sorted(cells),
            "ambiguous": not (math.isfinite(slope) and math.isfinite(residual)
                              and slope > 0 and residual >= 0),
        }
    return fits


def attribution_ok(fits):
    """False if there is no fit at all, or any fitted rung is implausible."""
    return bool(fits) and all(not f["ambiguous"] for f in fits.values())


def vanilla_contrast(summaries):
    """VAN vs FA1 per rung — the cost of the fa dispatch + cylindrical pose path
    at ONE ViT pass. Reported separately; never part of the orbit fit (B6)."""
    by_rung = {}
    for s in _usable(summaries):
        if s["family"] in ("VAN", "FA1") and s["rung"]:
            by_rung.setdefault(s["rung"], {})[s["family"]] = s
    out = {}
    for rung, cells in by_rung.items():
        if set(cells) != {"VAN", "FA1"}:
            continue
        van_t = 1.0 / cells["VAN"]["steps_s"]
        fa1_t = 1.0 / cells["FA1"]["steps_s"]
        out[rung] = {"van_s_per_step": van_t, "fa1_s_per_step": fa1_t,
                     "delta_s": fa1_t - van_t}
    return out


def marginal_contrast(summaries):
    """C8 − C4L step-time contrast per rung — a measured difference, reported
    separately from (and never as) the fitted model."""
    by_rung = {}
    for s in _usable(summaries):
        if s["family"] in ("C4L", "C8") and s["rung"]:
            by_rung.setdefault(s["rung"], {})[s["family"]] = s
    out = {}
    for rung, cells in by_rung.items():
        if set(cells) != {"C4L", "C8"}:
            continue
        delta = 1.0 / cells["C8"]["steps_s"] - 1.0 / cells["C4L"]["steps_s"]
        extra = N_ORBIT_PASSES["C8"] - N_ORBIT_PASSES["C4L"]
        out[rung] = {"delta_s": delta, "extra_passes": extra, "s_per_pass": delta / extra}
    return out


def worker_contrast(summaries):
    """Per cell: the 0-vs-6-worker pair (input-starvation probe). Both halves
    must be present — a lone half says nothing about the dataloader."""
    by_cell = {}
    for s in _usable(summaries, workers=None):
        by_cell.setdefault(s["cell"], {})[s.get("workers")] = s
    out = {}
    for cell, halves in by_cell.items():
        if not {0, MATRIX_WORKERS} <= set(halves):
            continue
        w0, w6 = halves[0]["steps_s"], halves[MATRIX_WORKERS]["steps_s"]
        out[cell] = {"steps_s_w0": w0, "steps_s_w6": w6, "speedup_w6_over_w0": w6 / w0}
    return out


def ddp_scaling(summaries):
    """Per family: steps/s across rungs and strong-scaling efficiency (micro x N
    is pinned at 64, so ideal scaling is steps/s proportional to N)."""
    by_family = {}
    for s in _usable(summaries):
        by_family.setdefault(s["family"], []).append(s)
    out = {}
    for family, cells in by_family.items():
        cells = sorted(cells, key=lambda c: c["ngpu"])
        ref = cells[0]
        out[family] = [{
            "cell": c["cell"], "rung": c["rung"], "ngpu": c["ngpu"], "mb": c["mb"],
            "steps_s": c["steps_s"],
            "efficiency": c["steps_s"] / (ref["steps_s"] * c["ngpu"] / ref["ngpu"]),
        } for c in cells]
    return out


def grad_ckpt_cost(summaries):
    """CKPT4 (ViT grad-ckpt ON) vs C4L (OFF) at the same rung, or ``None``."""
    usable = {s["cell"]: s for s in _usable(summaries)}
    for cell, ckpt in sorted(usable.items(), key=lambda kv: _cell_key(kv[0])):
        if ckpt["family"] != "CKPT4":
            continue
        plain = usable.get(f"C4L_{ckpt['rung']}")
        if plain is None:
            continue
        return {
            "rung": ckpt["rung"], "ckpt_cell": cell, "no_ckpt_cell": plain["cell"],
            "no_ckpt_speedup": plain["steps_s"] / ckpt["steps_s"],
            "delta_s_per_step": 1.0 / ckpt["steps_s"] - 1.0 / plain["steps_s"],
            "delta_peak_mib": plain["peak_overall_mib"] - ckpt["peak_overall_mib"],
        }
    return None


# --------------------------------------------------------------------------- #
# markdown rendering (deterministic: no clock, input order irrelevant)
# --------------------------------------------------------------------------- #
def _fmt(value, spec=".4f"):
    return "—" if value is None else format(value, spec)


def _rung_ngpu(rung):
    return rung_of(f"X_{rung}")[2] or 0


def render_markdown(summaries, mode="matrix", complete=False, manifest=None, poller=None,
                    rejected=(), sources=()):
    """Status table always; derived attribution ONLY on a complete, all-OK run.
    The orbit fit is rendered for ``matrix`` manifests — the only mode that
    promises one; ``spot``/``workers`` report their own cells and contrasts."""
    summaries = sorted(summaries, key=_sort_key)
    lines = ["# exp_11 P0 profiling — measured throughput and peak VRAM", ""]
    if manifest:
        lines += [f"Run `{manifest['runid']}` · mode `{manifest['mode']}` · commit "
                  f"`{manifest['sha'][:12]}` — {len(manifest['expected'])} expected row(s).", ""]
    lines += [
        f"Steady-state rate measured INSIDE one fit (steps/s = "
        f"{WINDOW_HI - WINDOW_LO} / (t{WINDOW_HI} − t{WINDOW_LO}) from the runner's callback "
        "marks), so no startup, rendezvous or teardown time is included. Peak VRAM is the "
        "whole-run poller peak; every OK row's poller artifact is hash-verified.",
        "",
        "| Cell | rung MBxN | workers | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        step_time = 1.0 / s["steps_s"] if s["steps_s"] else None
        lines.append(
            f"| {s['cell']} | {s['rung'] or '—'} | {s.get('workers', '—')} | "
            f"{_fmt(s['steps_s'])} | {_fmt(step_time, '.2f')} | "
            f"{_fmt(s['peak_overall_mib'], 'd')} | {_fmt(s['peak_per_gpu_max_mib'], 'd')} | "
            f"{s['status']} | {s.get('note') or ''} |"
        )

    if poller:
        lines += ["", "## GPU utilisation / power over the step-10 → step-30 window", "",
                  "| Cell | workers | GPU UUID | ticks | mem max MiB | util mean % | power mean W |",
                  "|---|---|---|---|---|---|---|"]
        for key in sorted(poller, key=lambda k: (_cell_key(k[0]), -k[1])):
            for uuid in sorted(poller[key]):
                per = poller[key][uuid]
                lines.append(
                    f"| {key[0]} | {key[1]} | {uuid} | {per['ticks']} | "
                    f"{_fmt(per['mem_max_mib'], 'd')} | {_fmt(per['util_mean'], '.1f')} | "
                    f"{_fmt(per['power_mean_w'], '.1f')} |")

    workers = worker_contrast(summaries)
    if workers:
        lines += ["", "## Dataloader worker contrast (0 vs 6 workers, same cell)", "",
                  "| Cell | steps/s @0 | steps/s @6 | speedup |", "|---|---|---|---|"]
        for cell in sorted(workers, key=_cell_key):
            w = workers[cell]
            lines.append(f"| {cell} | {_fmt(w['steps_s_w0'])} | {_fmt(w['steps_s_w6'])} | "
                         f"{_fmt(w['speedup_w6_over_w0'], '.3f')} |")

    if not complete:
        lines += [
            "", "## Derived attribution — **WITHHELD**", "",
            "The submitted run is not complete and all-OK (see the status column), so the "
            "per-orbit-pass fit, scaling efficiency and grad-ckpt cost are NOT computed: a "
            "partial matrix cannot support a rung decision.",
        ]
    elif mode != "matrix":
        lines += ["", f"## Derived attribution — not applicable to a `{mode}` run", "",
                  f"A `{mode}` manifest does not carry the FA1+C4L+C8 set; its cells are "
                  "reported above and enter the matrix analysis only through their own run."]
    else:
        fits = orbit_pass_fit(summaries)
        lines += ["", "## Per-orbit-pass cost (step time vs ViT passes; exact FA1+C4L+C8 set)", ""]
        if fits:
            lines += ["| rung | s per orbit pass | unattributed residual (s) | points | R² | verdict |",
                      "|---|---|---|---|---|---|"]
            for rung in sorted(fits, key=_rung_ngpu):
                f = fits[rung]
                lines.append(
                    f"| {rung} | {_fmt(f['slope_s_per_pass'], '.3f')} | "
                    f"{_fmt(f['unattributed_residual_s'], '.3f')} | "
                    f"{f['n_points']} ({'+'.join(f['families'])}) | {_fmt(f['r2'], '.4f')} | "
                    f"{'AMBIGUOUS' if f['ambiguous'] else 'plausible'} |")
            lines += ["", "FA1 is `fa_invariant` with a single-angle orbit, so it shares the "
                      "cylindrical pose path with C4L/C8 and the slope is the cost of one "
                      "ADDITIONAL ViT pass. The intercept is an **unattributed residual** (the "
                      "fa base step, including its one pass); naming what it contains needs the "
                      "utilisation/power trace and the worker contrast above.", ""]
        else:
            lines.append("_not estimable — no rung has the complete FA1+C4L+C8 set._")

        vanilla = vanilla_contrast(summaries)
        lines += ["", "## Vanilla vs FA1 (pose-path + fa dispatch overhead, 1 ViT pass each)", ""]
        if vanilla:
            lines += ["| rung | VAN s/step | FA1 s/step | Δ s/step |", "|---|---|---|---|"]
            for rung in sorted(vanilla, key=_rung_ngpu):
                v = vanilla[rung]
                lines.append(f"| {rung} | {_fmt(v['van_s_per_step'], '.3f')} | "
                             f"{_fmt(v['fa1_s_per_step'], '.3f')} | {_fmt(v['delta_s'], '.3f')} |")
        else:
            lines.append("_no rung has both VAN and FA1._")

        contrasts = marginal_contrast(summaries)
        lines += ["", "## C8 − C4L marginal contrast (measured difference, not a fit)", ""]
        if contrasts:
            lines += ["| rung | Δ s/step | extra passes | s per extra pass |", "|---|---|---|---|"]
            for rung in sorted(contrasts, key=_rung_ngpu):
                c = contrasts[rung]
                lines.append(f"| {rung} | {_fmt(c['delta_s'], '.3f')} | {c['extra_passes']} | "
                             f"{_fmt(c['s_per_pass'], '.3f')} |")
        else:
            lines.append("_no rung has both C4L and C8._")

        lines += ["", "## DDP strong scaling at micro x N = 64", ""]
        scaling = ddp_scaling(summaries)
        if scaling:
            lines += ["| family | rung | GPUs | steps/s | efficiency vs smallest N |",
                      "|---|---|---|---|---|"]
            for family in sorted(scaling, key=lambda f: _cell_key(f"{f}_0x0")):
                for e in scaling[family]:
                    lines.append(f"| {family} | {e['rung']} | {e['ngpu']} | {_fmt(e['steps_s'])} | "
                                 f"{_fmt(e['efficiency'], '.3f')} |")
        else:
            lines.append("_no usable cells._")

        lines += ["", "## Gradient-checkpointing cost (CKPT4 vs C4L, same rung)", ""]
        gc = grad_ckpt_cost(summaries)
        if gc:
            lines += [
                f"- rung {gc['rung']}: disabling ViT grad-ckpt is "
                f"**{_fmt(gc['no_ckpt_speedup'], '.3f')}x** faster "
                f"({_fmt(gc['delta_s_per_step'], '.3f')} s/step of recompute removed).",
                f"- VRAM delta (no-ckpt − ckpt-on): {_fmt(gc['delta_peak_mib'], 'd')} MiB.",
            ]
        else:
            lines.append("_not estimable — the CKPT4/C4L pair is incomplete._")

    if rejected:
        lines += ["", "## Rejected rows (not from this run)", ""]
        lines += [f"- `{where}`: {msg}" for where, msg in sorted(rejected)]
    if sources:
        lines += ["", "## Source files", ""] + [f"- `{os.path.basename(s)}`" for s in sorted(sources)]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def scan_dir(dirpath):
    """Read every ``slurm_p0_*.out``; return ``(rows, malformed)``.

    Two P0RESULT lines in one file is MALFORMED, never last-wins."""
    rows, malformed = [], []
    for path in sorted(glob.glob(os.path.join(dirpath, "slurm_p0_*.out"))):
        base = os.path.basename(path)
        found, broken = [], False
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    row = parse_p0_line(line)
                except ValueError as exc:
                    malformed.append((base, f"{base}: {exc}"))
                    broken = True
                    continue
                if row is not None:
                    found.append(row)
        if broken:
            continue
        if len(found) > 1:
            cells = sorted({r["cell"] for r in found})
            malformed.append((base, f"{base}: duplicate P0RESULT lines ({len(found)}) "
                                    f"for {cells} — refusing last-wins"))
        elif found:
            rows.append(found[0])
    return rows, malformed


def _attach_cells(malformed, manifest):
    """Key malformed-file reports by cell where the file name identifies one."""
    out = []
    for where, msg in malformed:
        cell = next((e["cell"] for e in manifest["expected"] if e["cell"] in where), where)
        out.append((cell, msg))
    return out


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="collect exp_11 P0 profiling results")
    ap.add_argument("--manifest", required=True,
                    help="submission manifest written by p0_submit_matrix.sh")
    ap.add_argument("--dir", default=here, help="folder holding slurm_p0_*.out")
    ap.add_argument("--out", "-o", default=None,
                    help="output markdown (default <dir>/p0_report_<runid>.md)")
    ap.add_argument("--print", dest="to_stdout", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    rows, malformed = scan_dir(args.dir)
    admitted, rejected = admit_rows(rows, manifest)
    summaries = summarize(manifest, admitted, malformed=_attach_cells(malformed, manifest))
    summaries, poller = apply_poller_evidence(summaries, args.dir)

    complete = all_ok(summaries)
    mode = manifest["mode"]
    fits = orbit_pass_fit(summaries) if (complete and mode == "matrix") else {}
    ambiguous = complete and mode == "matrix" and not attribution_ok(fits)

    sources = [os.path.basename(p) for p in glob.glob(os.path.join(args.dir, "slurm_p0_*.out"))]
    md = render_markdown(summaries, mode=mode, complete=complete, manifest=manifest,
                         poller=poller, rejected=rejected, sources=sources)
    out = args.out or os.path.join(args.dir, f"p0_report_{manifest['runid']}.md")
    with open(out, "w") as fh:
        fh.write(md)
    if args.to_stdout:
        sys.stdout.write(md)

    bad = [s for s in summaries if s["status"] not in _OK_STATUSES]
    print(f"wrote {out}: mode={mode}, {len(summaries)} expected row(s), {len(bad)} not OK, "
          f"{len(rejected)} rejected row(s), {len(malformed)} malformed file(s)")
    if bad:
        for s in bad:
            print(f"  {s['status']:<10} {s['cell']} (w{s.get('workers')})  {s['note']}")
        return 1
    if ambiguous:
        print("  attribution AMBIGUOUS or absent: a matrix run must yield a plausible "
              "FA1+C4L+C8 fit at every complete rung")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
