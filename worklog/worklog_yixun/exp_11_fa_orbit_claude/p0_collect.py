#!/usr/bin/env python
"""exp_11 P0 profiling collector — manifest-bound ``P0RESULT`` -> ``p0_report.md``.

One 30-step job per cell. ``p0_runner``'s callback timestamps completed optimizer
steps 10 and 30 inside that single fit, the sbatch forwards those marks, and the
steady-state rate is

    steps/s = (30 - 10) / (t30_mono - t10_mono)

which contains no startup, rendezvous or teardown time at all.

Collection is bound to ONE submission manifest (round-2 review B4): a row is
admitted only if its run id, commit sha, job id, cell and config sha256 all match
what the submitter recorded. Every EXPECTED cell gets a row — PENDING, MISSING,
MALFORMED, INVALID, OOM, FAILED or OK — and anything short of all-OK makes the
collector withhold the derived attribution tables and exit nonzero, so a partial
matrix can never be read as a rung decision.

Derived columns: the per-orbit-pass slope needs the EXACT {VAN, C4L, C8} set at a
rung (two points always fit a line perfectly and would fake an R² of 1), the
fitted intercept is labelled an *unattributed residual* rather than a bottleneck
diagnosis until the utilisation/power trace is read, and physically implausible
fits are marked AMBIGUOUS. Usage:

    python p0_collect.py --manifest p0_manifest_<runid>.txt [--dir .] [--out ...]
"""
import argparse
import glob
import math
import os
import sys

WINDOW_LO, WINDOW_HI = 10, 30

# ViT forward passes per training step (orbit size; vanilla runs the ViT once).
N_ORBIT_PASSES = {"VAN": 1, "C4L": 4, "C8": 8, "C16": 16, "C32": 32, "CKPT4": 4}
# The fit needs exactly these three families at a rung — no subsets (B6).
FIT_FAMILIES = ("VAN", "C4L", "C8")
FAMILY_ORDER = ("VAN", "C4L", "C8", "C16", "C32", "CKPT4")

_INT_FIELDS = ("jobid", "maxsteps", "ngpu", "mb", "rc", "peak_overall_mib", "valid")
_FLOAT_FIELDS = ("wall_fit", "t10", "t30", "t10_mono", "t30_mono")
_STR_FIELDS = ("runid", "sha", "cell", "config_sha", "peak_per_uuid", "vram_csv")
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

    ``None`` for any line that is not a P0RESULT line; ``ValueError`` naming the
    offending key when a P0RESULT line is missing a field, carries a non-numeric
    or NON-FINITE number (``nan``/``inf`` must never reach the tables — B6), a
    negative peak, or a ``valid`` outside {0, 1}."""
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
    row["peak_per_uuid"] = parse_peak_per_uuid(row["peak_per_uuid"])
    return row


def parse_manifest(text):
    """Parse the submitter's manifest into ``{runid, sha, expected: [...]}``."""
    runid = sha = None
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
        elif parts[0] == "cell" and len(parts) == 5:
            cell, maxsteps, jobid, config_sha = parts[1:]
            if cell in seen:
                raise ValueError(f"manifest lists cell {cell} more than once")
            seen.add(cell)
            expected.append({
                "cell": cell, "maxsteps": int(maxsteps),
                "jobid": int(jobid) if jobid.isdigit() else jobid,
                "config_sha": config_sha,
            })
    if not runid or not sha:
        raise ValueError("manifest must declare both 'runid' and 'sha'")
    if not expected:
        raise ValueError("manifest declares no expected cells")
    return {"runid": runid, "sha": sha, "expected": expected}


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


# --------------------------------------------------------------------------- #
# manifest binding (B4)
# --------------------------------------------------------------------------- #
def admit_rows(rows, manifest):
    """Keep only rows that match the manifest exactly.

    Returns ``(admitted: {cell: row}, rejected: [(cell_or_jobid, reason)])``. A
    cell with two admitted candidates is dropped entirely: silently preferring
    one would mix launches."""
    by_cell = {e["cell"]: e for e in manifest["expected"]}
    jobids = {e["jobid"] for e in manifest["expected"]}
    admitted, rejected, dupes = {}, [], set()
    for row in rows:
        cell = row["cell"]
        if row["runid"] != manifest["runid"]:
            rejected.append((cell, f"runid {row['runid']} is not this run ({manifest['runid']})"))
            continue
        if row["sha"] != manifest["sha"]:
            rejected.append((cell, f"sha {row['sha'][:12]} != manifest sha {manifest['sha'][:12]}"))
            continue
        if row["jobid"] not in jobids:
            rejected.append((cell, f"jobid {row['jobid']} is not in the manifest"))
            continue
        expect = by_cell.get(cell)
        if expect is None:
            rejected.append((cell, f"cell {cell} is not expected by this manifest"))
            continue
        if expect["jobid"] != row["jobid"]:
            rejected.append((cell, f"jobid {row['jobid']} does not match cell {cell}"))
            continue
        if expect["config_sha"] != row["config_sha"]:
            rejected.append((cell, f"config_sha {row['config_sha'][:12]} != expected "
                                   f"{expect['config_sha'][:12]}"))
            continue
        if cell in admitted or cell in dupes:
            admitted.pop(cell, None)
            dupes.add(cell)
            rejected.append((cell, f"duplicate result rows for cell {cell}"))
            continue
        admitted[cell] = row
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


def _blank(cell, status, note):
    rung, mb, ngpu = rung_of(cell)
    return {"cell": cell, "family": family_of(cell), "rung": rung, "mb": mb, "ngpu": ngpu,
            "steps_s": None, "s_per_step": None, "peak_overall_mib": None,
            "peak_per_gpu_max_mib": None, "status": status, "note": note,
            "t10": None, "t30": None, "vram_csv": None, "jobid": None}


def summarize(manifest, admitted, malformed=()):
    """One row per EXPECTED cell, in canonical order. Nothing is ever dropped."""
    malformed_by_cell = {}
    for cell, message in malformed:
        malformed_by_cell.setdefault(cell, message)

    out = []
    for expect in manifest["expected"]:
        cell = expect["cell"]
        row = admitted.get(cell)
        if row is None:
            if cell in malformed_by_cell:
                out.append(_blank(cell, "MALFORMED", malformed_by_cell[cell]))
            elif not isinstance(expect["jobid"], int):
                out.append(_blank(cell, "MISSING", f"never submitted ({expect['jobid']})"))
            else:
                out.append(_blank(cell, "PENDING", f"no P0RESULT for job {expect['jobid']}"))
            continue

        rung, mb, ngpu = rung_of(cell)
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
            "cell": cell, "family": family_of(cell), "rung": rung,
            "mb": mb if mb is not None else row["mb"],
            "ngpu": ngpu if ngpu is not None else row["ngpu"],
            "steps_s": steps_s, "s_per_step": (1.0 / steps_s) if steps_s else None,
            "peak_overall_mib": row["peak_overall_mib"],
            "peak_per_gpu_max_mib": max(per_gpu) if per_gpu else row["peak_overall_mib"],
            "status": status, "note": note, "t10": row["t10"], "t30": row["t30"],
            "vram_csv": row["vram_csv"], "jobid": row["jobid"],
        })
    out.sort(key=lambda s: _cell_key(s["cell"]))
    return out


def all_ok(summaries):
    """True only when every expected cell produced a usable measurement."""
    return bool(summaries) and all(s["status"] in _OK_STATUSES for s in summaries)


# --------------------------------------------------------------------------- #
# poller window summary (B3): utilisation/power over the measured interval
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
    for key, name in (("util", "util"), ("power", "power")):
        try:
            out[name] = float(fields.get(key, "nan"))
        except ValueError:
            out[name] = float("nan")     # e.g. '[N/A]' on some GPUs
    return out


def summarize_poller(path, t10, t30, expected_uuids):
    """Per-UUID mem/util/power over the measured step-10 -> step-30 window.

    A UUID with zero in-window ticks is reported with ``ticks == 0`` — that is
    the missing rank-placement evidence, not something to average away."""
    per = {u: {"ticks": 0, "mem_max_mib": None, "util_mean": None,
               "power_mean_w": None} for u in expected_uuids}
    acc = {u: {"mem": [], "util": [], "power": []} for u in expected_uuids}
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return per
    for line in lines:
        tick = parse_poll_tick(line)
        if tick is None or tick["uuid"] not in acc:
            continue
        if not (t10 <= tick["ts"] <= t30):
            continue
        acc[tick["uuid"]]["mem"].append(tick["mem"])
        for key in ("util", "power"):
            if math.isfinite(tick.get(key, float("nan"))):
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


def poller_summaries(summaries, dirpath):
    """Window summaries for every OK cell that named a poller CSV."""
    out = {}
    for s in summaries:
        if s["status"] != "OK" or not s.get("vram_csv"):
            continue
        path = s["vram_csv"]
        if not os.path.isabs(path):
            path = os.path.join(dirpath, os.path.basename(path))
        uuids = _csv_uuids(path)
        if uuids:
            out[s["cell"]] = summarize_poller(path, s["t10"], s["t30"], uuids)
    return out


def _csv_uuids(path):
    try:
        with open(path, "r", errors="replace") as fh:
            ticks = (parse_poll_tick(line) for line in fh)
            return sorted({t["uuid"] for t in ticks if t})
    except OSError:
        return []


# --------------------------------------------------------------------------- #
# derived attribution (pure functions over summaries)
# --------------------------------------------------------------------------- #
def _usable(summaries):
    return [s for s in summaries if s["status"] == "OK" and s["steps_s"]]


def orbit_pass_fit(summaries):
    """Per rung: least-squares fit of step time against ViT orbit passes.

    ``step_time = unattributed_residual + slope * n_passes`` over the EXACT
    {VAN, C4L, C8} set — with only two of them the line is underdetermined and
    R² is 1 by construction, so such rungs are omitted entirely (B6). The
    intercept is an *unattributed residual*, not a bottleneck diagnosis: naming
    what it contains needs the utilisation/power trace and the worker-pair cell.
    ``ambiguous`` flags a physically impossible fit (non-positive slope or
    negative residual)."""
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
    """False if any fitted rung is physically implausible (report AMBIGUOUS)."""
    return bool(fits) and all(not f["ambiguous"] for f in fits.values())


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


def ddp_scaling(summaries):
    """Per family: steps/s across rungs and strong-scaling efficiency.

    micro x N is pinned at 64, so the global batch is constant and ideal scaling
    is steps/s proportional to N; the reference is the smallest GPU count."""
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


def render_markdown(summaries, complete=False, manifest=None, poller=None,
                    rejected=(), sources=()):
    """Status table always; derived attribution ONLY on a complete, all-OK run."""
    summaries = sorted(summaries, key=lambda s: _cell_key(s["cell"]))
    lines = ["# exp_11 P0 profiling — measured throughput and peak VRAM", ""]
    if manifest:
        lines += [f"Run `{manifest['runid']}` at commit `{manifest['sha'][:12]}` — "
                  f"{len(manifest['expected'])} expected cell(s).", ""]
    lines += [
        f"Steady-state rate measured INSIDE one fit (steps/s = "
        f"{WINDOW_HI - WINDOW_LO} / (t{WINDOW_HI} − t{WINDOW_LO}) from the runner's "
        "callback marks), so no startup, rendezvous or teardown time is included. "
        "Peak VRAM is the whole-run poller peak.",
        "",
        "| Cell | rung MBxN | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        step_time = s.get("s_per_step") or (1.0 / s["steps_s"] if s["steps_s"] else None)
        lines.append(
            f"| {s['cell']} | {s['rung'] or '—'} | {_fmt(s['steps_s'])} | "
            f"{_fmt(step_time, '.2f')} | {_fmt(s['peak_overall_mib'], 'd')} | "
            f"{_fmt(s['peak_per_gpu_max_mib'], 'd')} | {s['status']} | {s.get('note') or ''} |"
        )

    if poller:
        lines += ["", "## GPU utilisation / power over the step-10 → step-30 window", "",
                  "| Cell | GPU UUID | ticks | mem max MiB | util mean % | power mean W |",
                  "|---|---|---|---|---|---|"]
        for cell in sorted(poller, key=_cell_key):
            for uuid in sorted(poller[cell]):
                per = poller[cell][uuid]
                lines.append(
                    f"| {cell} | {uuid} | {per['ticks']} | {_fmt(per['mem_max_mib'], 'd')} | "
                    f"{_fmt(per['util_mean'], '.1f')} | {_fmt(per['power_mean_w'], '.1f')} |"
                )

    if not complete:
        lines += [
            "", "## Derived attribution — **WITHHELD**", "",
            "The submitted matrix is not complete and all-OK (see the status column), "
            "so the per-orbit-pass fit, scaling efficiency and grad-ckpt cost are NOT "
            "computed: a partial matrix cannot support a rung decision.",
        ]
        if rejected:
            lines += ["", "### Rejected rows (not from this run)", ""]
            lines += [f"- `{where}`: {msg}" for where, msg in sorted(rejected)]
        if sources:
            lines += ["", "## Source files", ""] + [f"- `{os.path.basename(s)}`" for s in sorted(sources)]
        return "\n".join(lines) + "\n"

    fits = orbit_pass_fit(summaries)
    lines += ["", "## Per-orbit-pass cost (step time vs ViT passes; exact VAN+C4L+C8 set)", ""]
    if fits:
        lines += ["| rung | s per orbit pass | unattributed residual (s) | points | R² | verdict |",
                  "|---|---|---|---|---|---|"]
        for rung in sorted(fits, key=lambda r: rung_of(f"X_{r}")[2] or 0):
            f = fits[rung]
            lines.append(
                f"| {rung} | {_fmt(f['slope_s_per_pass'], '.3f')} | "
                f"{_fmt(f['unattributed_residual_s'], '.3f')} | "
                f"{f['n_points']} ({'+'.join(f['families'])}) | {_fmt(f['r2'], '.4f')} | "
                f"{'AMBIGUOUS' if f['ambiguous'] else 'plausible'} |"
            )
        lines += ["", "The intercept is an **unattributed residual** (everything not "
                  "proportional to orbit passes). Naming what it contains requires the "
                  "utilisation/power trace above and the 0-vs-6-worker pair — it is not "
                  "a bottleneck diagnosis on its own.", ""]
    else:
        lines.append("_not estimable — no rung has the complete VAN+C4L+C8 set._")

    contrasts = marginal_contrast(summaries)
    lines += ["", "## C8 − C4L marginal contrast (measured difference, not a fit)", ""]
    if contrasts:
        lines += ["| rung | Δ s/step | extra passes | s per extra pass |", "|---|---|---|---|"]
        for rung in sorted(contrasts, key=lambda r: rung_of(f"X_{r}")[2] or 0):
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

    A file with two P0RESULT lines is MALFORMED, not last-wins: two results in
    one job log means the job (or the log) is not what the manifest describes."""
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
    ap.add_argument("--manifest", required=True, help="submission manifest written by p0_submit_matrix.sh")
    ap.add_argument("--dir", default=here, help="folder holding slurm_p0_*.out")
    ap.add_argument("--out", default=None, help="output markdown (default <dir>/p0_report.md)")
    ap.add_argument("--print", dest="to_stdout", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    rows, malformed = scan_dir(args.dir)
    admitted, rejected = admit_rows(rows, manifest)
    summaries = summarize(manifest, admitted, malformed=_attach_cells(malformed, manifest))

    complete = all_ok(summaries)
    poller = poller_summaries(summaries, args.dir) if admitted else {}
    fits = orbit_pass_fit(summaries) if complete else {}
    ambiguous = complete and not attribution_ok(fits)

    sources = [os.path.basename(p) for p in glob.glob(os.path.join(args.dir, "slurm_p0_*.out"))]
    md = render_markdown(summaries, complete=complete, manifest=manifest, poller=poller,
                         rejected=rejected, sources=sources)
    out = args.out or os.path.join(args.dir, "p0_report.md")
    with open(out, "w") as fh:
        fh.write(md)
    if args.to_stdout:
        sys.stdout.write(md)

    bad = [s for s in summaries if s["status"] not in _OK_STATUSES]
    print(f"wrote {out}: {len(summaries)} expected cell(s), {len(bad)} not OK, "
          f"{len(rejected)} rejected row(s), {len(malformed)} malformed file(s)")
    if bad:
        for s in bad:
            print(f"  {s['status']:<10} {s['cell']}  {s['note']}")
        return 1
    if ambiguous:
        print("  attribution AMBIGUOUS: a fitted rung is physically implausible")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
