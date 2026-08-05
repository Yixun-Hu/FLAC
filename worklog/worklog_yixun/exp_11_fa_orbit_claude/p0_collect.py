#!/usr/bin/env python
"""exp_11 P0 profiling collector — turn ``P0RESULT`` lines into ``p0_report.md``.

Each P0 cell is run TWICE by ``p0_profile.sbatch`` (``--max-steps`` 10 and 30)
and prints exactly one ``P0RESULT`` line. Pairing the two cancels the startup
cost (imports, VAE load, DDP rendezvous, first-batch compile), so

    steps/s = (30 - 10) / (wall_fit_30 - wall_fit_10)

is the steady-state training rate. Peak VRAM is the max over the pair.

Derived attribution columns (plan Rev 3 §10 P0.2), all computed here rather than
by hand: the per-orbit-pass cost from a least-squares fit of step time against
the number of ViT passes over {VAN, C4L, C8} at a fixed rung, DDP scaling
efficiency across rungs at fixed micro x N = 64, and the gradient-checkpointing
cost from CKPT4 vs C4L at the same rung.

Cells that OOMed, failed, or produced an invalid measurement are reported as
table rows with a status — never dropped, since a missing row would silently
bias the rung decision. Usage:

    python p0_collect.py [--dir <exp_11 folder>] [--out p0_report.md] [--print]
"""
import argparse
import glob
import os
import sys

STEPS_LO = 10
STEPS_HI = 30

# ViT forward passes per training step (the frame-averaging orbit size; the
# vanilla path runs the ViT conditioners once).
N_ORBIT_PASSES = {"VAN": 1, "C4L": 4, "C8": 8, "C16": 16, "C32": 32, "CKPT4": 4}
# Families entering the per-orbit-pass fit: identical recipe apart from the
# orbit (CKPT4 is excluded — grad-ckpt changes the per-pass cost itself).
FIT_FAMILIES = ("VAN", "C4L", "C8")
FAMILY_ORDER = ("VAN", "C4L", "C8", "C16", "C32", "CKPT4")

_INT_FIELDS = ("maxsteps", "ngpu", "mb", "rc", "peak_overall_mib", "valid")
_REQUIRED = _INT_FIELDS + ("cell", "wall_fit", "peak_per_uuid")


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

    Returns ``None`` for any line that is not a P0RESULT line; raises
    ``ValueError`` naming the offending key(s) when a P0RESULT line is missing a
    required field or carries a non-numeric value (a malformed measurement must
    never be silently treated as absent)."""
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
    try:
        row["wall_fit"] = float(row["wall_fit"])
    except ValueError:
        raise ValueError(f"P0RESULT wall_fit={row['wall_fit']!r} is not a number")
    row["peak_per_uuid"] = parse_peak_per_uuid(row["peak_per_uuid"])
    return row


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
# pairing and per-cell summary
# --------------------------------------------------------------------------- #
def pair_rows(rows):
    """Group rows into ``(cell, {'lo': row10, 'hi': row30})`` pairs.

    Returns ``(pairs, problems)`` where pairs are in canonical cell order and
    ``problems`` is a list of ``(cell, message)`` for cells that cannot be paired
    (missing half, duplicate submission, or a rung that changed between runs)."""
    by_cell = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)

    pairs, problems = [], []
    for cell in sorted(by_cell, key=_cell_key):
        buckets = {}
        for row in by_cell[cell]:
            buckets.setdefault(row["maxsteps"], []).append(row)
        dups = sorted(s for s, rs in buckets.items() if len(rs) > 1)
        if dups:
            problems.append((cell, f"duplicate result line(s) for max-steps {dups}"))
            continue
        unexpected = sorted(s for s in buckets if s not in (STEPS_LO, STEPS_HI))
        if unexpected:
            problems.append((cell, f"unexpected maxsteps {unexpected}"))
            continue
        have = sorted(buckets)
        if len(have) != 2:
            missing = STEPS_LO if STEPS_LO not in buckets else STEPS_HI
            problems.append((cell, f"missing the max-steps {missing} run (have {have})"))
            continue
        lo, hi = buckets[STEPS_LO][0], buckets[STEPS_HI][0]
        shape = [k for k in ("ngpu", "mb") if lo[k] != hi[k]]
        if shape:
            problems.append((cell, f"{'/'.join(shape)} differ between the paired runs"))
            continue
        pairs.append((cell, {"lo": lo, "hi": hi}))
    return pairs, problems


def steps_per_second(wall_lo, wall_hi, steps_lo=STEPS_LO, steps_hi=STEPS_HI):
    """Steady-state rate from the paired walls; the startup cost cancels."""
    delta = wall_hi - wall_lo
    if delta <= 0:
        raise ValueError(
            f"non-positive wall delta ({wall_hi} - {wall_lo} = {delta}); the "
            f"{steps_hi}-step run must take longer than the {steps_lo}-step run"
        )
    return (steps_hi - steps_lo) / delta


def summarize(pairs, problems=()):
    """One row per cell: rate, peaks and a status, in canonical order."""
    out = []
    for cell, pair in pairs:
        lo, hi = pair["lo"], pair["hi"]
        peak_overall = max(lo["peak_overall_mib"], hi["peak_overall_mib"])
        per_gpu = [v for row in (lo, hi) for v in row["peak_per_uuid"].values()]
        rung, mb, ngpu = rung_of(cell)
        steps_s, note = None, ""
        if any(r["rc"] == 3 for r in (lo, hi)):
            status = "OOM"
        elif any(r["rc"] != 0 for r in (lo, hi)):
            status = "FAILED"
        elif any(r["valid"] == 0 for r in (lo, hi)):
            status = "INVALID"
            note = "measurement invalid"
        else:
            try:
                steps_s = steps_per_second(lo["wall_fit"], hi["wall_fit"])
                status = "OK"
            except ValueError as exc:
                status, note = "INVALID", str(exc)
        out.append({
            "cell": cell, "family": family_of(cell), "rung": rung,
            "mb": mb if mb is not None else lo["mb"],
            "ngpu": ngpu if ngpu is not None else lo["ngpu"],
            "steps_s": steps_s, "peak_overall_mib": peak_overall,
            "peak_per_gpu_max_mib": max(per_gpu) if per_gpu else peak_overall,
            "wall_lo": lo["wall_fit"], "wall_hi": hi["wall_fit"],
            "status": status, "note": note,
        })
    for cell, message in problems:
        rung, mb, ngpu = rung_of(cell)
        out.append({
            "cell": cell, "family": family_of(cell), "rung": rung, "mb": mb,
            "ngpu": ngpu, "steps_s": None, "peak_overall_mib": None,
            "peak_per_gpu_max_mib": None, "wall_lo": None, "wall_hi": None,
            "status": "INCOMPLETE", "note": message,
        })
    out.sort(key=lambda s: _cell_key(s["cell"]))
    return out


# --------------------------------------------------------------------------- #
# derived attribution (pure functions over summaries)
# --------------------------------------------------------------------------- #
def _usable(summaries):
    return [s for s in summaries if s["status"] == "OK" and s["steps_s"]]


def orbit_pass_fit(summaries):
    """Per rung: least-squares fit of step time vs ViT orbit passes.

    ``step_time = intercept + slope * n_passes`` over {VAN, C4L, C8} — the slope
    is the marginal cost of one orbit pass, the intercept everything else in the
    step (data, VAE, DiT, optimizer). Rungs with fewer than two usable families
    are omitted (a line needs two points)."""
    by_rung = {}
    for s in _usable(summaries):
        if s["family"] in FIT_FAMILIES and s["rung"]:
            by_rung.setdefault(s["rung"], []).append(s)
    fits = {}
    for rung, cells in by_rung.items():
        pts = sorted((N_ORBIT_PASSES[c["family"]], 1.0 / c["steps_s"]) for c in cells)
        if len(pts) < 2:
            continue
        n_mean = sum(p[0] for p in pts) / len(pts)
        t_mean = sum(p[1] for p in pts) / len(pts)
        denom = sum((p[0] - n_mean) ** 2 for p in pts)
        if denom == 0:
            continue
        slope = sum((p[0] - n_mean) * (p[1] - t_mean) for p in pts) / denom
        intercept = t_mean - slope * n_mean
        ss_tot = sum((p[1] - t_mean) ** 2 for p in pts)
        ss_res = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts)
        fits[rung] = {
            "slope_s_per_pass": slope, "intercept_s": intercept,
            "n_points": len(pts),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "families": [c["family"] for c in sorted(cells, key=lambda c: c["cell"])],
        }
    return fits


def ddp_scaling(summaries):
    """Per family: steps/s across rungs and the strong-scaling efficiency.

    micro x N is pinned at 64, so the global batch is constant and ideal scaling
    is steps/s proportional to N; efficiency is measured against the smallest
    GPU count available for that family."""
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
            "rung": ckpt["rung"],
            "ckpt_cell": cell, "no_ckpt_cell": plain["cell"],
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


def render_markdown(summaries, sources=(), scan_problems=()):
    summaries = sorted(summaries, key=lambda s: _cell_key(s["cell"]))
    lines = [
        "# exp_11 P0 profiling — measured throughput and peak VRAM",
        "",
        "Steady-state rate from the paired 10/30-step runs "
        f"(steps/s = {STEPS_HI - STEPS_LO} / (wall_{STEPS_HI} − wall_{STEPS_LO})), "
        "so startup cost cancels. Peak VRAM is the max over the pair.",
        "",
        "| Cell | rung MBxN | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        step_time = 1.0 / s["steps_s"] if s["steps_s"] else None
        lines.append(
            f"| {s['cell']} | {s['rung'] or '—'} | {_fmt(s['steps_s'])} | "
            f"{_fmt(step_time, '.2f')} | {_fmt(s['peak_overall_mib'], 'd')} | "
            f"{_fmt(s['peak_per_gpu_max_mib'], 'd')} | {s['status']} | {s['note'] or ''} |"
        )

    fits = orbit_pass_fit(summaries)
    lines += ["", "## Per-orbit-pass cost (step time vs ViT passes; VAN/C4L/C8)", ""]
    if fits:
        lines += ["| rung | s per orbit pass | residual step cost (s) | points | R² |",
                  "|---|---|---|---|---|"]
        for rung in sorted(fits, key=lambda r: rung_of(f"X_{r}")[2] or 0):
            f = fits[rung]
            lines.append(
                f"| {rung} | {_fmt(f['slope_s_per_pass'], '.3f')} | "
                f"{_fmt(f['intercept_s'], '.3f')} | {f['n_points']} "
                f"({'+'.join(f['families'])}) | {_fmt(f['r2'], '.4f')} |"
            )
    else:
        lines.append("_not estimable — fewer than two usable families at any rung._")

    lines += ["", "## DDP strong scaling at micro x N = 64", ""]
    scaling = ddp_scaling(summaries)
    if scaling:
        lines += ["| family | rung | GPUs | steps/s | efficiency vs smallest N |",
                  "|---|---|---|---|---|"]
        for family in sorted(scaling, key=lambda f: _cell_key(f"{f}_0x0")):
            for e in scaling[family]:
                lines.append(
                    f"| {family} | {e['rung']} | {e['ngpu']} | {_fmt(e['steps_s'])} | "
                    f"{_fmt(e['efficiency'], '.3f')} |"
                )
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

    if scan_problems:
        lines += ["", "## Collection problems", ""]
        lines += [f"- {msg}" for _, msg in sorted(scan_problems)]  # msg names its file
    if sources:
        lines += ["", "## Source files", ""]
        lines += [f"- `{os.path.basename(s)}`" for s in sorted(sources)]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def scan_dir(dirpath):
    """Read every ``slurm_p0_*.out`` in ``dirpath``; return ``(rows, problems)``."""
    rows, problems = [], []
    for path in sorted(glob.glob(os.path.join(dirpath, "slurm_p0_*.out"))):
        base = os.path.basename(path)
        found = []
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    row = parse_p0_line(line)
                except ValueError as exc:
                    problems.append((base, f"{base}: {exc}"))
                    row = None
                if row is not None:
                    found.append(row)
        if not found:
            problems.append((base, f"{base}: no P0RESULT line (job died before the result?)"))
        elif len(found) > 1:
            problems.append((base, f"{base}: {len(found)} P0RESULT lines; using the last"))
            rows.append(found[-1])
        else:
            rows.append(found[0])
    return rows, problems


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="collect exp_11 P0 profiling results")
    ap.add_argument("--dir", default=here, help="folder holding slurm_p0_*.out")
    ap.add_argument("--out", default=None, help="output markdown (default <dir>/p0_report.md)")
    ap.add_argument("--print", dest="to_stdout", action="store_true")
    args = ap.parse_args(argv)

    rows, scan_problems = scan_dir(args.dir)
    summaries = summarize(*pair_rows(rows))
    sources = [os.path.basename(p) for p in glob.glob(os.path.join(args.dir, "slurm_p0_*.out"))]
    md = render_markdown(summaries, sources=sources, scan_problems=scan_problems)

    out = args.out or os.path.join(args.dir, "p0_report.md")
    with open(out, "w") as fh:
        fh.write(md)
    if args.to_stdout:
        sys.stdout.write(md)
    print(f"wrote {out}: {len(summaries)} cell(s), {len(scan_problems)} collection problem(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
