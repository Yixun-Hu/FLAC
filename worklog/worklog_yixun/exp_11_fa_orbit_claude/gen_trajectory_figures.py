#!/usr/bin/env python3
"""exp_11 trajectory figures — acoustic metrics vs training step, per arm.

Reads the on-disk screen/conf metric JSONs (K8, s42 screens; 5-seed conf at
40k rendered as mean +/- sd) and emits a fully self-contained HTML page with
inline-SVG line charts (one panel per metric, one series per arm), light/dark
via prefers-color-scheme + data-theme, hover crosshair, and a table view.
Planner one-off; reviewed with the closing package. Numbers come only from the
metric JSONs — this page is presentation, never a source.
"""
import json, glob, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import exp11_validate_rows as V          # noqa: E402  (the harvest gate, see validated_band)

REPO = os.path.dirname(os.path.abspath(__file__)) + "/../../.."
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_orbit_02_trajectories.html")
METRICS = [("T60", "T60 (%)", True), ("C50", "C50 (dB)", True),
           ("EDT", "EDT (ms)", True), ("RIR_to_GT_RIR_R@1", "R@1 (%)", False),
           ("RIR_to_GT_RIR_R@5", "R@5 (%)", False), ("RIR_to_GT_RIR_R@10", "R@10 (%)", False)]
# categorical slots 1-5 (validated reference instance, documented order), light/dark
ARMS = [
    ("C4L",  "C4 (bridge)",   "#2a78d6", "#3987e5", "exp11_C4L",  4,  ""),
    ("C8",   "C8",            "#eb6834", "#d95926", "exp11_C8",   8,  ""),
    ("C16",  "C16",           "#1baf7a", "#199e70", "exp11_C16",  16, ""),
    ("C32",  "C32",           "#eda100", "#c98500", "exp11_C32",  32, "6 3"),
    ("C4bf", "C4 legacy-loop", "#e87ba4", "#d55181", None,        4,  "2 3"),
    ("VANL", "vanilla (fa recipe)", "#008300", "#008300", "exp11_VANL", 0, "8 4"),
    ("P1v",  "P1 vanilla legacy", "#4a3aa7", "#9085e9", "POINT",     0,  "3 2"),
]

def validated_band(base, arm, k, suffix, validate=None, refusals=None):
    """Q10 (>40k) band points, harvested ONLY through the row validator.

    Re-pin review, fix 5: this used to draw a band whenever five files with the
    right name shape existed — it never proved the five were seeds 42-46 exactly
    once, never read a sidecar, and never required one checkpoint/config/source
    SHA across them, so five duplicates or a mixed-pin block became a band. Every
    step now goes through ``validate_cell(..., contract="traj")``, which is where
    the five unique seeds, the common provenance (CELL_IDENTITY_FIELDS), the
    2,500 grid, the >40000 floor and the 100000 ceiling are all enforced. A step
    with ANY problem is refused outright and disclosed — never silently narrowed
    into a partial band."""
    validate = validate or V.validate_cell
    by_step = {}
    for f in sorted(glob.glob(f"{base}/*_metrics_*_traj_S*_s4[2-6]_K{k}_*{suffix}*.json")):
        if f.endswith(".screenmeta.json"):
            continue
        by_step.setdefault(int(f.split("_traj_S")[1].split("_")[0]), []).append(f)
    band = {}
    for st, paths in sorted(by_step.items()):
        rows, problems = validate(sorted(paths), arm, st, k, contract="traj")
        if problems:
            if refusals is not None:
                refusals.append({"arm": arm, "K": k, "step": st, "files": len(paths),
                                 "problems": problems})
            continue
        vals = {}
        for mk, _, _ in METRICS:
            xs = [r["metrics"][mk] for r in rows if mk in r.get("metrics", {})]
            if len(xs) == len(rows) and len(xs) > 1:
                vals[mk] = (statistics.mean(xs), statistics.stdev(xs))
        if vals:
            band[st] = vals
    return band


def value_extent(data, mkey):
    """Every value a panel actually DRAWS: points, conf whiskers, band envelope.

    Re-pin review, fix 5: the scale was computed from the points and the conf
    MEANS only, so a band (or even a conf error bar) could render outside the
    plotted range. Both generators scale from this one function."""
    vals = [d["pts"][s][mkey] for d in data.values() for s in d["pts"] if mkey in d["pts"][s]]
    for d in data.values():
        if d.get("conf") and mkey in d["conf"]:
            m, sd = d["conf"][mkey]
            vals += [m - sd, m + sd]
        for b in (d.get("band") or {}).values():
            if mkey in b:
                m, sd = b[mkey]
                vals += [m - sd, m + sd]
    return (min(vals), max(vals)) if vals else (0.0, 1.0)


def series(k=8, refusals=None):
    data = {}
    for key, label, cl, cd, tree, n, dash in ARMS:
        pts, conf, band = {}, None, {}
        if tree == "POINT":
            pts = {}
            base = f"{REPO}/outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints"
            if k == 8:
                for f in glob.glob(f"{base}/*_metrics_*_exp07_P1_screen_S*_ema.json"):
                    if f.endswith(".screenmeta.json"): continue
                    step = int(f.split("_S")[1].split("_")[0])
                    if step <= 40000:
                        m = json.load(open(f)); m = m.get("metrics", m)
                        pts[step] = m
            p1conf = ({"T60": (8.993, 0.011), "C50": (1.0093, 0.0035), "EDT": (40.650, 0.101),
                       "RIR_to_GT_RIR_R@1": (5.173, 0.138), "RIR_to_GT_RIR_R@5": (15.430, 0.197),
                       "RIR_to_GT_RIR_R@10": (23.409, 0.056)}
                      if k == 8 else
                      {"T60": (10.287, 0.026), "C50": (1.0884, 0.0088), "EDT": (43.437, 0.397),
                       "RIR_to_GT_RIR_R@1": (4.990, 0.115), "RIR_to_GT_RIR_R@5": (15.111, 0.129),
                       "RIR_to_GT_RIR_R@10": (22.509, 0.181)})
            data[key] = dict(label=label, cl=cl, cd=cd, dash=dash, pts=pts, conf=p1conf)
            continue
        if tree:
            base = f"{REPO}/outputs_FLAC/{tree}/FLAC_{tree}/{tree}/checkpoints"
            suffix = f"a{n}" if n else "vanilla"
            for f in glob.glob(f"{base}/*_metrics_*_screen_S*_s42_K{k}_*{suffix}*.json"):
                if f.endswith(".screenmeta.json"): continue
                m = json.load(open(f)); m = m.get("metrics", m)
                pts[int(f.split("_S")[1].split("_")[0])] = m
            cf = [json.load(open(f)) for f in glob.glob(f"{base}/*_metrics_*conf_S40000_s4[2-6]_K{k}_*{suffix}*.json")
                  if not f.endswith(".screenmeta.json")]
            cf = [c.get("metrics", c) for c in cf]
            if len(cf) == 5:
                conf = {k: (statistics.mean([c[k] for c in cf]), statistics.stdev([c[k] for c in cf]))
                        for k, _, _ in METRICS}
            # Q10: above 40k the curve is 5-seed, so it carries a band. Below and
            # at 40k it stays the single-seed screen record plus the conf dot —
            # deliberately NOT mixed, or the shading would mean two different
            # things on the same line. Harvested through the validator, never by
            # counting files (see validated_band).
            band = validated_band(base, key, k, suffix, refusals=refusals)
        else:  # legacy C4: backfill screens + historical 5-seed conf row values
            base = f"{REPO}/outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
            for f in glob.glob(f"{base}/*exp11_C4backfill_S*_s42_K{k}_*a4.json"):
                if f.endswith(".screenmeta.json"): continue
                m = json.load(open(f)); m = m.get("metrics", m)
                pts[int(f.split("_S")[1].split("_")[0])] = m
            conf = ({"T60": (8.202, 0.017), "C50": (0.9778, 0.0015),
                     "EDT": (38.793, 0.074), "RIR_to_GT_RIR_R@1": (5.387, 0.075),
                     "RIR_to_GT_RIR_R@5": (16.456, 0.038), "RIR_to_GT_RIR_R@10": (24.198, 0.164)}
                    if k == 8 else
                    {"T60": (9.543, 0.054), "C50": (1.0559, 0.0040),
                     "EDT": (41.754, 0.347), "RIR_to_GT_RIR_R@1": (5.166, 0.166),
                     "RIR_to_GT_RIR_R@5": (16.071, 0.241), "RIR_to_GT_RIR_R@10": (23.721, 0.150)})
        data[key] = dict(label=label, cl=cl, cd=cd, dash=dash, pts=pts, conf=conf,
                         band=band if tree else {})
    return data

def svg_panel(data, mkey, mlabel, lower_better, W=560, H=340):
    P = dict(l=56, r=132, t=18, b=40)
    xs = sorted({s for d in data.values() for s in d["pts"]}
                | {s for d in data.values() for s in d.get("band", {})} | {40000})
    lo, hi = value_extent(data, mkey)       # points + conf whiskers + band envelope
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    def X(s): return P["l"] + (s - xs[0]) / (xs[-1] - xs[0]) * (W - P["l"] - P["r"])
    def Y(v): return P["t"] + (hi - v) / (hi - lo) * (H - P["t"] - P["b"])
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{mlabel} vs training step">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        out.append(f'<line x1="{P["l"]}" x2="{W-P["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{P["l"]-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    ticks = [10000, 20000, 30000, 40000]
    if xs[-1] > 40000:
        ticks += [t for t in (50000, 60000, 70000, 80000, 90000, 100000) if t <= xs[-1]]
    for s in ticks:
        out.append(f'<text x="{X(s):.1f}" y="{H-P["b"]+16}" class="tick" text-anchor="middle">{s//1000}k</text>')
    out.append(f'<text x="{(P["l"]+W-P["r"])/2:.0f}" y="{H-8}" class="axis">training step</text>')
    for key, d in data.items():
        pxy = sorted((s, m[mkey]) for s, m in d["pts"].items() if mkey in m)
        if not pxy: continue
        path = " ".join(f"{'M' if i==0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pxy))
        dash = f' stroke-dasharray="{d["dash"]}"' if d["dash"] else ""
        out.append(f'<path d="{path}" class="ln s-{key}"{dash}/>')
        for s, v in pxy:
            out.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3" class="pt s-{key}">'
                       f'<title>{d["label"]} @{s}: {v:.3f}</title></circle>')
        if d["conf"]:
            m, sd = d["conf"][mkey]
            x = X(40000) + 6
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(m-sd):.1f}" y2="{Y(m+sd):.1f}" class="err s-{key}"/>'
                       f'<circle cx="{x:.1f}" cy="{Y(m):.1f}" r="4.5" class="pt s-{key}">'
                       f'<title>{d["label"]} 40k conf (5-seed): {m:.3f} ± {sd:.3f}</title></circle>')
        # Q10 band: mean line + shaded +-1 sd above 40k. Shading (not error bars)
        # because at 2,500-step density 24 bars per arm would be unreadable; the
        # band carries the same information as an envelope.
        bxy = sorted((s, d["band"][s][mkey]) for s in d.get("band", {}) if mkey in d["band"][s])
        if bxy:
            anchor = [(40000, (d["conf"][mkey][0], d["conf"][mkey][1]))] if d["conf"] else []
            seq = anchor + bxy
            up = " ".join(f"{'M' if i==0 else 'L'}{X(s):.1f},{Y(m+sd):.1f}"
                          for i, (s, (m, sd)) in enumerate(seq))
            dn = " ".join(f"L{X(s):.1f},{Y(m-sd):.1f}" for s, (m, sd) in reversed(seq))
            out.append(f'<path d="{up} {dn} Z" class="band s-{key}" fill-opacity="0.15" stroke="none"/>')
            mid = " ".join(f"{'M' if i==0 else 'L'}{X(s):.1f},{Y(m):.1f}"
                           for i, (s, (m, sd)) in enumerate(seq))
            out.append(f'<path d="{mid}" class="ln s-{key}"{dash}/>')
            for s, (m, sd) in bxy:
                out.append(f'<circle cx="{X(s):.1f}" cy="{Y(m):.1f}" r="3" class="pt s-{key}">'
                           f'<title>{d["label"]} @{s} (5-seed): {m:.3f} ± {sd:.3f}</title></circle>')
        lx, ly = (bxy[-1][0], bxy[-1][1][0]) if bxy else pxy[-1]
        d.setdefault("_lbl", {})[mkey] = (
            X(lx) + 14,
            Y(ly if bxy else (d["conf"][mkey][0] if d["conf"] else ly)) + 4)
    # collision pass: sort by y, push apart to >=14px, then emit
    lbls = sorted(((d["_lbl"][mkey][1], d["_lbl"][mkey][0], key, d["label"]) for key, d in data.items()
                   if "_lbl" in d and mkey in d["_lbl"]), key=lambda t: t[0])
    adj = []
    for y, x, key, lab in lbls:
        if adj and y - adj[-1][0] < 14: y = adj[-1][0] + 14
        adj.append((y, x, key, lab))
    for y, x, key, lab in adj:
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" class="lbl s-{key}">{lab}</text>')
    arrow = "lower is better" if lower_better else "higher is better"
    out.append(f'<text x="{P["l"]}" y="{P["t"]-4}" class="axis">{mlabel} — {arrow}; dot at 40k = 5-seed mean ± sd</text>')
    out.append("</svg>")
    return "".join(out)

def band_cell(entry):
    """`mean ± sd [lo, hi]` — the extrema the shading spans, not just its centre."""
    if not entry:
        return "<td>—</td>"
    m, sd = entry
    return f"<td>{m:.3f} ± {sd:.3f}<br><span class=\"note\">[{m - sd:.3f}, {m + sd:.3f}]</span></td>"


def main():
    sections, refusals = [], []
    for k in (8, 1):
        data = series(k, refusals=refusals)
        sections.append((k, data))
    data = sections[0][1]
    css_series = "\n".join(
        f'.s-{k}{{stroke:{d["cl"]};fill:{d["cl"]}}} .lbl.s-{k}{{fill:{d["cl"]};stroke:none}}\n'
        f'[data-theme="dark"] .s-{k},.dark-auto .s-{k}{{stroke:{d["cd"]};fill:{d["cd"]}}}'
        for k, d in data.items())
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.4rem"><svg width="26" height="8">'
        f'<line x1="0" x2="26" y1="4" y2="4" class="ln s-{k}"'
        + (f' stroke-dasharray="{d["dash"]}"' if d["dash"] else "") + "/></svg>"
        f'<span class="note">{d["label"]}</span></span>' for k, d in data.items())
    blocks = ""
    for k, dk in sections:
        pan = "".join(f'<figure>{svg_panel(dk, mk, ml, lb)}</figure>' for mk, ml, lb in METRICS)
        note = "" if k == 8 else "<p class=\"note\">K=1 note: P1 legacy has no sub-40k K=1 raws (exp_07 screened P1 at K=8 only) — only its published 5-seed 40k point; C4 legacy-loop K=1 backfill exists at 20k/30k (the 40k cell was refused at the campaign pin and is omitted).</p>"
        blocks += f'<h2 style="font-size:1.05rem">K = {k}</h2>{note}<main>{pan}</main>'
    panels = blocks
    # Table view: the plotted screen values AND, for every 5-seed row, the band
    # EXTREMA that the shading actually spans (re-pin review, fix 5) — a reader
    # can now read the envelope off the page instead of measuring the shading.
    rows = []
    for k, d in data.items():
        for s in sorted(d["pts"]):
            m = d["pts"][s]
            rows.append(f"<tr><td>{d['label']}</td><td>{s}</td><td>screen (s42)</td>" +
                        "".join(f"<td>{m.get(mk, float('nan')):.3f}</td>" for mk, _, _ in METRICS) +
                        "</tr>")
        if d["conf"]:
            rows.append(f"<tr><td>{d['label']}</td><td>40000</td><td>conf 5-seed</td>" +
                        "".join(band_cell(d["conf"].get(mk)) for mk, _, _ in METRICS) + "</tr>")
        for s in sorted(d.get("band") or {}):
            b = d["band"][s]
            rows.append(f"<tr><td>{d['label']}</td><td>{s}</td><td>traj 5-seed</td>" +
                        "".join(band_cell(b.get(mk)) for mk, _, _ in METRICS) + "</tr>")
    refusal_note = ""
    if refusals:
        items = "".join(
            f"<li>{r['arm']} K={r['K']} S{r['step']}: {r['files']} file(s) present, refused — "
            f"{r['problems'][0]}</li>" for r in refusals)
        refusal_note = ('<p class="note"><strong>Refused trajectory blocks (disclosed):</strong> '
                        "a >40k step is drawn only when its whole cell passes "
                        "<code>validate_cell(contract=&quot;traj&quot;)</code> — five unique seeds "
                        "42–46, one checkpoint/config/source SHA, on the 2,500 grid at or below "
                        f"100000. These did not:</p><ul class=\"note\">{items}</ul>")
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>exp_11 trajectories — acoustic metrics vs training step</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#d9dee5}}
@media(prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#3a4048}}}}
[data-theme="dark"]{{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#3a4048}}
[data-theme="light"]{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#d9dee5}}
body{{margin:0 auto;max-width:76rem;padding:1.5rem;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
.grid{{stroke:var(--line);stroke-width:1}} .tick,.axis{{fill:var(--mut);font-size:11px;stroke:none}}
.ln{{fill:none;stroke-width:2}} .pt{{stroke:var(--bg);stroke-width:2}} .err{{stroke-width:2}}
.lbl{{font-size:11.5px;font-weight:600}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}}
figure{{margin:0;overflow-x:auto}} svg{{width:100%;height:auto}}
table{{border-collapse:collapse;font-size:.85rem;margin-top:1rem}} td,th{{border:1px solid var(--line);padding:.25rem .5rem}}
details{{margin-top:1rem}} .note{{color:var(--mut);font-size:.85rem}}
</style></head><body>
<h1 style="font-size:1.2rem">exp_11 — acoustic performance vs training step (K=8, full unseen split)</h1>
<p class="note">Lines up to 40k: single-seed (s42) fa-protocol screens every 2,500–5,000 steps, each arm under its own orbit.
The 40k marker is the 5-seed confirmatory block (mean ± sd). Beyond 40k (Q10 extension to 100k) the line is the
5-seed trajectory mean with a shaded ±1 sd band — shading rather than error bars because at 2,500-step density the
bars would collide. A step is drawn only when all five seeds are present, so a partially-landed block never appears
as a narrower band.
Terminal dots: 5-seed confirmatory mean ± sd at the 40k checkpoint. C4 legacy-loop (dashed pink) is the historical
recipe — background context, not the inferential comparator; C32 extends as its screens land; VANL (green, dashed) trains now — its screens backfill after the post-C32 re-pin;
P1 vanilla legacy (violet, dotted): exp_07 EMA screens at that arm's 10k cadence (s42; finer 2.5k-grid points
were never run for P1 — 12.5k/15k/…/37.5k do not exist as raws) with its published 5-seed 40k dot.</p>
<div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin:.4rem 0 .8rem">{legend}</div>
{refusal_note}
{panels}
<details><summary>Table view (all plotted values; 5-seed rows carry the band extrema)</summary>
<div style="overflow-x:auto"><table><tr><th>arm</th><th>step</th><th>source</th>
{''.join(f'<th>{ml}</th>' for _, ml, _ in METRICS)}</tr>
{''.join(rows)}</table></div></details>
<style>{css_series}</style></body></html>"""
    open(OUT, "w").write(html)
    npts = sum(len(d["pts"]) for d in data.values())
    nband = sum(len(d.get("band") or {}) for _, dk in sections for d in dk.values())
    print(f"wrote {OUT} ({npts} screen points, "
          f"{sum(1 for d in data.values() if d['conf'])} conf endpoints, "
          f"{nband} validated 5-seed trajectory points)")
    for r in refusals:
        print(f"  REFUSED {r['arm']} K={r['K']} S{r['step']} ({r['files']} file(s)): {r['problems'][0]}")

if __name__ == "__main__":
    main()
