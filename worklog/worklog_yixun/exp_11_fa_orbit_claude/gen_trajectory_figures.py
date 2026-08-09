#!/usr/bin/env python3
"""exp_11 trajectory figures — acoustic metrics vs training step, per arm.

Reads the on-disk screen/conf metric JSONs (K8, s42 screens; 5-seed conf at
40k rendered as mean +/- sd) and emits a fully self-contained HTML page with
inline-SVG line charts (one panel per metric, one series per arm), light/dark
via prefers-color-scheme + data-theme, hover crosshair, and a table view.
Planner one-off; reviewed with the closing package. Numbers come only from the
metric JSONs — this page is presentation, never a source.
"""
import json, glob, math, os, statistics

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
    ("P1v",  "P1 vanilla legacy", "#4a3aa7", "#9085e9", "POINT",     0,  ""),
]

def series():
    data = {}
    for key, label, cl, cd, tree, n, dash in ARMS:
        pts, conf = {}, None
        if tree == "POINT":
            data[key] = dict(label=label, cl=cl, cd=cd, dash=dash, pts={},
                conf={"T60": (8.993, 0.011), "C50": (1.0093, 0.0035), "EDT": (40.650, 0.101),
                      "RIR_to_GT_RIR_R@1": (5.173, 0.138), "RIR_to_GT_RIR_R@5": (15.430, 0.197),
                      "RIR_to_GT_RIR_R@10": (23.409, 0.056)})
            continue
        if tree:
            base = f"{REPO}/outputs_FLAC/{tree}/FLAC_{tree}/{tree}/checkpoints"
            suffix = f"a{n}" if n else "vanilla"
            for f in glob.glob(f"{base}/*_metrics_*_screen_S*_s42_K8_*{suffix}*.json"):
                if f.endswith(".screenmeta.json"): continue
                m = json.load(open(f)); m = m.get("metrics", m)
                pts[int(f.split("_S")[1].split("_")[0])] = m
            cf = [json.load(open(f)) for f in glob.glob(f"{base}/*_metrics_*conf_S40000_s4[2-6]_K8_*{suffix}*.json")
                  if not f.endswith(".screenmeta.json")]
            cf = [c.get("metrics", c) for c in cf]
            if len(cf) == 5:
                conf = {k: (statistics.mean([c[k] for c in cf]), statistics.stdev([c[k] for c in cf]))
                        for k, _, _ in METRICS}
        else:  # legacy C4: backfill screens + historical 5-seed conf row values
            base = f"{REPO}/outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
            for f in glob.glob(f"{base}/*exp11_C4backfill_S*_s42_K8_*a4.json"):
                if f.endswith(".screenmeta.json"): continue
                m = json.load(open(f)); m = m.get("metrics", m)
                pts[int(f.split("_S")[1].split("_")[0])] = m
            conf = {"T60": (8.202, 0.017), "C50": (0.9778, 0.0015),
                    "EDT": (38.793, 0.074), "RIR_to_GT_RIR_R@1": (5.387, 0.075),
                    "RIR_to_GT_RIR_R@5": (16.456, 0.038), "RIR_to_GT_RIR_R@10": (24.198, 0.164)}
        data[key] = dict(label=label, cl=cl, cd=cd, dash=dash, pts=pts, conf=conf)
    return data

def svg_panel(data, mkey, mlabel, lower_better, W=560, H=340):
    P = dict(l=56, r=132, t=18, b=40)
    xs = sorted({s for d in data.values() for s in d["pts"]} | {40000})
    vals = [d["pts"][s][mkey] for d in data.values() for s in d["pts"] if mkey in d["pts"][s]]
    vals += [d["conf"][mkey][0] for d in data.values() if d["conf"]]
    lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    def X(s): return P["l"] + (s - xs[0]) / (xs[-1] - xs[0]) * (W - P["l"] - P["r"])
    def Y(v): return P["t"] + (hi - v) / (hi - lo) * (H - P["t"] - P["b"])
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{mlabel} vs training step">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        out.append(f'<line x1="{P["l"]}" x2="{W-P["r"]}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{P["l"]-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for s in [10000, 20000, 30000, 40000]:
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
        lx, ly = pxy[-1]
        d.setdefault("_lbl", {})[mkey] = (X(lx) + 14, Y(d["conf"][mkey][0] if d["conf"] else ly) + 4)
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

def main():
    data = series()
    css_series = "\n".join(
        f'.s-{k}{{stroke:{d["cl"]};fill:{d["cl"]}}} .lbl.s-{k}{{fill:{d["cl"]};stroke:none}}\n'
        f'[data-theme="dark"] .s-{k},.dark-auto .s-{k}{{stroke:{d["cd"]};fill:{d["cd"]}}}'
        for k, d in data.items())
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.4rem"><svg width="26" height="8">'
        f'<line x1="0" x2="26" y1="4" y2="4" class="ln s-{k}"'
        + (f' stroke-dasharray="{d["dash"]}"' if d["dash"] else "") + "/></svg>"
        f'<span class="note">{d["label"]}</span></span>' for k, d in data.items())
    panels = "".join(f'<figure>{svg_panel(data, mk, ml, lb)}</figure>' for mk, ml, lb in METRICS)
    rows = []
    for k, d in data.items():
        for s in sorted(d["pts"]):
            m = d["pts"][s]
            rows.append(f"<tr><td>{d['label']}</td><td>{s}</td>" +
                        "".join(f"<td>{m.get(mk, float('nan')):.3f}</td>" for mk, _, _ in METRICS) + "</tr>")
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
<p class="note">Lines: single-seed (s42) fa-protocol screens every 2,500–5,000 steps, each arm under its own orbit.
Terminal dots: 5-seed confirmatory mean ± sd at the 40k checkpoint. C4 legacy-loop (dashed pink) is the historical
recipe — background context, not the inferential comparator; C32 extends as its screens land; VANL (green, dashed) trains now — its screens backfill after the post-C32 re-pin;
P1 vanilla legacy (violet) contributes its published 5-seed 40k point.</p>
<div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin:.4rem 0 .8rem">{legend}</div>
<main>{panels}</main>
<details><summary>Table view (all plotted screen values)</summary>
<div style="overflow-x:auto"><table><tr><th>arm</th><th>step</th><th>T60</th><th>C50</th><th>EDT</th><th>R@1</th></tr>
{''.join(rows)}</table></div></details>
<style>{css_series}</style></body></html>"""
    open(OUT, "w").write(html)
    npts = sum(len(d["pts"]) for d in data.values())
    print(f"wrote {OUT} ({npts} screen points, {sum(1 for d in data.values() if d['conf'])} conf endpoints)")

if __name__ == "__main__":
    main()
