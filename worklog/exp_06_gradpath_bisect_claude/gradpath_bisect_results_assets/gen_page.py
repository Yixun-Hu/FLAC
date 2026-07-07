"""Generates gradpath_bisect_01_results.html (source: _results.md)."""
import os, math
S1 = {'R1b': [(0, 8.609), (200, 9.071), (400, 9.282), (625, 9.195)],
      'V1p': [(0, 8.609), (200, 9.089), (400, 9.253), (625, 9.235)]}
S2 = [('L1', 5e-7, 9.087), ('L2 (V1p)', 5e-6, 9.235), ('L3', 2e-5, 9.596), ('L4', 4.2e-5, 9.866), ('L5 restart', 5e-5, 10.099)]
BASE = 8.609
def s1chart():
    W,H,LM,BM = 560,240,64,34; PW,PH = W-LM-20, H-BM-30
    lo, hi = 8.5, 9.45
    def X(st): return LM + st/625*PW
    def Y(v): return 26 + (hi-v)/(hi-lo)*PH
    COL = {'R1b': 'var(--s1)', 'V1p': 'var(--s2)'}
    s=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="T60 vs optimizer step">']
    s.append(f'<line class="axis" x1="{LM}" y1="{Y(BASE)}" x2="{W-16}" y2="{Y(BASE)}" stroke-dasharray="4 3"/><text x="{W-16}" y="{Y(BASE)-5}" text-anchor="end">baseline 8.609</text>')
    for st in (0,200,400,625):
        s.append(f'<text x="{X(st)}" y="{H-12}" text-anchor="middle">{st}</text>')
    for name, pts in S1.items():
        poly = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a,b in pts)
        s.append(f'<polyline points="{poly}" fill="none" stroke="{COL[name]}" stroke-width="2"/>')
        for a,b in pts:
            s.append(f'<circle class="mark" cx="{X(a)}" cy="{Y(b)}" r="4.5" fill="{COL[name]}"><title>{name} step {a}: T60 {b}</title></circle>')
        s.append(f'<text class="val" x="{X(625)+6}" y="{Y(pts[-1][1])+4}" fill="{COL[name]}">{name}</text>')
    s.append(f'<text x="{LM}" y="14" class="val">S1 — K=8 T60 vs optimizer step: ~75–80% of the damage lands by step 200</text>')
    s.append(f'<text x="{LM}" y="{H-1}" ></text><text x="{X(300)}" y="{H-1}" text-anchor="middle">optimizer step</text></svg>')
    return "".join(s)
def s2chart():
    W,H,LM,BM = 640,250,64,40; PW,PH = W-LM-30, H-BM-30
    lo, hi = 8.5, 10.3
    lrs = [x[1] for x in S2]
    lmin, lmax = math.log10(min(lrs)), math.log10(max(lrs))
    def X(lr): return LM + (math.log10(lr)-lmin)/(lmax-lmin)*PW
    def Y(v): return 26 + (hi-v)/(hi-lo)*PH
    s=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="T60 vs learning rate (log scale)">']
    s.append(f'<line class="axis" x1="{LM}" y1="{Y(BASE)}" x2="{W-16}" y2="{Y(BASE)}" stroke-dasharray="4 3"/><text x="{W-16}" y="{Y(BASE)-5}" text-anchor="end">baseline 8.609</text>')
    poly = " ".join(f"{X(lr):.1f},{Y(v):.1f}" for _,lr,v in S2)
    s.append(f'<polyline points="{poly}" fill="none" stroke="var(--s6)" stroke-width="2"/>')
    for name,lr,v in S2:
        s.append(f'<circle class="mark" cx="{X(lr)}" cy="{Y(v)}" r="5" fill="var(--s6)"><title>{name} (lr {lr:g}): T60 {v}</title></circle>')
        s.append(f'<text class="val" x="{X(lr)}" y="{Y(v)-10}" text-anchor="middle">{v}</text>')
        s.append(f'<text x="{X(lr)}" y="{H-24}" text-anchor="middle">{lr:g}</text>')
        s.append(f'<text x="{X(lr)}" y="{H-10}" text-anchor="middle">{name}</text>')
    s.append(f'<text x="{LM}" y="14" class="val">S2 — K=8 T60 vs learning rate (log axis): monotone WORSE with lr; no arm near the gate</text></svg>')
    return "".join(s)
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp_06 — gradient-path bisection</title><link rel="stylesheet" href="gradpath_bisect_results_assets/results.css"></head><body>
<h1>exp_06 — Why fine-tuning can't recover T60/EDT, and the lr axis</h1>
<p class="sub">Full split everywhere; screens K=8 seed 42 (pre-registered ordering-only) · 2026-07-06. Numbers mirror <a href="gradpath_bisect_results.md">_results.md</a>.</p>
<div class="verdict bad"><b>Q1 answered — it's convergence, not corruption.</b> ~75–80% of the T60 shift lands within 200 optimizer steps and then flattens: the model rapidly converges toward the optimum of the objective we actually train — which is not where the released weights sit.</div>
<figure>{s1chart()}<figcaption>Interval-checkpoint evals of the exp_03/05 controls (identical with and without frozen BN → the T60 path is BN-independent).</figcaption></figure>
<div class="verdict bad"><b>Q2 answered — no tested lr recovers the gate; damage is monotone-increasing in lr.</b> The schedule-faithful InverseLR restart (the original recipe restored, plus every repair from exp_03–05) is the <i>most</i> destructive arm. The 5e-6 constant used since exp_03 was conservative, not the culprit.</div>
<figure>{s2chart()}<figcaption>Five arms, otherwise identical recipes (freeze-bn, effective batch 128, 625 steps, seed 42). Finalist thresholds (T60 ≤ 8.65 or EDT ≤ 37.3): none reached.</figcaption></figure>
<h2>S3 — lineage audit</h2>
<table><thead><tr><th>Candidate</th><th>Test</th><th>Outcome</th></tr></thead><tbody>
<tr><td>Training-path code drift</td><td>git diff vs <code>upstream/master</code> (AmandineBtto/FLAC)</td><td><span class="chip pass">ELIMINATED</span> — only the fork's equi-test file + a device_map nit</td></tr>
<tr><td>Target truncation</td><td>energy beyond the 10240 loss window</td><td><span class="chip pass">dead</span> (0.08% mean)</td></tr>
<tr><td>Augmentation target bias</td><td>paired aug-vs-raw metrics, train split</td><td><span class="chip warnc">partial-EDT only</span> — T60 bias ~0.05, 10× too small</td></tr></tbody></table>
<div class="verdict mixed"><b>Unified conclusion.</b> Under upstream-identical code, training walks monotonically away from the released point — the released FLAC_EMA is not near this objective's optimum on the data/environment we possess. Surviving explanations sit outside recipe space: dataset-version or library lineage, or source-side checkpoint selection. Fine-tune-based H3-vs-exp_01 is closed; Routes A (matched comparison) / B (from-scratch fa_invariant) stand.</div>
<p class="foot">Falsified across exp_03–06: Adam transient · EMA · batch noise · BN mutation · max_len · warmup · lr (×100 range incl. schedule restart). Generated by <code>gradpath_bisect_results_assets/gen_page.py</code>.</p>
</body></html>"""
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gradpath_bisect_01_results.html')
open(out,'w').write(html); print('wrote', out)
