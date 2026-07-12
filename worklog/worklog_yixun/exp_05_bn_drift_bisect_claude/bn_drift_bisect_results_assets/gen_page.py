"""Generates bn_drift_bisect_01_results.html (sources: _results.md + committed B0/B1 JSONs)."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
reps = [json.load(open(os.path.join(EXP, f'bn_drift_B0_train_seed{s}.json'))) for s in (42,43,44)]
ev = json.load(open(os.path.join(EXP, 'bn_drift_B0_eval_seed42.json')))
layers = list(reps[0]['per_layer'].keys())
def driftchart():
    W,LM,RH = 780,235,20; PW=W-LM-60; H=52+RH*len(layers)+18; hi=0.85
    def X(v): return LM+min(v,hi)/hi*PW
    s=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="per-layer BN input drift, train vs eval loader">']
    for gv in (0.05,0.2,0.4,0.6,0.8):
        s.append(f'<line class="axis" x1="{X(gv)}" y1="40" x2="{X(gv)}" y2="{40+RH*len(layers)}"/><text x="{X(gv)}" y="{52+RH*len(layers)}" text-anchor="middle">{gv}</text>')
    for i,L in enumerate(layers):
        y=40+RH*i+3; ms=[r['per_layer'][L]['mean_shift_max'] for r in reps]; mn,mx=min(ms),max(ms); mean=sum(ms)/3
        evv=ev['per_layer'][L]['mean_shift_max']
        s.append(f'<text x="{LM-8}" y="{y+10}" text-anchor="end">{L.replace("cnn.","")}</text>')
        s.append(f'<rect class="mark" x="{LM}" y="{y}" width="{X(mean)-LM:.1f}" height="12" rx="4" fill="var(--s1)"><title>{L}: train mean_shift_max {mean:.3f} (repeats {mn:.3f}–{mx:.3f})</title></rect>')
        s.append(f'<line x1="{X(mn)}" y1="{y+6}" x2="{X(mx)}" y2="{y+6}" stroke="var(--ink)" stroke-width="1.5"/>')
        s.append(f'<circle class="mark" cx="{X(evv)}" cy="{y+6}" r="4" fill="var(--s3)" stroke="var(--surface)" stroke-width="2"><title>{L}: EVAL loader {evv:.3f}</title></circle>')
    s.append(f'<text x="{LM}" y="16" class="val">BN-input drift per layer — standardized max mean-shift (bars: train loader, 3-repeat range; dots: eval loader)</text>')
    s.append(f'<rect x="{LM}" y="24" width="12" height="10" rx="3" fill="var(--s1)"/><text x="{LM+18}" y="33">train</text>')
    s.append(f'<circle cx="{LM+70}" cy="29" r="4" fill="var(--s3)"/><text x="{LM+80}" y="33">eval (unseen rooms — discrimination reference)</text>')
    s.append('</svg>'); return "".join(s)
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp_05 — BN drift bisection</title><link rel="stylesheet" href="bn_drift_bisect_results_assets/results.css"></head><body>
<h1>exp_05 — BN buffers as a drift probe: bisection & the per-metric damage decomposition</h1>
<p class="sub">Probe TDD'd + review-hardened (fail-fast load, device-correct, no-mutation asserted) · 2026-07-06. Numbers mirror <a href="bn_drift_bisect_results.md">_results.md</a>; raw JSONs in this folder.</p>
<h2>B0 — the drift landscape</h2>
<figure>{driftchart()}<figcaption>All 20 layers exceed the 0.05 pre-registered threshold on the train loader, with monotone depth amplification (stem 0.082 → layer4 0.357); the eval loader (different rooms) drifts 2–4× more — the probe discriminates. Repeats agree to ±0.001–0.02. Hover for exact values.</figcaption></figure>
<h2>B1 — max_len grid: <span class="chip pass">loader exonerated</span></h2>
<table><thead><tr><th>max_len (alternatives: single seed-42 probes)</th><th class="num">stem drift</th><th class="num">worst-layer drift</th><th>verdict</th></tr></thead><tbody>
<tr><td><b>9600 (shipped)</b> — 3-repeat mean</td><td class="num"><b>0.082</b></td><td class="num"><b>0.357</b></td><td><span class="chip pass">✓ clear optimum</span></td></tr>
<tr><td>4800</td><td class="num">0.647</td><td class="num">1.121</td><td><span class="chip fail">✗ far worse</span></td></tr>
<tr><td>10240</td><td class="num">0.117</td><td class="num">1.393</td><td><span class="chip fail">✗ layer4 explodes</span></td></tr>
<tr><td>19200</td><td class="num">0.620</td><td class="num">1.685</td><td><span class="chip fail">✗ worst</span></td></tr></tbody></table>
<p>Dispersion check: observed max shift 0.085 vs predicted EMA-tail noise 0.024 (≈3.5×) — residual drift is real but small; EMA-tail refuted as sole cause.</p>
<h2>V1′ — BN-frozen control: <span class="chip fail">✗ gate fail</span> — but the decomposition is the result</h2>
<table><thead><tr><th>Metric (K=1, baseline)</th><th class="num">R1b — unfrozen FT</th><th class="num">W0 — lr=0, BN only</th><th class="num">V1′ — frozen FT</th><th>verdict</th></tr></thead><tbody>
<tr><td>EDT (39.95)</td><td class="num">43.27</td><td class="num">41.10</td><td class="num">41.33</td><td>largely BN-mediated; gradient residual ≈ +1.4 ms</td></tr>
<tr><td>C50 (1.046)</td><td class="num">1.078</td><td class="num">1.050</td><td class="num"><b>1.010</b></td><td><span class="chip pass">✓ BETTER than baseline</span> (frozen stats + trainable affine)</td></tr>
<tr><td>T60 (9.97)</td><td class="num">10.47</td><td class="num">10.13</td><td class="num">10.52</td><td><span class="chip fail">gradient-driven, BN-independent</span></td></tr>
<tr><td>R@1</td><td class="num">baseline</td><td class="num">baseline</td><td class="num">baseline</td><td>never damaged</td></tr></tbody></table>
<div class="verdict mixed"><b>Outcome.</b> The fixable channel is fixed (freeze-bn recipe: EDT mostly recovered, C50 beats baseline at both K); the unfixable-from-artifact channel is isolated (T60, gradient path) and handed to exp_06 with a short suspect list. Falsified across exp_03–05: Adam transient · EMA · batch noise (sole) · BN mutation (sole) · max_len · lr magnitude (single value).</div>
<p class="foot">Instrument: <code>tools/bn_drift_probe.py</code> (fail-fast provenance; buffers bit-identity asserted in every probe) · Generated by <code>bn_drift_bisect_results_assets/gen_page.py</code>.</p>
</body></html>"""
out = os.path.join(EXP, 'bn_drift_bisect_01_results.html')
open(out,'w').write(html); print('wrote', out)
