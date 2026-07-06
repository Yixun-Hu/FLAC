"""Generates fa_invariant_cond_01_results.html (source: _results.md)."""
import os
def zbars(title, rows, xmax):
    # rows: (label, z, pass_bool); horizontal bars, direct labels
    W,LM,RH = 720,190,24; PW = W-LM-70; H = 40+RH*len(rows)+16
    s=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">']
    x2 = LM + 2.0/xmax*PW
    s.append(f'<line class="axis" x1="{x2}" y1="26" x2="{x2}" y2="{26+RH*len(rows)}"/><text x="{x2}" y="{38+RH*len(rows)}" text-anchor="middle">2σ gate</text>')
    for i,(lab,z,ok) in enumerate(rows):
        y = 26+RH*i+4; w = min(z,xmax)/xmax*PW
        col = 'var(--good)' if ok else 'var(--crit)'
        s.append(f'<text x="{LM-8}" y="{y+11}" text-anchor="end">{lab}</text>')
        s.append(f'<rect class="mark" x="{LM}" y="{y}" width="{w:.1f}" height="14" rx="4" fill="{col}"><title>{lab}: {z}σ {"PASS" if ok else "FAIL"}</title></rect>')
        s.append(f'<text class="val" x="{LM+w+6}" y="{y+11}">{z}σ {"✓" if ok else "✗"}</text>')
    s.append(f'<text x="{LM}" y="14" class="val">{title}</text></svg>')
    return "".join(s)
r1 = [("K=1 T60", 9.70, False), ("K=1 C50", 5.19, False), ("K=1 EDT", 11.90, False), ("K=1 R@1", 1.82, True),
      ("K=8 T60", 57.56, False), ("K=8 C50", 16.74, False), ("K=8 EDT", 62.41, False), ("K=8 R@1", 3.65, False)]
r1b= [("K=1 T60", 6.22, False), ("K=1 C50", 2.65, False), ("K=1 EDT", 8.06, False), ("K=1 R@1", 0.16, True),
      ("K=8 T60", 38.57, False), ("K=8 C50", 6.16, False), ("K=8 EDT", 41.88, False), ("K=8 R@1", 0.11, True)]
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp_03 — hard invariant conditioning</title><link rel="stylesheet" href="fa_invariant_cond_results_assets/results.css"></head><body>
<h1>exp_03 — Route 1: hard invariant conditioning (frame averaging + cylindrical poses)</h1>
<p class="sub">DINOv3 untouched · 6 TDD rounds, 7 Codex reviews, 83→104 tests · 2026-07-04/05. Numbers mirror <a href="fa_invariant_cond_results.md">_results.md</a> and the R1/R1b gate entries in <a href="fa_invariant_cond_worklog.md">the lab notebook</a>.</p>
<h2>H1 — conditioning-level hard symmetry: <span class="chip pass">✓ PROVEN</span></h2>
<div><span class="kpi"><div class="n">4.9×10⁻⁸</div><div class="l">relative conditioning deviation on C₄ (real DINOv3 + real data)</div></span>
<span class="kpi"><div class="n">×1200</div><div class="l">VAE-decoder amplification of float dust (the entire e2e residual)</div></span>
<span class="kpi"><div class="n">200–400×</div><div class="l">margin below the vanilla model's exp_02 gap</div></span>
<span class="kpi"><div class="n">11 / 6337</div><div class="l">real eval items exercising the degenerate fallback (reviews made it invariance-correct)</div></span></div>
<div class="verdict good"><b>The mechanism works by construction.</b> <span class="eq">c<sub>inv</sub>(x) = 1/|G| Σ<sub>g∈G</sub> f(g·x)</span> over G = C₄ for the ViT path (exact on 90°·k; panorama roll is integer), plus intrinsically invariant pose features <span class="eq">(r, z, Δφ)</span> exact at <i>any</i> angle. Determinism control: exactly 0. Off-subgroup 45° residual (ViT path only): ~0.2 zero-shot — the pre-registered known limitation.</div>
<h2>R0 — zero-shot fa_invariant on the frozen checkpoint (K=1)</h2>
<table><thead><tr><th></th><th class="num">T60↓</th><th class="num">C50↓</th><th class="num">EDT↓</th><th class="num">R@1↑</th></tr></thead><tbody>
<tr><td>baseline (vanilla)</td><td class="num">9.969</td><td class="num">1.046</td><td class="num">39.95</td><td class="num">6.83</td></tr>
<tr><td>R0 zero-shot fa_invariant</td><td class="num">10.082</td><td class="num"><b>1.038</b></td><td class="num">42.02</td><td class="num">5.38</td></tr></tbody></table>
<p>Mild OOD degradation, as predicted — symmetrized conditioning is off-distribution for the frozen DiT; the fine-tune's job was to close ~0.1 T60 / ~2 ms EDT / ~1.4 pp R@1.</p>
<h2>R1 / R1b — vanilla control fine-tunes: <span class="chip fail">✗ GATE FAIL ×2 → registered stop</span></h2>
<figure>{zbars('R1 (lr 5e-6 const, effective batch 8, 10k steps): deviation from exp_01 baseline', r1, 65)}</figure>
<figure>{zbars('R1b (amended: effective batch 128 = original parity, same 80k-sample budget)', r1b, 65)}<figcaption>Batch parity recovered ~35–40% of the regression and brought retrieval exactly to baseline (green) — but T60/C50/EDT stay far outside the 2σ gate. Per the pre-registered stop: R2–R4 were not launched; an fa_invariant result would be uninterpretable through a failing control.</figcaption></figure>
<h2>Gate-failure diagnostics</h2>
<table><thead><tr><th>Hypothesis</th><th>Test</th><th>Outcome</th></tr></thead><tbody>
<tr><td>EMA-vs-online weights</td><td>eval EMA-stripped <code>FLAC.ckpt</code> (corrected after a loader-confound first attempt)</td><td><span class="chip warnc">≤15% of effect</span></td></tr>
<tr><td>Effective-batch noise</td><td>R1 (batch 8) → R1b (batch 128)</td><td><span class="chip warnc">~35–40% recovery</span></td></tr>
<tr><td>Learning-rate magnitude</td><td>5e-6 is 8× below the original schedule's end value</td><td><span class="chip info">excluded (this exp)</span></td></tr></tbody></table>
<div class="verdict mixed"><b>Headline finding.</b> The released FLAC checkpoint could not be non-destructively fine-tuned by any recipe tested here — the blocker sits deeper than symmetry, and it retroactively explains the pre-revert "inconclusive FA" result: the confound was never frame averaging.</div>
<p class="foot">Trail: plan + plan review + 6 per-round reviews + integrative GO-WITH-CONDITIONS in this folder · commits in <code>commits_fa_invariant_cond.md</code> · Generated by <code>fa_invariant_cond_results_assets/gen_page.py</code>.</p>
</body></html>"""
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fa_invariant_cond_01_results.html')
open(out,'w').write(html); print('wrote', out)
