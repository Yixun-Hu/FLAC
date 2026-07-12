"""Generates fa_matched_01_results.html (source: fa_matched_results.md).
Numbers mirror aggregate_results.py over the committed per-seed JSONs, EXCEPT two context
constants pulled from prior experiments: A_BASE (released baseline) from exp_01, and Chart C's
'vanilla C4 mean' from exp_02's Metric-1 gaps (0.221354/0.193486/0.213683 -> mean 0.2095079).
Palette: shared reference instance in results.css (already dataviz-validated in exp_01-06; unchanged here)."""
import os

# ---------------- data (from aggregate_results.py / committed JSONs) ----------------
# Chart A: T60 above the released baseline (pp) — lower bar = less fine-tune damage
A_BASE = {'K=1': 9.969, 'K=8': 8.609}
A_ROWS = {'K=1': {'A-V': 10.6473, 'A-F': 10.3716}, 'K=8': {'A-V': 9.3549, 'A-F': 8.9156}}
# Chart B: M5 training-seed downgrade (K=8). ratio = worst|dseed| / |FA_eff|; downgrade if >= 0.5
B_ROWS = [('T60', 0.1878, 0.4552), ('C50', 0.0185, 0.0229), ('EDT', 0.6398, 0.5239)]
# Chart C: Metric-1 relative-L2 (log axis). vanilla = exp_02 C4-gap mean; A-F = exp_08 C4-gap mean.
C_ROWS = [('vanilla C₄ mean (exp_02)', 0.2095079, 'crit'),
          ('A-F 45° off-C₄ (structural)', 0.2064, 'serious'),
          ('registered bf16 floor', 0.00931, 'floor'),
          ('A-F C₄ mean {90,180,270}°', 0.0023352, 'good')]
# Chart D: T60 vs panorama-rotation angle
D_K1 = [(0, 10.3935), (45, 10.8272), (90, 10.394), (180, 10.3936), (270, 10.3944)]
D_K8 = [(0, 8.9141), (90, 8.913)]

SC = {'good': 'var(--good)', 'serious': 'var(--serious)', 'crit': 'var(--crit)', 'floor': 'var(--ink3)'}


def chartA():
    W, H, LM, BM, TM = 560, 240, 46, 46, 30
    PW, PH = W - LM - 20, H - BM - TM
    hi = 0.8
    groups = ['K=1', 'K=8']
    gw = PW / len(groups)
    bw = 40
    COL = {'A-V': 'var(--s3)', 'A-F': 'var(--s1)'}

    def Y(v):
        return TM + (1 - v / hi) * PH
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="T60 above released baseline by K">']
    # y grid + labels
    for gv in (0, 0.2, 0.4, 0.6, 0.8):
        s.append(f'<line class="axis" x1="{LM}" y1="{Y(gv):.1f}" x2="{W-16}" y2="{Y(gv):.1f}"/>')
        if gv != 0:
            s.append(f'<text x="{LM-6}" y="{Y(gv)+3:.1f}" text-anchor="end">{gv:.1f}</text>')
    s.append(f'<text x="{LM-6}" y="{Y(0)+3:.1f}" text-anchor="end" class="val">base</text>')
    for gi, g in enumerate(groups):
        cx = LM + gi * gw + gw / 2
        pairs = [('A-V', A_ROWS[g]['A-V']), ('A-F', A_ROWS[g]['A-F'])]
        for bi, (name, val) in enumerate(pairs):
            above = val - A_BASE[g]
            x = cx - bw - 2 + bi * (bw + 4)
            y = Y(above)
            s.append(f'<rect class="mark" x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{Y(0)-y:.1f}" rx="4" fill="{COL[name]}">'
                     f'<title>{name} {g}: T60 {val:.3f} = baseline+{above:.3f}</title></rect>')
            s.append(f'<text class="val" x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle">+{above:.2f}</text>')
        s.append(f'<text x="{cx:.1f}" y="{H-24}" text-anchor="middle" class="val">{g}</text>')
    # legend
    s.append(f'<rect x="{W-150}" y="8" width="10" height="10" rx="2" fill="var(--s3)"/><text x="{W-136}" y="17">A-V vanilla</text>')
    s.append(f'<rect x="{W-150}" y="22" width="10" height="10" rx="2" fill="var(--s1)"/><text x="{W-136}" y="31">A-F fa_invariant</text>')
    s.append(f'<text x="{LM}" y="14" class="val">A — T60 above released baseline (pp)</text>')
    s.append(f'<text x="{LM+PW/2:.0f}" y="{H-6}" text-anchor="middle">context count K</text></svg>')
    return "".join(s)


def chartB():
    W, H, LM, BM, TM = 680, 235, 150, 30, 30
    PW, PH = W - LM - 40, H - BM - TM
    hi = 1.4
    rh = 30

    def X(v):
        return LM + v / hi * PW
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="M5 training-seed downgrade ratio, K=8">']
    # threshold at 0.5
    xt = X(0.5)
    s.append(f'<line class="axis" x1="{xt:.1f}" y1="{TM-4}" x2="{xt:.1f}" y2="{TM+len(B_ROWS)*rh:.0f}" stroke="var(--crit)" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{xt:.1f}" y="{TM-8}" text-anchor="middle" fill="var(--crit)">downgrade ≥ 0.5</text>')
    for gv in (0, 0.5, 1.0):
        s.append(f'<text x="{X(gv):.1f}" y="{TM+len(B_ROWS)*rh+16:.0f}" text-anchor="middle">{gv:.1f}</text>')
    for i, (name, dseed, feff) in enumerate(B_ROWS):
        ratio = dseed / feff
        y = TM + i * rh
        col = 'var(--good)' if ratio < 0.5 else 'var(--serious)'
        tag = 'survives' if ratio < 0.5 else 'indeterminate'
        s.append(f'<rect class="mark" x="{LM}" y="{y:.1f}" width="{X(ratio)-LM:.1f}" height="{rh-10}" rx="4" fill="{col}">'
                 f'<title>{name}: worst|Δseed| {dseed:.3f} / |FA_eff| {feff:.3f} = {ratio:.2f} → {tag}</title></rect>')
        s.append(f'<text x="{LM-8}" y="{y+rh/2-2:.1f}" text-anchor="end" class="val">{name}</text>')
        s.append(f'<text class="val" x="{X(ratio)+6:.1f}" y="{y+rh/2-2:.1f}" fill="{col}">{ratio:.2f} {tag}</text>')
    s.append(f'<text x="12" y="14" class="val">B — M5 seed check (K=8): swing ÷ FA effect</text>')
    s.append(f'<text x="{LM+PW/2:.0f}" y="{H-4}" text-anchor="middle">training-seed swing as a fraction of the FA effect</text></svg>')
    return "".join(s)


def chartC():
    import math
    W, H, LM, BM, TM = 620, 210, 210, 34, 28
    PW, PH = W - LM - 60, H - BM - TM
    lo, hi = math.log10(0.001), math.log10(0.3)
    rh = 38

    def X(v):
        return LM + (math.log10(v) - lo) / (hi - lo) * PW
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Metric-1 relative L2, log axis">']
    for gv in (0.001, 0.01, 0.1):
        s.append(f'<line class="axis" x1="{X(gv):.1f}" y1="{TM-2}" x2="{X(gv):.1f}" y2="{TM+len(C_ROWS)*rh-8:.0f}"/>')
        s.append(f'<text x="{X(gv):.1f}" y="{TM+len(C_ROWS)*rh+6:.0f}" text-anchor="middle">{gv:g}</text>')
    for i, (name, val, kind) in enumerate(C_ROWS):
        y = TM + i * rh
        col = SC[kind]
        if kind == 'floor':
            s.append(f'<line x1="{X(val):.1f}" y1="{TM-2}" x2="{X(val):.1f}" y2="{TM+len(C_ROWS)*rh-8:.0f}" stroke="{col}" stroke-dasharray="3 3"/>')
            s.append(f'<text x="{X(val):.1f}" y="{y+rh/2:.1f}" text-anchor="middle" class="val" fill="{col}">floor {val:.5f}</text>')
        else:
            s.append(f'<rect class="mark" x="{LM}" y="{y:.1f}" width="{max(2,X(val)-LM):.1f}" height="{rh-14}" rx="4" fill="{col}">'
                     f'<title>{name}: relL2 {val:.5f}</title></rect>')
            s.append(f'<text class="val" x="{X(val)+6:.1f}" y="{y+(rh-14)/2+3:.1f}" fill="{col}">{val:.5f}</text>')
        s.append(f'<text x="{LM-8}" y="{y+(rh-14)/2+3:.1f}" text-anchor="end">{name}</text>')
    s.append(f'<text x="{LM-200}" y="12" class="val">C — Metric-1 rel-L2 (log scale)</text>')
    s.append(f'<text x="{LM+PW/2:.0f}" y="{H-4}" text-anchor="middle">relative L2 of the RIR waveform (rotated vs unrotated)</text></svg>')
    return "".join(s)


def chartD():
    W, H, LM, BM, TM = 560, 220, 50, 44, 30
    PW, PH = W - LM - 60, H - BM - TM
    lo, hi = 8.5, 11.0
    angles = [0, 45, 90, 180, 270]

    def X(a):
        return LM + angles.index(a) / (len(angles) - 1) * PW

    def Y(v):
        return TM + (hi - v) / (hi - lo) * PH
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="T60 vs rotation angle">']
    for gv in (9, 10, 11):
        s.append(f'<line class="axis" x1="{LM}" y1="{Y(gv):.1f}" x2="{W-16}" y2="{Y(gv):.1f}"/>')
        s.append(f'<text x="{LM-6}" y="{Y(gv)+3:.1f}" text-anchor="end">{gv}</text>')
    for a in angles:
        s.append(f'<text x="{X(a):.1f}" y="{H-24}" text-anchor="middle">{a}°</text>')
    # C4 mean guide lines
    for series, col, lab in ((D_K1, 'var(--s1)', 'K=1'), (D_K8, 'var(--s2)', 'K=8')):
        c4 = [v for a, v in series if a != 45]
        my = Y(sum(c4) / len(c4))
        s.append(f'<line x1="{LM}" y1="{my:.1f}" x2="{W-60}" y2="{my:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="2 3" opacity=".6"/>')
        for a, v in series:
            isc4 = (a != 45)
            fill = col if isc4 else 'var(--serious)'
            s.append(f'<circle class="mark" cx="{X(a):.1f}" cy="{Y(v):.1f}" r="5" fill="{fill}">'
                     f'<title>{lab} {a}°: T60 {v:.3f}{"" if isc4 else "  (off-C₄, structural)"}</title></circle>')
        s.append(f'<text class="val" x="{W-56}" y="{my+4:.1f}" fill="{col}">{lab}</text>')
    s.append(f'<text x="{X(45):.1f}" y="{Y(10.8272)-9:.1f}" text-anchor="middle" fill="var(--serious)" class="val">45° breaks</text>')
    s.append(f'<text x="{LM}" y="14" class="val">D — T60 vs panorama rotation angle</text>')
    s.append(f'<text x="{LM+PW/2:.0f}" y="{H-6}" text-anchor="middle">panorama yaw rotation angle</text></svg>')
    return "".join(s)


html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp_08 — fa_invariant vs vanilla (matched fine-tune)</title>
<link rel="stylesheet" href="fa_matched_results_assets/results.css"></head><body>
<h1>exp_08 — Frame-averaged geometry vs vanilla, at a matched fine-tune</h1>
<p class="sub">Full unseen split (6337 items / 17 rooms); eval seeds 42–46; both arms <code>--cond-autocast bf16</code> · 2026-07-09.
Numbers mirror <a href="fa_matched_results.md">fa_matched_results.md</a> via <code>aggregate_results.py</code>.
<b>Analyst: Opus 4.8</b> (main session; the SOP’s analysis seat, formerly Fable 5).</p>

<div>
<span class="kpi"><span class="n" style="color:var(--good)">PASS</span><span class="l">H-A2 · exact C₄ invariance</span></span>
<span class="kpi"><span class="n" style="color:var(--good)">PASS</span><span class="l">H-A3 · metric flatness</span></span>
<span class="kpi"><span class="n">−0.44</span><span class="l">T60 vs control (K=8, seed-robust)</span></span>
<span class="kpi"><span class="n">0.0023</span><span class="l">C₄ rel-L2 (floor 0.0093)</span></span>
</div>

<div class="verdict good"><b>Minimum project goal — achieved on a trained model.</b> The fine-tuned <code>fa_invariant</code> model passes the cylindrical sanity check <i>exactly</i> (H-A2+H-A3) and does so at no cost to the headline T60 metric — in fact a training-seed-robust T60 <i>gain</i> over its matched vanilla control.</div>

<h2>The accuracy trade (H-A1)</h2>
<figure>{chartA()}<figcaption>Distance of each fine-tuned arm above the released baseline (exp_01 EMA), lower = less damage. A-F sits closer to baseline than its matched vanilla control at both K — the first intervention across exp_03–08 to pull a fine-tuned model back toward baseline T60.</figcaption></figure>

<div class="verdict mixed"><b>Strict H-A1 = FAIL (6/6 T60/C50/EDT cells outside 2σ<sub>c</sub>), but the pre-registered “FA materially worse” reading does not describe this.</b> A-F is <b>superior</b> on T60 at both K (−0.28 / −0.44, many σ<sub>c</sub>) and equivalent on R@1; it regresses EDT (+0.5–1.0 ms) and C50 (+0.02 dB) at both K. The M5 seed check (K=8 only) downgrades the <i>K=8</i> EDT/C50 regressions to indeterminate — but the <i>K=1</i> EDT/C50 regressions were not seed-tested and stand. This is not a non-inferiority claim.</div>

<figure>{chartB()}<figcaption>M5 retrained both arms at training-seed 43 and screened at <b>K=8 only</b>. For each metric, the training-seed swing as a fraction of the FA effect; the pre-registered rule downgrades a cell to <i>indeterminate</i> once the swing reaches half the effect. Only T60’s gain clears the bar (reproduced superior at both seeds, −0.46 / −0.35); the K=8 EDT and C50 costs dissolve into training-seed variance. K=1 was not seed-tested (its EDT/C50 regressions remain strict).</figcaption></figure>

<table><thead><tr><th>K</th><th>Metric</th><th class="num">A-F</th><th class="num">A-V (bf16 mirror)</th><th class="num">Δ</th><th class="num">d/σ<sub>c</sub></th><th>Verdict</th></tr></thead><tbody>
<tr><td>1</td><td>T60</td><td class="num">10.372</td><td class="num">10.647</td><td class="num">−0.276</td><td class="num">−3.3</td><td><span class="chip pass">superior</span></td></tr>
<tr><td>1</td><td>C50</td><td class="num">1.0317</td><td class="num">1.0091</td><td class="num">+0.023</td><td class="num">+2.5</td><td><span class="chip fail">regression (strict; K=1 not seed-tested)</span></td></tr>
<tr><td>1</td><td>EDT</td><td class="num">42.270</td><td class="num">41.246</td><td class="num">+1.025</td><td class="num">+7.3</td><td><span class="chip fail">regression (strict; K=1 not seed-tested)</span></td></tr>
<tr><td>1</td><td>R@1</td><td class="num">6.663</td><td class="num">6.767</td><td class="num">−0.104</td><td class="num">−0.4</td><td><span class="chip info">equivalent</span></td></tr>
<tr><td>8</td><td>T60</td><td class="num">8.916</td><td class="num">9.355</td><td class="num">−0.439</td><td class="num">−49.3</td><td><span class="chip pass">superior (seed-robust)</span></td></tr>
<tr><td>8</td><td>C50</td><td class="num">0.9476</td><td class="num">0.9261</td><td class="num">+0.021</td><td class="num">+7.4</td><td><span class="chip warnc">regress → indeterminate (M5)</span></td></tr>
<tr><td>8</td><td>EDT</td><td class="num">39.111</td><td class="num">38.613</td><td class="num">+0.498</td><td class="num">+16.2</td><td><span class="chip warnc">regress → indeterminate (M5)</span></td></tr>
<tr><td>8</td><td>R@1</td><td class="num">6.842</td><td class="num">6.994</td><td class="num">−0.152</td><td class="num">−0.8</td><td><span class="chip info">equivalent</span></td></tr>
</tbody></table>
<p class="sub" style="margin:.2rem 0 0">All 6 T60/C50/EDT cells lie outside 2σ<sub>c</sub> → strict H-A1 <b>FAIL</b>: 2 superior (T60), 4 regression. M5 (K=8 only) downgrades the two K=8 early-field regressions to indeterminate; the two K=1 early-field regressions were not seed-tested and stand.</p>

<h2>Exact C₄ invariance (H-A2 / H-A3)</h2>
<div class="grid2">
<figure>{chartC()}<figcaption>Metric-1: rel-L2 of the predicted RIR, rotated vs unrotated panorama. On the C₄ subgroup A-F is ~90× below the vanilla yaw gap and under the registered bf16 floor (0.0093). The 45° off-subgroup value (≈ vanilla) is the pre-registered <i>structural</i> residual — patch tokens aren’t roll-equivariant off-group.</figcaption></figure>
<figure>{chartD()}<figcaption>Metric-2: downstream T60 is flat across all C₄ angles (|Δ| ≤ 0.001) and breaks only at 45°, mirroring the Metric-1 picture. Same shape at K=8 (0°/90° shown).</figcaption></figure>
</div>

<table><thead><tr><th>Angle</th><th class="num">K=1 rel-L2</th><th class="num">K=8 rel-L2</th><th>vs floor 0.0093</th></tr></thead><tbody>
<tr><td>0° (identity)</td><td class="num">0.0</td><td class="num">0.0</td><td><span class="chip pass">exact</span></td></tr>
<tr><td>90°</td><td class="num">0.00233</td><td class="num">0.00231</td><td><span class="chip pass">pass</span></td></tr>
<tr><td>180°</td><td class="num">0.00233</td><td class="num">—</td><td><span class="chip pass">pass</span></td></tr>
<tr><td>270°</td><td class="num">0.00235</td><td class="num">—</td><td><span class="chip pass">pass</span></td></tr>
<tr><td>45° (off-C₄)</td><td class="num">0.20640</td><td class="num">—</td><td><span class="chip warnc">structural</span></td></tr>
</tbody></table>

<h2>What exp_08 licenses about exp_07</h2>
<div class="verdict mixed"><b>Measured at a damaged operating point.</b> Both arms sit below the released baseline (A-F K=8 T60 8.92 vs 8.61) — exp_08 is a matched <i>marginal</i> comparison that deliberately sidesteps the exp_03–06 lineage blocker. The FA effect may not transfer to a from-scratch/undamaged model; that is exactly the exp_07 question. It <i>does</i> de-risk exp_07: no evidence FA is intrinsically harmful, and its one seed-robust effect is a T60 gain.</div>
<p class="sub" style="margin-top:.6rem">The exp_07 question, sharpened: does from-scratch training reach the released Table-1 numbers (lineage), and does from-scratch <code>fa_invariant</code> keep the T60 gain + exact invariance while the seed-indeterminate early-field cost stays small? Framing, cost (~40 GPU-days at paper parity), and the recommendation (hybrid, reconcile against the in-flight <code>FLAC_vanilla291k</code> GPU-0 run) are in <a href="fa_matched_analysis.md">fa_matched_analysis.md</a>.</p>

<p class="foot">Arms: A-V = reused exp_05 V1′ (recipe-equivalent); A-F = trained M1, matched recipe + <code>--cond-method fa_invariant</code>. Pre-registered gates in <a href="plan_fa_matched.md">plan_fa_matched.md</a>. Generated by <code>fa_matched_results_assets/gen_page.py</code>.</p>
</body></html>"""

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fa_matched_01_results.html')
open(out, 'w').write(html)
print('wrote', out)
