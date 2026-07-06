"""Generates yaw_noninvariance_01_results.html (source of truth: _results.md)."""
import os
ANGLES = [0, 90, 180, 270]
M2 = {  # accuracy vs GT per angle (baseline = alpha 0)
 'T60':  {0: 9.99,  90: 10.38, 180: 10.72, 270: 10.44},
 'EDT':  {0: 40.11, 90: 43.58, 180: 46.39, 270: 44.07},
 'C50':  {0: 1.047, 90: 1.074, 180: 1.189, 270: 1.126},
 'R@1':  {0: 6.71,  90: 6.30,  180: 6.38,  270: 6.11}}
M1 = {0: 0.0, 90: 0.221, 180: 0.193, 270: 0.214}  # waveform rel-L2 vs P0
M1ac = {0:(0.0,0.0,0.0), 90:(3.34,0.563,18.65), 180:(3.33,0.550,20.11), 270:(3.41,0.600,19.98)}
def linechart(metric, unit, color, floor_sigma):
    vals = M2[metric]; lo = min(vals.values()); hi = max(vals.values()); pad = (hi-lo)*0.25 or 1
    lo -= pad; hi += pad
    W,H,LM,BM = 340,190,52,28; PW,PH = W-LM-14, H-BM-26
    def X(a): return LM + ANGLES.index(a)/(len(ANGLES)-1)*PW
    def Y(v): return 26 + (hi-v)/(hi-lo)*PH
    pts = " ".join(f"{X(a):.1f},{Y(vals[a]):.1f}" for a in ANGLES)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{metric} vs rotation angle">']
    for a in ANGLES:
        s.append(f'<line class="axis" x1="{X(a)}" y1="26" x2="{X(a)}" y2="{26+PH}"/><text x="{X(a)}" y="{H-8}" text-anchor="middle">{a}°</text>')
    s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    for a in ANGLES:
        s.append(f'<circle class="mark" cx="{X(a)}" cy="{Y(vals[a])}" r="4.5" fill="{color}"><title>α={a}°: {metric} = {vals[a]}{unit}</title></circle>')
        s.append(f'<text class="val" x="{X(a)}" y="{Y(vals[a])-9}" text-anchor="middle">{vals[a]}</text>')
    s.append(f'<text x="{LM}" y="14" class="val">{metric} ({unit}) vs conditioning rotation α — worst at 180°</text>')
    s.append('</svg>'); return "".join(s)
def m1bars():
    W,H,LM = 420,170,52; PW,PH = W-LM-16, H-54
    hi = 0.25
    def X(i): return LM + i*(PW/4) + 14
    def bh(v): return v/hi*PH
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="invariance gap by angle">']
    s.append(f'<line class="axis" x1="{LM}" y1="{28+PH}" x2="{W-10}" y2="{28+PH}"/>')
    for i,a in enumerate(ANGLES):
        v = M1[a]; x = X(i); h = bh(v)
        s.append(f'<rect class="mark" x="{x}" y="{28+PH-h}" width="42" height="{h}" rx="4" fill="var(--s1)"><title>α={a}°: rel-L2(P_α, P_0) = {v}</title></rect>')
        s.append(f'<text class="val" x="{x+21}" y="{28+PH-h-6}" text-anchor="middle">{v:.3f}</text>')
        s.append(f'<text x="{x+21}" y="{40+PH}" text-anchor="middle">{a}°</text>')
    s.append(f'<text x="{LM}" y="14" class="val">Metric 1 — waveform rel-L2 between P_α and P_0 (no GT involved)</text>')
    s.append('</svg>'); return "".join(s)
rows2 = "".join(f'<tr><td>{"baseline (α=0)" if a==0 else f"rot{a}"}</td><td class="num">{M2["T60"][a]}</td><td class="num">{M2["C50"][a]}</td><td class="num">{M2["EDT"][a]}</td><td class="num">{M2["R@1"][a]}</td></tr>' for a in ANGLES)
rows1 = "".join(f'<tr><td>{a}°</td><td class="num">{M1[a]}</td><td class="num">{M1ac[a][0]}</td><td class="num">{M1ac[a][1]}</td><td class="num">{M1ac[a][2]}</td></tr>' for a in ANGLES)
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>exp_02 — FLAC yaw non-invariance</title><link rel="stylesheet" href="yaw_noninvariance_results_assets/results.css"></head><body>
<h1>exp_02 — FLAC is not yaw-invariant (frozen checkpoint, unseen K=1)</h1>
<p class="sub">Full 6337-item split · frozen <code>FLAC_EMA</code> · seed 42 · pristine <code>0bd5da0</code> · 2026-07-04/05. A rigid yaw rotation of the whole scene leaves the true mono RIR unchanged — an invariant model would predict identically. Numbers mirror <a href="yaw_noninvariance_results.md">_results.md</a>.</p>
<div class="verdict good"><b>Control:</b> the α=0 run is <b>bit-identical</b> to baseline (7/7 metrics; comparator max|Δ| = 0.0 over all 6337×10240 samples). Everything below is caused by rotation alone.</div>
<div class="verdict bad"><b>Verdict — hypothesis confirmed.</b> Rotating only the conditioning degrades every metric (Metric 2, up to 18σ of the exp_01 noise floor) and moves the prediction itself by ~20% rel-L2 (Metric 1) — ~5× larger than the net accuracy loss, so Metric 2 alone <i>understates</i> the symmetry defect.</div>
<h2>What the rotation actually does — inputs and outputs</h2>
<figure><img src="yaw_noninvariance_results_assets/depth_c4.png" alt="C4-rotated depth panorama: radial distance rolls; per-pixel vector azimuth field is identical at every angle" style="max-width:100%"><figcaption><b>The conditioning input under C₄ yaw rotation</b> (sample 0). Left: radial distance ‖p‖ — invariant per-pixel values, so the room geometry <i>rolls</i> horizontally with α (red line: target-source azimuth, moving with the scene). Right: per-pixel vector azimuth — the field is <b>identical at every α</b>: the column roll and the per-pixel vector rotation cancel exactly, which is precisely the geometric-consistency property of a valid equirectangular map (a roll/rotation sign error would visibly break this pattern; cf. the in-repo self-check).</figcaption></figure>
<figure><img src="yaw_noninvariance_results_assets/rir_rotation.png" alt="Predicted RIR waveform, difference trace, and Schroeder decay before/after rotation" style="max-width:100%"><figcaption><b>The model output before/after rotation</b> — sample #3689, chosen at the 90th percentile of the per-sample P₁₈₀-vs-P₀ rel-L2 gap (0.388; dataset median 0.146) so it is representative-strong, not the maximum. Same starting noise, same GT; only the conditioning was rotated. The difference trace (middle) would be a flat zero for an invariant model; the Schroeder panel (bottom) shows the rotated-conditioning predictions decaying differently — the visible face of the T60/EDT damage.</figcaption></figure>
<h2>Metric 2 — accuracy vs ground truth under rotated conditioning</h2>
<table><thead><tr><th>run</th><th class="num">T60 (%)↓</th><th class="num">C50 (dB)↓</th><th class="num">EDT (ms)↓</th><th class="num">R@1 (%)↑</th></tr></thead><tbody>{rows2}</tbody></table>
<div class="grid2"><figure>{linechart('T60','%','var(--s1)',0.04)}<figcaption>T60 degradation peaks at 180° (+0.73 = 18σ).</figcaption></figure>
<figure>{linechart('EDT','ms','var(--s2)',0.37)}<figcaption>EDT: +3.5 to +6.3 ms (9–17σ). Separate panels — different units, one axis each.</figcaption></figure></div>
<h2>Metric 1 — invariance gap P<sub>α</sub> vs P<sub>0</sub></h2>
<table><thead><tr><th>α</th><th class="num">waveform rel-L2</th><th class="num">T60 gap (%)</th><th class="num">C50 gap (dB)</th><th class="num">EDT gap (ms)</th></tr></thead><tbody>{rows1}</tbody></table>
<figure>{m1bars()}<figcaption>The prediction moves ~20% in relative energy under a rotation that should change nothing. α=0 is exactly 0 (determinism + pairing control).</figcaption></figure>
<h2>Reference targets set for the fix</h2>
<p>A yaw-invariant FLAC must achieve Metric-1 ≡ 0 at all α (canonicalization/frame-averaging gives this by construction on column-quantized angles) while Metric 2 at α=0 stays within ~2σ of baseline (T60 9.99, C50 1.047, EDT 40.11).</p>
<p class="foot">Per-run JSONs: <code>metrics_json/</code>, <code>metric1_rot*.json</code> · comparator: <code>compare_predictions.py</code> (Opus-written, Codex-reviewed) · Generated by <code>yaw_noninvariance_results_assets/gen_page.py</code>.</p>
</body></html>"""
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'yaw_noninvariance_01_results.html')
open(out,'w').write(html); print('wrote', out)
