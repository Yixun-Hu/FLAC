#!/usr/bin/env python3
"""gen_closing_page.py - renders fa_scratch_01_results.html (exp_07 closing page).

Presentation layer ONLY: every number is transcribed from fa_scratch_results.md /
the gate aggregations logged in fa_scratch_worklog.md (numbers appear there first).
Tables + verdict blocks, no charts (curve tables carry the trajectory story).
"""
import html, os

REL = {"K=8": {"T60": ("8.609", "0.012"), "C50": ("0.9682", "0.0030"), "EDT": ("37.10", "0.07"), "R@1": ("7.06", "0.10")},
       "K=1": {"T60": ("9.969", "0.039"), "C50": ("1.0460", "0.0064"), "EDT": ("39.95", "0.37"), "R@1": ("6.83", "0.22")}}
GATE = {  # metric -> (ours, std, sigma_c_delta, tier) per K, ckpt 87500
 "K=8": {"T60": ("8.2930", "0.0106", "-19.8", "SUPERIOR"), "C50": ("0.9660", "0.0015", "-0.65", "EQUIV"),
         "EDT": ("35.9513", "0.0532", "-13.1", "SUPERIOR"), "R@1": ("6.9592", "0.1353", "-0.60", "EQUIV")},
 "K=1": {"T60": ("9.5401", "0.0231", "-9.5", "SUPERIOR"), "C50": ("1.0323", "0.0060", "-1.6", "SUPERIOR"),
         "EDT": ("38.7283", "0.2263", "-2.8", "SUPERIOR"), "R@1": ("6.8108", "0.1766", "-0.07", "EQUIV")}}
CURVE = [  # step, T60, C50, EDT, R@1 (P1 EMA s42)
 (10000, 11.784, 1.3775, 47.481, 1.783), (20000, 8.442, 1.0954, 45.418, 2.919), (30000, 9.200, 1.0958, 43.428, 4.166),
 (40000, 8.989, 1.0076, 40.620, 5.192), (50000, 8.647, 0.9854, 37.649, 5.539), (55000, 8.510, 0.9506, 36.992, 5.997),
 (57500, 8.493, 0.9625, 36.427, 6.060), (60000, 8.893, 1.0146, 38.991, 6.328), (62500, 8.815, 0.9486, 38.161, 5.870),
 (65000, 8.887, 0.9604, 38.461, 6.044), (67500, 8.771, 0.9734, 36.952, 6.281), (70000, 8.079, 0.9390, 37.228, 6.233),
 (75000, 9.116, 0.9407, 39.575, 6.675), (80000, 8.804, 0.9332, 37.132, 6.833), (85000, 8.906, 0.9569, 38.023, 6.249),
 (87500, 8.307, 0.9643, 35.973, 6.975), (90000, 8.785, 1.0099, 36.598, 6.849), (92500, 8.994, 0.9300, 37.761, 6.533),
 (95000, 8.488, 0.9619, 37.133, 6.407), (97500, 9.012, 0.9946, 38.547, 6.596), (100000, 9.552, 0.9432, 39.039, 6.754)]
TIER_STYLE = {"SUPERIOR": "background:#0a5c2e;color:#fff", "EQUIV": "background:#1a4f7a;color:#fff"}

rows8 = "".join(
    f"<tr><td>{m}</td><td><b>{GATE['K=8'][m][0]} ± {GATE['K=8'][m][1]}</b></td><td>{REL['K=8'][m][0]} ± {REL['K=8'][m][1]}</td>"
    f"<td style='{TIER_STYLE[GATE['K=8'][m][3]]}'>{GATE['K=8'][m][3]} ({GATE['K=8'][m][2]}σ)</td>"
    f"<td><b>{GATE['K=1'][m][0]} ± {GATE['K=1'][m][1]}</b></td><td>{REL['K=1'][m][0]} ± {REL['K=1'][m][1]}</td>"
    f"<td style='{TIER_STYLE[GATE['K=1'][m][3]]}'>{GATE['K=1'][m][3]} ({GATE['K=1'][m][2]}σ)</td></tr>"
    for m in ["T60", "C50", "EDT", "R@1"])

curve_rows = "".join(
    ("<tr style='background:#3a2f00;font-weight:bold'>" if s == 87500 else "<tr>") +
    f"<td>{s:,}</td><td>{t:.3f}</td><td>{c:.4f}</td><td>{e:.3f}</td><td>{r:.3f}</td></tr>"
    for s, t, c, e, r in CURVE)

page = f"""<meta charset="utf-8"><title>exp_07 closing — FULL Table-1 parity</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;background:#111;color:#eee}}
table{{border-collapse:collapse;margin:12px 0;width:100%}}td,th{{border:1px solid #444;padding:5px 9px;text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#222}}td:first-child,th:first-child{{text-align:left}}
.hero{{background:#0a5c2e;color:#fff;padding:14px 18px;border-radius:8px;font-size:1.05em}}
.note{{background:#222;border-left:4px solid #1a4f7a;padding:8px 14px;margin:10px 0}}
h2{{border-bottom:1px solid #333;padding-bottom:4px}}
</style>
<h1>exp_07 fa_scratch — closing results</h1>
<div class="hero"><b>FULL TABLE-1 PARITY.</b> Checkpoint <code>exp07_P1 / epoch=19-step=87500</code>
(vanilla, DDP 32/GPU×2 + SyncBN-64 + ViT grad-ckpt, seed 42, 87.5k steps): 5-seed confirmed at both K —
<b>8/8 cells SUPERIOR or ≤1σ<sub>c</sub>-equivalent</b> vs released FLAC (5 strictly superior; none worse).
Maximum project goal closed 2026-07-28.</div>
<h2>1 · Gate — ckpt 87,500 vs released Table-1 (5 eval seeds, full 6,337-item unseen split)</h2>
<table><tr><th>Metric</th><th>K=8 ours</th><th>K=8 released</th><th>verdict</th><th>K=1 ours</th><th>K=1 released</th><th>verdict</th></tr>{rows8}</table>
<div class="note">σ<sub>c</sub> = √(σ²<sub>ours</sub> + σ²<sub>released</sub>); tiers pre-registered in <code>plan_bv_parity.md</code>.
T60/C50/EDT: lower is better; R@1: higher. Selection on seed 42, confirmation on held-out seeds 43–46 (the plan's PARITY protocol).
Secondary within-original-budget checkpoint 57,500: T60/C50/EDT superior-or-equiv at both K (5-seed), R@1 out — the program's first composite-rule qualifier.</div>
<h2>2 · P1 selection curve (EMA, K=8, seed 42) — checkpoint of record highlighted</h2>
<table><tr><th>step</th><th>T60 ↓</th><th>C50 ↓</th><th>EDT ↓</th><th>R@1 ↑</th></tr>{curve_rows}</table>
<h2>3 · Program findings</h2>
<div class="note"><b>Attribution (single-delta):</b> at the identical recipe, vanilla (P1) tracks the 8×8 anchor while fa_invariant (B-F) plateaus ~2× worse with conditioning demonstrably active (cfg-0 lift probe) and 3.5× step cost — <b>frame-averaged equivariance is a fine-tune-stage property</b> (exp_08 route), not a from-scratch one.</div>
<div class="note"><b>Mechanism:</b> BN-statistics quality (SyncBN batch 64 = paper) closes the EDT gap the BN-8 recipe could not close in 100k steps (curve statistic 37.664 vs 40.087; inverted to 35.95 at the confirmed ckpt); training budget beyond 67.5k closes R@1 (both arms).</div>
<div class="note"><b>Scope:</b> claim = "our recipe surpasses the released checkpoint" under deliberate, disclosed deviations (DDP/SyncBN/grad-ckpt/flash-env/87.5k budget) — not a replication of the release's training. Full caveats in <code>fa_scratch_analysis.md</code>.</div>
<p style="color:#888">Numbers first recorded in <code>fa_scratch_results.md</code> / <code>fa_scratch_worklog.md</code>; this page is presentation only. Generated by <code>gen_closing_page.py</code>.</p>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_scratch_01_results.html")
with open(out, "w") as f:
    f.write(page)
print("wrote", out, len(page), "bytes")
