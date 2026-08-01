#!/usr/bin/env python3
"""gen_closing_page.py - renders fa_finetune_01_results.html (exp_09 closing page).
Presentation only; numbers transcribed from fa_finetune_results.md."""
import os

GATE = {"K=8": {"T60": ("8.4652", "0.0058", "-10.8", "SUPERIOR"), "C50": ("0.9582", "0.0010", "-3.2", "SUPERIOR"),
                "EDT": ("37.4968", "0.0813", "+3.8", "OUT"), "R@1": ("6.9244", "0.0700", "-1.1", "NONINF")},
        "K=1": {"T60": ("9.8271", "0.0612", "-2.0", "SUPERIOR"), "C50": ("1.0337", "0.0025", "-1.8", "SUPERIOR"),
                "EDT": ("40.8740", "0.3393", "+1.8", "NONINF"), "R@1": ("6.8581", "0.1108", "+0.1", "EQUIV")}}
REL = {"K=8": {"T60": "8.609", "C50": "0.9682", "EDT": "37.10", "R@1": "7.06"},
       "K=1": {"T60": "9.969", "C50": "1.0460", "EDT": "39.95", "R@1": "6.83"}}
SWEEP = [("0°", 8.4649, 0.9584, 37.5091, 6.8802), ("90°", 8.4647, 0.9585, 37.5111, 6.8960),
         ("180°", 8.4651, 0.9585, 37.5103, 6.8960), ("270°", 8.4648, 0.9585, 37.5132, 6.9118),
         ("45° (neg. control)", 9.0959, 1.0864, 40.3434, 5.2391)]
TIER = {"SUPERIOR": "background:#0a5c2e;color:#fff", "EQUIV": "background:#1a4f7a;color:#fff",
        "NONINF": "background:#5a4a12;color:#fff", "OUT": "background:#6b1f1f;color:#fff"}

rows = "".join(
    f"<tr><td>{m}</td><td><b>{GATE['K=8'][m][0]} ± {GATE['K=8'][m][1]}</b></td><td>{REL['K=8'][m]}</td>"
    f"<td style='{TIER[GATE['K=8'][m][3]]}'>{GATE['K=8'][m][3]} ({GATE['K=8'][m][2]}σ)</td>"
    f"<td><b>{GATE['K=1'][m][0]} ± {GATE['K=1'][m][1]}</b></td><td>{REL['K=1'][m]}</td>"
    f"<td style='{TIER[GATE['K=1'][m][3]]}'>{GATE['K=1'][m][3]} ({GATE['K=1'][m][2]}σ)</td></tr>"
    for m in ["T60", "C50", "EDT", "R@1"])
sweep_rows = "".join(
    f"<tr><td>{a}</td><td>{t:.3f}</td><td>{c:.4f}</td><td>{e:.3f}</td><td>{r:.3f}</td></tr>"
    for a, t, c, e, r in SWEEP)

page = f"""<meta charset="utf-8"><title>exp_09 closing — equivariant FLAC at Table-1 level</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;background:#111;color:#eee}}
table{{border-collapse:collapse;margin:12px 0;width:100%}}td,th{{border:1px solid #444;padding:5px 9px;text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#222}}td:first-child,th:first-child{{text-align:left}}
.hero{{background:#0a5c2e;color:#fff;padding:14px 18px;border-radius:8px}}
.warn{{background:#5a4a12;color:#fff;padding:10px 14px;border-radius:8px;margin:10px 0}}
.note{{background:#222;border-left:4px solid #1a4f7a;padding:8px 14px;margin:10px 0}}h2{{border-bottom:1px solid #333;padding-bottom:4px}}</style>
<h1>exp_09 fa_finetune — closing results</h1>
<div class="hero"><b>C₄-equivariant FLAC — registered tier NEGATIVE (strict G2 fail); exploratory Table-1 reading: 4 SUPERIOR + 1 EQUIV + 2 NONINF + 1 OUT.</b>
Checkpoint <code>exp09_Fw / epoch=20-step=95000</code> (fa fine-tune of the exp_07 anchor; evaluate with
<code>--cond-method fa_invariant</code>): C₄ sweep max spread 0.032 at the checkpoint of record, both K (10⁻²–10⁻³ of the 45°-control break); vs released Table-1 (5 seeds, both K):
T60 &amp; C50 SUPERIOR at both K; R@1 NONINF (K=8) / EQUIV (K=1); EDT OUT +0.40 ms (K=8) / NONINF (K=1). One TRAINING seed, five eval seeds. The registered G2 anchor-preservation gate FAILED (T60/EDT both K) — this comparison is the unregistered exploratory reading on a fallback candidate.</div>
<h2>1 · G1 — C₄ rotation sweep at the checkpoint of record (95000, K=8 shown; K=1 in results.md; seed 42)</h2>
<table><tr><th>rotation</th><th>T60</th><th>C50</th><th>EDT</th><th>R@1</th></tr>{sweep_rows}</table>
<div class="note">Metric-level flatness across the C₄ orbit (max spread 0.032 across both K); the 45° control breaks by construction. Registered-G1 departures (conditioning rel-L2, waveform floor) recorded in the results file.</div>
<h2>2 · Gate — Fw-95000 vs released Table-1 (5 eval seeds, full split, fa eval)</h2>
<table><tr><th>Metric</th><th>K=8 ours</th><th>K=8 released</th><th>verdict</th><th>K=1 ours</th><th>K=1 released</th><th>verdict</th></tr>{rows}</table>
<div class="warn"><b>Protocol-error record:</b> all fa-arm evals before 2026-07-30 ~19:50 ran with vanilla eval-time conditioning
(<code>--cond-method</code> never passed) — the interim "fine-tune damage" reading and the non-flat first sweep were artifacts of that
mismatch and are retired. Discovered via the pre-registered G1 sweep; full record in <code>fa_finetune_results.md</code> / <code>_worklog.md</code>.</div>
<h2>3 · Findings</h2>
<div class="note"><b>Warm vs reset:</b> Adam-moment reuse across the conditioning switch is immaterial (probe pick: warm, by a small margin). <b>Control (G4 per metric):</b> matched-step mean F−V = T60 −0.71 / C50 +0.015 / EDT +0.73 / R@1 +0.13 (F better on T60/R@1, comparable C50, worse EDT) — the fa cost concentrates in EDT; G4 cannot override the registered G2 fail. <b>Program:</b> the large apparent damage HERE was an eval-protocol artifact; strong-anchor fa fine-tuning shows a smaller, mixed delta. exp_03–06's released-lineage blocker findings are NOT overturned.</div>
<p style="color:#888">Numbers first recorded in <code>fa_finetune_results.md</code>; page generated by <code>gen_closing_page.py</code>.</p>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_finetune_01_results.html")
open(out, "w").write(page)
print("wrote", out, len(page), "bytes")
