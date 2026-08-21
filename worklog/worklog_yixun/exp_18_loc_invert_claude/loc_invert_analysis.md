# loc_invert_analysis — exp_18 registered campaign (Planner: Claude Fable 5, 2026-08-20; R4 exploratory addendum to follow)

## Answer to the registered research question
*Can a generative forward acoustic model be inverted, without localization training, to recover the source position of a held-out RIR in an unseen room?*

**Yes — with a sharply regime-dependent comparison against non-generative retrieval.** Vanilla FLAC's analysis-by-synthesis achieves top-1 ≈ 0.50 (pooled median error 0.0 m) on the full 6,337-query unseen split, and this is invariant to context size (K_ctx=8: 0.5007±0.0008; K_ctx=1: 0.5029±0.0032). Against the pre-registered information-matched baseline it wins in both regimes (p≈0.01 at K=8; p≈0 at K=1). Against the nearest-context retrieval control the story splits: with dense context coverage (8 refs over ~10 candidates) retrieval is stronger (0.689 vs 0.501); with sparse context (1 ref) retrieval collapses to chance (0.108) while FLAC is unchanged — **the generator adds genuine, transferable positional information precisely in the few-shot regime the model exists for.**

## Reliability assessment — HIGH for the claims as stated
1. **Protocol integrity:** machine-verified registration (manifest ancestry + byte-equality + locked fields) on every headline run; τ and all constants frozen from seen data before any unseen inspection; fail-closed identity gates (6,337/6,337 every run, zero substitutions on unseen); split/candidate-manifest digests pinned; atomic publication.
2. **Numerical integrity:** the driver's generation path is bit-identical to `eval_FLAC`'s (parity 0.0, M×K, real data); the scorer is deterministic (mean readout; measured sampled-noise ≈7e-5 removed); the calibration replay reproduced 1,194×~80 published similarities bitwise (`all_match=true`) — the pipeline is regenerable to the bit.
3. **Statistical integrity:** three seeds within ±0.003 on every headline metric; 17-room clustered CIs; paired per-query tests; conclusions identical under every aggregator (τ landscape flat).
4. **The gates caught real defects rather than passing them:** a silent-RIR substitution (seen split, position 1194 — invisible to all prior pipelines), duplicate-position source labels (2 seen rooms), and the context-twin leak — none touched the unseen headline; all are documented dataset properties now.

## Mechanistic reading
- **The 38% context-member failure mode at K=8** (predictions on candidates whose measured RIR was in the conditioning) collapses to 0.047 at K=1 — *below* the 1/9 chance rate. Interpretation: the generator's output at a context-covered position imitates that context RIR (which resembles h_obs at room level), poisoning the AGREE ranking when many candidates are context-covered. This is the direct mechanism behind the dense-regime loss to retrieval, and it suggests the deficit is scorer-confusability, not absent positional information — exactly what R4's waveform metrics probe.
- **Power statistic ≈ 460** (candidate identity moves similarities two orders above sampling noise): conditioning is load-bearing. The formal constant-source control (R3, pre-registered slice) runs before the HTML report as the final wiring proof.
- FLAC's seen-vs-unseen performance is nearly identical (dev median 0.0, mean 1.07 vs unseen 1.08) — the positional information transfers; it was never seen-room memorization.

## Threats to validity (stated, none judged conclusion-threatening)
Released FLAC_EMA only (one architecture, one training run) — cross-arm generalization is exp_20's question; discrete M=10 candidate sets make success@0.5m ≈ top-1 (metric degeneracy documented); `flash_attn` absent in this env (fallback attention; internally consistent, recorded in every provenance); bf16 conditioning is batch-composition-dependent (~0.6% — frozen manifest keeps per-query composition fixed; autocast-off diagnostic available); AGREE scorer shares training data lineage with FLAC's training set (both AR train split) — mitigated by the train-split-only scorer choice and now cross-checked by R4's AGREE-free metrics.

## Recommended next steps
1. **R4** (in flight, frozen): if a waveform metric (M1@Δ8 dev top-1 0.608 is the candidate) beats AGREE inversion AND its own metric-matched retrieval on unseen, the dense-regime deficit reattributes to the scorer, strengthening the world-model claim.
2. **exp_20 cross-arm**: same protocol over B-F/P1/YAW (+cyl) — does equivariance/augmentation widen the sparse-context margin? Inputs already on the NAS.
3. R3 constant-source control + heatmap gallery + integrated HTML: tomorrow, with R4 folded in.
