# Plan — exp_05_bn_drift_bisect (BN buffers as a gradient-free data-drift probe)

**Author:** Fable 5 (Planner) · **Coder:** Opus 4.8 max (TDD) · **Reviewer:** Codex gpt-5.5 xhigh · **Date:** 2026-07-06
**Status:** AWAITING plan review + Yixun approval.

## 0. Mechanism (from exp_04)

W0 (lr=0) regressed T60/EDT via RIR-encoder BatchNorm running-stat adaptation alone ⇒ our loader's reference-RIR statistics ≠ the released training's. BN layers store `running_mean`/`running_var` per channel — comparing them against the batch statistics our data produces at each layer localizes the drift without any training.

## 1. Code to write (TDD; one cycle) — `tools/bn_drift_probe.py` + tests

New module `tools/bn_drift_probe.py` (repo-root `tools/` package; sys.path fix per stale-site-packages memory) — probe core is model-agnostic:

```python
def bn_drift_metrics(running_mean, running_var, batch_mean, batch_var, eps=1e-5):
    """Per-channel standardized drift: mean_shift = |bm - rm| / sqrt(rv + eps);
    var_ratio = bv / (rv + eps). Returns dict of tensors + scalar summaries
    (max/mean over channels)."""

class BNInputRecorder:
    """Forward-pre-hooks on every nn.BatchNorm1d/2d of a module; accumulates
    streaming per-channel mean/var (Welford) of the INPUTS over N batches in
    eval mode (running stats untouched — pure measurement)."""

def probe_rir_encoder(model_ckpt, dataset_config, n_batches, batch_size, device, md_variant=None):
    """Load released conditioner, register recorder on the RIRConditioner's ResNet,
    stream context_audio through the conditioner's own preprocessing path
    (MultiConditioner only_ids=('context_audio',)), return per-layer drift report."""
```

Tests (`src/tests/test_bn_drift_probe.py`, RED first — plan-review findings 3/4 baked in):
| Test | Pins |
|---|---|
| `test_bn_drift_metrics_zero` | batch stats == running stats ⇒ mean_shift 0, var_ratio 1 exactly |
| `test_bn_drift_metrics_known` | hand-computed shift/ratio on 3-channel example, atol 1e-6 |
| `test_recorder_welford_unbiased` | streamed mean/var == torch mean/var(**unbiased — matches PyTorch running_var semantics**) over concatenated batches, atol 1e-5; running stats bit-unchanged after probing (eval mode, no mutation) |
| `test_recorder_conv_bn_stem` | Conv2d(1ch stem)→BN module: recorder captures BN **inputs** (= post-conv activations), pinned against manual forward (finding 3: AudioResNet18 replaces conv1 with 1-channel input) |
| `test_recorder_finds_all_bns` | torchvision resnet18 ⇒ exactly **20 BatchNorm2d, 0 BatchNorm1d** |
| `test_probe_variant_hook` | `md_variant` callable applied per metadata dict before conditioning |
| `test_probe_report_provenance` | report includes ckpt path, clean-load missing/unexpected counts, BN-buffer count == 20 (finding 4: must be the exact `FLAC_EMA.ckpt` EMA buffers — the exp_03 EMA-load confound must be impossible here) |

Estimator noise (finding 5): B0 runs 3× with different data order; report per-layer max mean-shift with min/max across repeats as error bars.

## 2. Bisection procedure (no src/ changes — variants via dataset-config copies + md_variant hooks)

0. **B-1 pilot (plan-review answer 5):** ONE real train batch through the exact `FLAC_EMA.ckpt` conditioner, all BN pre-hook stats dumped, no-mutation asserted, worst-layers mini-table — catches wrong hooks/ckpt/variance convention before the full tool run.
1. **B0 baseline probe:** drift report for the as-is train loader (`acousticroom_train.json`), N≥200 batches × batch 16, ×3 repeats (error bars). Also probe the *eval* loader for reference. Output: per-layer table + worst-channel ranking → `bn_drift_report_baseline.json`.
2. **B1 knob grid** (each = one probe run, ~minutes; finding 2 revisions): reference-RIR `max_len` ∈ {4800, 9600, **10240 (= model sample_size — the 9600-vs-10240 config mismatch is suspect #1)**, 19200-as-fixed-length}; padding side {end (current), start}; truncation {head (current), onset-aligned — **defined**: truncate starting at the first sample whose |amplitude| exceeds 1% of the file peak}; amplitude normalization {none (current), peak, rms}; context sampling {random-without-replacement (current), fixed-seed}. Diagnostics recorded per variant: silence/padding fraction, clamp/min-max stats. **Variants with max_len > 9600 require config-copy reload (md_variant cannot recover samples already truncated at metadata time)** — config copies live in the exp folder.
3. **Selection:** configuration minimizing summary drift (target: max standardized mean-shift < 0.05 and var_ratio ∈ [0.9, 1.1] on all layers — thresholds pre-registered; the baseline B0 numbers will contextualize).
4. **Interpretation guard:** if NO configuration approaches zero ⇒ drift is upstream (data files / STFT path / deeper lineage) → stop after documenting the landscape; options B/C from exp_04 analysis take over.

## 3. Gated validation runs (only if a near-zero configuration is found)

| # | Run | Gate |
|---|---|---|
| V1 | lr=0 null run under corrected loader (625 steps, eff. batch 128) + 10 gate evals | **must PASS the exp_01 2σ gate** (this is the causal test of the bisection) |
| V2 | vanilla control fine-tune (R1b recipe, corrected loader) + gate evals | PASS ⇒ fine-tuning unblocked |
| V3+ | resume blocked pipeline: fa_invariant fine-tune, H3 evals, bf16-floor re-registration, H1/H2 rotation sweeps | exp_03 §6 H1/H2/H3 criteria verbatim |

Stop rules (**V1 attribution corrected per plan-review finding 1**): V1 (lr=0) FAIL after zeroed BN drift ⇒ the instrument/BN-update model is incomplete (something else about train-mode dynamics, micro-batching, or export still matters) — NOT evidence about gradient/target/loss drift; stop + analyze the instrument. Target/loss-path drift only becomes a candidate explanation at V2 (lr>0) failure with V1 passing. V2 marginal ⇒ pause for Yixun (same 1.5σ/2σ definition as exp_04). Every launch: command into `_command.md` at launch; ETAs reported per the wait-time rule.

## 4. Important caveat (pre-registered; corrected per plan review)

Zeroing BN drift proves we matched the *reference-RIR input statistics*; it does not by itself prove the whole pipeline matches the original. The causal ladder is: **V1 pass** validates the bisection at the BN/data level; **V1 fail** indicts the instrument/BN-update model (lr=0 excludes gradient/target explanations by construction); **V2 fail with V1 passing** is what would confine remaining drift to the gradient/target/loss path.

## 5. Risks

- The original knob values may be outside the grid (continuous preprocessing differences) — mitigated by reading the drift *structure* first (B0) and extending the grid once, not fishing.
- Probe must not mutate running stats (eval mode + pinned by test) or the instrument corrupts the checkpoint's evidence.
- torchaudio version differences could make bit-level wav loading differ from the original environment — detectable in B0 as uniform low-level drift; documented if seen.
