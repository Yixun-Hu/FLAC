# Issue report — open issues, caveats, pending decisions

Updated at every session handoff/compaction (CLAUDE.md protocol). Closed items move to the bottom with resolution notes.

## Awaiting Yixun (decisions)

1. **exp_07 B-V gate verdict** (imminent) — pre-registered strict 6/6 2σ_c gate vs released Table-1; per GO step 5, any non-pass profile stops here. Package will include: verdict table, T60 selection-curve band (oscillation [8.34, 9.15] brackets the released 8.609), 291k corroborating row, R@1-advisory shortfall (~5.8–6.5 vs 6.86), and options (proceed to B-F / extend / investigate).

## Open issues / caveats (technical)

1. **R@1 trails at 67.5k** (advisory, non-gating): retrieval-grade specificity converges slowest; may simply need the full budget or reflect residual data/env lineage (§5.ii of the audit).
2. **T60 endpoint = draw from an oscillating band** — InverseLR keeps lr ≈ 4.84e-5 at 67.5k (by design), so late-training metrics oscillate; endpoint luck vs checkpoint selection is quantified by the 2.5k-cadence selection curve.
3. **DINOv3 initializer provenance** (audit §5.iii) — authors' snapshot revision unknowable; ours pinned (`114c1379…`/`4610ad75…`), enforced fail-closed. Affects absolute-parity reads only.
4. **Screen-watcher summary extraction bug** (cosmetic) — one-line summaries came out empty; numbers extracted from tee'd logs instead. Watcher retired; fix only if reused.
5. **`unwrap_model.py` imports `stable_audio_tools`** (pre-existing upstream issue, CLAUDE.md §Checkpoints) — needs adapting before use.
6. **rir2rir SOP copy divergence** — `rir2rir/worklog/experiment_SOP.md` has the reviewer-model update but NOT the worklog_<username> namespace rule (that repo's layout untouched; another session active there).
7. **Yixun's `FLAC_vanilla291k` run** — not B-V-certifiable (data folder `single_channel_ir` ≠ `_1`, micro 16×4, third-party file copies); corroborating row only, under our protocol.

## Resolved (recent)

- ~~eff-batch 128 assumption~~ → corrected to 64 (exp_07 audit; README example adjudicated ckpt-incompatible).
- ~~`train.py` hardcoded max_steps~~ → `--max-steps` flag, TDD round 1 (`e85ebde`), CLAUDE.md HAA note updated.
- ~~M0 registered ladder OOM~~ → documented amendment; common pair 8×8 (eff 64) both arms.
- ~~worklog layout~~ → per-user namespace `worklog/worklog_yixun/` (announcement 03, `cb85fd0`); 5 scripts converted to marker-walk repo-root discovery.
