# Issue report — open issues, caveats, pending decisions

Updated at every session handoff/compaction (CLAUDE.md protocol). Closed items move to the bottom with resolution notes.

## Awaiting Yixun (decisions)

1. **wandb: RESOLVED (2026-07-16 ~05:15).** Yixun added yh4742's key to `~/.bashrc`; verified → **yh4742@princeton.edu** / entity `yh4742-princeton-university`. Caveat handled: the export sits below `.bashrc`'s interactive guard, so non-interactive shells must extract it directly (launch scripts self-extract; plain `source ~/.bashrc` silently keeps the old yixunhu21 key). **B-F launches `LOGGER=wandb`** (project `FLAC_exp07_BF`, run `exp07_BF`). Env note: Yixun said `conda activate flac`; exp_07 stays on `rir2rir` for manifest identity (flagged).
2. **Next decision point (~Jul 28, nothing blocking before):** B-F verdict (B-F vs 8×8 B-V matched + vs released). GPU-1 queue fixed by Yixun 2026-07-16: **extend (→100k, stops regardless) → B-F (go GIVEN) → P1**. The old Jul-17 continue-to-135k decision is moot; further B-V extension deferred (resumable). Extend best-ckpt report still delivered ~Jul 17 16:00 (informational).

## Open issues / caveats (technical)

1. **R@1 trails at 67.5k** (advisory, non-gating): retrieval-grade specificity converges slowest; may simply need the full budget or reflect residual data/env lineage (§5.ii of the audit).
2. **T60 endpoint = draw from an oscillating band** — InverseLR keeps lr ≈ 4.84e-5 at 67.5k (by design), so late-training metrics oscillate; endpoint luck vs checkpoint selection is quantified by the 2.5k-cadence selection curve.
3. **DINOv3 initializer provenance** (audit §5.iii) — authors' snapshot revision unknowable; ours pinned (`114c1379…`/`4610ad75…`), enforced fail-closed. Affects absolute-parity reads only.
4. **Screen-watcher summary extraction bug** (cosmetic) — one-line summaries came out empty; numbers extracted from tee'd logs instead. Watcher retired; fix only if reused.
5. **`unwrap_model.py` imports `stable_audio_tools`** (pre-existing upstream issue, CLAUDE.md §Checkpoints) — needs adapting before use.
6. **rir2rir SOP copy divergence** — `rir2rir/worklog/experiment_SOP.md` has the reviewer-model update but NOT the worklog_<username> namespace rule (that repo's layout untouched; another session active there).
7. **Yixun's `FLAC_vanilla291k` run** — not B-V-certifiable (data folder `single_channel_ir` ≠ `_1`, micro 16×4, third-party file copies); corroborating row only, under our protocol.
8. **Main session is silently ALTERNATING models mid-session** (found 2026-07-16 ~17:35, Fable 5). This session's transcript (`eacb1737-…jsonl`) holds **38 `claude-fable-5` + 36 `claude-opus-4-8`** assistant records, none sidechain — i.e. the harness swaps the serving model between turns (usage-limit failover), *not* only on an explicit `/model`. Consequences: (a) the handoff hook fires on each real family flip and is **behaving correctly** — the `fable -> claude-opus-4-8` reminder at 17:30:47 was a true detection, not a bug; (b) an incoming model can hold **no memory of the other model's turns in the same session** — the Fable turn at ~17:30 had no context for the Opus turns that stopped the extend at 13:35, wired SyncBN, and resumed the second leg at 16:39, and only recovered them by reading the docs + `ps`. **Therefore: the four handoff docs are the only reliable cross-model channel *within* a session, not just across sessions — write state to them at the moment it changes, and re-verify live state (`ps`/`nvidia-smi`/newest ckpt) before quoting any ETA.** Role-attribution rule stands (flag the authoring model in artifact by-lines).
9. **`cylindrical-dinov3`: two decisions awaiting Yixun (both non-blocking; defaults chosen, work continues).**
   - **(a) `pytest` is not installed in the `flac` conda env.** exp_01's tests currently run only via a *session-local scratchpad symlink shim* off the `rir2rir` env (same Python 3.10.20 / torch 2.7.0+cu126 / transformers 4.57.0; torch+transformers verified still resolving to **flac's** site-packages; nothing installed, nothing in site-packages touched). **This shim will not survive the session.** Durable fix: `pip install -e ".[test]"` into `flac` (the `test` extra is already declared in the new `pyproject.toml`). Not done unilaterally — mutating Yixun's env is his call, and `flac` is the env he reserved for fresh experiments.
   - **(b) Test location.** The SOP requires *asking* where tests go the first time they're added to a project; the Codex reviewer explicitly rejected "Yixun is away" as a substitute. Defaulted to repo-root `tests/` (the Codex transcript's own recommended layout). Trivially movable — nothing depends on it.
10. **`cylindrical-dinov3` has no `CLAUDE.md` and no `announcement/` directives yet** — its SOP copy references `worklog/worklog_<username>/announcement/*.md` ("read every announcement before planning"), but that folder is empty. FLAC's `02_test_driven_development.md` and `03_worklog_username_namespace.md` look project-agnostic and probably should be copied over; `01_always_use_full_FLAC_eval_configs.md` becomes relevant the moment the backbone is evaluated through FLAC. Offered to Yixun; awaiting his call.

## Resolved (recent)

- ~~eff-batch 128 assumption~~ → corrected to 64 (exp_07 audit; README example adjudicated ckpt-incompatible).
- ~~`train.py` hardcoded max_steps~~ → `--max-steps` flag, TDD round 1 (`e85ebde`), CLAUDE.md HAA note updated.
- ~~M0 registered ladder OOM~~ → documented amendment; common pair 8×8 (eff 64) both arms.
- ~~worklog layout~~ → per-user namespace `worklog/worklog_yixun/` (announcement 03, `cb85fd0`); 5 scripts converted to marker-walk repo-root discovery.
