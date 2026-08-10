# Issue report — open issues, caveats, pending decisions

Updated at every session handoff/compaction (CLAUDE.md protocol). Closed items move to the bottom with resolution notes. **Last refresh: 2026-08-10 (Opus 5 max seat, on the Fable 5 → Opus 5 model change).**

## Awaiting Yixun (decisions)

1. **Cross-machine metrics consolidation — HALF-DONE (open since 2026-08-08).** This box force-added 152+ raw metric JSONs plus `A6000_METRICS_SHA256SUMS.txt`, so `model_comparison.md` regenerates from committed data here. The cluster session's neuronic/exp_11 row evidence is committed as a *manifest only* — the raws stay on its disk — so `gen_model_comparison.py`'s row-regression guard (their upgrade, behaving correctly) refuses to regenerate on this box: six rows would lose their numbers. **Options: (a) the cluster force-adds its raws, mirroring this box's pattern; (b) the generator learns published-value carry-forward with a provenance note.** Nothing is lost meanwhile — new rows are recorded in the owning experiments' results files.
2. **exp_11 recipe decision** (cluster thread): its P0 found C8+ infeasible without gradient checkpointing; the rung/recipe choice is with Yixun.
3. **Paper column selection.** Three confirmed flavors exist (anchor 87.5k / equivariant Fw-95k / decay-tail S93750). Open: which headline, and whether to include a fa-from-scratch column — post-retraction it is a viable peer *per step* at ~3.5× step cost, but exp_10's matched-COMPUTE readout shows the fine-tune route dominates it.

## Open issues / caveats (technical)

1. **Late checkpoints are band draws, not trajectory points.** InverseLR holds lr ≈ 4.8e-5 through the whole run, so adjacent checkpoints swing ~±0.5 T60 / ~±2 EDT. This distorted three readings before it was quantified (B-V's band-max 67.5k endpoint; fa-scratch's band-best 40k spike; fa-scratch's band-worst 67.5k endpoint). exp_13 showed a decaying tail halves the band but lands on a different metric trade point rather than reproducing a wide-band best draw. **Mitigation, now standing: pre-registered window selection + held-out eval-seed confirmation; never quote a single screen as a result.**
2. **Eval-protocol flags must be declared per arm.** `--cond-method vanilla|fa_invariant` mismatch yields plausible-but-catastrophic numbers in both directions (see HANDOFF "Load-bearing facts"). Root cause of exp_09's protocol error and the retracted exp_07 B-F conclusion. Now mandated in every launch/screen manifest.
3. **DINOv3 initializer provenance** — the authors' snapshot revision is unknowable; ours is pinned (`114c1379…` / `4610ad75…`) and enforced fail-closed. Affects absolute-parity readings only.
4. **`unwrap_model.py` imports `stable_audio_tools`** (pre-existing upstream issue) — needs adapting before use; blocks external distribution of any checkpoint until then.
5. **rir2rir SOP copy divergence** — that repo's SOP has the reviewer-model update but not the `worklog_<username>` namespace rule; another session is active there.
6. **Yixun's `FLAC_vanilla291k` run** is not B-V-certifiable under our protocol (data folder `single_channel_ir` ≠ `_1`, micro 16×4) — corroborating row only.
7. **Main session silently ALTERNATES models mid-session** (found 2026-07-16). The harness swaps the serving model between turns on usage-limit failover, not only on `/model`, and an incoming model holds no memory of the other's turns. **The four handoff docs are therefore the only reliable cross-model channel *within* a session** — write state at the moment it changes and re-verify live state before quoting ETAs. Role-attribution rule stands (flag the authoring model in artifact by-lines).
8. **⚠️ `codex exec -s read-only` does NOT protect the environment** (found 2026-07-16). A reviewer ran `pip install -e ".[test]"` with bare `pip` (conda BASE, py3.13) and the sandbox did not block it: 32 packages including the whole torch/CUDA stack landed in base site-packages (~6.1 GB). **Standing mitigation: every review prompt must explicitly forbid installing/modifying environments;** any pip verification must be `--dry-run` with an explicit interpreter path. *Still awaiting Yixun:* whether to strip that ~6 GB from base — not done unilaterally because several entries were upgrades of packages base legitimately had, and fresh-vs-upgraded cannot be distinguished post-hoc. Disk is not urgent.
9. **Pre-existing `flac` env conflict** (not ours): `pip check` reports `flac 1.0 requires wandb==0.15.4, but wandb 0.26.1 is installed`.
10. **`transformers==4.57.0` is YANKED on PyPI.** Harmless today (already installed; an exact `==` pin still resolves a yanked release), but a fresh env build will warn and could break if the release is fully removed. Both FLAC and `cylindrical-dinov3` pin it.
11. **`cylindrical-dinov3` has no `CLAUDE.md` and an empty `announcement/`**, while its SOP copy tells the reader to read every announcement first. FLAC's announcements 02/03 look project-agnostic and probably should be copied over; 01 becomes relevant once the backbone is evaluated through FLAC. Offered to Yixun; awaiting his call.
12. **Concurrent writers on `check-equivariance-necessity`** — the cluster session pushes exp_11 work to the same branch; two rebase conflicts have already occurred (resolved by regenerating artifacts rather than picking a side). Always `git pull --rebase` before committing; never rewrite files another session owns.
13. **Sibling-checkout numbering collision.** `~/codespace/exp-12-arms` (and exp-08/09/10-* siblings) run their own experiment numbering against the same GPUs; `exp12A_c3c4`/`exp12C_ray12` there are unrelated to this repo's exp_12 (mem_probe, closed). Verify `readlink /proc/<pid>/cwd` before attributing any run to this worktree.

## Resolved (recent)

- ~~Codex API key 401-dead (2026-08-08 ~01:00)~~ → **re-authenticated by Yixun the same morning; probe OK.** All exp_13 reviews that ran meanwhile used the declared Opus 5 max fallback seat (valid per SOP, flagged in each artifact) — and that seat caught exp_13's decisive scheduler-clobber bug before launch.
- ~~exp_07 parity mandate ("B-V should at least match FLAC")~~ → **FULL PARITY achieved**: ckpt 87,500, 8/8 cells, both K, 5-seed.
- ~~"fa-from-scratch plateaus ~2× worse" (exp_07 B-F)~~ → **RETRACTED 2026-08-01** (`exp_07…/fa_scratch_CORRECTION_addendum.md`): the screens had run under vanilla eval; under its own protocol B-F@40k is on par with vanilla at matched steps. Surviving: the 3.5× step cost.
- ~~"fa beats vanilla 12/14 at matched steps" (exp_10, 40k)~~ → **narrowed to band-parity** after the A3 diagnostic showed 40k was a band-best spike (neighbors 8.86–9.73 on both sides).
- ~~exp_10's open matched-COMPUTE estimand~~ → **closed** (fa@25k ≡ anchor compute: T60 tied, everything else far behind).
- ~~wandb account~~ → yh4742@princeton.edu key in `~/.bashrc` (below the interactive guard; scripts self-extract).
- ~~`train.py` hardcoded max_steps~~ → `--max-steps` flag (TDD).
- ~~worklog layout~~ → per-user namespace `worklog/worklog_yixun/` (announcement 03).
- ~~`cylindrical-dinov3` test location / missing pytest~~ → `src/tests/`, pytest installed by Yixun 2026-07-16.
