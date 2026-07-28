# Commits — exp_07_fa_scratch

Base: `0bd5da0`. Branch: `check-equivariance-necessity`.

| Order | SHA | Summary |
|---|---|---|
| 1 | `b2e2000` | scaffold — from-scratch fa_invariant (Route B) |
| 2 | `79b9791` | plan — two matched from-scratch arms; budget decision table for Yixun |
| 3 | `8db486a` | plan review (REQUEST-CHANGES) + revision — 67.5k-step anchor discovered in FLAC.ckpt; --max-steps round; eval protocol; thresholds |
| 4 | `8ae9837` | config-identity audit round (Yixun's go-condition): released-ckpt probe v2 (counters/optimizer/scheduler/config-diff/ViT pin), arm configs BV/BF + asserts v4 (init-identity + fail-closed pin gate, red/green-proven), audit doc + launch manifest, plan eff-64 corrections, gpt-5.6-sol review→reverify→reverify2 loop closed; SOP reviewer-model update rides along |

Notes: audit authored by Fable 5 (main session; seat restored from Opus 4.8 per Yixun's `/model` switch). Reviewer for this and all future rounds: Codex `gpt-5.6-sol` xhigh (per Yixun 2026-07-10; CLI 0.144.1).
| 5 | _(this commit)_ | bookkeeping — record audit-round SHA `8ae9837` (amend during commit 4 changed its hash; lesson: never amend a SHA into a commit that contains it) |
| 6 | `e85ebde` | TDD round 1 — --max-steps flag (build_trainer_kwargs/construct_trainer + 10 tests, 121-suite green); review round closed |

## Full commit list (4d07611..closure, auto-generated at closure)

- `64a8cf3` exp_08: plan — matched comparison reusing V1' as control; A-F arm + H-A1/A2/A3 gates
- `a4d7b45` exp_08: plan review (APPROVE-WITH-CHANGES) + revision — A-V bf16 eval mirror added
- `a022385` exp_08: Yixun's pre-approval findings solved — M5 train-seed sensitivity pair + tiered H-A1 bands
- `f94b206` exp_08: approved; params + launch commands
- `779bc70` exp_08: M1.5 mirror — bf16 shifts A-V T60 +0.12 (confound confirmed real); comparator registered
- `b942357` exp_08: H-A1 verdict — strict FAIL, T60 superior (near-baseline K=8), EDT/C50 regressions; M3/M4 commands
- `50f58e6` exp_08: H-A2 PASS + H-A3 PASS — minimum project goal achieved on fine-tuned model; M5 launching
- `a3e8cf5` exp_08: closure — fa_invariant matched fine-tune; H-A2/H-A3 PASS (min goal on trained model), strict H-A1 FAIL with seed-robust T60 gain
- `83d58db` exp_08: record closure SHA a3e8cf5 in commits log
- `8ae9837` exp_07: config-identity audit — B-F≡B-V proven (init-identical); B-V≡released config up to no-op keys + pinned-ViT caveat; recipe corrected to eff-batch 64
- `21294cc` exp_07: record audit-round SHA 8ae9837 in commits log
- `e85ebde` exp_07: TDD round 1 — --max-steps flag for train.py (test-first; gpt-5.6-sol review round closed)
- `aa2015e` exp_07: record TDD round-1 SHA e85ebde in commits log
- `70dea5a` exp_07: M0 — registered ladder OOMs (64x1/32x2/16x4, FA ViT pass); documented amendment extends to 8x8 -> FITS both arms
- `cb85fd0` worklog: per-user namespace — everything moves to worklog/worklog_yixun/ except the portable experiment_SOP.md (Yixun directive, announcement 03)
- `ecb8352` exp_07: online-weights eval-config copy (use_ema=false) — committed before the first 10k screen per plan §3
- `50cd944` exp_05_cylvit: CylindricalViT geometry encoder + matched A/B configs and tests
- `b33bfb3` exp_05_cylvit/exp_06: worklog — plans, drivers, run scripts, results, figures
- `68f3caa` Fix test_eval_paths comparator import after worklog_yixun namespace move
- `b6f4030` worklog: seed master_experiment_tracker + issue_report (session-handoff protocol, Yixun directive; CLAUDE.md rule is local)
- `fa2b814` exp_07: B-V gate — strict FAIL 1/6, stopped per GO step 5; endpoint-draw vs systematic-EDT decomposition; 291k corroborates lineage signature
- `df35c4e` Merge pull request #1 from Yixun-Hu/Yaw-equi-ViT
- `79d9e39` tracker: note Yaw-equi-ViT merge (PR #1) + post-merge verification
- `ed011dd` exp_07 P0: full 21-point selection curve — selection alone cannot reach parity (EDT floor 38.29 vs 37.10; R@1 max 6.22 vs 7.06); plan_bv_parity (P1 micro-parity rerun) drafted, review in flight
- `67b8fce` exp_07 P1 plan: review round applied — best-observed-point wording, corrected micro-hypothesis framing, branch-and-estimand table (tiered PARITY/STRONG/DIRECTIONAL/NULL, R@1 required for Q5 parity), 8x8 stays sole B-F control, 16x4 rung added, hard-abort-only discipline
- `c40908c` exp_07 phase-2 scaffolding + model-change handoff hook
- `1de9e37` exp_07: reverify-round fixes (hook + P1a probe) + handoff refresh (Opus->Fable switch; extend launched)
- `cb0acfc` hook: reverify2 residual fix (non-dict message type guard); review round closed
- `87e6cdb` exp_07 extend: bvext_screen.sh (reviewed SHIP) + command-log entries
- `d0db803` exp_07 worklog: extend launch + S70000 first point (all metrics improve vs endpoint; R@1 6.49 = lineage max)
- `919a2f6` exp_07: B-F pre-staged launch kit (reviewed, fixes applied) + queue reorder extend->B-F->P1
- `b9adf60` exp_07 B-F: wandb ON (yh4742 key verified) + non-interactive .bashrc workaround
- `b3e067d` exp_07: extend STOPPED at ckpt 77500 (restart contract logged) + B-F DDP params sheet for Yixun verification
- `c584d80` exp_07: SyncBN mandate folded into params sheet (Yixun-approved, launch-gated); extend kit ready for wandb resume
- `f362673` exp_07: --sync-batchnorm wiring (TDD, 40 tests) + B-F DDP+SyncBN launch kit (review-clean)
- `87881bf` handoff: SyncBN batch closed, M1 watcher armed, extend second-leg state
- `e40760a` handoff: model-alternation finding (docs are the intra-session channel) + cylindrical-dinov3 sibling repo state
- `685356d` handoff: extend leg-2 verified state + 72.5-77.5k screen numbers (R@1 6.596 lineage max)
- `9db23a9` handoff: cylindrical-dinov3 exp_01 R1+R2 closed (plan approved r4); 2 decisions for Yixun
- `9fccf7a` handoff: S80000 screen — R@1 6.738, 3rd consecutive lineage max (gap to released 0.32)
- `4622721` issue_report: both cylindrical-dinov3 decisions resolved by Yixun + 3 new findings
- `c7535cf` handoff: S90000 R@1 6.817 (4th lineage max, gap 0.24); M1 ETA pulled in to ~23:30 tonight; post-100k selcurve auto-plan
- `6596121` exp_07: training env switched to conda 'flac' (Yixun) — pre-flight green, flash-attn kernel delta disclosed; M1 watcher re-armed on flac; evals stay rir2rir
- `38c3223` exp_07 EXTEND COMPLETE: full 13-pt curve — R@1 7.054@92.5k = released parity (under-training confirmed for R@1); EDT sole systematic gap; 92.5k = recommended ckpt; 4-seed confirm running
- `bc0ad10` exp_07: 92.5k 5-seed confirm — R@1 6.921±0.186 vs released 7.06 = 0.66σ_c → EQUIVALENCE tier; parity confirmed under protocol
- `a18e684` exp_07 M1: co-tenancy policy (Yixun) — per-GPU free-VRAM gate 45,087 MiB (review-corrected units) + co-tenant disclosure; single-rank fit proxy (reviewed; sampler guard, env enforcement, 124/137 timeout, explicit headroom classification)
- `e9ae24e` exp_07 M1 rung report: BN=64 rung (micro-32 B-F) OOMs solo on 48GB under BOTH allocators — true demand >=47GB (ViT-dominated); options (BN=32 compromise / ViT grad-ckpt for BN=64 / single-GPU 8x8) to Yixun; HOLD
- `b5f3038` exp_07: Yixun decision — BN=64 via ViT gradient checkpointing (use_reentrant=False); params + handoff updated; wiring in flight
- `cfdd7b3` exp_07: ViT gradient checkpointing — explicit per-layer adapter (TDD, execution-proven) + BF config key
- `f59f5a4` exp_07 B-F LAUNCHED (co-tenant GO): reverify SHIP + P0 formally retracted; launch gate -> MIN_FREE_MB threshold
- `824c2b6` handoff: B-F re-anchored — 0.079 steps/s co-tenant, done ~Jul 28, verdict ~Jul 29; health nominal
- `8510022` env policy final (Yixun): flac for everything incl. evals; env-bridge eval running to quantify the flash-DiT eval delta
- `eee666f` env-bridge verdict: flac vs rir2rir eval IDENTICAL to 4 decimals — chain comparability unconditional
- `3ba3498` exp_07 B-F 10k/20k screens + cfg0 lift probe: conditioning ACTIVE (lift >= BV absolute), trajectory globally slow, R@1 disproportionately late; ride to 50k futility review
- `82eb8e9` handoff: live status Jul 21 — B-F 41%, aug291k ended (100k), new third-party co-tenants noted, ETA ~Jul 27/28
- `9d6eab0` exp_07 B-F 30k: slope frozen (~2x EDT, 0.15x R@1) — futility options pre-staged for the 50k review
- `7b15104` exp_07: B-F STOPPED at 40k (Yixun futility call) -> P1 kit: BVp1 config (single-delta vs BF = cond_method+frame_avg_angles ONLY), p1_ddp_launch.sh (identical recipe + contract pre-flight), stop record + plan amendment
- `e50d098` p1 launch: fix contract-diff window (BVp1-vs-BV = 6 lines incl. replaced-line pairs; gate fired correctly on my miscount)
- `980ce59` handoff: B-F stopped @40,249 (futility) -> P1 attribution arm LIVE (acwm8gvt); monitors + review armed
- `1f49800` P1 kit review: live run VALID; amendment precision fixes (sequential decomposition, semantic copy, total-FA-effect estimand); script contract-gate fix deferred to run end (no editing a running script)
- `cb95f98` handoff: P1 re-anchored — 0.259 steps/s (3.5x B-F; attribution datapoint #1), done ~Jul 26 night
- `b0afa1c` exp_07 P1@10k: RECIPE INNOCENT (P1 ~= 8x8 anchor; EDT even better) — B-F's 2x gap attributes to fa-from-scratch; +P1 online-eval config
- `3d35396` exp_07 P1@20k: attribution VERDICT — recipe innocent (T60 8.44 < released), fa-from-scratch guilty (single-delta carries full crater); R@1 watch flagged
- `c4778cf` exp_07 P1@30k: R@1 watch resolved (4.17 > anchor) — recipe clean on all metrics
- `ccea7ec` exp_07: commit all tee'd screen/probe eval logs + B-V loss plot (SOP terminal-output provenance)
- `1b2f063` handoff refresh (model change fable->opus, authored by Opus 4.8): all screen/probe logs pushed (ccea7ec); P1 live ~33k, attribution verdict in, endpoint pending
- `1b56e7a` SOP: Coder/Claude-reviewer model -> Opus 5 max (Yixun); P1 launch script: resume support + review-prescribed parsed-object contract gate
- `39da10b` handoff: P1 death (session-restart teardown) + resume-from-32.5k recovery recorded; restart live-verify lesson; SOP Opus 5 update noted
- `61b8d7e` handoff: P1 resume confirmed (lr45v31g); exclusive-GPU rate 2x -> done ~Jul 26 evening, verdict ~Jul 27
- `a131fdc` exp_07 P1@50k: EDT 37.649 — best B-V-family EDT on record (8x8 floor 38.29); BN=64 hypothesis revived; late-curve stat + endpoint gate adjudicate
- `27ead43` exp_07 P1@60k: R@1 6.33 (+0.57 vs anchor matched-step); late-curve screens queued for endpoint block
- `2e2ac02` exp_07 P1 COMPLETE: EDT STRONG PASS (37.664, 81% gap closure); FIRST composite qualifier (57.5k: all 3 error metrics sub-released); 18-eval gate block overnight
- `3b6e1d2` exp_07 P1 GATE VERDICT: 57.5k SUPERIOR to released on T60/C50/EDT at both K (6/8 cells); R@1 sole open gap
- `9252742` tracker/issue: maximum goal ACHIEVED on 3/4 metrics (P1@57.5k 5-seed superior); closing fork to Yixun (extend-for-R@1 recommended)
- `860a320` issue_report: drop stale Jul-17 queue item
- `4e23737` p1 launch: MAXSTEPS override for the R@1 extension (Yixun closing-fork pick: extend then close)
- `3607c73` handoff (compact trigger): P1 extension killed by restart teardown AGAIN (~15:35, ckpt 72500) -> re-resumed 16:32; 70k ext screen T60 8.079 lineage-best, R@1 6.23; crossing rule stands
- `5e45c93` exp_07 R@1 extension: 75k R6.675 (+0.44/5k, crossing 6.9 nears); second teardown kill documented + leg-3 resume
- `48cf8a4` exp_07 ext 80k: R@1 6.833 (0.07 below crossing), C50 0.9332 program-best, EDT ~released
- `7f252cb` exp_07 ext 85k: R@1 oscillation dip (6.25 after 6.83 peak) — precedent-zone hunt continues
- `3630fce` exp_07 ext 90k: R@1 6.849 third near-miss; EDT 36.60 sub-released; 92.5k precedent step next
- `ae8bc1a` exp_07 FULL PARITY: ckpt 87500 5-seed confirmed both K — 8/8 cells SUPERIOR/EQUIV vs released Table-1; MAXIMUM PROJECT GOAL CLOSED
- `7e9246a` exp_07 ext 92.5k/95k: below the 87.5k peak (92.5k C50 0.9300 program-best); 87.5k stands as ckpt of record
- `e0dd598` exp_07 CLOSING: fa_scratch_results.md (full-parity headline + all arms) + fa_scratch_analysis.md (reliability, honest scope, next steps); HANDOFF -> closing state
- `d3bc39d` exp_07: closing HTML results page + generator (tables/verdict blocks; numbers transcribed from results.md)
