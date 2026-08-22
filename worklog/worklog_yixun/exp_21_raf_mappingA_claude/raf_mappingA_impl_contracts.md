# exp_21 implementation contracts (r1) — from approved plan Rev 2

Binding for the Coder; plan Rev 2 sections are normative — these pins add implementation specifics. TDD in `src/tests/test_mappingA_*.py`; path-scoped commits "exp_21 r1: <desc> (TDD cycle N)"; ledger `commits_raf_mappingA.md`; Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>. Fences: this worktree only (`~/codespace/exp-21-raf-mapping-a`); never touch `~/codespace/FLAC` or `~/codespace/exp-19-raf-finetune`; no installs, no GPU, synthetic fixtures only; real corpus read-only.

## A. `data/RAF/mappingA_common.py`
- `cluster_placements(groups) -> clusters` — complete-linkage over tx-group rx-centroids, cap 0.05 m, no transitive chaining; deterministic ordering; medoid template per cluster (plan §2 M2 rules).
- `match_mics(template_rx [36,3], group_rx [36,3]) -> assignment` — Hungarian 36×36; unique; per-match displacement; FAIL the group if p95 > 0.01 m, any > 0.02 m, or ambiguity margin < 3× (next-nearest-mic distance / matched displacement); returns displacements for the record.
- `select_targets(placement_poses, seed) -> pose` — stable-hash-uniform (sha256 over room/placement-id/pose-key), documented estimand.
- `stable_item_context(...)` — per-item deterministic K=8 (exp_19 machinery), excluding every group sharing target xyz.

## B. `data/RAF/prepare_mappingA.py`
CLI `--raf-root --output-dir --split-dir --rooms --seed 0 --n-placements 16 --k 8 --readback-record --non-canonical`.
1. Correspondence + eligibility (≥9 passing source-xyz-distinct groups) → FPS over validated placement centroids → 16/room → 36 mic-slots × 1 hash target = 1,152 items.
2. **Audio union**: enumerate exact target+context capture set (~10,368); amplitude audit on raw resampled peaks (×3 no-clip ≤0.999; nothing < −60 dBFS post-scale; finite/nonzero) — ANY violation aborts with a measured report (Yixun stop-and-ask; never drop items); resample 48k→22,050 float32 ×3; read-back verification; files shared with the Mapping-H publication are reused byte-identically with provenance (id, hash, exp_19 generation).
3. Outputs: `data/RAF/mappingA_eval.json` (+ splits-record with correspondence stats, displacement distributions, target/context distance distributions, exclusions) + runtime `mappingA_metadata.json` (item → mic slot, rx_target_p + raw RAF rx height, tx*_p, per-context {capture id, tx_j_p, rx_j_p}, depth file, displacements).
4. **Static manifest validator** (plan §3): importable + CLI-run at publish; all M5 conditions.
5. Publication: DISJOINT roots (`<output>/mappingA/<Room>/…`, split files under `data/RAF/mappingA_*`), marker kinds `mappingA_prepare`/`mappingA_depth` via extended `publish.py` (new kinds + registered identities per plan §4; H/A composition tests: H→A, A→H, republish, injected crash — both flavors stay valid).

## C. `data/RAF/render_depth.py` listener mode
`--positions-from mappingA`: renders at `rx_target_p` (1,152 maps, dedup identical positions if any); QA parameterized with independent raw RAF rx height (nadir gate), transmitter endpoints for the recorded sightline diagnostic, listener-position provenance label; same miss cap 0.0025 / mask-derived QA / no-flipud; canonical identity = plan §4 list.

## D. `src/configs/dataset_configs/custom_metadata/RAF_A_md.py`
AR_md semantics with the plan §3 EXACT formulas (per-context own-rx); `md` contract identical shapes to RAF_md (target [1,10240], context audio [8,1,9600], source [3], source_vit [1,3], context_poses [8,3], depth [3,256,512]); provenance int64 `context_capture_ids` + `sample_target_id`; deterministic eval draws; mandatory `mappingA` publication verification (test-only opt-out constant, no env var); real-conditioner pass test under vanilla AND fa_invariant + C₄ rotation-invariance test on Mapping-A metadata.

## E. Config + stats
`src/configs/dataset_configs/RAF/eval/raf_mappingA.json` (id "RAF", RAF_A_md, deterministic true, expected 1,152). Per-placement metric collection: a small offline aggregator `data/RAF/mappingA_stats.py` consuming per-scene + per-item records (or retained per-item metrics via the stream sidecar) implementing plan §6: within-placement aggregation, room-stratified cluster bootstrap, paired placement-level randomization for arm contrasts; hand-checkable tests.

## F. Cycle plan (guide, adapt sensibly)
1 clustering+matching; 2 target/context selection; 3 union enumeration + amplitude audit; 4 resample/write + provenance; 5 manifest+validator+records; 6 publish.py flavor kinds + composition tests; 7 listener render mode; 8 RAF_A_md core; 9 RAF_A_md gate + conditioner/C4 tests; 10 config + stats aggregator; 11 integration on synthetic fixture end-to-end.
