# exp_38 data_efficiency — commands (every launch is appended here AT LAUNCH TIME; failed / superseded runs stay, marked)

Conventions: dev worktree `~/codespace/exp-38-data-efficiency` (branch `exp-38-data-efficiency`); **campaign tree** `~/codespace/exp-38-campaign` (a detached worktree at the pushed campaign pin `<PIN>`, never committed in while anything runs); conda env `flac`; `E=worklog/worklog_yixun/exp_38_data_efficiency_claude`. Nothing below has been launched for real yet (2026-11-07): only the rungs marked DONE.

## Done (Coder, 2026-11-07, validation ladder)
```bash
# rung 3 — subset audit v2 + verify (CPU, read-only AR tree)
python $E/exp38_subset_audit.py --out $E/data_efficiency_subset_audit.json        # PASS, 39 s
python $E/exp38_subset_audit.py --verify $E/data_efficiency_subset_audit.json     # PASS
# rung 4 — CPU init digest (identical for every arm: 378440d0…)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python $E/exp38_init_digest.py --run de_bf_f050 --json <scratch>/init.json
# rung 5 — every launch path dry-run: $E/data_efficiency_2026-11-07_21-36-46_dry_run_all_paths.log + …_dry_run_sbatch.log
# code-review r1 fix round (2026-11-07 22:18) — subset audit v3 (realpath dropped from the bound map, HIGH 8)
python $E/exp38_subset_audit.py --out $E/data_efficiency_subset_audit.json        # PASS (v2 kept as *_v2_superseded_r1.json)
# code-review r2 fix round (2026-11-07 23:51 -> 2026-11-08 00:05) — the AR content manifest (BLOCKING 3): 445,414 files,
# 17.87 GB, per-file sha256 grouped into 729 room groups; pinned sha256(text) 369bd4fc… in exp38_registry.py
python $E/exp38_content_manifest.py --build <scratch>/ar_content_manifest.txt.gz --threads 16   # 14 min over CIFS
python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz        # every byte re-hashed
# rung 6 — GPU-0 single-rank smokes (Yixun-allowed, ≤ 15 min; GPU 1 untouched)
MODE=SMOKE RUN=de_yaw_f025 SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-07-13 FAIL (smoke loss floor 50, fixed) -> 21-15-35 PASS
MODE=SMOKE RUN=de_bf_f025  SMOKE_GPU=0 SMOKE_STEPS=20 bash $E/exp38_train_local.sh   # 21-10-54 PASS
```

## Done (campaign tree at `bfc4ea4b`, 2026-11-08 01:36–03:44) — superseded by the decisions below
```bash
python $E/exp38_registry.py untracked --platform a6000                     # PASS
python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz   # PASS
python $E/exp38_eval.py env-receipt --write && python $E/exp38_eval.py prepare-e0             # eval_8828062f4d436732
python $E/exp38_eval.py run --gpu 0 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42   # 01:57-02:55
python $E/exp38_eval.py run --gpu 1 --cells de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42   # 02:55-03:44
python $E/exp38_eval.py e0      # FAIL: E0a K8 + K1 (acoustics exact; AGREE retrieval R@1 / R@5 / R@10 differ by
#                                 1-6 items of 6,337); E0b-d exact -> D6 asked (data_efficiency_e0a_decision.md)
```

## Campaign under Yixun's decisions of 2026-11-08 (D2 = neuronic, D6 = "Accept, label R@1")
All seven trainings (six primary + the mandatory bridge `de_p1_f025`) run on neuronic
(`data_efficiency_neuronic_commands.md`); `MODE=FULL` refuses on this box (`TRAINING_PLATFORM`). This box runs
E0 and every evaluation cell, FROM the campaign's evaluation snapshot `~/codespace/exp-38-snapshots/eval_<closure16>/`
(created by `prepare-e0`; a new closure gets a new snapshot, the old one is inert). The campaign identity is the
**closure digest** (`exp38_registry.py closure`); the E0 cells and the result cells bind the campaign tree's
HEAD, so the tree stays frozen at the pin for the whole campaign (records written there are untracked; they are
copied to the dev worktree and committed from there — with the farm's E0 / result cells force-added, which the
offline table validator re-admits). E0a passes only with the exact (ours, reference) values pinned in
`data_efficiency_e0a_decision.json`; everything else is type-strictly exact.
```bash
# 0. move the campaign tree to the new pin (no exp_38 job may be running in it)
PIN=<PIN>
git -C ~/codespace/exp-38-campaign fetch origin exp-38-data-efficiency   # shared object store; harmless if present
git -C ~/codespace/exp-38-campaign checkout --detach $PIN
cd ~/codespace/exp-38-campaign
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
E=worklog/worklog_yixun/exp_38_data_efficiency_claude
test "$(git rev-parse HEAD)" = "$PIN" && test -z "$(git status --porcelain --untracked-files=no)"
python $E/exp38_registry.py closure                    # the closure digest of this pin
#    archive the superseded E0 at bfc4ea4b: the cells (receipts are written once; a cell refuses existing
#    outputs) inside the ignored outputs tree, the sealed E0 receipt OUTSIDE the tree (the untracked allowlist
#    admits no sub-directories in $E); the .ckpt links, the environment receipt and the logs stay
A=outputs_FLAC/exp38_archive/E0_bfc4ea4b_2026-11-08; mkdir -p $A
for g in E0a E0b E0c E0d; do mkdir -p $A/$g && mv outputs_FLAC/exp38_data_efficiency/eval/$g/*.json $A/$g/; done
mkdir -p ~/codespace/exp-38-archive/E0_bfc4ea4b_2026-11-08
mv $E/data_efficiency_e0_receipt.json{,.sha256} ~/codespace/exp-38-archive/E0_bfc4ea4b_2026-11-08/
python $E/exp38_registry.py untracked --platform a6000                      # UNTRACKED PASS (links by content)
python $E/exp38_eval.py env-receipt --check                                 # the sealed receipt of 01:56 still holds
#    (a fresh tree instead: the links + `env-receipt --write`, as in the first campaign, and the content verify:)
python $E/exp38_content_manifest.py --verify $E/data_efficiency_ar_content_manifest.txt.gz   # ~15 min, optional here

# 1. E0 at the pin (one A6000 card suffices: both cards are the registered A6000 model, every cell receipt
#    records its card; ≈ 1 h 50 min sequentially on one card, or ≈ 1 h on both)
python $E/exp38_eval.py prepare-e0          # sha256 + bytes of the 4 references; creates eval_<closure16>
python $E/exp38_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42 \
                                            de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42
python $E/exp38_eval.py e0            # -> the sealed $E/data_efficiency_e0_receipt.json (PASS / FAIL + every difference)
python $E/exp38_eval.py e0-check      # every later consumer re-runs this (incl. re-admitting the 8 E0 cells)
#    then copy the E0 receipt + the 8 cells' JSONs to neuronic (neuronic sheet step 0) and start the trainings

# 2. per neuronic-trained run, once it is published to the NAS (neuronic sheet step 4), in any order:
#    the seven: de_yaw_f025 de_yaw_f050 de_yaw_f075 de_bf_f025 de_bf_f050 de_bf_f075 de_p1_f025
RUN=de_bf_f025
python $E/exp38_eval.py prepare $RUN          # TRAINING_COMPLETE (neuronic, every field + retained records) at the
#                                              E0 closure, re-hash + bytes, full-state audit; farm copies
python $E/exp38_eval.py run --gpu 0 --cells ${RUN}_K8_s42 ${RUN}_K8_s43 ${RUN}_K8_s44 ${RUN}_K8_s45 ${RUN}_K8_s46 &
python $E/exp38_eval.py run --gpu 1 --cells ${RUN}_K1_s42 ${RUN}_K1_s43 ${RUN}_K1_s44 ${RUN}_K1_s45 ${RUN}_K1_s46; wait
#    (or all ten on one card: run --gpu 1 --cells <the 10>); a failed / interrupted cell keeps its
#    <metrics stem>.claim dir (archive it with the partial outputs to retry)
python $E/exp38_collect.py            # -> $E/data_efficiency_results.{md,json}; exit 1 while cells are pending
python $E/exp38_collect.py --partial  #    (publishes the admitted runs with PENDING rows, exit 0)
#    the FLAC 25/50/75 % rows are exp_14's records: their retrieval columns carry † (exp_14's evaluator, D6)
```

## Neuronic (D2 = neuronic: all seven trainings): see `data_efficiency_neuronic_commands.md`
