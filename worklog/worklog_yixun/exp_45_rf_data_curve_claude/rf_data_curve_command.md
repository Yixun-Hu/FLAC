# exp_45 rf_data_curve — commands (A6000 box; every launch is appended here AT LAUNCH TIME; failed / superseded runs stay, marked)

Conventions: **tree 45** `~/codespace/exp-45-campaign` = a clean, detached checkout of the frozen launch commit `X`
(`CAMPAIGN_ROOT`; never committed in, never moved while the campaign lasts: `cmd_run` requires HEAD == the E0
receipt's head); **publication worktree** `~/codespace/exp-45-publish` (branch `exp-45-rf-data-curve`), from which
every records commit (`Y`, the results) is made and pushed on Yixun's go; **tree 38** = exp_38's own campaign tree at
its E0 head (exp_38 step 4 runs there, from exp_38's own sheet). Card locks are exp_38's
(`~/codespace/.exp38_card<i>.lock`): trees 38 and 45 serialize on the same cards. `E=worklog/worklog_yixun/exp_45_rf_data_curve_claude`.
Nothing below has been run yet.

```bash
# 0. tree 45 at X (once): the links, the environment receipt, E0 (L4b; N9: before any SMOKE)
X=<the frozen launch commit X>
git -C ~/codespace/FLAC fetch origin exp-45-rf-data-curve
git -C ~/codespace/FLAC worktree add --detach ~/codespace/exp-45-campaign $X
cd ~/codespace/exp-45-campaign
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; unset PYTHONOPTIMIZE PYTHONPATH
E=worklog/worklog_yixun/exp_45_rf_data_curve_claude
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms AcousticRooms
ln -s /home/yixunhu/codespace/FLAC/weights/AGREE weights/AGREE
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors weights/FLAC/VAE.safetensors
ln -s /home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt weights/FLAC/VAE.ckpt
python $E/exp45_registry.py untracked --platform a6000
python $E/exp45_eval.py env-receipt --write
python $E/exp45_eval.py prepare-e0          # sha256 + bytes of the 4 E0 references; creates eval_<closure16>
python $E/exp45_eval.py run --gpu 1 --cells de_E0a_K8_s42 de_E0b_K8_s42 de_E0c_K8_s42 de_E0d_K8_s42 \
                                            de_E0a_K1_s42 de_E0b_K1_s42 de_E0c_K1_s42 de_E0d_K1_s42
python $E/exp45_eval.py e0 && python $E/exp45_eval.py e0-check    # -> the sealed $E/rf_data_curve_e0_receipt.json

# 1. records commit Y from the publication worktree (tree 45 stays at X): the E0 receipt + sidecar and, per E0 cell,
#    its metrics JSON, stream sidecar and eval receipt (24 files), byte for byte, at the same relative paths
cd ~/codespace/exp-45-publish && git pull --ff-only
cp ~/codespace/exp-45-campaign/$E/rf_data_curve_e0_receipt.json{,.sha256} $E/
for g in E0a E0b E0c E0d; do mkdir -p outputs_FLAC/exp45_rf_data_curve/eval/$g
  cp ~/codespace/exp-45-campaign/outputs_FLAC/exp45_rf_data_curve/eval/$g/*_metrics_1_1.0_de_${g}_K[18]_s42*.json \
     outputs_FLAC/exp45_rf_data_curve/eval/$g/; done          # metrics + stream + receipt: 6 per gate, 24 in all
git add -f $E/rf_data_curve_e0_receipt.json{,.sha256} outputs_FLAC/exp45_rf_data_curve/eval/E0{a,b,c,d}
#    commit "exp_45 records: E0 PASS at X" -> Y; check closure(Y) == closure(X); push on Yixun's go; then neuronic

# 2. the RF-100 bridge (N5) in tree 45, after E0: verify + link the checkpoint, K8 s42 first, then the rest;
#    exp_43's admission record + its ten reference cells arrive as a bundle dir with BRIDGE_MANIFEST.sha256
cd ~/codespace/exp-45-campaign
python $E/exp45_eval.py prepare-bridge
python $E/exp45_eval.py run --gpu 1 --cells exp45_RF100bridge_K8_s42
python $E/exp45_eval.py run --gpu 1 --cells exp45_RF100bridge_K8_s43 exp45_RF100bridge_K8_s44 exp45_RF100bridge_K8_s45 \
    exp45_RF100bridge_K8_s46 exp45_RF100bridge_K1_s42 exp45_RF100bridge_K1_s43 exp45_RF100bridge_K1_s44 \
    exp45_RF100bridge_K1_s45 exp45_RF100bridge_K1_s46
python $E/exp45_eval.py bridge --bundle <exp_43 reference bundle dir>    # -> sealed $E/rf_data_curve_bridge.json

# 3. (L8) per neuronic-trained run, once published to the NAS (neuronic sheet step 4):
RUN=exp45_RFf025_l40_2x32       # then exp45_RFf050_l40_2x32, exp45_RFf075_l40_2x32
python $E/exp45_eval.py prepare $RUN          # TRAINING_COMPLETE (incl. the loss record) at the E0 closure; farm copies
python $E/exp45_eval.py run --gpu 0 --cells ${RUN}_K8_s42 ${RUN}_K8_s43 ${RUN}_K8_s44 ${RUN}_K8_s45 ${RUN}_K8_s46 &
python $E/exp45_eval.py run --gpu 1 --cells ${RUN}_K1_s42 ${RUN}_K1_s43 ${RUN}_K1_s44 ${RUN}_K1_s45 ${RUN}_K1_s46; wait
python $E/exp45_collect.py            # all 30 + 10 cells admitted OFFLINE, else it refuses (exit 2)
#    then the exp_45 results commit from the publication worktree (force-added farm evidence under
#    outputs_FLAC/exp45_rf_data_curve/eval/, subject "exp_45 results publication ..."), pushed on Yixun's go
```
