# Command — exp-09 Stage A blessed audit run (recorded BEFORE launch)

```bash
cd /home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl
set -o pipefail
WT_SHA=$(git rev-parse HEAD)   # = the records commit; re-verified in result review
CUDA_VISIBLE_DEVICES='' /home/yixunhu/miniconda3/envs/flac/bin/python \
  worklog/worklog_yixun/exp_09_cyl_no_ssl/audit_convention.py \
  --out worklog/worklog_yixun/exp_09_cyl_no_ssl/audit_convention.json \
  --checkpoint /home/yixunhu/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32 \
  --cyl-repo /home/yixunhu/codespace/cylindrical-dinov3 \
  --worktree /home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl \
  --data-root /home/yixunhu/codespace/xRIR_code/data \
  --scene Cafe --sub Cafe_idx_0 --wav S001_R0044_hybrid_IR.wav \
  --geometry 256 512 \
  --expect-fingerprint 81038cc90f3f295277016e2a8981867ed752b9a081183a8a9541204a892cad5b \
  --expect-package-sha 1f2c015905980a070c01a9aebce68bdebe00dbd2 \
  --expect-package-path-prefix /home/yixunhu/codespace/cylindrical-dinov3/src/cylindrical_dinov3/ \
  --expect-worktree-sha "$WT_SHA" \
  --expect-clean-worktree \
  --expect-checkpoint-revision 114c1379950215c8b35dfcd4e90a5c251dde0d32 \
  --expect-weights-sha256 4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d \
  2>&1 | tee worklog/worklog_yixun/exp_09_cyl_no_ssl/audit_convention_run.log
```
(`--out` is worktree-relative here and abspath-canonicalized by the runner from this
cwd — equivalent to the absolute path; both land in this folder.)
