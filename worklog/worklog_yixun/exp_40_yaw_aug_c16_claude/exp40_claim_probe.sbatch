#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=exp40-claim-probe
# ============================================================================
# exp40_claim_probe.sbatch -- plan §14 B1: two tasks on two DIFFERENT nodes race an
# O_CREAT|O_EXCL create of the same claim path on the shared filesystem, 20 rounds; PASS
# iff every round has exactly one winner. Records the mount source/type. CPU only.
#   sbatch --output=<repo>/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/slurm_claim_probe_%j.out \
#          worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_claim_probe.sbatch
# then: python worklog/.../exp40_claim_probe_verdict.py (inline below via --verdict)
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan §14 B1.
# ============================================================================
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
DIR="$REPO/outputs_FLAC/exp40_yaw_aug_c16/claims/probe_${SLURM_JOB_ID}"
mkdir -p "$DIR"
findmnt -n -o SOURCE,FSTYPE,OPTIONS --target "$DIR" > "$DIR/mount.txt"
srun --ntasks=2 --ntasks-per-node=1 /n/fs/gatrdp/envs/flac/bin/python - "$DIR" <<'PYEOF'
import json, os, socket, sys, time
d = sys.argv[1]
rank = int(os.environ["SLURM_PROCID"])
res = []
for i in range(20):
    while time.time() % 2 > 0.05:            # both tasks fire at the same wall-clock tick
        time.sleep(0.005)
    try:
        fd = os.open(os.path.join(d, f"round{i}.claim"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        os.write(fd, f"{socket.gethostname()} {rank}\n".encode()); os.fsync(fd); os.close(fd)
        res.append(True)
    except FileExistsError:
        res.append(False)
    time.sleep(0.2)
json.dump({"rank": rank, "host": socket.gethostname(), "won": res}, open(os.path.join(d, f"rank{rank}.json"), "w"))
PYEOF
/n/fs/gatrdp/envs/flac/bin/python - "$DIR" "$REPO/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/receipts/claim_probe_neuronic.json" <<'PYEOF'
import json, os, sys, time
d, out = sys.argv[1:]
r = [json.load(open(os.path.join(d, f"rank{i}.json"))) for i in (0, 1)]
ok = r[0]["host"] != r[1]["host"] and all(a != b for a, b in zip(r[0]["won"], r[1]["won"]))
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"verdict": "PASS" if ok else "FAIL", "hosts": [x["host"] for x in r], "rounds": len(r[0]["won"]),
           "mount": open(os.path.join(d, "mount.txt")).read().strip(), "dir": d, "t": time.time()},
          open(out, "w"), indent=1)
print("CLAIM PROBE", "PASS" if ok else "FAIL", out)
PYEOF
