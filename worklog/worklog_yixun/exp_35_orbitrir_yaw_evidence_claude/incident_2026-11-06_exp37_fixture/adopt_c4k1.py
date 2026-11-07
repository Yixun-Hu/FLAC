"""Incident recovery (2026-11-06, Planner): adopt the H35 re-run of exp35_BF_C4_S40000_K1_s42_th90 (run in the scratch worktree
~/codespace/exp-35-k1-recover at H35 5b065801, dumps into a scratch NAS root) as the missing original ONLY if (a) its metrics JSON
hashes to the pre-incident digest 9674cfff… and (b) its waveform COMPLETE marker equals the original NAS dump's marker (bitwise
waveforms); then copy the JSON into the exp_35 farm with exclusive creation and re-hash. Refuses otherwise; never overwrites."""
import hashlib, os, sys
from pathlib import Path

STEM = "epoch=8-step=40000_metrics_1_1.0_exp35_BF_C4_S40000_K1_s42_th90_fa_invariant_a4_rot90"
SRC = Path(f"/home/yixunhu/codespace/exp-35-k1-recover/outputs_FLAC/exp25_depth_canon/BF/{STEM}.json")
SRC_DONE = Path(f"/media/diskstation/yixunhu/FLAC/.exp35_k1_recover_dumps/{STEM}/COMPLETE")
ORIG_DONE = Path(f"/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms/{STEM}/COMPLETE")
DST = Path(f"/home/yixunhu/codespace/exp-35-orbitrir-yaw/outputs_FLAC/exp25_depth_canon/BF/{STEM}.json")
WANT = "9674cfff19a4541cd8df76ef4a3cd92096a953981516a596d8a303540f528282"
b = SRC.read_bytes()
got = hashlib.sha256(b).hexdigest()
print("re-run metrics JSON", got, len(b), "B; == pre-incident:", got == WANT)
src_marker, orig_marker = SRC_DONE.read_text().strip(), ORIG_DONE.read_text().strip()
print("waveform COMPLETE markers: re-run", src_marker[:16], "original", orig_marker[:16], "equal:", src_marker == orig_marker)
if got != WANT or src_marker != orig_marker:
    sys.exit("REFUSED: the re-run does not reproduce the original; nothing written (declare unrecoverable; register new evidence)")
if os.path.lexists(DST):
    sys.exit(f"REFUSED: {DST} exists; nothing written")
with open(DST, "xb") as f:
    f.write(b); f.flush(); os.fsync(f.fileno())
after = hashlib.sha256(DST.read_bytes()).hexdigest()
print("adopted", DST, after, "OK" if after == WANT else "MISMATCH")
sys.exit(0 if after == WANT else 1)
