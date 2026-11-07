"""Incident recovery (2026-11-06, Planner): rebuild exp_35's overwritten metrics JSON exp35_BF_C4_S40000_K8_s42_th90 from
chain I's B-F replay exp36_BF_C4_S40000_K8_s42_th90 (the same evaluation at H36_I3, no-flash; the replay comparator's
equality excludes only source_sha and eval_name). Swaps exactly those two values (each must occur exactly once) and writes the
result into the exp_35 farm ONLY if it hashes to the pre-incident digest ef2840c7… (5,309 B) recorded before the incident
(the Planner's 15:40 pin and the leaf's 19:26:27 log); refuses otherwise, never overwrites."""
import hashlib, os, sys
from pathlib import Path

SRC = Path("/home/yixunhu/codespace/exp-36-run/outputs_FLAC/exp25_depth_canon/BF/"
           "epoch=8-step=40000_metrics_1_1.0_exp36_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90.json")
DST = Path("/home/yixunhu/codespace/exp-35-orbitrir-yaw/outputs_FLAC/exp25_depth_canon/BF/"
           "epoch=8-step=40000_metrics_1_1.0_exp35_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90.json")
WANT = "ef2840c746b871268bddf8ae36609d1ee33186074a9afb01612c5db97606d82f"
b = SRC.read_bytes()
for old, new in ((b'"exp36_BF_C4_S40000_K8_s42_th90"', b'"exp35_BF_C4_S40000_K8_s42_th90"'),
                 (b'"6be49f46ec31e899f1bee01f72c6a5827e701c6c"', b'"5b065801e9d1c1844461ec84cd0558702c714326"')):
    if b.count(old) != 1:
        sys.exit(f"REFUSED: {old!r} occurs {b.count(old)} times in the replay JSON")
    b = b.replace(old, new)
got = hashlib.sha256(b).hexdigest()
print("candidate", got, len(b), "B", "== pre-incident:", got == WANT)
if got != WANT:
    sys.exit("REFUSED: the rebuilt bytes are not the pre-incident file; nothing written")
if os.path.lexists(DST):
    sys.exit(f"REFUSED: {DST} exists; nothing written")
with open(DST, "xb") as f:
    f.write(b); f.flush(); os.fsync(f.fileno())
after = hashlib.sha256(DST.read_bytes()).hexdigest()
print("restored", DST, after, "OK" if after == WANT else "MISMATCH")
sys.exit(0 if after == WANT else 1)
