"""Incident restore (2026-11-06, Planner): put the exp_35 phase-(i) BF artifacts back to their PRE-INCIDENT bytes and quarantine
every fake the exp_37 test fixture wrote (20:21-20:52 EST). Refuses unless each current file is exactly the fake recorded in
affected_sha256_at_report.txt (21:08) and each restore source hashes to the pre-incident digest recorded by the leaf at 19:26:27
(and, where available, the Planner's own earlier pins). Fakes are MOVED into a quarantine folder, never deleted; originals are
copied in with no-replace semantics (the fake is moved away first) and re-hashed after the copy. Every action is printed."""
import hashlib, os, shutil, sys
from pathlib import Path

I = Path("/home/yixunhu/codespace/exp-37-loc-efficiency/worklog/worklog_yixun/exp_37_loc_efficiency_claude/incident_2026-11-06")
R, Q = I / "restore", I / "quarantine"
FARM = Path("/home/yixunhu/codespace/exp-35-orbitrir-yaw/outputs_FLAC/exp25_depth_canon/BF")
LEAF = Path("/home/yixunhu/codespace/exp-37-b3-synthesis/outputs_FLAC/exp25_depth_canon/BF")
NAS = Path("/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms")
SCR = Path(sys.argv[1])  # holds the T K8 metrics JSON rebuilt from its T-dup twin
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
P = "epoch=8-step=40000_metrics_1_1.0_"
# pre-incident digests: (metrics json, receipt, NAS manifest) from the leaf's 19:26:27 log (= reconstruct.py ORIG)
ORIG = {
 "exp35_BF_T_S40000_K1_s42_fa_invariant_a4": ("1e1768e3adf0d6513b705fa23f5a3ea7d9f357479755fb9d3fe9cc9d2a9efb46", "fa529324216c88832466a31d1b86fb4e4c8409aa037f45debdde1bc30befebef", "7dde0aa70ce01f0e108f12b6fce4eb72a0bf30ca79f2690853db21c0ab7fbf8f"),
 "exp35_BF_T_S40000_K8_s42_fa_invariant_a4": ("f167470e3501bcc7c4f116b0c26164523a8bd140b55c5e72e935d88d3f373adb", "79817d0b100005ed91172e80ea09ea3c1d250aeb549e0f0ef29bdaea16f141ad", "dda03a0e318f65b216108c166e9cae10bdd44bd3473811c56dbfd10f844894d2"),
 "exp35_BF_C4_S40000_K1_s42_th90_fa_invariant_a4_rot90": ("9674cfff19a4541cd8df76ef4a3cd92096a953981516a596d8a303540f528282", "983f9d5f9842a1256ad697663725b174fd4ae1405d56e097b6c4e4cd90e6c6da", "4923d48e954b3b23b7788d42d93ae99f7288aa1c76001ed9ef84d27a821713ee"),
 "exp35_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90": ("ef2840c746b871268bddf8ae36609d1ee33186074a9afb01612c5db97606d82f", "431ceba60189490e85d53d942f43025c493a863cddec928ec7555d34ed5b16fd", "975cb72125977b6724cda82501bff4b356accb80278db7f8341c69566630fd5c"),
}
# the Planner's own independent pins (computed 2026-11-06 ~15:40 / from the exp_36 chain's pinned dump evidence)
assert ORIG["exp35_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90"][1] == "431ceba60189490e85d53d942f43025c493a863cddec928ec7555d34ed5b16fd"
fakes = {}
for line in (I / "affected_sha256_at_report.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 2 and len(parts[0]) == 64:
        fakes[parts[-1]] = parts[0]
def refuse(msg):
    print("REFUSED:", msg); sys.exit(2)
plan = []
for suf, (m, r, man) in ORIG.items():
    for kind, cur, want_src in (("receipt", FARM / f"{P}{suf}.cell.json", R / f"{P}{suf}.cell.json"),
                                ("manifest", NAS / f"{P}{suf}" / "manifest.json", R / f"{P}{suf}.manifest.json")):
        want = r if kind == "receipt" else man
        if not want_src.is_file() or sha(want_src) != want:
            refuse(f"restore source {want_src} does not hash to the pre-incident {want[:12]}")
        if fakes.get(str(cur)) is None or sha(cur) != fakes[str(cur)]:
            refuse(f"{cur} is not exactly the recorded fake")
        plan.append((cur, want_src, want))
    cur = FARM / f"{P}{suf}.json"
    if fakes.get(str(cur)) is None or sha(cur) != fakes[str(cur)]:
        refuse(f"{cur} is not exactly the recorded fake")
    src = {"exp35_BF_T_S40000_K1_s42_fa_invariant_a4": R / f"{P}{suf}.json",
           "exp35_BF_T_S40000_K8_s42_fa_invariant_a4": SCR / f"{P}{suf}.json"}.get(suf)
    if src is not None and (not src.is_file() or sha(src) != m):
        refuse(f"restore source {src} does not hash to the pre-incident {m[:12]}")
    plan.append((cur, src, m))   # src None: the metrics JSON stays quarantined (missing) until recovered
leaf_files = sorted(LEAF.glob(f"{P}exp37_BF_ORB*")) if LEAF.is_dir() else []
nas_dirs = sorted(NAS.glob(f"{P}exp37_BF_ORB*"))
for p in leaf_files + [d / "manifest.json" for d in nas_dirs]:
    if fakes.get(str(p)) is None or sha(p) != fakes[str(p)]:
        refuse(f"{p} is not exactly the recorded fake")
for d in nas_dirs:
    if sorted(x.name for x in d.iterdir()) != ["manifest.json"]:
        refuse(f"{d} holds more than the fake manifest")
if len(leaf_files) != 16 or len(nas_dirs) != 8:
    refuse(f"expected 16 leaf files and 8 NAS dirs, found {len(leaf_files)} / {len(nas_dirs)}")
for sub in ("exp35_farm", "nas_manifests", "leaf_farm", "nas_exp37_dirs"):
    (Q / sub).mkdir(parents=True, exist_ok=False)
for cur, src, want in plan:
    dst = Q / ("nas_manifests" if cur.name == "manifest.json" else "exp35_farm") / (cur.parent.name + ".manifest.json" if cur.name == "manifest.json" else cur.name)
    shutil.move(str(cur), str(dst)); print("quarantined", cur, "->", dst)
    if src is not None:
        with open(src, "rb") as fi, open(cur, "xb") as fo:
            fo.write(fi.read()); fo.flush(); os.fsync(fo.fileno())
        got = sha(cur)
        if got != want:
            refuse(f"after the copy {cur} hashes to {got}, not {want}")
        print("restored", cur, want[:12])
    else:
        print("left MISSING (unrecoverable so far):", cur)
for p in leaf_files:
    shutil.move(str(p), str(Q / "leaf_farm" / p.name)); print("quarantined", p)
if LEAF.is_dir() and not any(LEAF.iterdir()):
    LEAF.rmdir(); print("removed the empty leaf dir", LEAF)
for d in nas_dirs:
    shutil.move(str(d), str(Q / "nas_exp37_dirs" / d.name)); print("quarantined", d)
print("DONE")
