"""Read-only reconstruction of the overwritten exp_35 artifacts, verified against the digests the leaf recorded at
19:26:27 EST (before the incident). Writes ONLY into ./restore/ in this scratchpad."""
import hashlib, json, sys
from pathlib import Path

FARM = Path("/home/yixunhu/codespace/exp-35-orbitrir-yaw/outputs_FLAC/exp25_depth_canon/BF")
DUMPS = Path("/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms")
OUT = Path("restore")
H35 = "5b065801e9d1c1844461ec84cd0558702c714326"
# from the leaf's loc_efficiency_2026-11-06_19-26-27_B3_reused_exp35_phase_i_cells.log (metrics, stream, receipt, manifest)
ORIG = {
 "exp35_BF_T_S40000_K1_s42_fa_invariant_a4": ("1e1768e3adf0d6513b705fa23f5a3ea7d9f357479755fb9d3fe9cc9d2a9efb46", "31aa02dcd5dcb2f6d80247f13e8a4e76ae28bc050f7f2c8f353e2b0808e0793d", "fa529324216c88832466a31d1b86fb4e4c8409aa037f45debdde1bc30befebef", "7dde0aa70ce01f0e108f12b6fce4eb72a0bf30ca79f2690853db21c0ab7fbf8f"),
 "exp35_BF_T_S40000_K8_s42_fa_invariant_a4": ("f167470e3501bcc7c4f116b0c26164523a8bd140b55c5e72e935d88d3f373adb", "8d8018cabfd803be559b4224ef7c8833fe0af7b96542cbce65cd0a3e1d1d4af6", "79817d0b100005ed91172e80ea09ea3c1d250aeb549e0f0ef29bdaea16f141ad", "dda03a0e318f65b216108c166e9cae10bdd44bd3473811c56dbfd10f844894d2"),
 "exp35_BF_C4_S40000_K1_s42_th90_fa_invariant_a4_rot90": ("9674cfff19a4541cd8df76ef4a3cd92096a953981516a596d8a303540f528282", "56660b9bf810102c16d17b39b526a0267549521661cec36a8a5deb5f83b4038f", "983f9d5f9842a1256ad697663725b174fd4ae1405d56e097b6c4e4cd90e6c6da", "4923d48e954b3b23b7788d42d93ae99f7288aa1c76001ed9ef84d27a821713ee"),
 "exp35_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90": ("ef2840c746b871268bddf8ae36609d1ee33186074a9afb01612c5db97606d82f", "e671270fc3cd61f5a47dd68f6f41707db68d6e62b4f2750ae52e9e7f9b7085a8", "431ceba60189490e85d53d942f43025c493a863cddec928ec7555d34ed5b16fd", "975cb72125977b6724cda82501bff4b356accb80278db7f8341c69566630fd5c"),
}
sha = lambda b: hashlib.sha256(b).hexdigest()

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ok = True
    for suffix, (m_sha, s_sha, r_sha, man_sha) in ORIG.items():
        stem = f"epoch=8-step=40000_metrics_1_1.0_{suffix}"
        stream_bytes = (FARM / f"{stem}.stream.json").read_bytes()
        stream_ok = sha(stream_bytes) == s_sha
        stream = json.loads(stream_bytes)
        wave = (DUMPS / stem / "COMPLETE").read_text().strip()
        manifest = {"schema_version": 1, "n": 6337, "dtype": "float32", "shape": [6337, 10240], "sha256": wave,
                    "input_hash": stream["input_hash"], "metrics_stem": stem, "source_sha": H35,
                    "ckpt_path": "outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt",
                    "metrics_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.json", "metrics_sha256": m_sha,
                    "stream_sidecar_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.stream.json",
                    "stream_sidecar_sha256": s_sha, "canon_sidecar_path": None, "canon_sidecar_sha256": None}
        raw = json.dumps(manifest, indent=2).encode()
        man_ok = sha(raw) == man_sha
        if man_ok:
            (OUT / f"{stem}.manifest.json").write_bytes(raw)
        print(f"{suffix}: stream intact {stream_ok}; manifest reconstructed byte-exact {man_ok}")
        ok &= stream_ok and man_ok
    # the K1 T metrics JSON from its T-dup twin (bitwise-deterministic re-run)
    dup = (FARM / "epoch=8-step=40000_metrics_1_1.0_exp35_BF_T_S40000_K1_s42_dup_fa_invariant_a4.json").read_bytes()
    cand = dup.replace(b'"exp35_BF_T_S40000_K1_s42_dup"', b'"exp35_BF_T_S40000_K1_s42"')
    t_ok = sha(cand) == ORIG["exp35_BF_T_S40000_K1_s42_fa_invariant_a4"][0]
    if t_ok:
        (OUT / "epoch=8-step=40000_metrics_1_1.0_exp35_BF_T_S40000_K1_s42_fa_invariant_a4.json").write_bytes(cand)
    print("T K1 metrics JSON reconstructed byte-exact from the T-dup:", t_ok)
    sys.exit(0 if ok and t_ok else 1)
