"""exp_23: widen the exp_19 CYL init (EMA of BB-CYL AR-40k) to 6 input channels.

Both shared-backbone patch convs (`source_vit`, `context_poses_vit` point at the SAME module at
runtime, but the state dict carries both keys) get channels [3:6] = 0. Every other tensor is
copied byte-for-byte. The output is what `--pretrained-ckpt-path` loads strict=True into the
widened CYLORI model, so step 0 of the finetune is EXACTLY the CYL arm's step 0.
"""
import hashlib, sys, torch
src, dst = sys.argv[1], sys.argv[2]
blob = torch.load(src, map_location="cpu")
sd = blob["state_dict"]
keys = [k for k in sd if k.endswith("vit.embeddings.patch_embeddings.weight")]
assert len(keys) == 2, keys
for k in keys:
    w = sd[k]; assert tuple(w.shape[1:]) == (3, 16, 16), w.shape
    nw = torch.zeros(w.shape[0], 6, 16, 16, dtype=w.dtype); nw[:, :3] = w; sd[k] = nw
torch.save(blob, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print(f"{h}  {dst}")
for k in keys: print(k, tuple(sd[k].shape), "nonzero extra:", int(torch.count_nonzero(sd[k][:, 3:])))
