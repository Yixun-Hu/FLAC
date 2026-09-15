"""Panorama convention audit: are the known point positions INSIDE the room that the depth
panorama describes, under the md's column->azimuth convention (theta_u = 2pi(u+.5)/W - pi,
row0 = up after the md's preprocessing)?  Scan mirror x offset hypotheses; the physically
correct convention should maximise the inside fraction."""
import json, glob, os, sys, numpy as np
H, W = 256, 512
def cols_for(theta, mirror, k):
    t = -theta if mirror else theta
    u = (t + np.pi) * W / (2*np.pi) - 0.5
    return (np.round(u).astype(int) + k) % W
def rows_for(elev):
    return np.clip(np.round((-elev + np.pi/2) * H/np.pi - 0.5).astype(int), 0, H-1)
def audit(depth, pts, label, ks=range(0, W, 8), tol=0.1):
    pts = np.asarray(pts, float); n = np.linalg.norm(pts, axis=1)
    th = np.arctan2(pts[:,1], pts[:,0]); el = np.arcsin(np.clip(pts[:,2]/np.maximum(n,1e-9), -1, 1))
    r3 = rows_for(el); req = H//2  # equator rows
    rho = np.linalg.norm(pts[:,:2], axis=1)
    best = None; table = {}
    for m in (0,1):
        for k in ks:
            u = cols_for(th, m, k)
            in3 = np.mean(n <= depth[r3, u]*(1+tol) + 0.15)
            ineq = np.mean(rho <= depth[req, u]*(1+tol) + 0.15)   # horizontal ray, furniture-robust-ish
            table[(m,k)] = (in3, ineq)
            if best is None or ineq + in3 > best[0]: best = (ineq+in3, m, k, in3, ineq)
    a = table[(0,0)]; mir = table[(1,0)]; rol = table[(0, W//2)]; mr = table[(1, W//2)]
    print(f"{label:34s} n={len(pts):4d} | formula(m0,k0): 3D {a[0]:.2f} eq {a[1]:.2f} | mirror(m1,k0): {mir[0]:.2f}/{mir[1]:.2f} | roll-pi: {rol[0]:.2f}/{rol[1]:.2f} | mirror+roll: {mr[0]:.2f}/{mr[1]:.2f} | BEST m={best[1]} k={best[2]:3d} ({best[3]:.2f}/{best[4]:.2f})")
    return table

if sys.argv[1] == 'haa':
    root = '/media/diskstation/yixunhu/FLAC/HAA'
    sc = json.load(open(root+'/metadata/scenes_metadata.json')); po = json.load(open(root+'/metadata/poses_metadata.json'))
    for s in sc:
        d = np.load(f'{root}/{s}/depth_images/{s}_depth_image.npy').astype(float)
        d = np.flipud(d).copy()            # exactly what HAA_md does
        spk = np.array(sc[s]['speaker_xyz'])
        pts = [np.array(p) - spk for p in po[s].values()]
        audit(d, pts, f'HAA {s}')
        # equator depth at the four cardinal columns (formula convention): u=0:-x, 128:-y, 256:+x, 384:+y
        eq = d[H//2]; print(f"     equator depth: -x(u0) {eq[0]:.2f}  -y(u128) {eq[128]:.2f}  +x(u256) {eq[256]:.2f}  +y(u384) {eq[384]:.2f}   | receivers rel. speaker: x[{min(p[0] for p in pts):.2f},{max(p[0] for p in pts):.2f}] y[{min(p[1] for p in pts):.2f},{max(p[1] for p in pts):.2f}]")
else:
    root = '/home/yixunhu/data_cache/AcousticRooms'
    scenes = sorted(os.listdir(root+'/depth_map'))
    rng = np.random.default_rng(0)
    picks = []
    for s in scenes[:12]:
        sids = sorted(os.listdir(f'{root}/depth_map/{s}'))
        for sid in sids[:2]:
            recs = sorted(glob.glob(f'{root}/depth_map/{s}/{sid}/*.npy'))
            if recs: picks.append((s, sid, recs[0]))
    for s, sid, dp in picks:
        r = int(os.path.basename(dp).split('.')[0])
        d = np.load(dp).astype(float)      # AR_md: no flip
        metas = glob.glob(f'{root}/metadata/{s}/{sid}/S*_R00{r}.json')
        pts = []; rec = None
        for mf in metas:
            m = json.load(open(mf)); rec = np.array(m['rec_loc']); pts.append(np.array(m['src_loc']) - rec)
        if len(pts) < 5: continue
        audit(d, pts, f'AR {s}/{sid}/R{r}')
