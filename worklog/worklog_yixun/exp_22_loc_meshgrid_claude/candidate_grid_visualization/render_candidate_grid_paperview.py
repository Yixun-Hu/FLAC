"""Candidate-grid figure for Cafe_idx_1 query 0, re-rendered from the paper camera
(az 78 / elev 55 / dist 2.8 / fov 14 / lookat 13.15 11.79 1.35 / zcut 2.40 — identical
to render_pipeline_panel.py). Points are the FROZEN registered z-band bank
(candidates_Cafe_Cafe_idx_1.npz, q0 indices, 5,295 candidates) — not a preview lattice.
Occlusion-aware: candidates behind mesh geometry are drawn faint."""
import open3d as o3d, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- room render (verbatim camera/shading from render_pipeline_panel.py) ---
m = o3d.io.read_triangle_mesh('/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/Cafe/Cafe_idx_1.obj')
V = np.asarray(m.vertices); T = np.asarray(m.triangles)
Tk = T[V[:,2][T].min(axis=1) < 2.40]
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(Tk))
lo = np.asarray(mesh.get_min_bound()); hi = np.asarray(mesh.get_max_bound())
diag = float(np.linalg.norm(hi-lo))
lookat = np.array([13.15, 11.79, 1.35])
az, el = np.deg2rad(78), np.deg2rad(55)
eye = lookat + 2.8*diag*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
fwd = lookat-eye; fwd/=np.linalg.norm(fwd)
right = np.cross(fwd,[0,0,1]); right/=np.linalg.norm(right); up = np.cross(right,fwd)
W,H = 2000,1350; fov = np.deg2rad(14); asp = W/H
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
xs = np.linspace(-1,1,W)*np.tan(fov/2)*asp; ys = np.linspace(1,-1,H)*np.tan(fov/2)
gx, gy = np.meshgrid(xs, ys)
dirs = fwd[None,None]+gx[...,None]*right[None,None]+gy[...,None]*up[None,None]; dirs/=np.linalg.norm(dirs,axis=-1,keepdims=True)
ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye,(H,W,3)),dirs],-1).astype(np.float32)))
t = ans['t_hit'].numpy(); nrm = ans['primitive_normals'].numpy(); hit = np.isfinite(t)
hp = eye[None,None]+dirs*t[...,None]
L = np.array([0.15,-0.15,1.0]); L/=np.linalg.norm(L)
lam = np.abs(np.einsum('ijk,k->ij', nrm, L))
img = np.zeros((H,W,4))
flat = np.abs(nrm[...,2])>0.8; low = hp[...,2]<0.12
tcn = np.cross(V[Tk][:,1]-V[Tk][:,0], V[Tk][:,2]-V[Tk][:,0]); tcn/=(np.linalg.norm(tcn,axis=1,keepdims=True)+1e-9)
steepf = np.abs(tcn[:,2])<0.35
prim = ans['primitive_ids'].numpy()
fsteep = np.zeros(len(Tk)+1,bool); fsteep[:-1]=steepf
is_wall = np.where(hit, fsteep[np.where(hit,prim,len(Tk))], False)
wall_c=np.array([0.82,0.84,0.87]); floor_c=np.array([0.93,0.92,0.89]); furn_c=np.array([0.73,0.70,0.64])
base = np.where(is_wall[...,None], wall_c, np.where((flat&low)[...,None], floor_c, furn_c))
img[...,:3]=np.clip(base*(0.60+0.40*lam)[...,None],0,1); img[...,3]=hit.astype(float)
d0=np.where(hit,t,np.nan)
edge=((np.abs(np.diff(d0,axis=1,prepend=d0[:,:1]))>0.20)|(np.abs(np.diff(d0,axis=0,prepend=d0[:1]))>0.20))&hit
img[edge,:3]*=0.45

# --- data: frozen registered bank + query 0 identity ---
base_c = np.load('outputs_loc/exp22/g1_audit/candidates_Cafe_Cafe_idx_1.npz')['base_candidates']
j = json.load(open('outputs_loc/exp22/g1_audit/candidates_Cafe_Cafe_idx_1.json'))
q0 = (j.get('queries') or j.get('records'))[0]
pts = base_c[np.array(q0['candidate_indices_z_band'])]
receiver = np.array(q0['receiver'])
gt = np.array(json.load(open('/media/diskstation/yixunhu/FLAC/AcousticRooms/metadata/Cafe/Cafe_idx_1/S006_R008.json'))['src_loc'])
d1 = json.load(open('outputs_loc/exp22/d1_context_manifest.json'))
r0 = (d1['records'] if 'records' in d1 else d1)[0]
ctx = np.array([[float(v) for v in fp.split(',')] for fp in r0['context_fingerprints']]) + receiver
near_i = np.argmin(np.linalg.norm(pts-gt, axis=1)); near = pts[near_i]
oracle = float(np.linalg.norm(near-gt))

def project(P):
    rel = P - eye
    px, py, pz = rel@right, rel@up, rel@fwd
    return (W/2*(1+px/(pz*np.tan(fov/2)*asp)), H/2*(1-py/(pz*np.tan(fov/2))), pz)

def occluded(P, eps=0.06):
    rel = P - eye; dist = np.linalg.norm(rel, axis=1); dn = rel/dist[:,None]
    rays = np.concatenate([np.broadcast_to(eye,(len(P),3)), dn], 1).astype(np.float32)
    th = sc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    return np.isfinite(th) & (th < dist - eps)

# --- compose ---
PAD_R = 460  # legend margin
Wt = W + PAD_R
fig = plt.figure(figsize=(Wt/200, (H+230)/200), dpi=200)
ax = fig.add_axes([0, 0, W/Wt, H/(H+230)]); ax.axis('off')
ax.imshow(img, interpolation='bilinear', zorder=1); ax.set_xlim(0,W); ax.set_ylim(H,0)

zlevels = sorted(set(np.round(pts[:,2],3)))
blues = ['#9dc3f0','#3f8ef2','#1a4f9c','#2f74d0']
occ = occluded(pts)
order = np.argsort(-project(pts)[2])  # far first
counts = {}
for zi, z in enumerate(zlevels):
    counts[z] = int((np.round(pts[:,2],3)==z).sum())
for i in order:
    x, y, _ = project(pts[i:i+1]); zi = zlevels.index(round(float(pts[i,2]),3))
    a = 0.14 if occ[i] else 0.85
    ax.scatter(x, y, s=13, c=blues[zi], alpha=a, linewidths=0, zorder=3)
for P, kw in [(ctx, dict(marker='D', s=150, c='#e0402a', edgecolors='white', linewidths=1.2, zorder=6)),
              (receiver[None], dict(marker='^', s=260, c='#1a2433', edgecolors='white', linewidths=1.4, zorder=7)),
              (gt[None], dict(marker='*', s=560, c='#f5a300', edgecolors='#7a5200', linewidths=1.2, zorder=8)),
              (near[None], dict(marker='P', s=260, c='#18a05c', edgecolors='white', linewidths=1.2, zorder=7))]:
    x, y, _ = project(np.atleast_2d(P)); ax.scatter(x, y, **kw)
nx, ny, _ = project(near[None])
ax.annotate(f"{oracle:.3f} m", (float(nx[0])+18, float(ny[0])-16), color='#e0700a', fontsize=15, fontweight='bold', zorder=9)

fig.text(0.012, 0.965, 'Actual Candidate Grid for One Localization Query', fontsize=26, fontweight='bold', color='#1a2433', va='top')
fig.text(0.012, 0.912, f"Cafe/Cafe_idx_1  ·  query 0  ·  S006_R008_hybrid_IR.wav  ·  0.5 m spacing  ·  {len(pts)} candidates (registered z-band bank)", fontsize=14, color='#5a6472', va='top')
handles = [Line2D([],[], marker='s', ls='', mfc='#d9dce1', mec='none', ms=13, label='Room mesh (cutaway, z < 2.4 m)')]
handles += [Line2D([],[], marker='o', ls='', mfc=blues[i], mec='none', ms=9, label=f"Candidate grid, z={z:.1f} m ({counts[z]})") for i, z in enumerate(zlevels)]
handles += [Line2D([],[], marker='^', ls='', mfc='#1a2433', mec='white', ms=13, label='Receiver'),
            Line2D([],[], marker='D', ls='', mfc='#e0402a', mec='white', ms=10, label='Context sources (8)'),
            Line2D([],[], marker='*', ls='', mfc='#f5a300', mec='#7a5200', ms=17, label='Target source (ground truth)'),
            Line2D([],[], marker='P', ls='', mfc='#18a05c', mec='white', ms=12, label='Nearest grid point')]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 0.90), fontsize=12.5, frameon=True, framealpha=0.95, edgecolor='#c8ccd2')
fig.text(0.995, 0.03, 'Occluded candidates drawn faint  ·  Ground truth is not inserted into the grid', ha='right', fontsize=11.5, color='#8a919b')
out = 'worklog/worklog_yixun/exp_22_loc_meshgrid_claude/candidate_grid_visualization/candidate_grid_case_cafe_idx_1_q0.png'
fig.savefig(out, dpi=200, facecolor='white'); plt.close(fig)
print(f'wrote {out}: {len(pts)} candidates at z levels {zlevels}, oracle {oracle:.3f} m')
