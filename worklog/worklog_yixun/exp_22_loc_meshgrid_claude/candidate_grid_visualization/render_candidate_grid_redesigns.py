"""Four redesigns of the Cafe_idx_1 q0 candidate-grid figure.
Source of truth: frozen registered bank (candidates_Cafe_Cafe_idx_1.npz, q0 z-band
indices), the official room OBJ, and the pair metadata / D1 record for q0.
No candidate is invented, moved, or interpolated; only v4 exaggerates the vertical
separation between layers, and says so on the figure."""
import open3d as o3d, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
plt.rcParams['font.family']='serif'; plt.rcParams['font.serif']=['P052','Palatino','TeX Gyre Pagella','DejaVu Serif']

OUT = 'worklog/worklog_yixun/exp_22_loc_meshgrid_claude/candidate_grid_visualization/'
NAVY='#1a2433'; RED='#e0402a'; ORANGE='#f5a300'; GREEN='#18a05c'
GT_NOTE = 'Ground truth is not inserted into the candidate grid'

# ---------- data ----------
base_c = np.load('outputs_loc/exp22/g1_audit/candidates_Cafe_Cafe_idx_1.npz')['base_candidates']
j = json.load(open('outputs_loc/exp22/g1_audit/candidates_Cafe_Cafe_idx_1.json'))
q0 = (j.get('queries') or j.get('records'))[0]
pts = base_c[np.array(q0['candidate_indices_z_band'])]
receiver = np.array(q0['receiver'])
gt = np.array(json.load(open('/media/diskstation/yixunhu/FLAC/AcousticRooms/metadata/Cafe/Cafe_idx_1/S006_R008.json'))['src_loc'])
d1 = json.load(open('outputs_loc/exp22/d1_context_manifest.json'))
r0 = (d1['records'] if 'records' in d1 else d1)[0]
ctx = np.array([[float(v) for v in fp.split(',')] for fp in r0['context_fingerprints']]) + receiver
near = pts[np.argmin(np.linalg.norm(pts-gt, axis=1))]
oracle = float(np.linalg.norm(near-gt)); assert abs(oracle-0.258) < 5e-4
zlevels = sorted(set(np.round(pts[:,2],3)))
layer = {z: pts[np.round(pts[:,2],3)==z] for z in zlevels}
EXPECT = {1.0:1239, 1.5:1346, 2.0:1348, 2.5:1362}
for z in zlevels: assert len(layer[z])==EXPECT[z], (z, len(layer[z]))
assert len(pts)==5295
near_z = round(float(near[2]),3)

# ---------- room mesh + scenes ----------
m = o3d.io.read_triangle_mesh('/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/Cafe/Cafe_idx_1.obj')
V = np.asarray(m.vertices); T = np.asarray(m.triangles)
Tk = T[V[:,2][T].min(axis=1) < 2.40]
mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(Tk))
lo = np.asarray(mesh.get_min_bound()); hi = np.asarray(mesh.get_max_bound())
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# ---------- top-down orthographic room image (shared by v1/v3) ----------
RES = 0.02
xg = np.arange(lo[0]-0.3, hi[0]+0.3, RES); yg = np.arange(lo[1]-0.3, hi[1]+0.3, RES)
GX, GY = np.meshgrid(xg, yg)
orig = np.stack([GX, GY, np.full_like(GX, hi[2]+1.0)], -1)
drc = np.zeros_like(orig); drc[...,2] = -1.0
ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([orig, drc], -1).astype(np.float32).reshape(-1,6)))
th = ans['t_hit'].numpy().reshape(GX.shape)
hitm = np.isfinite(th)
hz = np.where(hitm, hi[2]+1.0-th, np.nan)   # surface height seen from above
room_img = np.ones(GX.shape+(3,))
wall = hitm & (hz > 2.0)
furn = hitm & (hz > 0.15) & (hz <= 2.0)
floor = hitm & (hz <= 0.15)
room_img[floor] = [0.965,0.960,0.945]
room_img[furn]  = [0.845,0.830,0.795]
room_img[wall]  = [0.68,0.70,0.73]
ed = np.zeros_like(hitm)
d_ = np.where(hitm, hz, 0)
ed[(np.abs(np.diff(d_,axis=0,prepend=d_[:1]))>0.12)|(np.abs(np.diff(d_,axis=1,prepend=d_[:,:1]))>0.12)] = True
room_img[ed & hitm] *= 0.62
EXT = [xg[0], xg[-1], yg[0], yg[-1]]
# surface height lookup for top-down occlusion of a 3-D point
def covered(P):
    ix = np.clip(((P[:,0]-xg[0])/RES).astype(int), 0, len(xg)-1)
    iy = np.clip(((P[:,1]-yg[0])/RES).astype(int), 0, len(yg)-1)
    s = hz[iy, ix]
    return np.isfinite(s) & (s > P[:,2] + 0.05)

def major_grid(ax, step=2.0):
    for x in np.arange(np.ceil(xg[0]/step)*step, xg[-1], step): ax.axvline(x, color='#b9bec6', lw=0.5, alpha=0.35, zorder=2)
    for y in np.arange(np.ceil(yg[0]/step)*step, yg[-1], step): ax.axhline(y, color='#b9bec6', lw=0.5, alpha=0.35, zorder=2)

def draw_markers_xy(ax, sizes=1.0, proj_note=False):
    ax.scatter(*ctx[:,:2].T, marker='D', s=110*sizes, c=RED, edgecolors='white', linewidths=1.0, zorder=8)
    ax.scatter(*receiver[:2], marker='^', s=190*sizes, c=NAVY, edgecolors='white', linewidths=1.1, zorder=9)
    ax.scatter(*gt[:2], marker='*', s=420*sizes, c=ORANGE, edgecolors='#7a5200', linewidths=1.0, zorder=10)
    ax.scatter(*near[:2], marker='P', s=190*sizes, c=GREEN, edgecolors='white', linewidths=1.0, zorder=9)

def target_inset(ax_in, half=1.25, show_layer=None, title=None):
    """True-geometry inset: 0.5 m cells centred near GT, dashed GT->nearest + 0.258 m."""
    x0, y0 = gt[0]-half, gt[1]-half
    ax_in.imshow(room_img, extent=EXT, origin='lower', zorder=1)
    for gxl in np.arange(np.floor(x0/0.5)*0.5, gt[0]+half+0.5, 0.5): ax_in.axvline(gxl, color='#9aa2ad', lw=0.6, alpha=0.8, zorder=2)
    for gyl in np.arange(np.floor(y0/0.5)*0.5, gt[1]+half+0.5, 0.5): ax_in.axhline(gyl, color='#9aa2ad', lw=0.6, alpha=0.8, zorder=2)
    if show_layer is not None:
        L = layer[show_layer]; mloc = (np.abs(L[:,0]-gt[0])<half+0.3)&(np.abs(L[:,1]-gt[1])<half+0.3)
        ax_in.scatter(*L[mloc][:,:2].T, s=34, c='#3f8ef2', alpha=0.75, linewidths=0, zorder=4)
    ax_in.plot([gt[0], near[0]],[gt[1], near[1]], ls='--', lw=2.2, c='#e0700a', zorder=9)
    ax_in.scatter(*gt[:2], marker='*', s=480, c=ORANGE, edgecolors='#7a5200', linewidths=1.1, zorder=8)
    ax_in.scatter(*near[:2], marker='P', s=230, c=GREEN, edgecolors='white', linewidths=1.1, zorder=8)
    mx, my = (gt[0]+near[0])/2, (gt[1]+near[1])/2
    ax_in.annotate('0.258 m', xy=(mx, my), xytext=(mx+0.52, my-0.52), color='#e0700a',
                   fontsize=12, fontweight='bold', ha='left', va='top', zorder=10,
                   arrowprops=dict(arrowstyle='-', color='#e0700a', lw=0.9, shrinkA=0, shrinkB=4))
    ax_in.set_xlim(x0, gt[0]+half); ax_in.set_ylim(y0, gt[1]+half); ax_in.set_aspect('equal')
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for s_ in ax_in.spines.values(): s_.set_color('#6a7280')
    if title: ax_in.set_title(title, fontsize=11, color='#3a424d')

BLUE = '#3f8ef2'
# ============================ V1: layer slices ============================
fig, axs = plt.subplots(2, 2, figsize=(12.6, 10.6), dpi=200)
for axp, z in zip(axs.flat, zlevels):
    axp.imshow(room_img, extent=EXT, origin='lower', zorder=1)
    major_grid(axp)
    L = layer[z]; cov = covered(L)
    axp.scatter(*L[~cov][:,:2].T, s=6, c=BLUE, alpha=0.33, linewidths=0, zorder=4)
    axp.scatter(*L[cov][:,:2].T,  s=6, c=BLUE, alpha=0.06, linewidths=0, zorder=3)
    draw_markers_xy(axp, sizes=0.75)
    axp.set_title(f"z = {z:.1f} m   ·   {len(L):,} candidates", fontsize=14)
    axp.set_xlim(EXT[0],EXT[1]); axp.set_ylim(EXT[2],EXT[3]); axp.set_aspect('equal'); axp.set_xticks([]); axp.set_yticks([])
ins = fig.add_axes([0.858, 0.395, 0.13, 0.21]); target_inset(ins, show_layer=near_z, title='target neighborhood\n(0.5 m cells)')
handles = [Line2D([],[],marker='o',ls='',mfc=BLUE,mec='none',ms=7,alpha=0.7,label='Candidate (this layer)'),
           Line2D([],[],marker='o',ls='',mfc=BLUE,mec='none',ms=7,alpha=0.15,label='Candidate under furniture'),
           Line2D([],[],marker='^',ls='',mfc=NAVY,mec='white',ms=11,label='Receiver (XY projection)'),
           Line2D([],[],marker='D',ls='',mfc=RED,mec='white',ms=9,label='Context sources, 8 (XY projection)'),
           Line2D([],[],marker='*',ls='',mfc=ORANGE,mec='#7a5200',ms=15,label='Target source, ground truth (XY projection)'),
           Line2D([],[],marker='P',ls='',mfc=GREEN,mec='white',ms=11,label='Nearest grid point')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11.5, frameon=False, bbox_to_anchor=(0.44, -0.005))
fig.subplots_adjust(left=0.015, right=0.845, top=0.975, bottom=0.075, wspace=0.04, hspace=0.10)
fig.savefig(OUT+'candidate_grid_v1_layer_slices.png', facecolor='white'); plt.close(fig)
print('v1 written:', {z: len(layer[z]) for z in zlevels})

# ============================ V2: focused-layer 3D ============================
lookat = np.array([13.15, 11.79, 1.35]); az, el = np.deg2rad(78), np.deg2rad(55)
diag = float(np.linalg.norm(hi-lo))
eye = lookat + 2.8*diag*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
fwd = lookat-eye; fwd/=np.linalg.norm(fwd)
right = np.cross(fwd,[0,0,1]); right/=np.linalg.norm(right); up = np.cross(right,fwd)
W,H = 2000,1350; fov = np.deg2rad(14); asp = W/H
xs = np.linspace(-1,1,W)*np.tan(fov/2)*asp; ys = np.linspace(1,-1,H)*np.tan(fov/2)
gx2, gy2 = np.meshgrid(xs, ys)
dirs = fwd[None,None]+gx2[...,None]*right[None,None]+gy2[...,None]*up[None,None]; dirs/=np.linalg.norm(dirs,axis=-1,keepdims=True)
ans2 = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye,(H,W,3)),dirs],-1).astype(np.float32)))
t2 = ans2['t_hit'].numpy(); nrm2 = ans2['primitive_normals'].numpy(); hit2 = np.isfinite(t2)
hp2 = eye[None,None]+dirs*t2[...,None]
Lz = np.array([0.15,-0.15,1.0]); Lz/=np.linalg.norm(Lz)
lam2 = np.abs(np.einsum('ijk,k->ij', nrm2, Lz))
img2 = np.ones((H,W,3))
flat2 = np.abs(nrm2[...,2])>0.8; low2 = hp2[...,2]<0.12
tcn = np.cross(V[Tk][:,1]-V[Tk][:,0], V[Tk][:,2]-V[Tk][:,0]); tcn/=(np.linalg.norm(tcn,axis=1,keepdims=True)+1e-9)
fsteep = np.zeros(len(Tk)+1,bool); fsteep[:-1]=np.abs(tcn[:,2])<0.35
prim2 = ans2['primitive_ids'].numpy()
is_wall2 = np.where(hit2, fsteep[np.where(hit2,prim2,len(Tk))], False)
b2 = np.where(is_wall2[...,None], [[0.82,0.84,0.87]], np.where((flat2&low2)[...,None], [[0.94,0.93,0.90]], [[0.76,0.73,0.67]]))
shade2 = np.clip(b2*(0.62+0.38*lam2)[...,None],0,1)
img2 = np.where(hit2[...,None], shade2, 1.0)
d0=np.where(hit2,t2,np.nan)
edge2=((np.abs(np.diff(d0,axis=1,prepend=d0[:,:1]))>0.20)|(np.abs(np.diff(d0,axis=0,prepend=d0[:1]))>0.20))&hit2
img2[edge2] *= 0.5
def project(P):
    rel = np.atleast_2d(P) - eye
    px, py, pz = rel@right, rel@up, rel@fwd
    return W/2*(1+px/(pz*np.tan(fov/2)*asp)), H/2*(1-py/(pz*np.tan(fov/2))), pz
def occl3d(P, eps=0.06):
    rel = np.atleast_2d(P) - eye; dist = np.linalg.norm(rel,axis=1); dn = rel/dist[:,None]
    tt = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(eye,(len(dn),3)),dn],1).astype(np.float32)))['t_hit'].numpy()
    return np.isfinite(tt) & (tt < dist-eps)
fig = plt.figure(figsize=(12.3, 8.3), dpi=200)
ax = fig.add_axes([0.0, 0.0, 0.80, 1.0]); ax.axis('off')
ax.imshow(img2, zorder=1); ax.set_xlim(0,W); ax.set_ylim(H,0)
SEL='#5b3fd4'
for z in zlevels:
    L = layer[z]; xq, yq, _ = project(L); oc = occl3d(L)
    if z == near_z:
        ax.scatter(xq[~oc], yq[~oc], s=6.5, c=SEL, alpha=0.55, linewidths=0, zorder=5)
        ax.scatter(xq[oc], yq[oc], s=6.5, c=SEL, alpha=0.08, linewidths=0, zorder=4)
    else:
        ax.scatter(xq[~oc], yq[~oc], s=5, c='#8a919b', alpha=0.10, linewidths=0, zorder=3)
        ax.scatter(xq[oc], yq[oc], s=5, c='#8a919b', alpha=0.04, linewidths=0, zorder=3)
gtx, gty, _ = project(gt); nxx, nyy, _ = project(near)
ax.plot([gtx[0],nxx[0]],[gty[0],nyy[0]], ls='--', lw=2.2, c='#e0700a', zorder=9)
ax.annotate('0.258 m', ((gtx[0]+nxx[0])/2+30,(gty[0]+nyy[0])/2-14), color='#e0700a', fontsize=13, fontweight='bold', zorder=10)
cxx, cyy, _ = project(ctx); ax.scatter(cxx, cyy, marker='D', s=120, c=RED, edgecolors='white', linewidths=1.1, zorder=8)
rxx, ryy, _ = project(receiver); ax.scatter(rxx, ryy, marker='^', s=210, c=NAVY, edgecolors='white', linewidths=1.2, zorder=8)
ax.scatter(gtx, gty, marker='*', s=470, c=ORANGE, edgecolors='#7a5200', linewidths=1.1, zorder=10)
ax.scatter(nxx, nyy, marker='P', s=210, c=GREEN, edgecolors='white', linewidths=1.1, zorder=9)
handles = [Line2D([],[],marker='o',ls='',mfc=SEL,mec='none',ms=7,alpha=0.8,label=f'Selected layer z = {near_z:.1f} m ({len(layer[near_z]):,})'),
           Line2D([],[],marker='o',ls='',mfc='#8a919b',mec='none',ms=6,alpha=0.35,label='Other layers (1.0 / 2.0 / 2.5 m), de-emphasized'),
           Line2D([],[],marker='^',ls='',mfc=NAVY,mec='white',ms=11,label='Receiver'),
           Line2D([],[],marker='D',ls='',mfc=RED,mec='white',ms=9,label='Context sources (8)'),
           Line2D([],[],marker='*',ls='',mfc=ORANGE,mec='#7a5200',ms=15,label='Target source (ground truth)'),
           Line2D([],[],marker='P',ls='',mfc=GREEN,mec='white',ms=11,label='Nearest grid point'),
           Line2D([],[],ls='--',c='#e0700a',lw=2,label='GT to nearest grid point (0.258 m)')]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 0.965), fontsize=11.5, frameon=True, framealpha=0.95, edgecolor='#c8ccd2')
ins2 = fig.add_axes([0.815, 0.09, 0.165, 0.30]); target_inset(ins2, show_layer=near_z, title='target neighborhood\n(0.5 m cells)')
fig.text(0.985, 0.015, GT_NOTE, ha='right', fontsize=10, color='#8a919b')
fig.savefig(OUT+'candidate_grid_v2_focused_layer_3d.png', facecolor='white'); plt.close(fig)
print('v2 written: selected layer', near_z, len(layer[near_z]))

# ============================ V3: XY coverage map ============================
keys = np.round(pts[:,:2], 3)
uniq, inv, cnts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
cov_colors = ['#c6dbef','#6baed6','#2171b5','#08306b']
fig = plt.figure(figsize=(12.8, 9.2), dpi=200)
ax = fig.add_axes([0.01, 0.03, 0.72, 0.94]); ax.axis('off')
ax.imshow(room_img, extent=EXT, origin='lower', zorder=1); major_grid(ax)
for n in (1,2,3,4):
    sel = uniq[cnts==n]
    if len(sel): ax.scatter(sel[:,0], sel[:,1], s=13, c=cov_colors[n-1], alpha=0.75, linewidths=0, zorder=4)
draw_markers_xy(ax, sizes=0.85)
ax.set_xlim(EXT[0],EXT[1]); ax.set_ylim(EXT[2],EXT[3]); ax.set_aspect('equal')
handles = [Line2D([],[],marker='o',ls='',mfc=cov_colors[n-1],mec='none',ms=8,label=f'{n} level{"s" if n>1 else ""}') for n in (1,2,3,4)]
leg1 = fig.legend(handles=handles, title='Available height levels', loc='upper right', bbox_to_anchor=(0.985, 0.94), fontsize=11.5, title_fontsize=12, frameon=True, edgecolor='#c8ccd2')
handles2 = [Line2D([],[],marker='^',ls='',mfc=NAVY,mec='white',ms=11,label='Receiver'),
            Line2D([],[],marker='D',ls='',mfc=RED,mec='white',ms=9,label='Context sources (8)'),
            Line2D([],[],marker='*',ls='',mfc=ORANGE,mec='#7a5200',ms=15,label='Target (ground truth)'),
            Line2D([],[],marker='P',ls='',mfc=GREEN,mec='white',ms=11,label='Nearest grid point')]
fig.legend(handles=handles2, loc='upper right', bbox_to_anchor=(0.985, 0.70), fontsize=11.5, frameon=True, edgecolor='#c8ccd2')
axs3 = fig.add_axes([0.755, 0.30, 0.225, 0.16]); axs3.axis('off')
axs3.set_title('Height layers (0.5 m apart)', fontsize=12, color='#3a424d')
for i, z in enumerate(zlevels):
    yy = 0.15 + i*0.24
    axs3.plot([0.02, 0.30], [yy, yy], c=cov_colors[min(i,3)], lw=3)
    axs3.text(0.34, yy, f'z = {z:.1f} m — {len(layer[z]):,} candidates', va='center', fontsize=10.5)
axs3.set_xlim(0,1); axs3.set_ylim(0,1.05)
ins3 = fig.add_axes([0.765, 0.055, 0.20, 0.22]); target_inset(ins3, show_layer=near_z, title='target neighborhood (0.5 m cells)')
fig.text(0.985, 0.012, GT_NOTE, ha='right', fontsize=10, color='#8a919b')
fig.savefig(OUT+'candidate_grid_v3_xy_layer_count.png', facecolor='white'); plt.close(fig)
print('v3 written: XY positions', len(uniq), 'sum check', int(cnts.sum()))

# ============================ V4: exploded schematic ============================
V4C = ['#9ecae1','#6baed6','#807dba','#54278f']
SHY, SHX, GAP = 0.36, 0.50, 9.0   # oblique shear + exaggerated layer gap (display units)
def obl(P, zdisp):
    return P[:,0] + SHX*(P[:,1]-lo[1]), zdisp + SHY*(P[:,1]-lo[1])
from contourpy import contour_generator
cg = contour_generator(x=GX, y=GY, z=hitm.astype(float))
outer = max(cg.lines(0.5), key=len)
fig = plt.figure(figsize=(12.2, 10.6), dpi=200)
ax = fig.add_axes([0.02, 0.02, 0.80, 0.96]); ax.axis('off')
for i, z in enumerate(zlevels):
    zd = i*GAP
    ox, oy = outer[:,0] + SHX*(outer[:,1]-lo[1]), zd + SHY*(outer[:,1]-lo[1])
    ax.plot(ox, oy, c='#b9bec6', lw=1.0, alpha=0.8, zorder=2+2*i)
    L = layer[z]; xE, yE = obl(L, zd)
    ax.scatter(xE, yE, s=5.5, c=V4C[i], alpha=0.55, linewidths=0, zorder=3+2*i)
    lx = lo[0] + SHX*(hi[1]-lo[1]); ly = zd + SHY*(hi[1]-lo[1])
    ax.text(lx-1.3, ly+0.7, f'z = {z:.1f} m — {len(L):,} candidates', fontsize=13, color=V4C[i] if i>0 else '#5a7fa5', fontweight='bold')
gx4, gy4 = obl(gt[None], (zlevels.index(1.5))*GAP); nx4, ny4 = obl(near[None], (zlevels.index(near_z))*GAP)
ax.scatter(gx4, gy4, marker='*', s=300, c=ORANGE, edgecolors='#7a5200', linewidths=1.0, zorder=20)
bx = hi[0] + SHX*(hi[1]-lo[1]) + 1.4
ax.add_patch(FancyArrowPatch((bx, 0), (bx, GAP), arrowstyle='<->', mutation_scale=14, color='#3a424d', lw=1.4))
ax.text(bx+0.5, GAP/2, 'real spacing\n0.5 m', fontsize=11.5, va='center', color='#3a424d')
ax.set_aspect('equal')
handles = [Line2D([],[],marker='o',ls='',mfc=V4C[i],mec='none',ms=7,label=f'z = {z:.1f} m') for i,z in enumerate(zlevels)]
handles.append(Line2D([],[],marker='*',ls='',mfc=ORANGE,mec='#7a5200',ms=15,label='Target (on its true layer height)'))
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.97), fontsize=11.5, frameon=True, edgecolor='#c8ccd2')
ins4 = fig.add_axes([0.815, 0.40, 0.17, 0.24]); target_inset(ins4, show_layer=near_z, title='target neighborhood\n(true geometry, 0.5 m cells)')
fig.text(0.985, 0.055, 'Vertical separation exaggerated for clarity', ha='right', fontsize=11.5, color='#3a424d', style='italic')
fig.text(0.985, 0.030, GT_NOTE, ha='right', fontsize=10, color='#8a919b')
fig.savefig(OUT+'candidate_grid_v4_exploded_layers.png', facecolor='white'); plt.close(fig)
print('v4 written')
print('VERIFY: total', len(pts), 'per-layer', {z: len(layer[z]) for z in zlevels}, 'oracle', round(oracle,3))
