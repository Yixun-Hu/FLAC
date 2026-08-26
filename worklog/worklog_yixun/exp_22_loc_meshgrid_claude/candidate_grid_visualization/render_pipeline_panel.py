"""exp_22 pipeline heatmap panel — Cafe_idx_1, paper camera (az 78 / elev 55 / dist 2.8 /
fov 14 / lookat 13.15 11.79 1.35 / zcut 2.40, render_room_obj.py convention).
Current variant: heat field only + colorbar (markers/legend deleted per Yixun 2026-08-25;
colorbar restored per Yixun 2026-08-25). Heat field is ILLUSTRATIVE (mock Gaussian mixture
at GT + offset lobe) until real P1 scores land."""
import open3d as o3d, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colorbar as mcb
plt.rcParams['font.family']='serif'; plt.rcParams['font.serif']=['P052','Palatino','TeX Gyre Pagella','DejaVu Serif']
m = o3d.io.read_triangle_mesh('/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/Cafe/Cafe_idx_1.obj')
V = np.asarray(m.vertices); T = np.asarray(m.triangles)
keep = V[:,2][T].min(axis=1) < 2.40
Tk = T[keep]
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
wall_c=np.array([0.82,0.84,0.87]); floor_c=np.array([0.91,0.88,0.82]); furn_c=np.array([0.73,0.70,0.64])
base = np.where(is_wall[...,None], wall_c, np.where((flat&low)[...,None], floor_c, furn_c))
img[...,:3]=np.clip(base*(0.60+0.40*lam)[...,None],0,1); img[...,3]=hit.astype(float)
d0=np.where(hit,t,np.nan)
edge=((np.abs(np.diff(d0,axis=1,prepend=d0[:,:1]))>0.20)|(np.abs(np.diff(d0,axis=0,prepend=d0[:1]))>0.20))&hit
img[edge,:3]*=0.45
gt = np.array(json.load(open('/media/diskstation/yixunhu/FLAC/AcousticRooms/metadata/Cafe/Cafe_idx_1/S001_R008.json'))['src_loc'])
fx, fy = hp[...,0], hp[...,1]
with np.errstate(invalid='ignore'):
    d2 = (fx-gt[0])**2 + (fy-gt[1])**2
    d2b = (fx-(gt[0]-3.2))**2 + (fy-(gt[1]+2.1))**2
    score = np.exp(-d2/(2*1.55**2)) + 0.30*np.exp(-d2b/(2*2.6**2))
score = np.nan_to_num(score); mx = score[hit].max() if hit.any() else 1
score = np.clip(score/mx, 0, 1)
cmap = colormaps['RdYlBu_r']
heat = cmap(score)[...,:3]
floorpix = hit & flat & low & ~is_wall
shade = (0.66+0.34*lam)[...,None]
img[floorpix,:3] = (heat*shade)[floorpix]
with np.errstate(invalid='ignore'):
    gl = ((np.abs((fx/0.5)-np.round(fx/0.5))<0.02)|(np.abs((fy/0.5)-np.round(fy/0.5))<0.02)) & floorpix
img[gl,:3]*=0.88
PAD = 230  # dedicated right margin strip for the colorbar (no overlap with the room)
Wt = W + PAD
fig=plt.figure(figsize=(Wt/200,H/200),dpi=200)
ax=fig.add_axes([0,0,W/Wt,1]); ax.axis('off')
ax.imshow(img,interpolation='bilinear'); ax.set_xlim(0,W); ax.set_ylim(H,0)
cax=fig.add_axes([(W+70)/Wt,0.16,90/Wt,0.68])
cb=mcb.ColorbarBase(cax,cmap=cmap,orientation='vertical'); cb.set_ticks([])
cb.outline.set_visible(False)
out='worklog/worklog_yixun/exp_22_loc_meshgrid_claude/candidate_grid_visualization/pipeline_panel_cafe_idx_1_heatmap.png'
fig.savefig(out,transparent=True,dpi=200); plt.close(fig)
import shutil; shutil.copy(out,'/home/yixunhu/codespace/-Neurips-2026-workshop-Sound-Localization/figs/pipeline_panel_cafe_idx_1_heatmap.png')
print('wrote panel: heat field + colorbar (no markers/legend)')
