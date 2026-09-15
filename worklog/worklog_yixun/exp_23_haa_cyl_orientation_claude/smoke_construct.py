"""exp_23 construction smoke (CPU): the CYLORI model builds, strict-loads the widened init, and
its conditioner output on a real HAA training sample is BIT-IDENTICAL to the CYL model's on the
same sample (zero extra weights => the facing field is inert at step 0). Also: the CYL config
still builds a 3-channel backbone and loads its init unchanged; rotate_scene_metadata turns
'facing' with the world; a non-zero extra weight makes the facing field matter."""
import json, sys, copy, torch, importlib.util
sys.path.insert(0, '.')
from src.models.factory import create_model_from_config
from src.models.utils import load_ckpt_state_dict
from src.data.yaw_rotation import rotate_scene_metadata
E = 'worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude'
torch.manual_seed(0)

def build(cfg_path, init_path):
    cfg = json.load(open(cfg_path))
    model = create_model_from_config(cfg)
    w = load_ckpt_state_dict(init_path)
    w = {k.replace('diffusion.', ''): v for k, v in w.items()}
    w = {k: v for k, v in w.items() if 'discriminator' not in k and 'losses' not in k}
    model.load_state_dict(w, strict=True)           # exactly train.py:139-147
    return model.eval()

cyl = build('worklog/worklog_yixun/exp_19_haa_finetune_claude/FLAC_HAA_finetune_CYL.json', 'outputs_FLAC/exp19_inits/HAA_init_CYL.ckpt')
ori = build(f'{E}/FLAC_HAA_finetune_CYLORI.json', 'outputs_FLAC/exp19_inits/HAA_init_CYLORI.ckpt')
vit_c = cyl.conditioner.conditioners['source_vit']; vit_o = ori.conditioner.conditioners['source_vit']
assert vit_c.vit is cyl.conditioner.conditioners['context_poses_vit'].vit and vit_o.vit is ori.conditioner.conditioners['context_poses_vit'].vit, "backbone not shared"
print('CYL  conv', tuple(vit_c.vit.embeddings.patch_embeddings.weight.shape), 'orientation_field', vit_c.orientation_field)
print('ORI  conv', tuple(vit_o.vit.embeddings.patch_embeddings.weight.shape), 'orientation_field', vit_o.orientation_field, 'scale', vit_o.orientation_scale, 'config.num_channels', vit_o.vit.config.num_channels)
assert tuple(vit_c.vit.embeddings.patch_embeddings.weight.shape)[1] == 3 and tuple(vit_o.vit.embeddings.patch_embeddings.weight.shape)[1] == 6

# one real HAA training sample through the exp_23 metadata module
spec = importlib.util.spec_from_file_location('md_ori', f'{E}/HAA_md_ori.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
info = {'path': 'HAA/hallwayBase/mono_rirs_22050Hz/0.wav', 'relpath': 'hallwayBase/mono_rirs_22050Hz/0.wav',
        'modalities': {'acoustic_context': {'load': True, 'max_context': 8, 'max_len': 9600}, 'depth': {'load': True}, 'poses': {'load': True}}}
import numpy as np; np.random.seed(0)
md = mod.get_custom_metadata(info, None)
print('md keys', sorted(md.keys()), 'facing', md['facing'].tolist(), 'depth', tuple(md['depth'].shape), 'source', md['source'].tolist())
md_cyl = {k: v for k, v in md.items() if k != 'facing'}
with torch.no_grad():
    out_c = cyl.conditioner([md_cyl], device='cpu', only_ids=['source_vit', 'context_poses_vit'])
    out_o = ori.conditioner([md], device='cpu', only_ids=['source_vit', 'context_poses_vit'])
for k in ('source_vit', 'context_poses_vit'):
    a, b = out_c[k][0], out_o[k][0]
    print(k, tuple(a.shape), 'max|diff| CYL vs CYLORI(zero extra) =', float((a - b).abs().max()))
    assert torch.equal(a, b), f'{k}: CYLORI at init is not bit-identical to CYL'
# facing must be able to matter: perturb the extra weights
with torch.no_grad():
    vit_o.vit.embeddings.patch_embeddings.weight[:, 3:].normal_(std=0.02)
    out_p = ori.conditioner([md], device='cpu', only_ids=['source_vit'])
    d = float((out_p['source_vit'][0] - out_o['source_vit'][0]).abs().max()); print('with non-zero extra weights, |diff| =', d); assert d > 0
    # and a different facing gives a different output (the field is really consumed)
    md2 = dict(md); md2['facing'] = torch.tensor([1.0, 0.0, 0.0])
    out_q = ori.conditioner([md2], device='cpu', only_ids=['source_vit'])
    d2 = float((out_q['source_vit'][0] - out_p['source_vit'][0]).abs().max()); print('facing (0,-1,0) vs (1,0,0): |diff| =', d2); assert d2 > 0
# rotation covariance of 'facing'
r = rotate_scene_metadata(md, 0.5 * np.pi, 512)
print('facing after +90deg yaw:', [round(x, 6) for x in r['facing'].tolist()], '(expect (1,0,0))')
assert torch.allclose(r['facing'], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
# equivariance of the widened (perturbed) backbone under physical yaw of the 6-channel field, one roll
from cylindrical_dinov3 import physical_yaw
c = (md['source'][None, :, None, None] - md['depth'][None]) / 1.0
x6 = torch.cat([c, (md['facing'] * vit_o.orientation_scale)[None, :, None, None].expand(1, 3, 256, 512)], 1)
with torch.no_grad():
    y0 = vit_o.vit(x6).pooler_output; y1 = vit_o.vit(physical_yaw(x6, 64)).pooler_output
print('pooled invariance under 45-deg physical yaw (widened, perturbed):', float((y0 - y1).norm() / y0.norm()))
print('SMOKE OK')
