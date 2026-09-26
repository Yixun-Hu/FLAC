"""exp_24 construction smoke (CPU): each cue arm builds on the STOCK DINOv3 backbone, strict-loads its
widened init, and its conditioner output on a real HAA training sample is BIT-IDENTICAL to its base arm
(P1ORI27 vs P1, YAWORI27 vs YAW) because the extra weights are zero. Also: a non-zero extra weight makes
the facing field matter, a different facing gives a different output, the cylindrical package is never
imported, and the second (shared-backbone) conditioner is widened once."""
import json, sys, torch, importlib.util, numpy as np
sys.path.insert(0, '.')
from src.models.factory import create_model_from_config
from src.models.utils import load_ckpt_state_dict
E = 'worklog/worklog_yixun/exp_24_haa_orientation_fairness_claude'; E23 = 'worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude'
torch.manual_seed(0)
def build(cfg_path, init_path):
    model = create_model_from_config(json.load(open(cfg_path)))
    w = load_ckpt_state_dict(init_path); w = {k.replace('diffusion.', ''): v for k, v in w.items()}
    w = {k: v for k, v in w.items() if 'discriminator' not in k and 'losses' not in k}
    model.load_state_dict(w, strict=True); return model.eval()
spec = importlib.util.spec_from_file_location('md_ori', f'{E23}/HAA_md_ori.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
info = {'path': 'HAA/hallwayBase/mono_rirs_22050Hz/0.wav', 'relpath': 'hallwayBase/mono_rirs_22050Hz/0.wav',
        'modalities': {'acoustic_context': {'load': True, 'max_context': 8, 'max_len': 9600}, 'depth': {'load': True}, 'poses': {'load': True}}}
np.random.seed(0); md = mod.get_custom_metadata(info, None); md_base = {k: v for k, v in md.items() if k != 'facing'}
print('facing', md['facing'].tolist())
PAIRS = {'P1ORI27': ('src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json', 'outputs_FLAC/exp19_inits/HAA_init_P1.ckpt', f'{E}/FLAC_HAA_finetune_P1ORI27.json', 'outputs_FLAC/exp19_inits/HAA_init_P1ORI.ckpt'),
         'YAWORI27': ('worklog/worklog_yixun/exp_19_haa_finetune_claude/FLAC_HAA_finetune_YAW.json', 'outputs_FLAC/exp19_inits/HAA_init_YAW.ckpt', f'{E}/FLAC_HAA_finetune_YAWORI27.json', 'outputs_FLAC/exp19_inits/HAA_init_YAWORI.ckpt')}
for arm, (bcfg, binit, ocfg, oinit) in PAIRS.items():
    base = build(bcfg, binit); ori = build(ocfg, oinit)
    vb = base.conditioner.conditioners['source_vit']; vo = ori.conditioner.conditioners['source_vit']
    assert vo.vit is ori.conditioner.conditioners['context_poses_vit'].vit, 'backbone not shared'
    print(arm, '| base conv', tuple(vb.vit.embeddings.patch_embeddings.weight.shape), '| cue conv', tuple(vo.vit.embeddings.patch_embeddings.weight.shape), '| field', vo.orientation_field, 'scale', vo.orientation_scale, '| config.num_channels', vo.vit.config.num_channels, '| backbone', type(vo.vit).__name__)
    assert vb.vit.embeddings.patch_embeddings.weight.shape[1] == 3 and vo.vit.embeddings.patch_embeddings.weight.shape[1] == 6
    with torch.no_grad():
        ob = base.conditioner([md_base], device='cpu', only_ids=['source_vit', 'context_poses_vit']); oo = ori.conditioner([md], device='cpu', only_ids=['source_vit', 'context_poses_vit'])
    for k in ('source_vit', 'context_poses_vit'):
        d = float((ob[k][0] - oo[k][0]).abs().max()); print(f'   {k}: max|diff| base vs cue(zero extra) = {d}'); assert torch.equal(ob[k][0], oo[k][0]), f'{arm} {k} not bit-identical at init'
    with torch.no_grad():
        vo.vit.embeddings.patch_embeddings.weight[:, 3:].normal_(std=0.02)
        op = ori.conditioner([md], device='cpu', only_ids=['source_vit']); d1 = float((op['source_vit'][0] - oo['source_vit'][0]).abs().max())
        md2 = dict(md); md2['facing'] = torch.tensor([1.0, 0.0, 0.0]); oq = ori.conditioner([md2], device='cpu', only_ids=['source_vit']); d2 = float((oq['source_vit'][0] - op['source_vit'][0]).abs().max())
    print(f'   non-zero extra weights: |diff| = {d1:.4g}; facing (0,-1,0) vs (1,0,0): |diff| = {d2:.4g}'); assert d1 > 0 and d2 > 0
assert 'cylindrical_dinov3' not in sys.modules, 'the vanilla arms must not import the cylindrical package'
print('SMOKE OK (vanilla cue arms)')
