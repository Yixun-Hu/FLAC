"""exp_37 shared CPU fakes for the localization executors (a tiny conditioner / DiT / pretransform / retrieval and
a frozen query) -- test support only, never imported by production code.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c (moved out of test_exp37_stage_split.py)."""
import numpy as np
import torch


class FakeConditioner:
    def __call__(self, metadata, device, only_ids=None):
        out = {}
        for key in only_ids:
            values = torch.stack([item[key].float().reshape(-1)[:3] for item in metadata]).unsqueeze(1).to(device)
            out[key] = [values, torch.ones(len(metadata), 1, device=device)]
        return out


class FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, cond=None, **sampler_kwargs):      # cfg_scale, batch_cfg (cfg 1.0: unused)
        return x * self.w + cond.mean(dim=1).view(-1, 1, 1)


class FakePretransform:
    downsampling_ratio = 1024

    def decode(self, z):
        return z.mean(dim=1, keepdim=True).repeat_interleave(1024, dim=-1) * 0.1


class FakeDiffusion:
    diffusion_objective, io_channels, dist_shift = "rectified_flow", 2, None

    def __init__(self):
        self.conditioner, self.model, self.pretransform = FakeConditioner(), FakeDiT(), FakePretransform()

    def get_conditioning_inputs(self, conditioning):
        return {"cond": torch.cat([conditioning[k][0].reshape(conditioning[k][0].shape[0], -1)
                                   for k in sorted(conditioning)], dim=1)}


class FakeModule:
    device = "cpu"

    def __init__(self):
        self.diffusion = FakeDiffusion()


class FakeRetrieval:
    def compute_audio_features(self, waveforms):
        return torch.nn.functional.normalize(waveforms[:, 0, :16] + torch.arange(16.0), dim=1)


def _fake_query(record, root):
    k = len(record["contexts"])
    gen = torch.Generator().manual_seed(7)
    poses = torch.arange(k * 3, dtype=torch.float32).reshape(k, 3) / 10 + 0.1
    metadata = {"scene": "S", "source": torch.tensor([1.0, 0.5, 0.2]), "source_vit": torch.tensor([[1.0, 0.5, 0.2]]),
                "context_poses": poses, "context_poses_vit": poses.clone(),
                "context_audio": torch.linspace(0, 1, k * 16).reshape(k, 1, 16), "depth": torch.rand(3, 4, 8, generator=gen)}
    return torch.linspace(-0.5, 0.5, 10240).reshape(1, 10240), metadata


CANDIDATES = np.array([[0.25 * i, 0.5, 1.0] for i in range(70)])                     # two batches of 64 + 6
GRID = np.array([[x, y, 1.0] for x in np.arange(0.0, 4.01, 0.5) for y in np.arange(0.0, 3.01, 0.5)])   # 63 points
