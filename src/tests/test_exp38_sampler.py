"""exp_38 B4b — what the DistributedSampler does at the two rungs and under the two launchers
(plan_data_efficiency.md r2 §7, §10 B4b; Codex plan r2 HIGH 5).

* PL 2.1 injects ``DistributedSampler(dataset, shuffle=True, seed=PL_GLOBAL_SEED, num_replicas=W,
  rank=r)`` via ``_get_distributed_sampler``; the rank/world it gets come from the cluster
  environment: ``LightningEnvironment`` (PL's subprocess DDP launcher, this box) or
  ``TorchElasticEnvironment`` (torchrun, neuronic). Same (W, r) -> same sampler, same worker seeds.
* Per epoch both rungs consume the same unique permutation prefix and drop the same unique suffix
  (1 / 1 / 31 items at 25 / 50 / 75 %); padding differs (W2: 1 / 1 / 1; W8: 7 / 7 / 1) but the
  padded entries lie in the dropped tails; rank lists are strided, so a global step is the
  rank-interleaving of perm[64t : 64t+64].
In-process only (a real two-process trace needs a CPU DDP harness; disclosed in the plan).
"""
import math
import os

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

FRACTIONS = {"025": 73473, "050": 145665, "075": 218463}


class _Toy(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _sampler(n, world, rank, seed=42, epoch=0):
    from pytorch_lightning.trainer.connectors.data_connector import _get_distributed_sampler
    dl = DataLoader(_Toy(n), batch_size=64 // world, shuffle=True, drop_last=True)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    s = _get_distributed_sampler(dl, shuffle=True, overfit_batches=0, num_replicas=world, rank=rank)
    s.set_epoch(epoch)
    return s


def _consumed(n, world, epoch=0):
    micro = 64 // world
    per_rank = [list(_sampler(n, world, r, epoch=epoch)) for r in range(world)]
    steps = len(per_rank[0]) // micro
    used = [[per_rank[r][t * micro:(t + 1) * micro] for r in range(world)] for t in range(steps)]
    return steps, used, per_rank


@pytest.mark.parametrize("tag", sorted(FRACTIONS))
def test_padding_and_unique_drops_per_fraction(tag):
    n = FRACTIONS[tag]
    expect_pad = {"025": (1, 7), "050": (1, 7), "075": (1, 1)}[tag]
    expect_drop = {"025": 1, "050": 1, "075": 31}[tag]
    for world, pad in zip((2, 8), expect_pad):
        assert math.ceil(n / world) * world - n == pad
        micro = 64 // world
        steps = (math.ceil(n / world)) // micro
        assert steps == {"025": 1148, "050": 2276, "075": 3413}[tag]
        assert n - steps * micro * world == expect_drop


def test_both_rungs_consume_the_same_unique_prefix_and_global_batches():
    n = 6421                                                      # small n, same arithmetic shape
    steps2, used2, _ = _consumed(n, 2)
    steps8, used8, _ = _consumed(n, 8)
    assert steps2 == steps8
    g = torch.Generator()
    g.manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    for t in range(steps2):
        # rank-interleaving of the step's ranks == perm[64t : 64t+64] for both rungs
        inter2 = [used2[t][r][j] for j in range(32) for r in range(2)]
        inter8 = [used8[t][r][j] for j in range(8) for r in range(8)]
        assert inter2 == perm[64 * t:64 * t + 64] == inter8
        assert sorted(sum(used2[t], [])) == sorted(sum(used8[t], []))


def test_set_epoch_reshuffles_and_is_seed_bound():
    a0, a1 = list(_sampler(1000, 2, 0, epoch=0)), list(_sampler(1000, 2, 0, epoch=1))
    assert a0 != a1 and a0 == list(_sampler(1000, 2, 0, epoch=0))
    assert a0 != list(_sampler(1000, 2, 0, seed=43, epoch=0))


def test_both_launch_environments_give_the_same_rank_and_world(monkeypatch):
    from lightning_fabric.plugins.environments import LightningEnvironment, TorchElasticEnvironment
    for rank in (0, 1):
        for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "GROUP_RANK", "NODE_RANK", "LOCAL_WORLD_SIZE"):
            monkeypatch.delenv(k, raising=False)
        # torchrun --standalone --nproc_per_node=2
        for k, v in {"RANK": rank, "WORLD_SIZE": 2, "LOCAL_RANK": rank, "GROUP_RANK": 0,
                     "LOCAL_WORLD_SIZE": 2, "TORCHELASTIC_RUN_ID": "x"}.items():
            monkeypatch.setenv(k, str(v))
        te = TorchElasticEnvironment()
        # PL's subprocess launcher: the strategy sets world size / global rank on LightningEnvironment
        le = LightningEnvironment()
        le.set_world_size(2)
        le.set_global_rank(rank)
        assert (te.world_size(), te.global_rank()) == (le.world_size(), le.global_rank()) == (2, rank)
        assert list(_sampler(5000, te.world_size(), te.global_rank())) == \
            list(_sampler(5000, le.world_size(), le.global_rank()))


def test_worker_seeding_depends_only_on_seed_rank_and_worker(monkeypatch):
    import random
    import numpy as np
    from lightning_fabric.utilities.seed import pl_worker_init_function

    def draw(rank, worker):
        # inside a DataLoader worker torch.initial_seed() is the loader's base seed + worker id;
        # emulate a fixed base seed so only (rank, worker) vary
        torch.manual_seed(10_000 + worker)
        pl_worker_init_function(worker, rank=rank)
        return random.random(), float(np.random.rand()), float(torch.rand(1))
    assert draw(1, 3) == draw(1, 3)
    assert draw(1, 3) != draw(0, 3) and draw(1, 3) != draw(1, 2)
