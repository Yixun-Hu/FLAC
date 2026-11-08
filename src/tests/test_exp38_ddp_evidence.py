"""exp_38 — the run-evidence callback under a REAL two-rank DDP fit (CPU / gloo; Codex code r1 HIGH 11).

The registry pins the DDP RNG digests at fit start as the FRESH seed-42 state (PL 2.1's
DDPStrategy.setup_distributed() calls lightning_fabric's reset_seed() after the model build), not
the post-build state a single-rank run records. This test runs the production launcher path --
``strategy="ddp_find_unused_parameters_true"`` with PL's subprocess launcher re-executing the script,
seed_everything(42, workers=True) BEFORE a model build that draws from torch's CPU generator,
exactly as train.py does -- with a toy module, and checks every rank's record: rank / local_rank /
world size / strategy, ranks agreeing on the init state, and the Python / NumPy / torch-CPU digests
equal to the registry's DDP pins. Disclosed limits: CUDA is not available here, so SyncBN (CUDA-only
in torch) and the CUDA digest are confirmed on the cards by SMOKE_DDP before any FULL run.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")
sys.path.insert(0, EXP)
import exp38_registry as reg  # noqa: E402

SCRIPT = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {repo!r})
    import pytorch_lightning as pl
    import torch
    from src.training.run_evidence import RunEvidenceCallback

    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(4)
            self.lin = torch.nn.Linear(4, 1)          # draws from torch's CPU generator

        def training_step(self, batch, i):
            x, y = batch
            return ((self.lin(self.bn(x)) - y) ** 2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    pl.seed_everything(42, workers=True)
    model = Toy()
    data = torch.utils.data.TensorDataset(torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4) / 100,
                                          torch.zeros(64, 1))
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)
    trainer = pl.Trainer(accelerator="cpu", devices=2, num_nodes=1, strategy="ddp_find_unused_parameters_true",
                         max_steps=2, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[RunEvidenceCallback({out!r})])
    trainer.fit(model, loader)
""")


def test_two_rank_ddp_evidence_follows_the_registered_rule(tmp_path):
    out = str(tmp_path / "evidence" / "run_evidence.json")
    script = tmp_path / "ddp_toy.py"
    script.write_text(SCRIPT.format(repo=REPO, out=out))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "LOCAL_RANK", "RANK", "WORLD_SIZE",
                                                                     "MASTER_", "PL_", "CUDA_VISIBLE"))}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    recs = reg.load_evidence(str(tmp_path / "evidence"), 2)
    for rank, e in recs.items():
        assert set(e) == reg.EVIDENCE_KEYS
        assert (e["rank"], e["local_rank"], e["world_size"], e["num_nodes"]) == (rank, rank, 2, 1)
        assert e["strategy"] == "DDPStrategy" and e["global_step_recorded"] == 1 and e["micro_batch"] == 8
        assert (e["syncbn_modules"], e["batchnorm_nonsync_modules"]) == (0, 1)   # SyncBN needs CUDA (SMOKE_DDP)
        ddp = reg.RNG_AT_FIT_START["ddp"]
        assert e["rng_at_fit_start"] == {k: ddp[k] for k in ("python", "numpy", "torch_cpu")}
        assert e["rng_at_fit_start"]["torch_cpu"] != reg.RNG_AT_FIT_START["single"]["torch_cpu"]
    assert recs[0]["state_dict_sha256"] == recs[1]["state_dict_sha256"]
