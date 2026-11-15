"""exp_38 — the observe-only run-evidence callback (src/training/run_evidence.py) and its
default-off ``--evidence-json`` flag (plan_data_efficiency.md r2 §10 B3).

Contracts: (1) flag absent/empty -> no extra callback, so every existing recipe is
unchanged; (2) the callback never changes numerics: its wrappers are pass-through and are
removed after the first optimizer step; (3) the evidence records what the plan's start gate
needs (state digest, Sobol + RNG digests, SyncBN / BatchNorm counts, non-reentrant ViT
checkpoint wrappers, flash / SDPA calls in src.models.transformer, conditioner call shapes).
"""
import configparser
import json
import os
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402
from src.training import run_evidence as rev  # noqa: E402
import src.models.transformer as T  # noqa: E402


def test_defaults_ini_registers_an_empty_evidence_flag():
    cp = configparser.ConfigParser()
    cp.read(os.path.join(REPO, "defaults.ini"))
    assert cp["DEFAULTS"]["evidence_json"] == "''"


@pytest.mark.parametrize("value", ["", None])
def test_no_flag_means_no_extra_callback(value):
    args = types.SimpleNamespace() if value is None else types.SimpleNamespace(evidence_json=value)
    assert train.extra_callbacks(args) == []


def test_flag_adds_exactly_the_evidence_callback(tmp_path):
    cbs = train.extra_callbacks(types.SimpleNamespace(evidence_json=str(tmp_path / "ev.json")))
    assert len(cbs) == 1 and isinstance(cbs[0], rev.RunEvidenceCallback)


def test_state_dict_digest_is_order_free_dtype_and_value_sensitive():
    a = {"x": torch.zeros(3), "y": torch.ones(2, dtype=torch.bfloat16), "s": torch.tensor(5)}
    b = {"s": torch.tensor(5), "y": torch.ones(2, dtype=torch.bfloat16), "x": torch.zeros(3)}
    assert rev.state_dict_digest(a) == rev.state_dict_digest(b)
    assert rev.state_dict_digest(a) != rev.state_dict_digest({**a, "x": torch.zeros(3, dtype=torch.float64)})
    assert rev.state_dict_digest(a) != rev.state_dict_digest({**a, "s": torch.tensor(6)})


def test_rng_digests_do_not_consume_rng():
    torch.manual_seed(0)
    before = torch.get_rng_state().clone()
    d1 = rev.rng_digests()
    assert torch.equal(before, torch.get_rng_state()) and d1 == rev.rng_digests()
    torch.rand(1)
    assert rev.rng_digests()["torch_cpu"] != d1["torch_cpu"]


class _Cond(torch.nn.Module):
    def forward(self, batch_metadata, device, only_ids=None):
        return {"n": len(batch_metadata)}


class _Wrapper(torch.nn.Module):
    """Stands in for DiffusionCondTrainingWrapper: .diffusion.conditioner, .rng, a BN, a ViT layer."""
    def __init__(self, sync=False):
        super().__init__()
        self.diffusion = torch.nn.Module()
        self.diffusion.conditioner = _Cond()
        self.bn = torch.nn.SyncBatchNorm(4) if sync else torch.nn.BatchNorm1d(4)
        self.layer = torch.nn.Linear(2, 2)
        from src.models.conditioners import _make_vit_ckpt_forward
        self.layer.forward = _make_vit_ckpt_forward(self.layer, self.layer.forward)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


def _trainer(rank=0, world=2, step=0):
    return types.SimpleNamespace(global_rank=rank, local_rank=rank, world_size=world, num_nodes=1,
                                 global_step=step, precision="bf16-mixed", accumulate_grad_batches=1,
                                 max_steps=40000, gradient_clip_val=0.0,
                                 strategy=types.SimpleNamespace(__class__=type("DDPStrategy", (), {})))


def test_evidence_records_the_first_step_and_unwraps(tmp_path):
    path = str(tmp_path / "ev" / "run_evidence.json")
    cb = rev.RunEvidenceCallback(path)
    w = _Wrapper(sync=True)
    orig_flash, orig_ckpt, orig_F = T.flash_attn_func, torch.utils.checkpoint.checkpoint, T.F
    tr = _trainer()
    cb.on_fit_start(tr, w)
    w.diffusion.conditioner([1] * 32, "cpu")
    for _ in range(3):
        w.diffusion.conditioner([1] * 32, "cpu", only_ids=["source_vit", "context_poses_vit"])
    T.F.scaled_dot_product_attention(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4))
    w.layer.train()
    x = torch.ones(1, 2, requires_grad=True)
    w.layer(x).sum().backward()                                   # one non-reentrant checkpoint call
    cb.on_train_batch_end(_trainer(step=0), w, None, (torch.zeros(32, 1, 8), []), 0)   # before step 1: no write
    assert not os.path.exists(path + ".rank0.json")
    cb.on_train_batch_end(_trainer(step=1), w, None, (torch.zeros(32, 1, 8), []), 0)
    ev = json.load(open(path + ".rank0.json"))
    assert ev["rank"] == 0 and ev["world_size"] == 2 and ev["micro_batch"] == 32
    assert ev["syncbn_modules"] == 1 and ev["batchnorm_nonsync_modules"] == 0
    assert ev["vit_ckpt_wrapped_layers"] == 1
    assert ev["checkpoint_calls"] == [{"use_reentrant": False}]
    assert ev["conditioner_calls"] == [[32, None]] + [[32, ["source_vit", "context_poses_vit"]]] * 3
    assert ev["sdpa_calls"] == 1 and ev["flash_calls"] == 0
    assert len(ev["state_dict_sha256"]) == 64 and ev["sobol"]["num_generated"] == 0
    assert T.flash_attn_func is orig_flash and torch.utils.checkpoint.checkpoint is orig_ckpt and T.F is orig_F
    assert "forward" not in vars(w.diffusion.conditioner)
    cb.on_train_batch_end(_trainer(step=2), w, None, (torch.zeros(32, 1, 8), []), 1)   # written once


def test_wrappers_are_pass_through(tmp_path):
    cb = rev.RunEvidenceCallback(str(tmp_path / "e.json"))
    w = _Wrapper()
    q = torch.randn(1, 2, 3, 4)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    cb.on_fit_start(_trainer(world=1), w)
    try:
        assert torch.equal(T.F.scaled_dot_product_attention(q, q, q), ref)
        assert T.F.softmax is torch.nn.functional.softmax                  # every other attribute forwarded
        assert w.diffusion.conditioner([1, 2], "cpu") == {"n": 2}
    finally:
        cb.teardown(None, w, "fit")
    assert T.F is torch.nn.functional


def test_flash_counter_when_flash_is_present(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(T, "flash_attn_func", lambda *a, **k: calls.append(1) or "out")
    cb = rev.RunEvidenceCallback(str(tmp_path / "e.json"))
    w = _Wrapper()
    cb.on_fit_start(_trainer(world=1), w)
    assert T.flash_attn_func(1, 2, 3) == "out" and calls == [1]
    cb.on_train_batch_end(_trainer(world=1, step=1), w, None, (torch.zeros(4, 1, 8), []), 0)
    ev = json.load(open(str(tmp_path / "e.json") + ".rank0.json"))
    assert ev["flash_attn_func_set"] is True and ev["flash_calls"] == 1
