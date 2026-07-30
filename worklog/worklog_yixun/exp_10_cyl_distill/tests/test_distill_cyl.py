"""S2 unit surface: field port, loss semantics, lr schedule, gate, key resolution."""
import json, math, os, pytest, torch
from conftest import load_mod, D

dc = load_mod("distill_cyl")


def test_field_port_matches_audited_line():
    B, H, W = 2, 4, 6
    coord = torch.randn(B, 3)
    depth = torch.randn(B, 3, H, W)
    got = dc.build_field(coord, depth, 5.0, 0)
    want = (coord.unsqueeze(1)[:, 0, :, None, None] - depth) / 5.0
    assert torch.equal(got, want) and got.shape == (B, 3, H, W)


def test_field_port_source_quote():
    src = open("/home/yixunhu/codespace/exp-10-cyl-distill/src/models/conditioners.py").read()
    assert "c = (coord[:, i, :, None, None] - depth_coord) / self.max_value" in src


def test_branch_loss_sum_and_fp32():
    s = torch.randn(2, dc.TOKENS, dc.DIM, dtype=torch.bfloat16)
    t = torch.randn(2, dc.TOKENS, dc.DIM, dtype=torch.bfloat16)
    L = dc.branch_loss(s, t)
    assert L.dtype == torch.float32
    zero = dc.branch_loss(s, s)
    assert zero.item() < 1e-3        # cos term ~0, mse 0 (bf16->fp32 cast exact copy)
    with pytest.raises(SystemExit):
        dc.branch_loss(torch.randn(2, 10, dc.DIM), torch.randn(2, 10, dc.DIM))


def test_lr_schedule_pins():
    assert abs(dc.lr_at(0, 10000) - 1e-4 / 500) < 1e-12          # warmup start
    assert abs(dc.lr_at(499, 10000) - 1e-4) < 1e-9               # warmup end
    assert abs(dc.lr_at(9999, 10000) - 1e-6) < 2e-8              # cosine floor
    mid = dc.lr_at(500 + (10000 - 500) // 2, 10000)
    assert 4e-5 < mid < 6e-5                                     # ~half amplitude


def test_gate_arithmetic():
    losses = [1.0] * 1000 + [0.4] * 9000
    g = dc.gate_from_losses(losses)
    assert g["pass"] is True and abs(g["early_mean_801_1000"] - 1.0) < 1e-12
    losses = [1.0] * 1000 + [0.6] * 9000
    assert dc.gate_from_losses(losses)["pass"] is False
    assert dc.gate_from_losses([1.0] * 500)["pass"] is False     # short run never passes


def test_resolve_meta_two_step_and_fail():
    assert dc.resolve_meta({"source_vit": 1}, "source_vit", "source") == 1
    assert dc.resolve_meta({"source": 2}, "source_vit", "source") == 2
    with pytest.raises(SystemExit):
        dc.resolve_meta({"other": 3}, "source_vit", "source")


def test_max_value_read_and_uniqueness(tmp_path):
    cfg = {"model": {"conditioning": {"configs": [
        {"type": "ViTCoordinates", "config": {"max_value": 1}},
        {"type": "ViTCoordinates", "config": {"max_value": 1}},
        {"type": "dist_embedder", "config": {}}]}}}
    p = tmp_path / "c.json"; p.write_text(json.dumps(cfg))
    assert dc.read_max_value(str(p)) == 1.0
    cfg["model"]["conditioning"]["configs"][1]["config"]["max_value"] = 2
    p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        dc.read_max_value(str(p))
