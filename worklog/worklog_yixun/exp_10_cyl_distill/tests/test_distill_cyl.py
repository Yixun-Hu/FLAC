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


def test_branch_loss_exact_formula():
    torch.manual_seed(0)
    s = torch.randn(2, dc.TOKENS, dc.DIM)
    t = torch.randn(2, dc.TOKENS, dc.DIM)
    L = dc.branch_loss(s, t)
    want = (1.0 - torch.nn.functional.cosine_similarity(s, t, dim=-1, eps=1e-8)).mean() \
           + torch.nn.functional.mse_loss(s, t, reduction="mean")
    assert torch.allclose(L, want, atol=0, rtol=0)      # EXACT — /2 or term-drop mutants die
    assert L.dtype == torch.float32
    with pytest.raises(SystemExit):
        dc.branch_loss(torch.randn(2, 10, dc.DIM), torch.randn(2, 10, dc.DIM))


def test_total_loss_is_sum_not_mean():
    torch.manual_seed(1)
    a, b = torch.randn(1, dc.TOKENS, dc.DIM), torch.randn(1, dc.TOKENS, dc.DIM)
    c, d = torch.randn(1, dc.TOKENS, dc.DIM), torch.randn(1, dc.TOKENS, dc.DIM)
    L, L_src, L_ctx = dc.total_loss(a, b, c, d)
    assert torch.allclose(L, L_src + L_ctx, atol=0, rtol=0)
    assert not torch.allclose(L, (L_src + L_ctx) / 2)   # mean mutant dies


def test_strip_prefix_count_refusal():
    ok = dc.strip_prefix(torch.randn(2, dc.TOKENS + dc.N_PREFIX, dc.DIM))
    assert ok.shape == (2, dc.TOKENS, dc.DIM)
    for bad_tokens in (dc.TOKENS, dc.TOKENS + 1, dc.TOKENS + dc.N_PREFIX + 1):
        with pytest.raises(SystemExit):
            dc.strip_prefix(torch.randn(2, bad_tokens, dc.DIM))


def test_lr_schedule_pins():
    assert abs(dc.lr_at(0, 10000) - 1e-4 / 500) < 1e-15          # warmup start
    assert dc.lr_at(499, 10000) == 1e-4                          # warmup end EXACT
    assert dc.lr_at(500, 10000) < 1e-4                           # no base repeat (r1 #6)
    assert abs(dc.lr_at(9999, 10000) - 1e-6) < 1e-18             # floor EXACT at last executed step
    mid = dc.lr_at(500 + (10000 - 500) // 2, 10000)
    assert 4e-5 < mid < 6e-5


def test_gate_arithmetic_and_window_boundaries():
    # heterogeneous fixture: early window (steps 801-1000, 0-idx 800..999) = 2.0,
    # neighbors 800 and 1000 (0-idx 799/1000) poisoned; late window = 0.9 with
    # poisoned neighbor at 0-idx 9799. Any window shift changes the means.
    losses = [5.0] * 1000 + [0.9] * 9000
    for i in range(800, 1000):
        losses[i] = 2.0
    losses[799] = 100.0
    losses[9799] = 100.0
    g = dc.gate_from_losses(losses)
    assert abs(g["early_mean_801_1000"] - 2.0) < 1e-12           # off-by-one -> 2.49
    assert abs(g["late_mean_9801_10000"] - 0.9) < 1e-12          # off-by-one -> ~1.4
    assert g["pass"] is True                                      # 0.9 < 1.0
    for i in range(9800, 10000):
        losses[i] = 1.01
    assert dc.gate_from_losses(losses)["pass"] is False           # 1.01 > 1.0 boundary
    assert dc.gate_from_losses([1.0] * 500)["pass"] is False


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
