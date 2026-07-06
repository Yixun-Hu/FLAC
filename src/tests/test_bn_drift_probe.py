"""Tests for the BN-drift probe (exp_05, cycle 1; RED first).

These pin the contract of ``tools/bn_drift_probe.py`` *before* it exists. The
probe treats the RIR encoder's BatchNorm running statistics as a gradient-free
data-drift instrument (exp_04 mechanism: the lr=0 null run regressed T60/EDT
purely through RIR-encoder BN running-stat adaptation, so our loader's
reference-RIR statistics differ from the released training's). Comparing the
stored ``running_mean``/``running_var`` against the batch statistics *our* data
produces at each BN layer localizes the drift with no training.

Seven tests are plan_bn_drift_bisect.md §1's table verbatim (plan-review
findings 3/4 baked in): BN **input** stats (torchvision ResNet18 BNs sit after
convs; the stem change only swaps conv1 to one input channel), the **unbiased**
(n-1) variance estimator that matches PyTorch ``running_var`` semantics, the
exact 20-BatchNorm2d / 0-BatchNorm1d ResNet18 stack, and a provenance report.
Two more harden the instrument per the bndrift code review: a CUDA smoke for
device-correct accumulators (finding 2) and the all-buffer snapshot guard
(finding 3); the stem test's BN is non-identity per finding 4.

Only ``test_recorder_finds_all_bns`` builds the real ``AudioResNet18`` (random
weights via torchvision ``resnet18(pretrained=False)`` -- no download); every
other test uses synthetic modules and tiny tensors so the suite runs with no
checkpoint and no dataset. ``probe_rir_encoder`` itself needs real data (the
B-1 pilot covers it) and is exercised only through its pure parts here.

Imports resolve to *this* FLAC checkout via conftest.py's sys.path prepend (the
editable ``rir2rir`` install otherwise maps ``src`` to a divergent sibling repo).
"""
import pytest
import torch
from torch import nn

from tools import bn_drift_probe


# --------------------------------------------------------------------------------------
# 1-2. bn_drift_metrics: standardized per-channel drift
# --------------------------------------------------------------------------------------

def test_bn_drift_metrics_zero():
    """batch stats == running stats => mean_shift 0, var_ratio 1 exactly.

    eps=0 isolates the ratio: with identical, positive variances bv/(rv+0) == 1.0
    bit-exactly, and the |bm-rm| numerator is 0 so mean_shift is 0 regardless.
    """
    rm = torch.tensor([0.0, 1.0, -2.0])
    rv = torch.tensor([1.0, 4.0, 9.0])
    out = bn_drift_probe.bn_drift_metrics(rm, rv, rm.clone(), rv.clone(), eps=0.0)

    assert torch.equal(out["mean_shift"], torch.zeros(3))
    assert torch.equal(out["var_ratio"], torch.ones(3))
    assert out["mean_shift_max"] == 0.0
    assert out["mean_shift_mean"] == 0.0
    assert out["var_ratio_min"] == 1.0
    assert out["var_ratio_max"] == 1.0


def test_bn_drift_metrics_known():
    """Hand-computed shift/ratio on a 3-channel example, atol 1e-6.

    running_var=[1,4,9] => sqrt=[1,2,3]. mean_shift=|bm-rm|/sqrt(rv):
        |0.5-0|/1=0.5, |2-1|/2=0.5, |0-2|/3=2/3.
    var_ratio=bv/rv: 2/1=2, 8/4=2, 3/9=1/3.  (eps=0 for exact hand values.)
    """
    rm = torch.tensor([0.0, 1.0, 2.0])
    rv = torch.tensor([1.0, 4.0, 9.0])
    bm = torch.tensor([0.5, 2.0, 0.0])
    bv = torch.tensor([2.0, 8.0, 3.0])
    out = bn_drift_probe.bn_drift_metrics(rm, rv, bm, bv, eps=0.0)

    assert torch.allclose(out["mean_shift"], torch.tensor([0.5, 0.5, 2.0 / 3.0]), atol=1e-6)
    assert torch.allclose(out["var_ratio"], torch.tensor([2.0, 2.0, 1.0 / 3.0]), atol=1e-6)
    assert out["mean_shift_max"] == pytest.approx(2.0 / 3.0, abs=1e-6)  # max([0.5,0.5,2/3])
    assert out["mean_shift_mean"] == pytest.approx((0.5 + 0.5 + 2.0 / 3.0) / 3.0, abs=1e-6)
    assert out["var_ratio_min"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert out["var_ratio_max"] == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# 3. BNInputRecorder: streaming Welford, unbiased variance, no mutation
# --------------------------------------------------------------------------------------

def test_recorder_welford_unbiased():
    """Streamed mean/var == torch mean/var(unbiased) over the concatenated batches,
    atol 1e-5; and all BN running buffers are bit-unchanged after probing.

    Unbiased (n-1) matches PyTorch ``running_var`` semantics. Batches of differing
    sizes exercise the parallel (chunked) combine, not a single-batch shortcut. The
    module is in eval mode: BN reads-but-does-not-update its running stats, and the
    forward-pre-hook only reads inputs -- so the buffers must stay bit-identical.
    """
    torch.manual_seed(0)
    bn = nn.BatchNorm2d(3).eval()
    with torch.no_grad():  # perturb buffers off their defaults so bit-identity is non-trivial
        bn.running_mean.copy_(torch.randn(3))
        bn.running_var.copy_(torch.rand(3) + 0.5)
    rm0 = bn.running_mean.clone()
    rv0 = bn.running_var.clone()
    nbt0 = bn.num_batches_tracked.clone()

    recorder = bn_drift_probe.BNInputRecorder(bn)
    batches = [torch.randn(4, 3, 5, 6), torch.randn(7, 3, 5, 6), torch.randn(2, 3, 5, 6)]
    with torch.no_grad():
        for b in batches:
            bn(b)

    allx = torch.cat(batches, dim=0)  # concatenate along batch dim -> [13, 3, 5, 6]
    ref_mean = allx.mean(dim=(0, 2, 3))
    ref_var = allx.var(dim=(0, 2, 3), unbiased=True)
    name = recorder.layer_names[0]  # bare module -> name ""
    rec_mean, rec_var = recorder.input_stats(name)
    assert torch.allclose(rec_mean, ref_mean, atol=1e-5)
    assert torch.allclose(rec_var, ref_var, atol=1e-5)

    # No mutation: running buffers bit-identical to the pre-probe snapshot.
    assert torch.equal(bn.running_mean, rm0)
    assert torch.equal(bn.running_var, rv0)
    assert torch.equal(bn.num_batches_tracked, nbt0)


def test_recorder_conv_bn_stem():
    """Conv2d(1-ch stem)->BN: the recorder captures BN **inputs** = post-conv
    activations, pinned against a manual conv forward (finding 3).

    AudioResNet18 replaces conv1 with a 1-channel input but leaves the BNs after
    the convs. Hooking the raw stem input (1 channel) instead of the BN input
    (4 post-conv channels) would give the wrong shape/stats and fail here.

    The BN is made strongly non-identity (distinct running stats + affine): a
    default eval-mode BN is near-identity, so an output-hook bug would slip
    under tolerance (code-review measured the gap at 5.66e-6 < atol 1e-5). Here
    pre-BN and post-BN stats differ by >> tolerance and both directions are
    pinned: match the pre-BN activations, measurably differ from the outputs.
    """
    torch.manual_seed(1)
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(4)
    with torch.no_grad():  # non-identity normalization + affine
        bn.running_mean.copy_(torch.tensor([5.0, -3.0, 2.0, 7.0]))
        bn.running_var.copy_(torch.tensor([0.25, 4.0, 9.0, 0.04]))
        bn.weight.copy_(torch.tensor([2.0, 0.5, 3.0, 1.5]))
        bn.bias.copy_(torch.tensor([10.0, -5.0, 3.0, 0.0]))
    stem = nn.Sequential(conv, bn).eval()

    recorder = bn_drift_probe.BNInputRecorder(stem)
    x = torch.randn(6, 1, 8, 8)
    with torch.no_grad():
        stem(x)

    assert recorder.layer_names == ["1"]  # the BN is index 1 of the Sequential
    rec_mean, rec_var = recorder.input_stats("1")
    recorder.remove()  # stop recording: the manual bn() below must not re-feed the hook

    with torch.no_grad():
        conv_out = conv(x)     # the BN's input inside the Sequential (pre-BN)
        bn_out = bn(conv_out)  # what an output-hook implementation would record

    assert torch.allclose(rec_mean, conv_out.mean(dim=(0, 2, 3)), atol=1e-5)
    assert torch.allclose(rec_var, conv_out.var(dim=(0, 2, 3), unbiased=True), atol=1e-5)
    # Discriminating margin: recorded stats differ grossly from post-BN stats,
    # so an output-hook implementation cannot pass the two asserts above.
    assert (rec_mean - bn_out.mean(dim=(0, 2, 3))).abs().max() > 1.0
    assert (rec_var - bn_out.var(dim=(0, 2, 3), unbiased=True)).abs().max() > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_recorder_cuda_smoke():
    """Review finding 2 (device correctness): CUDA hook inputs must stream into
    accumulators on the same device. The pre-fix recorder allocated CPU
    accumulators at registration and crashed here on the CUDA-CPU subtraction;
    the fix lazily initializes on the first hooked batch's device. Also asserts
    Welford correctness and buffer bit-identity on the GPU path (the CLI's
    default --device cuda)."""
    torch.manual_seed(3)
    bn = nn.BatchNorm2d(3).cuda().eval()
    with torch.no_grad():
        bn.running_mean.copy_(torch.randn(3, device="cuda"))
        bn.running_var.copy_(torch.rand(3, device="cuda") + 0.5)
    snap = bn_drift_probe.snapshot_buffers(bn)

    recorder = bn_drift_probe.BNInputRecorder(bn)
    batches = [torch.randn(4, 3, 5, 6, device="cuda"), torch.randn(3, 3, 5, 6, device="cuda")]
    with torch.no_grad():
        for b in batches:
            bn(b)

    rec_mean, rec_var = recorder.input_stats(recorder.layer_names[0])
    assert rec_mean.is_cuda and rec_var.is_cuda  # accumulators lived on the input device
    allx = torch.cat(batches, dim=0)
    assert torch.allclose(rec_mean, allx.mean(dim=(0, 2, 3)), atol=1e-5)
    assert torch.allclose(rec_var, allx.var(dim=(0, 2, 3), unbiased=True), atol=1e-5)
    bn_drift_probe.assert_buffers_unchanged(bn, snap)  # read-only on CUDA too


def test_snapshot_buffers_catches_mutation():
    """Review finding 3: the all-buffer no-mutation guard used around
    probe_rir_encoder's pass. snapshot_buffers covers every named buffer;
    assert_buffers_unchanged passes on an eval-mode forward (BN reads only) and
    raises, naming the buffer, after a train-mode forward mutates running stats.
    The raise also proves the snapshot holds clones, not references."""
    torch.manual_seed(2)
    mod = nn.Sequential(nn.Conv2d(2, 3, kernel_size=1, bias=False), nn.BatchNorm2d(3))
    snap = bn_drift_probe.snapshot_buffers(mod)
    assert set(snap) == {"1.running_mean", "1.running_var", "1.num_batches_tracked"}

    mod.eval()
    with torch.no_grad():
        mod(torch.randn(4, 2, 5, 5))
    bn_drift_probe.assert_buffers_unchanged(mod, snap)  # eval forward: no raise

    mod.train()
    with torch.no_grad():
        mod(torch.randn(4, 2, 5, 5))  # train-mode BN updates its running stats
    with pytest.raises(RuntimeError, match="running_mean"):
        bn_drift_probe.assert_buffers_unchanged(mod, snap)


def test_recorder_finds_all_bns():
    """The real AudioResNet18 (torchvision resnet18) => exactly 20 BatchNorm2d, 0
    BatchNorm1d; the recorder hooks exactly those 20 layers.

    resnet18 = 1 stem BN + 4 stages x (2 BasicBlocks x 2 BN) + 3 downsample BN
    = 1 + 16 + 3 = 20. Built with pretrained=False (random weights, no download).
    """
    from src.models.conditioners import AudioResNet18

    net = AudioResNet18(in_channels=1)
    n_bn2d = sum(1 for _, m in net.named_modules() if isinstance(m, nn.BatchNorm2d))
    n_bn1d = sum(1 for _, m in net.named_modules() if isinstance(m, nn.BatchNorm1d))
    assert n_bn2d == 20
    assert n_bn1d == 0

    recorder = bn_drift_probe.BNInputRecorder(net)
    assert recorder.n_layers == 20
    assert len(recorder.layer_names) == 20
    assert "cnn.bn1" in recorder.layer_names  # true torchvision BN-buffer name


# --------------------------------------------------------------------------------------
# 6-7. probe pure parts: md_variant hook + provenance report
# --------------------------------------------------------------------------------------

def test_probe_variant_hook():
    """md_variant callable is applied once per metadata dict, in order, before
    conditioning; None is an identity fast-path returning the same list object."""
    md1 = {"context_audio": torch.zeros(2, 1, 10), "tag": "a"}
    md2 = {"context_audio": torch.ones(2, 1, 10), "tag": "b"}
    seen = []

    def variant(md):
        seen.append(md["tag"])
        new = dict(md)
        new["tag"] = md["tag"].upper()
        return new

    lst = [md1, md2]
    out = bn_drift_probe._apply_md_variant(lst, variant)
    assert seen == ["a", "b"]
    assert [m["tag"] for m in out] == ["A", "B"]
    assert out[0] is not md1 and md1["tag"] == "a"  # originals untouched

    assert bn_drift_probe._apply_md_variant(lst, None) is lst  # None => identity


def test_probe_report_provenance():
    """The report carries ckpt path, clean-load missing/unexpected counts, and one
    entry per BN layer (finding 4: provenance must make the exact-checkpoint /
    BN-buffer-count evidence explicit). A 20-BatchNorm2d synthetic stack (matching
    the real AudioResNet18 count) drives build_report with no checkpoint/dataset.
    """
    torch.manual_seed(0)

    class _Stack(nn.Module):
        def __init__(self, n, c=4):
            super().__init__()
            self.bns = nn.ModuleList([nn.BatchNorm2d(c) for _ in range(n)])

        def forward(self, x):
            for bn in self.bns:
                x = bn(x)
            return x

    mod = _Stack(20, c=4).eval()
    recorder = bn_drift_probe.BNInputRecorder(mod)
    with torch.no_grad():
        for _ in range(2):
            mod(torch.randn(8, 4, 3, 3))

    report = bn_drift_probe.build_report(
        ckpt_path="weights/FLAC/FLAC_EMA.ckpt",
        load_missing=[],
        load_unexpected=["diffusion_ema.x", "losses.y"],
        recorder=recorder,
        n_batches=2,
        n_samples=16,
    )

    assert report["ckpt_path"] == "weights/FLAC/FLAC_EMA.ckpt"
    assert report["load_missing"] == 0
    assert report["load_unexpected"] == 2
    assert report["n_bn_layers"] == 20
    assert report["n_batches"] == 2
    assert report["n_samples"] == 16
    assert len(report["per_layer"]) == 20
    for stats in report["per_layer"].values():
        assert set(stats) == {
            "mean_shift_max", "mean_shift_mean", "var_ratio_min", "var_ratio_max", "n_channels",
        }
        assert stats["n_channels"] == 4
