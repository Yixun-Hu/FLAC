"""exp_46 D1 probe (open item 13 / Codex code r3 finding 3): collected only when named on a pytest command line (not a
test_*.py file). Run in the same process right after an upstream sampler test, it checks that lightning's GLOBAL rank --
the one shared rank_zero_only callable that lightning_fabric and pytorch_lightning re-export, and the reference the
training wrapper's rank-0 banner reads -- is back at its incoming value 0 (what E7 / E8's monkeypatch teardown restores).

Written by the exp_46 Coder (Claude Opus 5.5).
"""


def test_every_shared_rank_reference_is_back_at_zero_after_the_sampler_test():
    import lightning_fabric.utilities.rank_zero as fabric_rz
    import pytorch_lightning.utilities.rank_zero as pl_rz
    from src.training import diffusion as tdiff
    refs = {"lightning_fabric": fabric_rz.rank_zero_only, "pytorch_lightning": pl_rz.rank_zero_only,
            "the banner (src.training.diffusion)": tdiff.rank_zero_only}
    assert len({id(f) for f in refs.values()}) == 1, "the modules no longer share one callable"
    assert {name: getattr(f, "rank", None) for name, f in refs.items()} == {name: 0 for name in refs}
