"""exp_25 depth_canon round 3b -- the numerical-composition control and eps_cond
(``worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py``,
plan_depth_canon.md v8 §1 Tier 2 + eps_cond, §6 test 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §1/§6/§8.

CPU only. The pure helpers run on synthetic tensors; the stage runner drives a real tiny
``diffusion_cond`` wrapper built by the repo factories (``test_yaw_aug_training``'s
forward-consistent toy config); the loader is checked against ``eval_FLAC.evaluate_model``
on one toy checkpoint.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from src.tools import depth_canon_stats as stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROBE_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_25_depth_canon_claude",
                         "depth_canon_cond_probe.py")


def _load_probe():
    spec = importlib.util.spec_from_file_location("depth_canon_cond_probe", _PROBE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


# --------------------------------------------------------------------------- #
# offsets and fixed noise: deterministic from the seed
# --------------------------------------------------------------------------- #
def test_offsets_are_the_registered_set_and_deterministic_from_the_seed():
    a, b, c = (probe.draw_offsets(64, 512, s) for s in (42, 42, 43))
    assert list(a) == list(probe.OFFSET_KEYS) == ["d64", "d128", "d256", "d384", "random"]
    for key, d in zip(probe.OFFSET_KEYS[:4], (64, 128, 256, 384)):
        assert a[key] == [d] * 64
    assert a == b and a["random"] != c["random"]
    assert all(isinstance(d, int) and 1 <= d <= 511 for d in a["random"])


def test_random_offsets_cover_exactly_one_to_w_minus_one():
    assert set(probe.draw_offsets(20000, 512, 7)["random"]) == set(range(1, 512))


@pytest.mark.parametrize("bad", [dict(n_items=0), dict(img_w=384), dict(img_w=True), dict(n_items=2.0),
                                 dict(seed=-1)])
def test_draw_offsets_refuses_bad_arguments(bad):
    kw = dict(n_items=4, img_w=512, seed=1)
    kw.update(bad)
    with pytest.raises(ValueError):
        probe.draw_offsets(**kw)


def test_noise_is_one_seeded_tensor_per_item_independent_of_the_item_count():
    five = probe.noise_bank(5, (4, 16), seed=42)
    assert five.shape == (5, 4, 16) and five.dtype == torch.float32
    assert torch.equal(probe.noise_bank(3, (4, 16), seed=42), five[:3])
    assert torch.equal(probe.noise_bank(5, (4, 16), seed=42), five)
    assert not torch.equal(probe.noise_bank(5, (4, 16), seed=43), five)
    assert len({tuple(r.flatten().tolist()[:4]) for r in five}) == 5


# --------------------------------------------------------------------------- #
# stagewise comparison (pinned denominators via depth_canon_stats)
# --------------------------------------------------------------------------- #
def test_identical_tensors_compare_to_exactly_zero():
    a = torch.randn(4, 3, 8, generator=torch.Generator().manual_seed(0))
    r = probe.stage_rel(a, a.clone())
    assert r["relL2"].tolist() == [0.0] * 4 and r["relLinf"].tolist() == [0.0] * 4


def test_rel_uses_the_second_argument_as_reference_with_the_floor():
    a, b = torch.ones(2, 3), torch.zeros(2, 3)
    r = probe.stage_rel(a, b)
    assert r["relL2"] == pytest.approx([3 ** 0.5 / stats.REL_FLOOR] * 2)
    assert r["relLinf"] == pytest.approx([1.0 / stats.REL_FLOOR] * 2)
    r2 = probe.stage_rel(torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 1.0]]))
    assert r2["relL2"] == pytest.approx([1 / 5 ** 0.5])


def test_rel_is_scale_invariant_and_per_item():
    g = torch.Generator().manual_seed(1)
    a, b = (torch.randn(3, 2, 5, generator=g, dtype=torch.float64) for _ in range(2))
    r, rk = probe.stage_rel(a, b), probe.stage_rel(3.7 * a, 3.7 * b)   # float64: no rescale rounding
    assert np.allclose(r["relL2"], rk["relL2"], rtol=1e-12) and r["relL2"].shape == (3,)
    want = stats.rel_l2(a.reshape(3, -1).double().numpy(), b.reshape(3, -1).double().numpy())
    assert np.array_equal(r["relL2"], want)


def test_bf16_stages_are_compared_in_float64():
    a = torch.randn(2, 6).to(torch.bfloat16)
    r = probe.stage_rel(a, a.float().double() + 1.0)
    assert r["relL2"].dtype == np.float64 and r["relL2"].shape == (2,)


def test_conditioning_comparison_per_id_masks_and_all():
    g = torch.Generator().manual_seed(2)
    x, y = torch.randn(3, 4, 8, generator=g), torch.randn(3, 1, 8, generator=g)
    m = torch.ones(3, 4, dtype=torch.bool)
    a = {"ctx": [x, m], "src": [y, torch.ones(3, 1)]}
    same = probe.compare_cond(a, {"ctx": [x.clone(), m.clone()], "src": [y.clone(), torch.ones(3, 1)]})
    assert same["masks_equal"] is True and same["ids"] == ["ctx", "src"]
    assert all(v.tolist() == [0.0] * 3 for k in ("ctx", "src", "all") for v in same[k].values())
    m2 = m.clone()
    m2[1, 2] = False
    y2 = y.clone()
    y2[2] += 1.0
    diff = probe.compare_cond(a, {"ctx": [x, m2], "src": [y2, torch.ones(3, 1)]})
    assert diff["masks_equal"] is False and diff["ctx"]["relL2"].tolist() == [0.0] * 3
    assert diff["src"]["relL2"][2] > 0 and diff["all"]["relL2"][2] > 0
    flat = lambda c: torch.cat([c["ctx"][0].reshape(3, -1), c["src"][0].reshape(3, -1)], 1)  # noqa: E731
    assert np.array_equal(diff["all"]["relL2"], probe.stage_rel(     # (test, reference)
        flat(a), flat({"ctx": [x], "src": [y2]}))["relL2"])
    with pytest.raises(ValueError, match="conditioner ids"):
        probe.compare_cond(a, {"ctx": [x, m]})


def test_summaries_are_depth_canon_stats_paired_summary():
    v = np.array([0.0, 1e-7, 3e-6, 2e-5])
    assert probe.summarize({"relL2": v, "relLinf": 2 * v}) == {
        "relL2": stats.paired_summary(v), "relLinf": stats.paired_summary(2 * v)}


# --------------------------------------------------------------------------- #
# A = canon(md), B = canon(rot_d(md)) and the Tier-1 record of the probe items
# --------------------------------------------------------------------------- #
from src.data import yaw_rotation as yr  # noqa: E402
from test_depth_canon_eval import CLASSES, W16, _eval_md  # noqa: E402  (16 x 32 AR clouds)


def _items(n=5):
    return [_eval_md(i, CLASSES[i % len(CLASSES)]) for i in range(n)]


def test_canonical_pairs_are_the_operator_on_md_and_on_its_rotation():
    mds = _items()
    offsets = {"d3": [3] * 5, "d16": [16] * 5, "random": [1, 7, 31, 9, 22]}
    a_list, b_lists, tier1 = probe.canonical_pairs(mds, offsets, W16)
    assert list(b_lists) == list(offsets) and len(a_list) == len(tier1) == 5
    for i, md in enumerate(mds):
        ref, dj, tie_class = yr.canonicalize_scene_metadata(md, W16)
        assert all(torch.equal(a_list[i][k], ref[k]) for k in ("depth",) + yr.POSE_KEYS)
        assert tier1[i]["dj_A"] == dj and tier1[i]["class_A"] == tie_class
        assert tier1[i]["depth_sha_A"] == yr.canonical_depth_sha256(ref["depth"])
        for key, ds in offsets.items():
            rot = yr.rotate_scene_metadata(md, yr.offsets_to_radians([ds[i]], W16)[0], W16)
            want, dj_b, cls_b = yr.canonicalize_scene_metadata(rot, W16)
            assert all(torch.equal(b_lists[key][i][k], want[k]) for k in ("depth",) + yr.POSE_KEYS)
            assert tier1[i]["offsets"][key] == {"d": ds[i], "dj_B": dj_b, "class_B": cls_b,
                                                "depth_sha_equal": True, "class_equal": True,
                                                "dj_corotates": dj_b == (dj - ds[i]) % W16}
            assert tier1[i]["offsets"][key]["dj_corotates"] is True
    assert all(md["context_audio"] is a["context_audio"] for md, a in zip(mds, a_list))


def test_tier1_records_a_mismatch_it_did_not_create(monkeypatch):
    real = yr.canonical_depth_sha256
    calls = []

    def lying(depth):
        calls.append(1)
        return real(depth) if len(calls) % 2 else "0" * 64
    monkeypatch.setattr(yr, "canonical_depth_sha256", lying)
    _, _, tier1 = probe.canonical_pairs(_items(1), {"d3": [3]}, W16)
    assert tier1[0]["offsets"]["d3"]["depth_sha_equal"] is False


def test_canonical_pairs_refuses_misaligned_offsets_and_periodic_items():
    with pytest.raises(ValueError, match="one offset per item"):
        probe.canonical_pairs(_items(2), {"d3": [3]}, W16)
    with pytest.raises(yr.StabilizedTieError):
        probe.canonical_pairs([_eval_md(0, period=16)], {"d3": [3]}, W16)


# --------------------------------------------------------------------------- #
# the stage runner on a REAL tiny diffusion_cond wrapper (repo factories, CPU)
# --------------------------------------------------------------------------- #
from test_yaw_aug_training import _base_config, no_flash  # noqa: E402,F401  (fixture)


def _toy_module(seed=0):
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config
    torch.manual_seed(seed)
    cfg = _base_config()
    module = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
    return module.eval().requires_grad_(False)


def _toy_inputs(n=3):
    a_list, b_lists, _ = probe.canonical_pairs(_items(n), {"d5": [5] * n}, W16)
    return a_list, b_lists["d5"], probe.noise_bank(n, (4, 64), seed=42), torch.zeros(n, 1, 64)


def test_run_stages_is_one_eval_euler_step_with_the_given_noise(no_flash):
    module = _toy_module()
    a_list, _, noise, reals = _toy_inputs()
    keep = noise.clone()
    out = probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["fp32"])
    assert torch.equal(noise, keep)                                       # not mutated
    with torch.no_grad():
        cond = module.diffusion.conditioner(a_list, module.device)
        v = module.diffusion.model(noise, torch.ones(3), **module.diffusion.get_conditioning_inputs(cond),
                                   cfg_scale=1.0, batch_cfg=True)
    assert sorted(out["cond"]) == sorted(cond)
    assert all(torch.equal(out["cond"][k][0], cond[k][0]) for k in cond)
    assert torch.equal(out["velocity"], v) and torch.equal(out["latent"], noise - v)
    assert torch.equal(out["waveform"], out["latent"].clamp(-1.0, 1.0))  # no pretransform here


def test_run_stages_hands_the_sampler_exactly_the_given_noise(no_flash, monkeypatch):
    import src.inference.sampling as sampling
    module, seen = _toy_module(), []
    real = sampling.sample_discrete_euler
    monkeypatch.setattr(sampling, "sample_discrete_euler",
                        lambda model, x, steps, **kw: seen.append((x, steps, kw)) or real(model, x, steps, **kw))
    a_list, b_list, noise, reals = _toy_inputs()
    probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["eval"])
    probe.run_stages(module, b_list, noise, reals, probe.PROTOCOLS["eval"])
    assert [torch.equal(x, noise) for x, _, _ in seen] == [True, True]
    assert all(s == 1 and kw["cfg_scale"] == 1.0 and kw["batch_cfg"] is True and kw["disable_tqdm"]
               and kw["dist_shift"] is module.diffusion.dist_shift for _, s, kw in seen)


@pytest.mark.parametrize("name, autocast, dtype, precision", [
    ("fp32", False, None, "highest"), ("eval", True, torch.bfloat16, "medium")])
def test_protocols_set_cond_autocast_and_matmul_precision(no_flash, name, autocast, dtype, precision):
    module, seen = _toy_module(), []
    hook = module.diffusion.conditioner.register_forward_pre_hook(lambda m, a: seen.append(
        (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), torch.get_float32_matmul_precision())))
    before = torch.get_float32_matmul_precision()
    a_list, _, noise, reals = _toy_inputs()
    probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS[name])
    hook.remove()
    assert seen[0][0] is autocast and seen[0][2] == precision and (not autocast or seen[0][1] == dtype)
    assert torch.get_float32_matmul_precision() == before and not torch.is_autocast_enabled("cpu")


def test_identical_inputs_give_exactly_zero_at_every_stage(no_flash):
    module = _toy_module()
    a_list, b_list, noise, reals = _toy_inputs()
    s1, s2 = (probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["fp32"]) for _ in range(2))
    cmp = probe.compare_stages(s1, s2)
    assert list(cmp) == list(probe.STAGES)
    assert cmp["cond"]["all"]["relL2"].max() == 0.0 and cmp["cond"]["masks_equal"] is True
    assert all(cmp[s]["relL2"].max() == 0.0 for s in probe.STAGES[1:])
    rot = probe.compare_stages(probe.run_stages(module, b_list, noise, reals, probe.PROTOCOLS["fp32"]), s1)
    assert rot["cond"]["all"]["relL2"].max() < 1e-5   # float32 pose composition only


def test_a_non_rectified_flow_objective_is_refused(no_flash):
    module = _toy_module()
    module.diffusion.model.diffusion_objective = "v"
    a_list, _, noise, reals = _toy_inputs()
    with pytest.raises(ValueError, match="rectified_flow"):
        probe.run_stages(module, a_list, noise, reals, probe.PROTOCOLS["eval"])


# --------------------------------------------------------------------------- #
# loading: the same checkpoint path as eval_FLAC.evaluate_model
# --------------------------------------------------------------------------- #
import copy  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402

import eval_FLAC  # noqa: E402
from test_depth_canon_eval import _model_cfg  # noqa: E402

CANON_ON = {"canonicalize": {"enabled": True, "img_w": 512}}


class _Stop(Exception):
    pass


def _toy_ckpt(tmp_path, training, embedded="same"):
    cfg = _model_cfg(training)
    cfg.update(model_type="diffusion_cond", sample_size=16, sample_rate=22050, audio_channels=1)
    (tmp_path / "model.json").write_text(json.dumps(cfg))
    sd = {"diffusion.model.w": torch.ones(2), "diffusion.conditioner.c": torch.zeros(1),
          "diffusion_ema.ema_model.model.w": torch.full((2,), 3.0), "losses.x": torch.ones(1),
          "diffusion.pretransform.diffusion.k": torch.ones(1)}   # nested 'diffusion.' (replace-all)
    ckpt = {"state_dict": sd}
    if embedded is not None:
        ckpt["model_config"] = copy.deepcopy(cfg) if embedded == "same" else embedded
    torch.save(ckpt, str(tmp_path / "toy.ckpt"))
    return str(tmp_path / "model.json"), str(tmp_path / "toy.ckpt")


def _capture(monkeypatch, seen, stop=True):
    def model_factory(cfg):
        seen.append(("model_cfg", copy.deepcopy(cfg)))
        return types.SimpleNamespace(load_state_dict=lambda sd, strict=False: seen.append(
            ("sd", {k: v.clone() for k, v in sd.items()}, strict)) or ([], []))

    def wrapper_factory(cfg, model):
        seen.append(("wrapper_cfg", copy.deepcopy(cfg)))
        if stop:
            raise _Stop
        module = types.SimpleNamespace(device="cpu")
        module.eval = module.requires_grad_ = module.to = lambda *a: module
        return module
    monkeypatch.setattr(eval_FLAC, "create_model_from_config", model_factory)
    monkeypatch.setattr(eval_FLAC, "create_training_wrapper_from_config", wrapper_factory)


@pytest.mark.parametrize("use_ema", [True, False])
def test_the_probe_loads_a_checkpoint_exactly_as_evaluate_model(tmp_path, monkeypatch, use_ema):
    cfg_path, ckpt_path = _toy_ckpt(tmp_path, dict(CANON_ON, use_ema=use_ema))
    via_eval, via_probe = [], []
    _capture(monkeypatch, via_eval)
    with pytest.raises(_Stop):
        eval_FLAC.evaluate_model(cfg_path, str(tmp_path / "unused.json"), ckpt_path, steps=1,
                                 cfg_scale=1.0, device="cpu", canonicalize=True)
    _capture(monkeypatch, via_probe)
    with pytest.raises(_Stop):
        probe.load_probe_module(cfg_path, ckpt_path, "cpu")
    assert [s[0] for s in via_eval] == [s[0] for s in via_probe] == ["model_cfg", "sd", "wrapper_cfg"]
    assert via_eval[0][1] == via_probe[0][1] and via_eval[2][1] == via_probe[2][1]
    sd_e, sd_p = via_eval[1][1], via_probe[1][1]
    assert list(sd_e) == list(sd_p) and all(torch.equal(sd_e[k], sd_p[k]) for k in sd_e)
    assert via_eval[1][2] is via_probe[1][2] is False
    assert via_probe[2][1]["training"]["use_ema"] is False
    assert ("model.model.w" in sd_p) is use_ema and "pretransform.k" in sd_p


def test_load_reports_the_binding_and_the_weights_source(tmp_path, monkeypatch):
    cfg_path, ckpt_path = _toy_ckpt(tmp_path, dict(CANON_ON, use_ema=True))
    _capture(monkeypatch, [], stop=False)
    module, info = probe.load_probe_module(cfg_path, ckpt_path, "cpu")
    assert module.device == "cpu" and info["weights_source"] == "ema"
    assert (info["canonicalize_trained"], info["canonicalize_applied"], info["canonicalize_mismatch"],
            info["img_w"]) == (True, True, False, 512)
    cfg = json.load(open(cfg_path))
    assert info["model_config_sha256_file"] == info["model_config_sha256_embedded"] == \
        eval_FLAC.model_config_sha256(cfg)


def test_a_non_canonical_checkpoint_is_refused_before_the_model_unless_allowed(tmp_path, monkeypatch):
    cfg_path, ckpt_path = _toy_ckpt(tmp_path, {"use_ema": False})
    monkeypatch.setattr(eval_FLAC, "create_model_from_config", lambda cfg: pytest.fail("model built"))
    with pytest.raises(ValueError, match="TRAINED with canonicalize=False"):
        probe.load_probe_module(cfg_path, ckpt_path, "cpu")
    _capture(monkeypatch, [], stop=False)
    _, info = probe.load_probe_module(cfg_path, ckpt_path, "cpu", allow_canon_mismatch=True)
    assert (info["canonicalize_trained"], info["canonicalize_applied"], info["canonicalize_mismatch"]) == \
        (False, True, True)


def test_an_are_checkpoint_is_refused(tmp_path, monkeypatch):
    cfg_path, ckpt_path = _toy_ckpt(tmp_path, dict(CANON_ON, use_ema=False))
    monkeypatch.setattr(eval_FLAC, "resolve_are_from_checkpoint", lambda *a: (0.5, "config", {}))
    monkeypatch.setattr(eval_FLAC, "create_model_from_config", lambda cfg: pytest.fail("model built"))
    with pytest.raises(ValueError, match="ARE"):
        probe.load_probe_module(cfg_path, ckpt_path, "cpu")


# --------------------------------------------------------------------------- #
# the probe loop: real tiny wrapper, real stages, items at W = 512
# --------------------------------------------------------------------------- #
from test_depth_canon import _cloud, _scalar_depth  # noqa: E402


def _probe_md(i):
    s = torch.tensor([1.3, -0.7 + 0.1 * i, 0.2])
    ctx = torch.tensor([[0.9, 1.1, -0.3], [-1.2, 0.4 + 0.05 * i, 0.1]])
    peaks = [(1, (300 + 37 * i) % 512)] + ([(2, (300 + 37 * i) % 512)] if i % 3 == 1 else [])
    return {"depth": _cloud(_scalar_depth(4, 512, seed=200 + i, peaks=peaks)), "source": s,
            "source_vit": s[None], "context_poses": ctx, "context_poses_vit": ctx.clone(),
            "context_audio": torch.zeros(2, 1, 8), "padding_mask": torch.ones(64), "scene": "Toy",
            "idx": i, "relpath": f"single_channel_ir_1/Toy/Toy_idx_0/S00{i}_R001_hybrid_IR.wav"}


def _run_probe(n=5, bs=3):
    mds = [_probe_md(i) for i in range(n)]
    return probe.run_probe(_toy_module(), torch.zeros(n, 1, 64), mds, img_w=512, seed=42,
                           batch_size=bs, latent_len=64)


def test_run_probe_results_eps_cond_and_control(no_flash):
    rep = _run_probe()
    assert rep["offsets"] == probe.draw_offsets(5, 512, 42)
    for p in probe.PROTOCOLS:
        for key in probe.OFFSET_KEYS:
            res = rep["results"][p][key]
            assert res["waveform"]["summary"]["relL2"]["n"] == 5 and len(res["waveform"]["relL2"]) == 5
            assert set(res["cond"]["ids"]) == {"source", "context_poses"} and res["cond"]["masks_equal"]
            assert res["latent"]["summary"] == probe.summarize(
                {k: np.array(res["latent"][k]) for k in ("relL2", "relLinf")})
        assert all(max(rep["determinism"][p][s]["relL2"]) == 0.0 for s in probe.STAGES[1:])
    eps = max(v for key in probe.OFFSET_KEYS for i in rep["results"]["fp32"][key]["cond"]["ids"]
              for v in rep["results"]["fp32"][key]["cond"]["per_id"][i]["relL2"])
    assert rep["eps_cond"]["value"] == eps and rep["eps_cond"]["threshold"] == 1e-5
    assert rep["eps_cond"]["pass"] is (eps <= 1e-5) is True
    pooled = [v for key in probe.OFFSET_KEYS for v in rep["results"]["eval"][key]["waveform"]["relL2"]]
    assert rep["control_p99"]["eval"]["pooled"] == stats.paired_summary(np.array(pooled))["p99"]
    assert rep["control_p99"]["eval"]["d128"] == rep["results"]["eval"]["d128"]["waveform"]["summary"]["relL2"]["p99"]
    assert rep["tier1"] == {"n_pairs": 25, "depth_sha_equal": 25, "class_equal": 25, "dj_corotates": 25}
    assert [it["class_A"] for it in rep["items"]] == ["none", "same_column", "none", "none", "same_column"]
    assert rep["verdict"] == {"pass": True, "eps_cond": True, "tier1": True, "masks": True}


def test_one_noise_tensor_per_item_is_reused_for_every_pass(monkeypatch, no_flash):
    import src.inference.sampling as sampling
    seen, real = [], sampling.sample_discrete_euler
    monkeypatch.setattr(sampling, "sample_discrete_euler",
                        lambda model, x, steps, **kw: seen.append(x.clone()) or real(model, x, steps, **kw))
    _run_probe()
    bank = probe.noise_bank(5, (4, 64), 42)
    per_batch = 2 * (2 + len(probe.OFFSET_KEYS))          # protocols x (A, A again, five B)
    assert len(seen) == 2 * per_batch                      # batches of 3 + 2 items
    assert all(torch.equal(x, bank[:3]) for x in seen[:per_batch])
    assert all(torch.equal(x, bank[3:]) for x in seen[per_batch:])


def test_a_tier1_mismatch_fails_the_verdict(monkeypatch, no_flash):
    real, calls = yr.canonical_depth_sha256, []
    monkeypatch.setattr(yr, "canonical_depth_sha256",
                        lambda d: calls.append(1) or (real(d) if len(calls) != 3 else "0" * 64))
    rep = _run_probe()
    assert rep["tier1"]["depth_sha_equal"] == 24 and rep["verdict"]["tier1"] is False
    assert rep["verdict"]["pass"] is False


# --------------------------------------------------------------------------- #
# loader items in order, and the markdown summary
# --------------------------------------------------------------------------- #
class _ToyLoader:
    def __init__(self, n, bs):
        mds = [_probe_md(i) for i in range(n)]
        self.batches = [[torch.zeros(len(mds[j:j + bs]), 1, 64), tuple(mds[j:j + bs])]
                        for j in range(0, n, bs)]

    def __iter__(self):
        return iter(self.batches)


def test_collect_items_takes_the_first_n_in_loader_order():
    reals, mds = probe.collect_items(_ToyLoader(7, 3), 5)
    assert reals.shape == (5, 1, 64) and [md["idx"] for md in mds] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="only 4 items"):
        probe.collect_items(_ToyLoader(4, 3), 5)


def test_markdown_summary_reports_every_cell_and_the_verdict(no_flash):
    text = probe.markdown_summary(_run_probe())
    assert text.startswith("## depth_canon composition probe -- verdict PASS")
    assert "eps_cond (fp32 conditioning" in text and "Tier 1: 25/25" in text
    assert all(f"| {p} | {k} |" in text for p in probe.PROTOCOLS for k in probe.OFFSET_KEYS)
    assert "control p99 (eval, waveform relL2): d64" in text and "determinism floor (fp32" in text
