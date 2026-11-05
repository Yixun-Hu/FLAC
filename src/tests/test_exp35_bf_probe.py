"""exp_35 round 4: the B-F composition probe (``depth_canon_cond_probe.py --arm BF``, plan_orbitrir_yaw_evidence.md
r2 §5, §11 H3, r3 §12 B3): A = FA(md) without canonicalization, B = FA(rotate_d(md)) for the three quarter turns
d in {128, 256, 384} -- FA = ``invariant_conditioning(conditioner, md, device, (0, 90, 180, 270),
max_fwd_samples=64)`` under each protocol's conditioning autocast -- through the same four stages, protocols and
fixed per-item noise as the CANON probe; verdict = masks equal AND the determinism floor exact.

CPU only, on the probe tests' real tiny diffusion_cond wrapper (repo factories).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import copy
import json
import types

import numpy as np
import pytest
import torch

import eval_FLAC
import test_depth_canon_cond_probe as TP
from src.data import yaw_rotation as yr
from test_yaw_aug_training import no_flash  # noqa: F401 (fixture)

probe = TP.probe
BF_KEYS = ("d128", "d256", "d384")
ANGLES = (0.0, 90.0, 180.0, 270.0)


def _mds(n=5):
    return [TP._probe_md(i) for i in range(n)]


def _run(n=5, bs=3, module=None):
    return probe.run_probe_bf(module or TP._toy_module(), torch.zeros(n, 1, 64), _mds(n), img_w=512, seed=42,
                              batch_size=bs, latent_len=64)


def test_the_bf_offsets_are_the_three_quarter_turns():
    assert probe.BF_OFFSET_KEYS == BF_KEYS and probe.BF_ANGLES == ANGLES and probe.BF_FWD_CAP == 64
    assert probe.bf_offsets(4) == {"d128": [128] * 4, "d256": [256] * 4, "d384": [384] * 4}


def test_b_is_the_evaluators_own_fixed_rotation_and_a_is_untouched():
    mds = _mds(3)
    keep = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()} for md in mds]
    a_list, b_lists = probe.bf_pairs(mds, probe.bf_offsets(3), 512)
    assert list(b_lists) == list(BF_KEYS)
    for i, md in enumerate(mds):
        assert a_list[i] is md and all(torch.equal(md[k], keep[i][k]) for k in md if torch.is_tensor(md[k]))
    for key, deg in zip(BF_KEYS, (90.0, 180.0, 270.0)):
        want = eval_FLAC.apply_rotation_plan(mds, eval_FLAC.resolve_rotation_plan("fixed", deg, None, 42))
        for got, ref in zip(b_lists[key], want):                  # the C4 cell's own conditioning input
            assert sorted(got) == sorted(ref)
            assert all(torch.equal(got[k], ref[k]) for k in got if torch.is_tensor(got[k])), key


@pytest.mark.parametrize("name, autocast, dtype", [("fp32", False, None), ("eval", True, torch.bfloat16)])
def test_the_bf_stages_condition_through_the_frame_average(no_flash, monkeypatch, name, autocast, dtype):
    module, seen = TP._toy_module(), []
    real = yr.invariant_conditioning

    def spy(conditioner, metadata, device, angles, **kw):
        seen.append((conditioner, angles, kw, torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        return real(conditioner, metadata, device, angles, **kw)
    monkeypatch.setattr(yr, "invariant_conditioning", spy)
    mds = _mds(3)
    out = probe.run_stages(module, mds, probe.noise_bank(3, (4, 64), 42), torch.zeros(3, 1, 64),
                           probe.PROTOCOLS[name], cond_fn=probe.bf_conditioning)
    (cond, angles, kw, enabled, got_dtype), = seen
    assert cond is module.diffusion.conditioner and angles == ANGLES and kw == {"max_fwd_samples": 64}
    assert enabled is autocast and (not autocast or got_dtype == dtype)
    import contextlib
    ctx = torch.amp.autocast("cpu", dtype=dtype) if autocast else contextlib.nullcontext()
    with torch.no_grad(), probe.matmul_precision(probe.PROTOCOLS[name]["matmul_precision"]), ctx:
        want = real(module.diffusion.conditioner, mds, module.device, ANGLES, max_fwd_samples=64)
    assert all(torch.equal(out["cond"][k][0], want[k][0]) for k in want)       # the same call, the same context


def _pose_blind(conditioner, metadata, device, angles, max_fwd_samples=None):
    """An exactly invariant stand-in for the frame average: the conditioner on pose-free metadata."""
    flat = [dict(md, source=torch.zeros(3), context_poses=torch.zeros_like(md["context_poses"])) for md in metadata]
    return conditioner(flat, device)


def test_an_invariant_toy_gives_exactly_zero_at_every_stage(no_flash, monkeypatch):
    monkeypatch.setattr(yr, "invariant_conditioning", _pose_blind)
    rep = _run()
    for p in probe.PROTOCOLS:
        for key in BF_KEYS:
            assert rep["results"][p][key]["cond"]["all"]["summary"]["relL2"]["max"] == 0.0
            assert all(rep["results"][p][key][s]["summary"]["relL2"]["max"] == 0.0 for s in probe.STAGES[1:])
        assert rep["control_p99"][p] == {"d128": 0.0, "d256": 0.0, "d384": 0.0, "pooled": 0.0}
    assert rep["verdict"] == {"pass": True, "masks": True, "determinism": True}


def test_the_bf_report_has_no_off_orbit_key_and_no_canonical_tier(no_flash):
    rep = _run()
    assert list(rep["offsets"]) == list(BF_KEYS) and "random" not in rep["offsets"] and "d64" not in rep["offsets"]
    assert "tier1" not in rep and set(rep) == {"offsets", "items", "results", "pooled", "determinism", "eps_cond",
                                               "control_p99", "verdict"}
    for p in probe.PROTOCOLS:
        assert list(rep["results"][p]) == list(BF_KEYS)
        want = {k: rep["results"][p][k]["waveform"]["summary"]["relL2"]["p99"] for k in BF_KEYS}
        pooled = [v for k in BF_KEYS for v in rep["results"][p][k]["waveform"]["relL2"]]
        assert rep["control_p99"][p] == dict(want, pooled=TP.stats.paired_summary(np.array(pooled))["p99"])
        assert all(rep["determinism"][p][s]["summary"]["relL2"]["max"] == 0.0 for s in probe.STAGES[1:])
    eps = max(v for k in BF_KEYS for d in rep["results"]["fp32"][k]["cond"]["per_id"].values() for v in d["relL2"])
    assert set(rep["eps_cond"]) == {"value", "argmax"} and rep["eps_cond"]["value"] == eps   # reported, no verdict
    assert rep["verdict"] == {"pass": True, "masks": True, "determinism": True}
    assert rep["items"] == [{}] * 5


def test_a_nondeterministic_condition_fails_the_verdict(no_flash, monkeypatch):
    real, g = yr.invariant_conditioning, torch.Generator().manual_seed(5)

    def noisy(conditioner, metadata, device, angles, max_fwd_samples=None):
        out = real(conditioner, metadata, device, angles, max_fwd_samples=max_fwd_samples)
        out["source"][0] = out["source"][0] + 1e-3 * torch.randn(out["source"][0].shape, generator=g)
        return out
    monkeypatch.setattr(yr, "invariant_conditioning", noisy)
    rep = _run()
    assert rep["verdict"]["determinism"] is False and rep["verdict"]["pass"] is False and rep["verdict"]["masks"]


# --------------------------------------------------------------------------- #
# loading, the registered contract and main()
# --------------------------------------------------------------------------- #
def test_the_bf_load_never_canonicalizes(tmp_path, monkeypatch):
    cfg_path, ckpt_path = TP._toy_ckpt(tmp_path, {"use_ema": True})
    TP._capture(monkeypatch, [], stop=False)
    _, info, ckpt_load = probe.load_probe_module(cfg_path, ckpt_path, "cpu", arm="BF")
    assert (info["canonicalize_trained"], info["canonicalize_applied"], info["canonicalize_mismatch"],
            info["img_w"]) == (None, False, False, None)
    assert info["weights_source"] == "ema" and ckpt_load == eval_FLAC.read_checkpoint_exact(ckpt_path)[1]
    cfg_path, ckpt_path = TP._toy_ckpt(tmp_path, dict(TP.CANON_ON, use_ema=True))   # a CANON-trained checkpoint
    with pytest.raises(ValueError, match="TRAINED with canonicalize=True"):
        probe.load_probe_module(cfg_path, ckpt_path, "cpu", arm="BF")


def test_bf_refuses_canonicalization_before_anything_loads(monkeypatch, capsys):
    monkeypatch.setattr(probe, "load_probe_module", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(SystemExit) as err:
        probe.main(["--arm", "BF", "--ckpt-path", "/nonexistent.ckpt", "--allow-canon-mismatch"])
    assert err.value.code == 2 and "canonicalization" in capsys.readouterr().err


def test_the_registered_bf_contract_constants():
    R = probe.REGISTERED_BF
    assert (R["n_items"], R["batch_size"], R["seed"], R["device"]) == (64, 64, 42, "cuda")
    assert R["model_config"].endswith("worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json")
    assert R["model_config_sha256"] == "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328"
    assert R["dataset_config"] == probe.DEFAULT_DATASET_CONFIG
    assert R["dataset_config_sha256"] == "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55"
    assert probe.BF_DEFAULT_OUT.endswith("exp_35_orbitrir_yaw_evidence_claude/orbitrir_bf_cond_probe.json")


def _run_main_bf(tmp_path, monkeypatch, *, registered=True, diagnostic=False, extra=(), n_loader=7, bs=3):
    """main(--arm BF) on the toy: the registered contract re-pointed at the toy's files (every check runs)."""
    import hashlib
    TP._clean_git(monkeypatch)
    cfg_path, ckpt_path = TP._toy_ckpt(tmp_path, {"use_ema": True})
    cfg = dict(json.load(open(cfg_path)), sample_size=64)
    open(cfg_path, "w").write(json.dumps(cfg))
    torch.save({"state_dict": {"diffusion_ema.ema_model.model.w": torch.ones(2)}, "model_config": cfg}, ckpt_path)
    (tmp_path / "data.json").write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    if registered:
        monkeypatch.setattr(probe, "REGISTERED_BF", {
            "n_items": 5, "batch_size": bs, "seed": 42, "device": "cpu", "model_config": cfg_path,
            "model_config_sha256": sha(cfg_path), "dataset_config": str(tmp_path / "data.json"),
            "dataset_config_sha256": sha(str(tmp_path / "data.json"))})
    toy = TP._toy_module()
    monkeypatch.setattr(eval_FLAC, "create_model_from_config",
                        lambda c: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], [])))
    monkeypatch.setattr(eval_FLAC, "create_training_wrapper_from_config", lambda c, m: toy)
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config", lambda *a, **k: TP._ToyLoader(n_loader, bs))
    out = tmp_path / "bf_probe.json"
    rc = probe.main(["--arm", "BF", "--model-config", cfg_path, "--ckpt-path", ckpt_path, "--dataset-config",
                     str(tmp_path / "data.json"), "--device", "cpu", "--n-items", "5", "--batch-size", str(bs),
                     "--seed", "42", "--out", str(out)] + (["--diagnostic"] if diagnostic else []) + list(extra))
    return rc, (json.loads(out.read_text()) if out.exists() else None), out, ckpt_path


def test_bf_main_end_to_end_registered(tmp_path, monkeypatch, no_flash, capsys):
    import hashlib
    rc, rep, out, ckpt_path = _run_main_bf(tmp_path, monkeypatch)
    assert rc == 0 and rep["registered"] is True and rep["deviations"] == [] and rep["arm"] == "BF"
    prov = rep["provenance"]
    assert (prov["cond_method"], prov["frame_avg_angles"], prov["frame_avg_max_fwd_samples"]) == (
        "fa_invariant", list(ANGLES), 64)
    assert prov["ckpt_sha256"] == prov["ckpt_sha256_loaded"] == hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    assert prov["img_w"] == 512 and prov["load"]["canonicalize_applied"] is False and "random_offset_seed" not in prov
    assert prov["dirty"] == "" and prov["source_sha"] and "launch" not in prov            # finalization adds it
    assert rep["offsets"] == probe.bf_offsets(5) and [it["position"] for it in rep["items"]] == list(range(5))
    assert (tmp_path / "bf_probe.json.sha256").read_text() == \
        f"{hashlib.sha256(out.read_bytes()).hexdigest()}  bf_probe.json\n"
    text = capsys.readouterr().out
    assert "B-F composition probe -- verdict PASS" in text and "| eval | d128 |" in text


def test_a_bf_deviation_is_a_labelled_diagnostic_or_refused(tmp_path, monkeypatch, no_flash, capsys):
    sub = tmp_path / "a"
    sub.mkdir()
    rc, rep, _, _ = _run_main_bf(sub, monkeypatch, registered=False, diagnostic=True)
    assert rc == probe.DIAGNOSTIC_EXIT and rep["registered"] is False and rep["deviations"]
    sub = tmp_path / "b"
    sub.mkdir()
    monkeypatch.setattr(probe, "load_probe_module", lambda *a, **k: pytest.fail("loaded before refusing"))
    rc, rep, _, _ = _run_main_bf(sub, monkeypatch, registered=False)
    assert rc == 2 and rep is None and "not the registered" in capsys.readouterr().out
