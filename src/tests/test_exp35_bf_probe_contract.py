"""exp_35 round 4b: the collector side of the B-F composition control (plan_orbitrir_yaw_evidence.md r2 §5
``PROBE_CONTRACT_BF``, §11 H3, r3 §12 B3): the non-logged finalization (``--finalize-bf-probe``) hashes the closed
probe log, writes the launch provenance into ``orbitrir_bf_cond_probe.json``, regenerates its ``.sha256`` and
validates the artifact; ``load_control_bf`` returns the Tier 2-C4 reference only for a valid, registered control
of THIS campaign.

The artifacts are built by the probe's OWN ``run_probe_bf`` / ``build_report`` / ``write_report`` on the CPU toy
stack; the contract is re-pointed at the toy's cardinalities (5 items, batch 3, K = 2 contexts).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import copy
import functools
import hashlib
import json
import os
from unittest import mock

import pytest
import torch

import eval_FLAC
import test_depth_canon_collect as T
import test_depth_canon_cond_probe as TP

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
DIGEST = "c" * 64
TARGET = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py"
BF_SHA = MANIFEST["arms"]["BF"]["ckpt"]["sha256"]
BF_BYTES = 723_922_667


@functools.lru_cache(maxsize=1)
def _toy_run():
    with mock.patch("src.models.transformer.flash_attn_func", None):
        mds = [TP._probe_md(i) for i in range(5)]
        rep = TP.probe.run_probe_bf(TP._toy_module(), torch.zeros(5, 1, 64), mds, img_w=512, seed=42, batch_size=3,
                                    latent_len=64)
    stream = eval_FLAC.RotationStream()
    for md in mds:
        stream.record(md, None, 512)
    return rep, stream


def _prov(pin=PIN):
    """What main(--arm BF) records for a registered run of THIS campaign (toy cardinalities)."""
    bf = MANIFEST["arms"]["BF"]
    ds = next(c for c in MANIFEST["cells"] if c["split"] == "unseen" and c["K"] == 8)
    digest = C.model_config_digest(next(c for c in MANIFEST["cells"] if c["arm"] == "BF"))
    return {"ckpt_path": "outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt", "ckpt_sha256": BF_SHA,
            "ckpt_sha256_loaded": BF_SHA, "ckpt_bytes_loaded": BF_BYTES,
            "model_config": os.path.join(T.REPO, bf["model_config"]), "model_config_file_sha256": bf["model_config_sha256"],
            "dataset_config": ds["dataset_config"], "dataset_config_sha256": ds["dataset_config_sha256"],
            "split_json": "data/AR/unseen_eval.json",
            "split_json_sha256": eval_FLAC._file_sha256(os.path.join(T.REPO, "data/AR/unseen_eval.json")),
            "source_sha": pin, "dirty": "", "device": "cuda", "torch": torch.__version__, "cuda_device": "toy",
            "seed": 42, "noise": "per item", "n_items": 5, "batch_size": 3, "num_workers": 4, "img_w": 512,
            "latent_shape": [4, 64], "protocols": copy.deepcopy(TP.probe.PROTOCOLS),
            "load": {"weights_source": "ema", "canonicalize_trained": None, "canonicalize_applied": False,
                     "canonicalize_mismatch": False, "img_w": None, "model_config_sha256_file": digest,
                     "model_config_sha256_embedded": digest},
            "allow_canon_mismatch": False, "cond_method": "fa_invariant",
            "frame_avg_angles": [0.0, 90.0, 180.0, 270.0], "frame_avg_max_fwd_samples": 64}


def write_bf_probe(tmp, monkeypatch, mutate=None, registered=True, pin=PIN):
    """The probe's own report + sidecar, as main(--arm BF) writes it (before finalization)."""
    rep, stream = _toy_run()
    monkeypatch.setattr(C, "PROBE_CONTRACT_BF", dict(C.PROBE_CONTRACT_BF, n_items=5, batch_size=3, K=2))
    report = TP.probe.build_report(registered, [], _prov(pin), stream, copy.deepcopy(rep), arm="BF")
    if mutate:
        mutate(report)
    path = os.path.join(str(tmp), "orbitrir_bf_cond_probe.json")
    TP.probe.write_report(report, path)
    return path


def write_log(tmp, text=None):
    path = os.path.join(str(tmp), "orbitrir_bf_cond_probe.log")
    with open(path, "w") as f:
        f.write(text if text is not None else T.backend_log(TARGET) + "## B-F composition probe -- verdict PASS\n")
    return path


def _finalize(tmp, monkeypatch, **kw):
    path = write_bf_probe(tmp, monkeypatch, **{k: v for k, v in kw.items() if k in ("mutate", "registered", "pin")})
    log = kw.get("log") or write_log(tmp)
    return path, log, C.finalize_bf_probe(path, log, campaign_digest=DIGEST, base_dir=str(tmp))


def _load(path, tmp, receipts=None, digest=DIGEST, pin=PIN):
    return C.load_control_bf(path, pin=pin, campaign_digest=digest, manifest=MANIFEST, base_dir=str(tmp),
                             bf_receipts=receipts)


def test_the_bf_contract_mirrors_the_probe():
    P, R = C.PROBE_CONTRACT_BF, TP.probe.REGISTERED_BF
    assert {k: P[k] for k in ("n_items", "batch_size", "seed", "device")} == {
        k: R[k] for k in ("n_items", "batch_size", "seed", "device")}
    assert tuple(P["offset_keys"]) == TP.probe.BF_OFFSET_KEYS and P["protocols"] == TP.probe.PROTOCOLS
    assert P["frame_avg_angles"] == list(TP.probe.BF_ANGLES) and P["frame_avg_max_fwd_samples"] == TP.probe.BF_FWD_CAP
    assert (P["K"], P["config_K"], P["img_w"], P["ckpt_bytes"]) == (8, 8, 512, BF_BYTES) and P["target"] == TARGET
    assert C.BF_CONTROL_KEYS == {90: "d128", 180: "d256", 270: "d384"}


def test_finalization_writes_the_launch_provenance_and_a_fresh_sidecar(tmp_path, monkeypatch):
    path, log, out = _finalize(tmp_path, monkeypatch)
    raw = open(path, "rb").read()
    rep = json.loads(raw)
    with open(log, "rb") as f:
        log_sha = hashlib.sha256(f.read()).hexdigest()
    assert rep["provenance"]["launch"] == {
        "attention_backend": "sdpa", "overlay_sha256": C.NO_FLASH["overlay_sha256"],
        "launcher_sha256": C.NO_FLASH["launcher_sha256"], "campaign_digest": DIGEST,
        "log_path": log, "log_sha256": log_sha}
    assert open(path + ".sha256").read() == f"{hashlib.sha256(raw).hexdigest()}  orbitrir_bf_cond_probe.json\n"
    assert raw.decode() == json.dumps(rep, indent=1) + "\n" and out == path
    control, reasons, eps = _load(path, tmp_path)
    rep0 = _toy_run()[0]
    assert reasons == [] and control == {k: float(v) for k, v in rep0["control_p99"]["eval"].items()}
    assert eps == {"value": rep0["eps_cond"]["value"], "argmax": rep0["eps_cond"]["argmax"], "verdict": None}


@pytest.mark.parametrize("bad, needle", [
    (lambda t: T.backend_log("eval_FLAC.py"), "target"),                   # a cell log is no probe log
    (lambda t: "", "sentinel"),
    (lambda t: "No module named 'flash_attn'\n" + T.backend_log(TARGET), "flash_attn"),
], ids=["cell_log", "empty", "flash_missing"])
def test_finalization_refuses_a_log_without_the_backend_evidence(tmp_path, monkeypatch, bad, needle):
    path = write_bf_probe(tmp_path, monkeypatch)
    before = open(path, "rb").read()
    with pytest.raises(ValueError, match=needle):
        C.finalize_bf_probe(path, write_log(tmp_path, bad(tmp_path)), campaign_digest=DIGEST, base_dir=str(tmp_path))
    assert open(path, "rb").read() == before                               # nothing written


def test_finalization_refuses_a_second_pass_or_a_tampered_probe_output(tmp_path, monkeypatch):
    path, log, _ = _finalize(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="already finalized"):
        C.finalize_bf_probe(path, log, campaign_digest=DIGEST, base_dir=str(tmp_path))
    path = write_bf_probe(tmp_path, monkeypatch)
    open(path, "a").write(" ")                                              # edited after the probe wrote it
    with pytest.raises(ValueError, match="sidecar"):
        C.finalize_bf_probe(path, write_log(tmp_path), campaign_digest=DIGEST, base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="campaign digest"):
        C.finalize_bf_probe(write_bf_probe(tmp_path, monkeypatch), log, campaign_digest="nope", base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="another arm"):                    # a CANON report is not finalized
        C.finalize_bf_probe(write_bf_probe(tmp_path, monkeypatch, mutate=lambda r: r.update(arm="CANON")), log,
                            campaign_digest=DIGEST, base_dir=str(tmp_path))


def _flip(r, *path):
    d = r
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = not d[path[-1]]


EDITS = {   # mutation before finalization -> (the refusal names it)
    "unregistered": (dict(registered=False), "registered"),
    "deviation": (dict(mutate=lambda r: r["deviations"].append("x")), "deviations"),
    "other_pin": (dict(pin="b" * 40), "source_sha"),
    "dirty": (dict(mutate=lambda r: r["provenance"].update(dirty=" M src/x.py")), "dirty"),
    "other_checkpoint": (dict(mutate=lambda r: r["provenance"].update(ckpt_sha256="1" * 64)), "ckpt_sha256"),
    "other_loaded_bytes": (dict(mutate=lambda r: r["provenance"].update(ckpt_bytes_loaded=BF_BYTES + 1)),
                           "ckpt_bytes_loaded"),
    "device": (dict(mutate=lambda r: r["provenance"].update(device="cpu")), "device"),
    "seed": (dict(mutate=lambda r: r["provenance"].update(seed=7)), "seed"),
    "n_items": (dict(mutate=lambda r: r["provenance"].update(n_items=4)), "n_items"),
    "protocols": (dict(mutate=lambda r: r["provenance"]["protocols"]["eval"].update(cond_autocast="off")), "protocols"),
    "cond_method": (dict(mutate=lambda r: r["provenance"].update(cond_method="vanilla")), "cond_method"),
    "angles": (dict(mutate=lambda r: r["provenance"].update(frame_avg_angles=[0, 90, 180, 270])), "frame_avg_angles"),
    "cap": (dict(mutate=lambda r: r["provenance"].update(frame_avg_max_fwd_samples=32)), "frame_avg_max_fwd_samples"),
    "model_config": (dict(mutate=lambda r: r["provenance"].update(model_config_file_sha256="0" * 64)),
                     "model_config_file_sha256"),
    "dataset": (dict(mutate=lambda r: r["provenance"].update(dataset_config_sha256="0" * 64)), "dataset_config_sha256"),
    "canonicalized": (dict(mutate=lambda r: r["provenance"]["load"].update(canonicalize_applied=True)), "load"),
    "random_offsets_recorded": (dict(mutate=lambda r: r["provenance"].update(random_offset_seed=43)), "provenance"),
    "a_d64_key": (dict(mutate=lambda r: r["offsets"].update(d64=[64] * 5)), "offsets"),
    "a_random_key": (dict(mutate=lambda r: r["offsets"].update(random=[7] * 5)), "offsets"),
    "item_dropped": (dict(mutate=lambda r: r["items"].pop()), "items"),
    "stream_hash": (dict(mutate=lambda r: r.update(stream_input_hash="0" * 64)), "stream_input_hash"),
    "control_inconsistent": (dict(mutate=lambda r: r["control_p99"]["eval"].update(d128=0.5)), "control_p99"),
    "pooled_inconsistent": (dict(mutate=lambda r: r["control_p99"]["eval"].update(pooled=0.5)), "control_p99"),
    "eps_inconsistent": (dict(mutate=lambda r: r["eps_cond"].update(value=1.0)), "eps_cond"),
    "eps_with_a_verdict": (dict(mutate=lambda r: r["eps_cond"].update({"pass": True})), "eps_cond"),
    "verdict_inconsistent": (dict(mutate=lambda r: _flip(r, "verdict", "masks")), "verdict"),
    "results_missing_a_stage": (dict(mutate=lambda r: r["results"]["eval"]["d256"].pop("latent")), "results"),
}


@pytest.mark.parametrize("name", list(EDITS))
def test_an_unbound_or_inconsistent_bf_probe_is_not_a_control(tmp_path, monkeypatch, name):
    kw, needle = EDITS[name]
    path, _, _ = _finalize(tmp_path, monkeypatch, **kw)
    control, reasons, eps = _load(path, tmp_path)
    assert control is None and eps is None and any(needle in x for x in reasons), (name, reasons)


def test_a_failed_determinism_floor_makes_the_probe_invalid(tmp_path, monkeypatch):
    def nondet(r):
        r["determinism"]["eval"]["waveform"]["relL2"][0] = 1e-6
        r["verdict"].update({"determinism": False, "pass": False})
    path, _, _ = _finalize(tmp_path, monkeypatch, mutate=nondet)
    control, reasons, _ = _load(path, tmp_path)
    assert control is None and any("verdict" in x for x in reasons), reasons


LAUNCH_EDITS = {   # after finalization: the launch block, the log, the sidecar
    "backend": (lambda r, log: r["provenance"]["launch"].update(attention_backend="flash"), "attention_backend"),
    "overlay": (lambda r, log: r["provenance"]["launch"].update(overlay_sha256="0" * 64), "overlay_sha256"),
    "launcher": (lambda r, log: r["provenance"]["launch"].update(launcher_sha256="0" * 64), "launcher_sha256"),
    "log_sha": (lambda r, log: r["provenance"]["launch"].update(log_sha256="0" * 64), "log_sha256"),
    "log_edited": (lambda r, log: open(log, "a").write("later\n"), "log_sha256"),
    "log_gone": (lambda r, log: os.unlink(log), "log"),
    "no_launch": (lambda r, log: r["provenance"].pop("launch"), "launch"),
}


@pytest.mark.parametrize("name", list(LAUNCH_EDITS))
def test_the_launch_provenance_is_validated(tmp_path, monkeypatch, name):
    edit, needle = LAUNCH_EDITS[name]
    path, log, _ = _finalize(tmp_path, monkeypatch)
    rep = json.load(open(path))
    edit(rep, log)
    TP.probe.write_report(rep, path)                                       # a consistent sidecar again
    control, reasons, _ = _load(path, tmp_path)
    assert control is None and any(needle in x for x in reasons), (name, reasons)


def test_the_campaign_digest_and_the_receipts_bind_the_probe(tmp_path, monkeypatch):
    path, _, _ = _finalize(tmp_path, monkeypatch)
    assert _load(path, tmp_path, digest="e" * 64)[1]                       # another campaign's cells
    ok = [{"ckpt_sha256_loaded": BF_SHA, "ckpt_bytes_loaded": BF_BYTES}] * 3
    assert _load(path, tmp_path, receipts=ok)[1] == []
    assert any("ckpt_bytes_loaded" in x for x in _load(path, tmp_path, receipts=[dict(ok[0], ckpt_bytes_loaded=1)])[1])
    assert any("receipt" in x for x in _load(path, tmp_path, receipts=[])[1])
    open(path + ".sha256", "w").write("0" * 64 + "  orbitrir_bf_cond_probe.json\n")
    assert any("sidecar" in x for x in _load(path, tmp_path)[1])


def test_cli_finalize_bf_probe(tmp_path, monkeypatch, capsys):
    path = write_bf_probe(tmp_path, monkeypatch)
    log = write_log(tmp_path)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    argv = ["--finalize-bf-probe", path, "--probe-log", log, "--campaign-digest", DIGEST, "--pin", PIN, "--cells",
            str(cells), "--base-dir", str(tmp_path)]
    assert C.main(argv) == 0 and "BF PROBE VALID" in capsys.readouterr().out
    assert C.main(argv) == 1 and "REFUSED" in capsys.readouterr().out      # a second finalization
