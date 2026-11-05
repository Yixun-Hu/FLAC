"""exp_35 round 1b: the attention-backend evidence of an exp_35 cell -- the log check (the no-flash
launcher's sentinel + transformer.py's SDPA fallback pair) and receipt schema 2 (plan_orbitrir_yaw_evidence.md
r2 §3b, §11 B1): an exp_35 receipt appends ``attention_backend`` sdpa + the overlay and launcher sha256, and
its log is re-read at every verification; exp_25 receipts stay schema 1, byte for byte.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import json
import os

import pytest

import test_depth_canon_collect as T
from test_exp35_collect import EXP35_CASES, VANILLA_CASES, xrow

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN

import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

REPO = T.REPO
OVERLAY_SHA = "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"
LAUNCHER = os.path.join(REPO, "overlays", "no_flash", "exp35_no_flash_launch.py")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_the_backend_constants_pin_the_committed_overlay_and_launcher():
    nf = C.NO_FLASH
    assert nf["overlay"] == "overlays/no_flash/flash_attn.py" and nf["overlay_sha256"] == OVERLAY_SHA
    assert nf["launcher"] == "overlays/no_flash/exp35_no_flash_launch.py"
    assert _sha(os.path.join(REPO, nf["overlay"])) == nf["overlay_sha256"]
    assert _sha(os.path.join(REPO, nf["launcher"])) == nf["launcher_sha256"]
    assert C.backend_fields() == {"attention_backend": "sdpa", "no_flash_overlay_sha256": OVERLAY_SHA,
                                  "no_flash_launcher_sha256": nf["launcher_sha256"]}
    assert C.RECEIPT_KEYS_V2 == C.RECEIPT_KEYS + ("attention_backend", "no_flash_overlay_sha256",
                                                  "no_flash_launcher_sha256")


def test_the_real_launcher_output_passes_the_log_check():
    """The predicate and the launcher agree on the sentinel format (the real launcher, CPU)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               PYTHONPATH=os.path.dirname(LAUNCHER))
    p = subprocess.run([sys.executable, LAUNCHER, "--preflight"], env=env, cwd=REPO, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    text = p.stdout + p.stderr
    assert C.backend_log_reasons(text, target="--preflight") == []
    assert any("target" in x for x in C.backend_log_reasons(text))      # a preflight log is no cell log


def good_log(target="eval_FLAC.py", noise=True):
    nf = C.NO_FLASH
    return "".join([
        "pkg_resources is deprecated as an API.\n" if noise else "",
        f"{nf['overlay_line']}\n", f"{nf['fallback_line']}\n",
        f"{nf['sentinel']} attention_backend=sdpa overlay_sha256={nf['overlay_sha256']} "
        f"launcher_sha256={nf['launcher_sha256']} target={target}\n",
        "Using EMA model\n" if noise else "", "Evaluating: 100%|##########| 100/100\nEvaluation complete!\n"])


def _swap(text, old, new):
    assert old in text
    return text.replace(old, new)


BAD_LOGS = {
    "no_sentinel": lambda t: "".join(ln + "\n" for ln in t.splitlines() if not ln.startswith("EXP35_NO_FLASH")),
    "two_sentinels": lambda t: t + [ln for ln in t.splitlines() if ln.startswith("EXP35")][0] + "\n",
    "other_backend": lambda t: _swap(t, "attention_backend=sdpa", "attention_backend=flash"),
    "other_overlay": lambda t: _swap(t, f"overlay_sha256={OVERLAY_SHA}", "overlay_sha256=" + "0" * 64),
    "other_launcher": lambda t: _swap(t, f"launcher_sha256={C.NO_FLASH['launcher_sha256']}", "launcher_sha256=" + "1" * 64),
    "extra_field": lambda t: _swap(t, "target=eval_FLAC.py", "target=eval_FLAC.py extra=1"),
    "other_target": lambda t: _swap(t, "target=eval_FLAC.py", "target=other.py"),
    "no_overlay_line": lambda t: _swap(t, C.NO_FLASH["overlay_line"] + "\n", ""),
    "no_fallback_line": lambda t: _swap(t, C.NO_FLASH["fallback_line"] + "\n", ""),
    "pair_not_adjacent": lambda t: _swap(t, C.NO_FLASH["overlay_line"] + "\n", C.NO_FLASH["overlay_line"] + "\nx\n"),
    "two_pairs": lambda t: (C.NO_FLASH["overlay_line"] + "\n" + C.NO_FLASH["fallback_line"] + "\n") + t,
    "flash_missing": lambda t: "No module named 'flash_attn'\n" + t,
    "sentinel_first": lambda t: "".join(sorted(t.splitlines(True), key=lambda ln: not ln.startswith("EXP35"))),
}


def test_a_good_log_passes():
    assert C.backend_log_reasons(good_log()) == [] and C.backend_log_reasons(good_log(noise=False)) == []
    assert C.backend_log_reasons(good_log("/tmp/h/stub_eval.py"), target="/tmp/h/stub_eval.py") == []


@pytest.mark.parametrize("name", list(BAD_LOGS))
def test_every_bad_log_is_refused(name):
    assert C.backend_log_reasons(BAD_LOGS[name](good_log())), name


def _exp35_receipt(tmp_path, spec=("CTL", "T", 1, 42, 0)):
    r = xrow(*spec)
    rel = T.write_cell(tmp_path, r)
    path = os.path.join(str(tmp_path), rel[:-len(".json")] + ".cell.json")
    return r, rel, path, json.load(open(path))


@pytest.mark.parametrize("spec", [EXP35_CASES[0], EXP35_CASES[1], EXP35_CASES[4], VANILLA_CASES[0], VANILLA_CASES[2]],
                         ids=lambda s: f"{s[0]}_{s[1]}_K{s[2]}")
def test_an_exp35_cell_carries_receipt_schema_2_and_verifies(tmp_path, spec):
    r, rel, path, rec = _exp35_receipt(tmp_path, spec)
    assert rec["schema_version"] == 2 and list(rec) == list(C.RECEIPT_KEYS_V2)
    assert {k: rec[k] for k in C.RECEIPT_V2_KEYS} == C.backend_fields()
    assert T.verify(tmp_path, rel) == [] and T.verify(tmp_path, rel, dump_check="receipt") == []


def test_an_exp25_receipt_is_schema_1_exactly_as_before(tmp_path):
    r = T.row("P1-pin", "P1", K=8, seed=44, sub="R")
    rel = T.write_cell(tmp_path, r)
    rec = json.load(open(os.path.join(str(tmp_path), rel[:-len(".json")] + ".cell.json")))
    assert rec["schema_version"] == 1 and list(rec) == list(C.RECEIPT_KEYS)
    assert not set(C.RECEIPT_V2_KEYS) & set(rec)
    assert T.verify(tmp_path, rel) == []


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(schema_version=1), "schema_version"),
    (lambda d: d.update(attention_backend="flash"), "attention_backend"),
    (lambda d: d.update(no_flash_overlay_sha256="0" * 64), "no_flash_overlay_sha256"),
    (lambda d: d.update(no_flash_launcher_sha256="0" * 64), "no_flash_launcher_sha256"),
    (lambda d: d.pop("attention_backend"), "receipt keys"),
    (lambda d: d.update(log="logs/absent.log"), "log")], ids=["schema", "backend", "overlay", "launcher", "absent",
                                                           "log"])
def test_a_v2_receipt_off_its_contract_is_refused(tmp_path, edit, needle):
    r, rel, path, rec = _exp35_receipt(tmp_path, ("BF-pin", "C4", 8, 42, 180))
    T.edit_json(path, edit)
    reasons = T.verify(tmp_path, rel, dump_check="receipt")
    assert reasons and any(needle in x for x in reasons), reasons


def test_the_log_is_reread_at_every_verification(tmp_path):
    r, rel, path, rec = _exp35_receipt(tmp_path, ("BF-pin", "T", 1, 43, 0))
    log = os.path.join(str(tmp_path), rec["log"])
    open(log, "w").write(BAD_LOGS["flash_missing"](good_log()))
    assert any("flash_attn" in x for x in T.verify(tmp_path, rel, dump_check="receipt"))
    os.unlink(log)
    assert any("log" in x for x in T.verify(tmp_path, rel, dump_check="receipt"))


def _entry(r, tmp_path):
    return C.plan_cells(MANIFEST, [r], farm_root=T.FARM, dump_root=os.path.join(str(tmp_path), "dumps"),
                        canon_index=T.canon_index(), no_dump_blocks=[])[0]


def test_build_receipt_binds_the_backend_to_the_campaign(tmp_path):
    kw = dict(lane=0, log="logs/x.log", rc=0, pin=PIN, ckpt_bytes=T.CKPT_BYTES, campaign_digest=T.DIGEST,
              spawn_digest=T.DIGEST, post_digest=T.DIGEST)
    r35 = xrow("BF-pin", "R", 8, 42, None)
    T.write_cell(tmp_path, r35, receipt=False)
    e35 = _entry(r35, tmp_path)
    for bad in (None, dict(C.backend_fields(), attention_backend="flash"),
                dict(C.backend_fields(), no_flash_overlay_sha256="0" * 64), {}):
        with pytest.raises(ValueError, match="backend"):
            C.build_receipt(e35, r35, base_dir=str(tmp_path), ckpt_sha256=e35["sha256"], backend=bad, **kw)
    rec = C.build_receipt(e35, r35, base_dir=str(tmp_path), ckpt_sha256=e35["sha256"], backend=C.backend_fields(), **kw)
    assert rec["schema_version"] == 2 and list(rec) == list(C.RECEIPT_KEYS_V2)
    r25 = T.row("T", K=1, seed=43)
    T.write_cell(tmp_path, r25, receipt=False)
    e25 = _entry(r25, tmp_path)
    with pytest.raises(ValueError, match="backend"):
        C.build_receipt(e25, r25, base_dir=str(tmp_path), ckpt_sha256=e25["sha256"], backend=C.backend_fields(), **kw)
    assert list(C.build_receipt(e25, r25, base_dir=str(tmp_path), ckpt_sha256=e25["sha256"], **kw)) == list(C.RECEIPT_KEYS)


def test_cli_check_backend_log(tmp_path, capsys):
    good, bad = tmp_path / "good.log", tmp_path / "bad.log"
    good.write_text(good_log())
    bad.write_text(BAD_LOGS["two_pairs"](good_log()))
    assert C.main(["--check-backend-log", str(good)]) == 0 and "BACKEND OK" in capsys.readouterr().out
    assert C.main(["--check-backend-log", str(bad)]) == 1 and "BACKEND REFUSED" in capsys.readouterr().out
    assert C.main(["--check-backend-log", str(good), "--backend-target", "other.py"]) == 1
    assert C.main(["--check-backend-log", str(tmp_path / "absent.log")]) == 1


def test_cli_write_receipt_for_an_exp35_cell_needs_the_backend_flags(tmp_path, capsys):
    r = xrow("CTL", "T", 8, 42, 0)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(MANIFEST))
    out = str(tmp_path / "plan.json")
    assert C.main(["--plan", out, "--cells", str(cells), "--only", r["cell"], "--farm-root", T.FARM,
                   "--dump-root", str(tmp_path / "dumps"), "--campaign-digest", T.DIGEST]) == 0
    assert json.load(open(out))["cells"][0]["no_flash"] is True
    T.write_cell(tmp_path, r, receipt=False)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "c.log").write_text(good_log())
    argv = ["--write-receipt", r["cell"], "--plan-json", out, "--cells", str(cells), "--base-dir", str(tmp_path),
            "--lane", "1", "--log", "logs/c.log", "--rc", "0", "--pin", PIN, "--ckpt-sha",
            MANIFEST["arms"]["P1"]["ckpt"]["sha256"], "--ckpt-bytes", str(T.CKPT_BYTES), "--farm-root", T.FARM,
            "--dump-root", str(tmp_path / "dumps"), "--no-link-check", "--spawn-digest", T.DIGEST, "--post-digest",
            T.DIGEST, "--verify-after"]
    receipt = os.path.join(str(tmp_path), json.load(open(out))["cells"][0]["receipt"])
    assert C.main(argv) == 1 and not os.path.exists(receipt)                 # no backend flags: refused
    assert "RECEIPT REFUSED" in capsys.readouterr().out
    nf = C.NO_FLASH
    flags = ["--attention-backend", "sdpa", "--no-flash-overlay-sha256", nf["overlay_sha256"],
             "--no-flash-launcher-sha256", nf["launcher_sha256"]]
    (tmp_path / "logs" / "c.log").write_text(BAD_LOGS["no_sentinel"](good_log()))
    assert C.main(argv + flags) == 1 and not os.path.exists(receipt)         # the log fails the re-check
    (tmp_path / "logs" / "c.log").write_text(good_log())
    assert C.main(argv + flags) == 0 and json.load(open(receipt))["schema_version"] == 2
    assert "VALID" in capsys.readouterr().out
