"""exp_43 std_aug -- the training kit (exp43_kit.py), its launcher and the evidence driver.

Forked from src/tests/test_exp40_kit.py (plan_std_aug.md §5, §6 rows 6a-6d). Every exp_40 test
has its exp_43 counterpart here except the neuronic-only ones, which are deleted together with
the neuronic code paths (the list is in ``DELETED_EXP40_TESTS`` below, each with its reason).

CPU only, synthetic inputs, no GPU, no NAS, no Slurm: every external command the kit runs
is injected (stub runners) and every filesystem effect lands under ``tmp_path``. One test
builds a real run tree from this checkout's HEAD (``git archive``), one runs the real
prefigure parse, one runs PyTorch Lightning twice on the tiny CPU model (init parity).
"""
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude"
_EXP40 = _REPO / "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"      # byte-frozen source of the fork


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load("exp43_kit", _EXP / "exp43_kit.py")
drv = _load("exp43_evidence_driver", _EXP / "exp43_evidence_driver.py")
# the exp_40 golden harness (tiny CPU model + metadata builders), read-only and shared
cap = _load("exp40_yaw_aug_golden_capture", _REPO / "src/tests/exp40_yaw_aug_golden_capture.py")
RID = "exp43_RF_a6000_2x32"
STAMP = "2026-11-13_22-00-00"

# exp_40 tests with NO exp_43 counterpart, each with the reason (plan §5.1: every neuronic
# platform, grammar and CLI path is deleted from the exp_43 kit, so its tests go with it).
DELETED_EXP40_TESTS = {
    "test_claim_is_per_execution_and_exclusive": "neuronic execution claim (deleted)",
    "test_a_second_control_namespace_cannot_admit_the_same_execution": "neuronic control dir (deleted)",
    "test_submit_records_the_attempt_boundary": "neuronic submit transaction (deleted)",
    "test_control_dir_must_be_canonical_absolute": "neuronic control dir (deleted)",
    "test_live_or_ambiguous_execution_blocks_any_new_one": "neuronic submit transaction (deleted)",
    "test_post_sbatch_crash_is_adopted_not_resubmitted": "neuronic submit transaction (deleted)",
    "test_delayed_accounting_then_completed_owner": "neuronic Slurm accounting (deleted)",
    "test_claim_owner_completed_with_a_refused_duplicate_is_completed": "neuronic claim owner (deleted)",
    "test_owner_unknown_to_the_scheduler_or_completed_without_claim_is_ambiguous": "neuronic claim owner (deleted)",
    "test_failed_before_claiming_is_terminal_failed": "neuronic claim owner (deleted)",
    "test_malformed_or_truncated_state_is_an_explicit_ambiguous_refusal": "neuronic submission state (deleted)",
    "test_concurrent_submitters_from_two_checkouts_share_one_control_dir": "neuronic submit lock (deleted)",
    "test_claim_is_exclusive_under_true_concurrency": "neuronic execution claim (deleted)",
    "test_state_normalisation": "Slurm state parser (deleted)",
    "test_state_normalisation_ambiguous_is_fatal": "Slurm state parser (deleted)",
    "test_query_failure_is_fatal": "Slurm squeue / sacct queries (deleted)",
    "test_full_gates_relaxation_probe_and_completion": "neuronic FULL submission gates (deleted)",
    "test_time_request_and_partition_limit": "Slurm --time request / MaxTime (deleted)",
    "test_pulled_manifest_publication": "neuronic pull-then-publish path (deleted; A6000 publishes in-run)",
    "test_slurm_gate_inspects_only_the_allocated_devices_by_uuid": "Slurm-allocated GPU gate (deleted)",
    "test_claim_probe_barrier_rounds_and_receipt_validation": "two-node claim probe (deleted)",
    "test_claim_probe_verdict_fails_on_one_host_two_winners_or_outside_dir": "two-node claim probe (deleted)",
    "test_neuronic_time_request_binds_independent_identity": "neuronic PILOT -> FULL time request (deleted)",
    "test_probe_preflight_refuses_queued_checkout_drift": "claim-probe sbatch preflight (deleted)",
}
# exp_40 tests whose exp_43 counterpart carries a corrected name (old -> new).
RENAMED_EXP40_TESTS = {
    "test_full_argv_is_the_exp15_shape_with_the_c16_config": "test_full_argv_is_the_exp15_shape_with_the_arm_config",
    "test_driver_pin_is_enforced_on_a6000_and_recorded_on_neuronic":
        "test_driver_pin_is_enforced_on_a6000_and_neuronic_is_refused",
}


# --------------------------------------------------------------------------- #
# exp_43 6b: identity, closure, roots, neuronic removal, shared-vs-private policy
# --------------------------------------------------------------------------- #
_LAUNCHER = _EXP / "exp43_train_local.sh"


def test_every_exp40_test_has_a_counterpart_or_a_listed_deletion():
    import ast
    names = lambda p: {n.name for n in ast.parse(p.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    old, new = names(_REPO / "src/tests/test_exp40_kit.py"), names(Path(__file__))
    assert not set(DELETED_EXP40_TESTS) - old, "a listed deletion is not an exp_40 test"
    assert not set(DELETED_EXP40_TESTS) & new, "a test listed as deleted still exists"
    assert not set(RENAMED_EXP40_TESTS) - old and set(RENAMED_EXP40_TESTS.values()) <= new
    assert not set(RENAMED_EXP40_TESTS) & new and not set(RENAMED_EXP40_TESTS) & set(DELETED_EXP40_TESTS)
    missing = old - new - set(DELETED_EXP40_TESTS) - set(RENAMED_EXP40_TESTS)
    assert not missing, f"exp_40 tests with neither a counterpart nor a listed reason: {sorted(missing)}"


def test_experiment_identity_constants():
    assert kit.EXPREL == "worklog/worklog_yixun/exp_43_std_aug_claude"
    assert kit.PROJECT == "exp43_std_aug"
    assert kit.NAS_ROOT == "/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug"
    assert kit.RUN_ID_RE.pattern == r"^exp43_RF_a6000_2x32(_r[1-9][0-9]?)?$"
    assert kit.RUNTREE_MANIFEST == ".exp43_runtree_manifest.json"
    assert (kit.PLATFORM, kit.RUNG) == ("a6000", "2x32")
    assert set(kit.PLATFORMS) == {"a6000"} and set(kit.RUNGS) == {"2x32"}
    assert kit.PLATFORMS["a6000"]["modes"] == ("SMOKE", "DDP_SMOKE", "FULL")
    assert kit.SMOKE_STEPS == {"SMOKE": 20, "DDP_SMOKE": 30}
    assert kit.ARM_CONFIG == f"{kit.EXPREL}/FLAC_AR_YAWAUG_FLIP.json"
    assert kit.BASE_CONFIG == "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"
    # the E1 golden of the reference YAW draws (exp_15 / exp_17 pins): RF's offsets must equal it
    assert kit.FIXTURE == "src/tests/fixtures/exp40_yaw_aug_enabled_golden.json"


def test_every_pinned_tracked_file_hashes_to_its_pin():
    assert set(kit.PINS) == {kit.AR_MANIFEST, kit.ARM_CONFIG, kit.BASE_CONFIG, kit.DATASET_CONFIG,
                             "data/AR/train.json", kit.VAE}
    for rel, want in kit.PINS.items():
        if rel == kit.VAE:
            continue                                   # a run-tree link, verified by pin_reasons there
        assert hashlib.sha256((_REPO / rel).read_bytes()).hexdigest() == want, rel


@pytest.mark.parametrize("run_id,ok", [
    ("exp43_RF_a6000_2x32", True), ("exp43_RF_a6000_2x32_r1", True), ("exp43_RF_a6000_2x32_r99", True),
    ("exp43_RF_a6000_2x32_r0", False), ("exp43_RF_a6000_2x32_r100", False), ("exp43_RF_a6000_8x8", False),
    ("exp43_RF_neuronic_2x32", False), ("exp43_RF_neuronic_8x8", False), ("exp40_C16_a6000_2x32", False),
    ("exp43_C16_a6000_2x32", False), ("exp43_YAW_a6000_2x32", False), ("exp43_RF_a6000_2x32\n", False),
    ("", False)])
def test_run_id_grammar(run_id, ok):
    assert kit.run_id_ok(run_id) is ok
    assert (kit.validate_identity("a6000", "2x32", "FULL", run_id) == []) is ok


@pytest.mark.parametrize("platform,rung,mode,run_id", [
    ("neuronic", "2x32", "FULL", "exp43_RF_a6000_2x32"), ("neuronic", "8x8", "FULL", "exp43_RF_neuronic_8x8"),
    ("neuronic", "2x32", "PILOT", "exp43_RF_neuronic_2x32"), ("a6000", "8x8", "FULL", "exp43_RF_a6000_2x32"),
    ("a6000", "2x32", "PILOT", "exp43_RF_a6000_2x32"), ("a6000", "2x32", "FULL", "exp43_RF_neuronic_2x32"),
    ("a6000", "2x32", "FULL", "exp40_C16_neuronic_2x32"), ("a6000", "2x32", "FULL", "exp40_C16_a6000_2x32")])
def test_any_neuronic_or_foreign_identity_is_refused(platform, rung, mode, run_id):
    assert kit.validate_identity(platform, rung, mode, run_id)
    with pytest.raises(kit.Refused):
        kit.train_argv(platform, rung, mode, run_id, "/abs")


@pytest.mark.parametrize("argv", [
    ["neuronic-submit", "--run-id", "x"], ["neuronic-time", "--mode", "FULL"], ["claim", "--claims-dir", "x"],
    ["probe-task", "--dir", "x"], ["probe-verdict", "--dir", "x"], ["probe-preflight", "--repo", "x"],
    ["publish-cadence", "--src", "x"],
    ["gpus", "--platform", "a6000", "--rung", "2x32", "--mode", "FULL", "--lanes", "0,1", "--policy", "idle",
     "--expect-total", "2", "--slurm-allocated"],
    ["check", "--platform", "a6000", "--rung", "2x32", "--mode", "FULL", "--run-id", RID, "--save-root", "/s",
     "--sha", "a" * 40, "--repo", "/r", "--tree", "/t", "--receipts", "/x", "--out", "/o", "--hf-sealed", "/h",
     "--in-job"]])
def test_the_cli_offers_no_neuronic_path(argv):
    with pytest.raises(SystemExit):
        kit.main(argv)


def test_no_neuronic_code_path_survives_in_the_kit():
    for name in ("neuronic_submit", "neuronic_time_reasons", "claim", "probe_task", "probe_verdict",
                 "probe_receipt_reasons", "probe_preflight_reasons", "slurm_gpu_reasons", "visible_gpu_uuids",
                 "control_paths", "normalise_state", "reconcile", "execution_status", "full_time_request",
                 "parse_maxtime_lines", "parse_slurm_time", "pilot_rate", "record_saved_local",
                 "measure_environment_light", "NEURONIC_CONTROL", "ALLOWED_CONTROL", "PILOT_WINDOW"):
        assert not hasattr(kit, name), name


def test_closure_and_archive_lists():
    assert kit.ARCHIVE_PATHS == ("train.py", "defaults.ini", "src", "data/AR", kit.EXPREL, kit.BASE_CONFIG)
    assert kit.CLOSURE == ("train.py", "defaults.ini", "src", "data/AR", kit.BASE_CONFIG, kit.ARM_CONFIG,
                           f"{kit.EXPREL}/exp43_kit.py", f"{kit.EXPREL}/exp43_evidence_driver.py",
                           f"{kit.EXPREL}/exp43_train_local.sh", kit.AR_MANIFEST)
    for rel in kit.CLOSURE:
        assert "sbatch" not in rel and "neuronic" not in rel and "submit" not in rel, rel
        assert (_REPO / rel).exists(), rel
    for name in ("exp43_train.sbatch", "exp43_submit_neuronic.sh", "exp43_claim_probe.sbatch",
                 "exp43_submit_claim_probe.sh"):
        assert not (_EXP / name).exists(), f"{name} must not be copied into exp_43"
    assert kit.LINKS == ("AcousticRooms", kit.VAE)


def test_the_ar_manifest_is_a_pinned_byte_copy_of_exp40s():
    assert kit.AR_MANIFEST == f"{kit.EXPREL}/exp43_ar_manifest.json"
    got = hashlib.sha256((_REPO / kit.AR_MANIFEST).read_bytes()).hexdigest()
    assert got == hashlib.sha256((_EXP40 / "exp40_ar_manifest.json").read_bytes()).hexdigest()
    assert got == kit.AR_MANIFEST_SHA256 == kit.PINS[kit.AR_MANIFEST] \
        == "0c109cd2fe0c03381a1d87cee234da5782b0754b4af3385da0917c54e189f12f"
    man = kit.read_json(str(_REPO / kit.AR_MANIFEST))
    assert kit.digest_obj(man["rooms"]) == man["digest"]


def test_a_mutated_ar_manifest_is_refused_by_the_pin(tmp_path):
    tree = tmp_path / "tree"
    for rel in kit.PINS:
        src = _REPO / rel
        if src.is_file() and not src.is_symlink():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes(src.read_bytes())
    man = tree / kit.AR_MANIFEST
    assert not any(kit.AR_MANIFEST in r for r in kit.pin_reasons(str(tree)))
    man.write_bytes(man.read_bytes() + b" ")
    assert any(kit.AR_MANIFEST in r for r in kit.pin_reasons(str(tree)))


def test_shared_vs_private_runtime_policy():
    """plan §5.1 (Codex r1 finding 11): shared ON PURPOSE -- the GPU handshake directory, the
    per-GPU-UUID locks in it, and exp_40's card lock (cross-experiment serialisation: an
    exp_40 / exp_42 / exp_43 job never co-occupies a card); everything else private."""
    assert kit.HANDSHAKE_DIR == "/home/yixunhu/codespace/claude_gpu_yield"
    assert kit.CARDS_LOCK == os.path.expanduser("~/.exp40_cards.lock")
    assert kit.PUBLISH_LOCK == os.path.expanduser("~/.exp43_publish.lock")
    text = _LAUNCHER.read_text()
    assert 'CARDS_LOCK="$HOME/.exp40_cards.lock"' in text
    assert 'BASE="$REPO/outputs_FLAC/exp43_std_aug"' in text
    assert 'SAVE_ROOT="$REPO/outputs_FLAC/exp43_smoke"' in text
    assert 'EXPREL="worklog/worklog_yixun/exp_43_std_aug_claude"' in text
    assert 'NAS_DIR="/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug/$RUN_ID"' in text
    assert 'CKPT_DIR="$SAVE_ROOT/exp43_std_aug/$NS/checkpoints"' in text
    assert 'HF_SEALED="$BASE/hf_sealed"' in text and 'RECEIPTS="$REPO/$EXPREL/receipts"' in text
    assert 'mktemp -d /tmp/exp43.XXXXXX' in text
    assert 'LOGBASE="$REPO/$EXPREL/std_aug_${TS}_${MODE}"' in text
    assert 'KIT="$RUNTREE/$EXPREL/exp43_kit.py"; DRIVER="$RUNTREE/$EXPREL/exp43_evidence_driver.py"' in text
    assert "~/codespace/claude_gpu_yield/" in text                       # the shared handshake dir
    for bad in ("EXP40_", "exp40_kit", "exp40_evidence_driver", "exp40_smoke", "FLAC_exp40",
                "exp_40_yaw_aug_c16_claude", "exp40_yaw_aug_c16", "/tmp/exp40.", "neuronic", "sbatch",
                "PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40"):
        assert bad not in text, bad
    drv_text = (_EXP / "exp43_evidence_driver.py").read_text()
    for bad in ("EXP40_", "exp40_kit", "local-manifest", "neuronic"):
        assert bad not in drv_text, bad
    for var in ("EXP43_EVIDENCE_DIR", "EXP43_EVIDENCE_MODE", "EXP43_PUBLISH", "EXP43_RUN_ID"):
        assert var in drv_text and var in text, var
    assert "EXP43_DRIVER_VERSION" in text


def test_launcher_identity_gates():
    text = _LAUNCHER.read_text()
    assert 'case "$MODE" in SMOKE|DDP_SMOKE|FULL) ;;' in text
    assert '[ "$PLATFORM" = a6000 ] ||' in text and '[ "$RUNG" = 2x32 ] ||' in text
    assert "RUN_ID=exp43_RF_a6000_2x32[_r<n>]" in text
    assert '"experiment\\": \\"exp_43\\"' in text
    assert "admit --run-result" in text and "--hf-sealed" in text[text.index("admit --run-result"):]


def test_the_driver_refuses_the_retired_publication_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXP43_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("EXP43_EVIDENCE_MODE", "digest")
    monkeypatch.chdir(drv.ROOT)
    for publish in ("local-manifest", "", "NAS"):
        monkeypatch.setenv("EXP43_PUBLISH", publish)
        with pytest.raises(SystemExit) as e:
            drv.main()
        assert e.value.code == 86
        assert f"EXP43_PUBLISH must be nas or off, got {publish!r}" in capsys.readouterr().out


def test_driver_reasons_name_the_exp43_variable():
    assert any("EXP43_DRIVER_VERSION" in r for r in kit.driver_reasons("a6000", ""))


# --------------------------------------------------------------------------- #
# exp_43 6c: the RF treatment binding (unquantised offsets + flip bits, mirror spy)
# --------------------------------------------------------------------------- #
FLIP_CASES = ((0, 1), (1, 1), (2499, 1), (2500, 1), (39999, 1))      # (global_step, global_rank)


def test_treatment_constants_and_the_real_banner(capsys, monkeypatch):
    assert kit.YAW_BLOCK == {"enabled": True, "img_w": 512, "seed": 42, "flip": True}
    assert kit.YAW_KWARGS == {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
                              "yaw_aug_flip": True}
    assert kit.BANNER == "yaw_aug ENABLED img_w=512 seed=42 flip=0.5"
    assert (kit.HIST_BINS, kit.HIST_WIDTH) == (16, 32) and kit.HIST_BINS * kit.HIST_WIDTH == kit.YAW_BLOCK["img_w"]
    assert json.loads((_REPO / kit.ARM_CONFIG).read_text())["training"]["yaw_aug"] == kit.YAW_BLOCK
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, {"flip": True}))
    capsys.readouterr()
    w._print_yaw_aug_banner()
    assert capsys.readouterr().out.strip().splitlines() == [kit.BANNER]
    assert kit.start_gate("\n".join([kit.BANNER, STEP1]), 1)[0] == "pass"
    for other in ("yaw_aug ENABLED img_w=512 seed=42", "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"):
        assert kit.start_gate("\n".join([other, STEP1]), 1)[0] == "fail", other


def _contract_tree(tmp_path, mutate):
    t = tmp_path / f"tree{len(list(tmp_path.iterdir()))}"
    for rel in (kit.ARM_CONFIG, kit.BASE_CONFIG):
        (t / rel).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(_REPO / "src"), str(t / "src"))
    cfg = json.loads((_REPO / kit.ARM_CONFIG).read_text())
    mutate(cfg)
    (t / kit.ARM_CONFIG).write_text(json.dumps(cfg))
    (t / kit.BASE_CONFIG).write_bytes((_REPO / kit.BASE_CONFIG).read_bytes())
    return str(t)


def test_config_contract_through_the_real_factory(tmp_path):
    assert kit.config_contract_reasons(str(_REPO), sys.executable) == []
    assert kit.config_contract_reasons(_contract_tree(tmp_path, lambda c: None), sys.executable) == []
    for mutate, needle in [
            (lambda c: c["training"]["yaw_aug"].update(flip=False), "yaw_aug"),
            (lambda c: c["training"]["yaw_aug"].pop("flip"), "yaw_aug"),
            (lambda c: c["training"]["yaw_aug"].update(flip=1), "factory refuses"),
            (lambda c: c["training"]["yaw_aug"].update(offset_step=32), "factory refuses"),
            (lambda c: c.update(sample_rate=1), "exactly training.yaw_aug.flip"),
            (lambda c: c["training"].update(use_ema=False), "use_ema")]:
        got = kit.config_contract_reasons(_contract_tree(tmp_path, mutate), sys.executable)
        assert any(needle in x for x in got), (needle, got)


def _oracle_stream(steps, ranks, micro):
    """An independent oracle of the registered RF stream (plan §2.3): per (42, step, rank) one
    generator; ``micro`` offsets ``randint(0, 512)`` FIRST, then ``micro`` bits ``randint(0, 2)``."""
    import src.training.diffusion as D
    out = {}
    for r in ranks:
        h, hist, nflip, so, sf, verb = hashlib.sha256(), [0] * 16, 0, {}, {}, []
        for s in range(steps):
            g = torch.Generator()
            g.manual_seed(D._yaw_aug_step_seed(42, s, r))
            o = torch.randint(0, 512, (micro,), generator=g, dtype=torch.long).tolist()
            f = torch.randint(0, 2, (micro,), generator=g, dtype=torch.long).tolist()
            h.update(f"{s}:{','.join(map(str, o))}|{','.join(map(str, f))}\n".encode())
            for d in o:
                hist[d // 32] += 1
            nflip += sum(f)
            if s in kit.SAMPLED_STEPS:
                so[str(s)], sf[str(s)] = o, f
            verb.append([o, f])
        out[r] = {"digest": h.hexdigest(), "hist": hist, "n": steps * micro, "flips": nflip, "sampled": so,
                  "sampled_flips": sf, "verbatim": verb}
    return out


def test_the_stream_reconstruction_is_the_registered_rf_stream():
    got = kit.expected_stream(str(_REPO), 5, range(2), 32, verbatim=True)
    assert got == _oracle_stream(5, range(2), 32)
    nov = kit.expected_stream(str(_REPO), 5, range(2), 32, verbatim=False)
    assert all(nov[r]["verbatim"] == [] and dict(nov[r], verbatim=got[r]["verbatim"]) == got[r] for r in range(2))
    offs = [d for r in range(2) for o, _ in got[r]["verbatim"] for d in o]
    assert any(d % 32 for d in offs), "the RF offsets are not quantised"
    assert {b for r in range(2) for _, f in got[r]["verbatim"] for b in f} == {0, 1}


def test_rf_offsets_are_the_reference_yaw_draws_and_flips_are_drawn_after_them(full_stream):
    """plan §2.3: RF's offset draw stream IS the YAW configuration's (the E1 golden of the exp_15
    / exp_17 reference pins) -- the flip bits come after the offsets from the same generator."""
    fx = json.loads((_REPO / kit.FIXTURE).read_text())["record"]["real"]["offsets"]["2x32"]
    assert sorted(fx, key=int) == ["0", "1", "2", "3", "1000", "2499", "2500", "39999"]
    for step, per_rank in fx.items():
        for r in range(2):
            assert full_stream[r]["sampled"][step] == per_rank[r], (step, r)
    assert all(0 < full_stream[r]["flips"] < kit.ENDPOINT * 32 for r in range(2))


def _spy_apply(monkeypatch, w, mds, step, rank):
    """Run the REAL wrapper's _apply_yaw_aug at (step, rank); return the offsets that reached the
    rotation and the sample indices that reached the mirror (by object identity)."""
    import src.training.diffusion as D
    from src.data.yaw_rotation import yaw_column_shift
    rotated, offsets, mirrored = [], [], []
    rot, mir = D.rotate_scene_metadata, D.mirror_scene_metadata

    def spy_rot(md, alpha, img_w):
        out = rot(md, alpha, img_w)
        offsets.append(int(yaw_column_shift(alpha, img_w)))
        rotated.append(out)
        return out

    def spy_mir(md, img_w, *a, **k):
        mirrored.append(next(i for i, x in enumerate(rotated) if x is md))
        return mir(md, img_w, *a, **k)
    monkeypatch.setattr(D, "rotate_scene_metadata", spy_rot)
    monkeypatch.setattr(D, "mirror_scene_metadata", spy_mir)
    cap.attach_stub_trainer(w, step, rank)
    out = D.DiffusionCondTrainingWrapper._apply_yaw_aug(w, mds)
    monkeypatch.setattr(D, "rotate_scene_metadata", rot)
    monkeypatch.setattr(D, "mirror_scene_metadata", mir)
    return out, offsets, mirrored, rotated


def test_the_wrapper_mirrors_exactly_the_recomputed_bits(monkeypatch, full_stream):
    """The kit's reconstruction against the REAL training path: at every sampled (step, rank) the
    offsets reaching rotate_scene_metadata and the indices reaching mirror_scene_metadata are the
    recomputed stream's; flipped samples equal an independent index-arithmetic mirror."""
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, {"flip": True}))
    mds = [cap.make_md(m, i, 512) for i in range(32)]
    for step, rank in FLIP_CASES + ((0, 0), (39998, 0)):
        out, offsets, mirrored, rotated = _spy_apply(monkeypatch, w, mds, step, rank)
        bits = full_stream[rank]["sampled_flips"][str(step)]
        assert offsets == full_stream[rank]["sampled"][str(step)], (step, rank)
        assert mirrored == [i for i, b in enumerate(bits) if b], (step, rank)
        for i, b in enumerate(bits):
            if not b:
                assert out[i] is rotated[i]
                continue
            depth = rotated[i]["depth"]
            want = depth[:, :, torch.arange(511, -1, -1)].clone()
            want[1] = -want[1]
            assert torch.equal(out[i]["depth"], want)
            for key in ("source", "source_vit", "context_poses", "context_poses_vit"):
                v = rotated[i][key].clone()
                v[..., 1] = -v[..., 1]
                assert torch.equal(out[i][key], v), key
            assert out[i]["context_audio"] is mds[i]["context_audio"]


@pytest.fixture
def driver_restore():
    """Undo drv.install's class / module patches after a test."""
    import torch.utils.checkpoint as tcp
    import src.models.transformer as T
    import src.training.diffusion as D
    cls = D.DiffusionCondTrainingWrapper
    snap = {k: getattr(cls, k) for k in ("_apply_yaw_aug", "on_fit_start", "on_train_end", "training_step")}
    saved = (D.rotate_scene_metadata, D.mirror_scene_metadata, tcp.checkpoint, T.flash_attn_func)
    yield
    for k, v in snap.items():
        setattr(cls, k, v)
    D.rotate_scene_metadata, D.mirror_scene_metadata, tcp.checkpoint, T.flash_attn_func = saved


def _flip_wrapper(monkeypatch):
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()
    return m, cap.build_wrapper(m, cap.tiny_config(512, {"flip": True})), [cap.make_md(m, i, 512) for i in range(32)]


def _md_equal(a, b):
    return a.keys() == b.keys() and all(
        torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def test_mirror_spy_records_the_applied_bits_and_checks_them_per_step(tmp_path, monkeypatch, full_stream,
                                                                       driver_restore):
    import src.training.diffusion as D
    m, w, mds = _flip_wrapper(monkeypatch)
    plain = {}
    for step, rank in FLIP_CASES:
        cap.attach_stub_trainer(w, step, rank)
        plain[step] = D.DiffusionCondTrainingWrapper._apply_yaw_aug(w, mds)
    ev = drv.Evidence(str(tmp_path), verbatim=True)
    drv.install(ev)
    rng = (torch.get_rng_state(), cap.rng_digest(m))
    total = 0
    for step, rank in FLIP_CASES:
        cap.attach_stub_trainer(w, step, rank)
        out = D.DiffusionCondTrainingWrapper._apply_yaw_aug(w, mds)
        assert len(out) == 32 and all(_md_equal(a, b) for a, b in zip(out, plain[step])), step
        total += sum(full_stream[rank]["sampled_flips"][str(step)])
    assert torch.equal(rng[0], torch.get_rng_state()) and rng[1] == cap.rng_digest(m)
    for step, rank in FLIP_CASES:
        assert ev.sampled[str(step)] == full_stream[rank]["sampled"][str(step)]
        assert ev.sampled_flips[str(step)] == full_stream[rank]["sampled_flips"][str(step)]
    assert ev.flips_applied == total == ev.counts["mirror_calls"] > 0
    assert ev.counts["flip_anomalies"] == 0 and ev.counts["stream_mismatch_steps"] == 0
    assert ev.counts["apply_yaw_aug"] == len(FLIP_CASES)
    rows = [json.loads(x) for x in (tmp_path / "rank1.jsonl").read_text().splitlines()]
    assert [r["flips"] for r in rows] == [full_stream[1]["sampled_flips"][str(s)] for s, _ in FLIP_CASES]
    assert [r["offsets"] for r in rows] == [full_stream[1]["sampled"][str(s)] for s, _ in FLIP_CASES]
    ev.summary({"global_step": 1, "pid": 1})
    summ = json.loads((tmp_path / "rank1.summary.json").read_text())
    assert summ["flips_applied"] == total and summ["sampled_flips"] == ev.sampled_flips
    assert sum(summ["offset_histogram"]) == summ["n_offsets"] == 32 * len(FLIP_CASES)
    h = hashlib.sha256()
    for (step, _), r in zip(FLIP_CASES, rows):
        h.update(f"{step}:{','.join(map(str, r['offsets']))}|{','.join(map(str, r['flips']))}\n".encode())
    assert summ["stream_digest"] == h.hexdigest()
    D.mirror_scene_metadata(mds[0], 512)                          # a mirror call outside _apply_yaw_aug
    assert ev.counts["flip_anomalies"] == 1 and ev.counts["mirror_calls"] == total + 1


def test_mirror_spy_flags_bits_that_are_not_the_recomputed_ones(tmp_path, monkeypatch, full_stream,
                                                                 driver_restore):
    import src.training.diffusion as D
    m, w, mds = _flip_wrapper(monkeypatch)
    orig = D.draw_mirror_flips
    monkeypatch.setattr(D, "draw_mirror_flips", lambda n, g: 1 - orig(n, g))      # mirrors the complement
    ev = drv.Evidence(str(tmp_path), verbatim=False)
    drv.install(ev)
    cap.attach_stub_trainer(w, 2500, 1)
    D.DiffusionCondTrainingWrapper._apply_yaw_aug(w, mds)
    want = full_stream[1]["sampled_flips"]["2500"]
    assert ev.sampled_flips["2500"] == [1 - b for b in want]           # what was APPLIED is recorded
    assert ev.counts["stream_mismatch_steps"] == 1 and ev.counts["flip_anomalies"] == 0


def test_mirror_spy_flags_a_batch_that_drops_a_mirrored_sample(tmp_path, monkeypatch, driver_restore):
    import src.training.diffusion as D
    m, w, mds = _flip_wrapper(monkeypatch)
    real = D.DiffusionCondTrainingWrapper._apply_yaw_aug

    def buggy(self, metadata):                                  # mirrors, then returns sample 0 un-mirrored
        out = real(self, metadata)
        return [metadata[0]] + out[1:]
    monkeypatch.setattr(D.DiffusionCondTrainingWrapper, "_apply_yaw_aug", buggy)
    ev = drv.Evidence(str(tmp_path), verbatim=False)
    drv.install(ev)
    cap.attach_stub_trainer(w, 0, 0)
    D.DiffusionCondTrainingWrapper._apply_yaw_aug(w, mds)
    assert ev.counts["flip_anomalies"] == 1


def test_mirror_spy_refuses_a_rebound_mirror_function(tmp_path, monkeypatch, driver_restore):
    import src.training.diffusion as D
    monkeypatch.setattr(D, "mirror_scene_metadata", lambda md, img_w, *a, **k: md)
    with pytest.raises(SystemExit) as e:
        drv.install(drv.Evidence(str(tmp_path), verbatim=False))
    assert e.value.code == 86


def test_fit_start_records_the_flip_state(tmp_path, monkeypatch, driver_restore):
    import src.training.diffusion as D
    m, w, mds = _flip_wrapper(monkeypatch)
    ev = drv.Evidence(str(tmp_path), verbatim=False)
    drv.install(ev)
    cap.attach_stub_trainer(w, 0, 0)
    w.trainer.world_size, w.trainer.local_rank = 1, 0
    D.DiffusionCondTrainingWrapper.on_fit_start(w)
    rec = json.loads((tmp_path / "rank0.jsonl").read_text().splitlines()[0])
    assert rec["kind"] == "fit_start" and rec["yaw_aug"] == kit.YAW_BLOCK


def _noflip(recs, sums, exp):
    """The evidence a run WITHOUT the flip would produce (same offsets, every bit 0)."""
    for r, s in sums.items():
        h = hashlib.sha256()
        for row in recs[r][1:]:
            row["flips"] = [0] * len(row["flips"])
            h.update(f"{row['step']}:{','.join(map(str, row['offsets']))}|{','.join(map(str, row['flips']))}\n".encode())
        s.update(stream_digest=h.hexdigest(), flips_applied=0,
                 sampled_flips={k: [0] * len(v) for k, v in s["sampled_flips"].items()})
        s["counts"]["mirror_calls"] = 0


@pytest.mark.parametrize("mode,world", [("SMOKE", 1), ("DDP_SMOKE", 2)])
def test_smoke_evidence_requires_applied_flip_evidence(mode, world):
    import copy
    recs, sums, log, exp = _evidence(world=world)
    fx = json.loads((_REPO / kit.FIXTURE).read_text())["record"]["real"]["offsets"]["2x32"]
    assert kit.evidence_reasons(recs, sums, log, mode, "2x32", 6, exp, fx) == []
    snapped = {k: [[32 * (d // 32) for d in v] for v in per] for k, per in fx.items()}
    assert any("E1 fixture" in x for x in kit.evidence_reasons(recs, sums, log, mode, "2x32", 6, exp, snapped))
    for mutate, needle in [
            (lambda r, s: s[0].update(flips_applied=0), "flips_applied"),
            (lambda r, s: s[0].update(flips_applied=True), "flips_applied"),
            (lambda r, s: s[0]["counts"].update(mirror_calls=s[0]["flips_applied"] + 1), "mirror_calls"),
            (lambda r, s: s[0]["counts"].update(flip_anomalies=1), "flip_anomalies"),
            (lambda r, s: s[0]["counts"].update(stream_mismatch_steps=1), "stream_mismatch_steps"),
            (lambda r, s: r[0][2]["flips"].__setitem__(4, 1 - r[0][2]["flips"][4]), "verbatim"),
            (lambda r, s: s[0]["sampled_flips"]["3"].__setitem__(0, 1 - s[0]["sampled_flips"]["3"][0]), "sampled_flips"),
            (lambda r, s: s[0].update(stream_digest=exp[0]["digest"][::-1]), "stream_digest")]:
        r2, s2 = copy.deepcopy(recs), copy.deepcopy(sums)
        mutate(r2, s2)
        got = kit.evidence_reasons(r2, s2, log, mode, "2x32", 6, exp, None)
        assert any(needle in x for x in got), (needle, got)
    _noflip(recs, sums, exp)
    got = kit.evidence_reasons(recs, sums, log, mode, "2x32", 6, exp, None)
    assert any("flips_applied" in x for x in got) and any("stream_digest" in x for x in got), got


# --------------------------------------------------------------------------- #
# argv / recipe projection / prefigure parse
# --------------------------------------------------------------------------- #
def test_full_argv_is_the_exp15_shape_with_the_arm_config():
    a = kit.train_argv("a6000", "2x32", "FULL", RID, "/abs/save")
    assert a[:2] == ["--model-config", kit.ARM_CONFIG]
    pairs = dict(zip(a[::2], a[1::2]))
    assert pairs["--batch-size"] == "32" and pairs["--num-gpus"] == "2" and pairs["--accum-batches"] == "1"
    assert pairs["--strategy"] == "ddp_find_unused_parameters_true" and pairs["--sync-batchnorm"] == "true"
    assert pairs["--max-steps"] == "40000" and pairs["--checkpoint-every"] == "2500"
    assert pairs["--seed"] == "42" and pairs["--precision"] == "bf16-mixed" and pairs["--val-every"] == "-1"
    assert pairs["--experiment-name"] == RID and pairs["--save-dir"] == "/abs/save"


def test_ddp_smoke_differs_from_full_only_in_steps_and_namespace():
    full = kit.train_argv("a6000", "2x32", "FULL", RID, "/abs/a")
    smoke = kit.train_argv("a6000", "2x32", "DDP_SMOKE", RID, "/abs/b", STAMP)
    assert kit.recipe_projection(full) == kit.recipe_projection(smoke)
    assert dict(zip(smoke[::2], smoke[1::2]))["--checkpoint-every"] == "2500"      # plan §14 H5
    one = kit.train_argv("a6000", "2x32", "SMOKE", RID, "/abs/b", STAMP)
    assert "--sync-batchnorm" not in one and dict(zip(one[::2], one[1::2]))["--num-gpus"] == "1"


@pytest.mark.parametrize("args,needle", [
    (("lambda", "2x32", "FULL", RID), "platform"),
    (("a6000", "8x8", "FULL", "exp43_RF_a6000_8x8"), "rung"),
    (("a6000", "2x32", "PILOT", RID), "mode"),
    (("neuronic", "2x32", "SMOKE", "exp43_RF_neuronic_2x32"), "platform"),
    (("a6000", "2x32", "FULL", "exp43_RF_neuronic_2x32"), "does not match"),
    (("a6000", "2x32", "FULL", "exp14_x"), "does not match"),
])
def test_identity_refusals(args, needle):
    with pytest.raises(kit.Refused, match=needle):
        kit.train_argv(*args, save_root="/abs")


def test_relative_save_root_refused():
    with pytest.raises(kit.Refused, match="absolute"):
        kit.train_argv("a6000", "2x32", "FULL", RID, "rel/save")


def test_prefigure_parse_matches_the_expected_values_type_strict():
    for mode, stamp in (("FULL", ""), ("DDP_SMOKE", STAMP), ("SMOKE", STAMP)):
        argv = kit.train_argv("a6000", "2x32", mode, RID, "/abs/s", stamp)
        want = kit.expected_parse("a6000", "2x32", mode, RID, "/abs/s", stamp)
        # exp_45 compat edit (plan_rf_data_curve.md §3.1): the merged defaults.ini carries exp_38's
        # evidence_json = '' -- the ONLY key added to the parsed namespace; every other assertion stays
        assert "evidence_json" not in want
        assert kit.parsed_argv_reasons(str(_REPO), argv, want, sys.executable) == [
            "parsed evidence_json = '' (str), expected '<absent>'"]
        want = dict(want, evidence_json="")
        assert kit.parsed_argv_reasons(str(_REPO), argv, want, sys.executable) == []
    bad = dict(want, batch_size=8)
    assert kit.parsed_argv_reasons(str(_REPO), argv, bad, sys.executable)


# --------------------------------------------------------------------------- #
# immutable run tree (real git archive of HEAD)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def runtree(tmp_path_factory):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    base = tmp_path_factory.mktemp("rt")
    ar, vae = base / "AR", base / "vae.safetensors"
    ar.mkdir()
    vae.write_bytes(b"vae")
    links = {"AcousticRooms": str(ar), "weights/FLAC/VAE.safetensors": str(vae)}
    tree = base / "tree"
    kit.build_runtree(str(_REPO), head, str(tree), links)
    return head, tree, links


def _writable(path):
    os.chmod(path.parent, 0o755)
    if path.exists() and not path.is_symlink():
        os.chmod(path, 0o644)


def test_runtree_verifies(runtree):
    head, tree, links = runtree
    assert kit.verify_runtree(str(_REPO), head, str(tree), links) == []
    assert kit.verify_runtree(None, head, str(tree), links) == []          # carried manifest alone


def test_runtree_refuses_rebuild_and_wrong_links(runtree, tmp_path):
    head, tree, links = runtree
    with pytest.raises(kit.Refused, match="built once"):
        kit.build_runtree(str(_REPO), head, str(tree), links)
    with pytest.raises(kit.Refused, match="exactly"):
        kit.build_runtree(str(_REPO), head, str(tmp_path / "x"), {"AcousticRooms": links["AcousticRooms"]})


def test_runtree_detects_a_modified_file(runtree, tmp_path):
    head, tree, links = runtree
    f = tree / "train.py"
    _writable(f)
    orig = f.read_bytes()
    try:
        f.write_bytes(orig + b"\n# shadow\n")
        r = kit.verify_runtree(str(_REPO), head, str(tree), links)
        assert any("train.py differs" in x for x in r) and any("writable" in x for x in r)
    finally:
        f.write_bytes(orig)
        os.chmod(f, 0o444)
        os.chmod(f.parent, 0o555)


def test_runtree_detects_an_extra_file_and_a_retargeted_link(runtree, tmp_path):
    head, tree, links = runtree
    extra = tree / "src" / "sitecustomize.py"
    os.chmod(extra.parent, 0o755)
    try:
        extra.write_text("x = 1\n")
        os.chmod(extra, 0o444)
        assert any("untracked file src/sitecustomize.py" in x
                   for x in kit.verify_runtree(str(_REPO), head, str(tree), links))
    finally:
        os.chmod(extra, 0o644)
        extra.unlink()
        os.chmod(extra.parent, 0o555)
    other = dict(links, AcousticRooms=str(tmp_path))
    assert any("links" in x or "->" in x for x in kit.verify_runtree(str(_REPO), head, str(tree), other))


# --------------------------------------------------------------------------- #
# start watcher
# --------------------------------------------------------------------------- #
B = kit.BANNER
STEP1 = "Epoch 0:   0%|          | 1/4550 [00:04<6:16:50,  0.20it/s, train/loss=2.3]"
MEM = "Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {m}/2"


def test_start_gate_pass_and_pending():
    ok = "\n".join([MEM.format(r=0, m=1), MEM.format(r=1, m=2), B, "Epoch 0:   0%| | 0/4550 [", STEP1])
    assert kit.start_gate(ok, 2)[0] == "pass"
    assert kit.start_gate("\n".join([MEM.format(r=0, m=1), B]), 2)[0] == "pending"


@pytest.mark.parametrize("text", [
    "\n".join([MEM.format(r=0, m=1), MEM.format(r=1, m=2), STEP1, B]),           # step first
    "\n".join([MEM.format(r=0, m=1), "yaw_aug ENABLED img_w=512 seed=42", STEP1]),  # exp_15 banner
    "\n".join([MEM.format(r=0, m=1), B, STEP1]),                                  # one rank only
    "\n".join(["Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8", B, STEP1]),  # wrong world
])
def test_start_gate_failures(text):
    assert kit.start_gate(text, 2)[0] == "fail"


def test_watch_terminates_only_its_own_session(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("\n".join([STEP1]))
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        rc = kit.watch(str(log), 1, proc.pid, proc.pid, timeout_s=30, poll_s=0.1)
        assert rc == 8
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
    with pytest.raises(kit.Refused, match="watcher itself"):
        kit.watch(str(log), 1, os.getpid(), os.getpgrp(), timeout_s=1)


# --------------------------------------------------------------------------- #
# GPU gate and cooperative per-card locks
# --------------------------------------------------------------------------- #
U0, U1 = "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"
GPUS = f"0, {U0}, NVIDIA RTX A6000, 48000\n1, {U1}, NVIDIA RTX A6000, 48000\n"


def test_gpu_gate():
    g = kit.parse_gpu_query(GPUS)
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g, [], 2) == []
    busy = kit.parse_apps_query(f"{U1}, 4242, 3000\n")
    assert any("4242" in r for r in kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g, busy, 2))
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "shared", g, [], 2)[0].startswith("a primary FULL")
    assert kit.gpu_reasons("a6000", "SMOKE", "2x32", [0], "idle", g, busy, 2) == []
    assert kit.gpu_reasons("a6000", "SMOKE", "2x32", [0, 1], "idle", g, [], 2)        # 2 lanes for 1 rank
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", g[:1], [], 2)      # count
    wrong = kit.parse_gpu_query(f"0, {U0}, NVIDIA L40, 48000\n1, {U1}, NVIDIA L40, 48000\n")
    assert kit.gpu_reasons("a6000", "FULL", "2x32", [0, 1], "idle", wrong, [], 2)
    stray = kit.parse_apps_query("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 1, 1\n")
    assert any("unknown" in r for r in kit.gpu_reasons("a6000", "SMOKE", "2x32", [0], "idle", g, stray, 2))
    with pytest.raises(kit.Refused):
        kit.parse_gpu_query("garbage")


def test_gpu_locks_race_rollback_and_owner(tmp_path):
    t1, p1 = kit.acquire_gpu_locks(str(tmp_path), [U1], {"who": "a"})
    with pytest.raises(kit.Refused, match="held"):
        kit.acquire_gpu_locks(str(tmp_path), [U0, U1], {"who": "b"})           # U0 taken first, then U1 fails
    assert not (tmp_path / f"GPU_LOCK_{U0}").exists(), "the partial acquisition was not rolled back"
    with pytest.raises(kit.Refused, match="another token"):
        kit.release_gpu_locks(p1, "not-the-owner")
    kit.release_gpu_locks(p1, t1)
    assert not (tmp_path / f"GPU_LOCK_{U1}").exists()
    with pytest.raises(kit.Refused):
        kit.release_gpu_locks(p1, t1)                                          # already released
    t2, p2 = kit.acquire_gpu_locks(str(tmp_path), [U0, U1], {"who": "c"})
    assert len(p2) == 2
    with pytest.raises(kit.Refused):
        kit.acquire_gpu_locks(str(tmp_path), ["not-a-uuid"], {})


# --------------------------------------------------------------------------- #
# save-time NAS publication (plan §14 B2)
# --------------------------------------------------------------------------- #
def _ckpt(dirpath, step, payload=b"x"):
    p = dirpath / f"epoch=0-step={step}.ckpt"
    p.write_bytes(payload * 100)
    return p


def test_publish_saved_acknowledges_and_is_idempotent(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    man = kit.read_manifest(nas / RID / "MANIFEST.sha256")
    assert man == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))            # idempotent
    assert not list((nas / RID).glob(".*partial*"))


def test_publish_adopts_an_identical_orphan_and_refuses_a_clobber(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    (nas / RID).mkdir(parents=True)
    p = _ckpt(src, 5000)
    kit.mark_saved(str(p))
    (nas / RID / p.name).write_bytes(p.read_bytes())                            # crash orphan
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    q = _ckpt(src, 7500, b"y")
    kit.mark_saved(str(q))
    (nas / RID / q.name).write_bytes(b"different")
    with pytest.raises(kit.Refused, match="clobber"):
        kit.publish_saved(str(q), RID, timeout_s=60, dest_root=str(nas))


def test_publish_refuses_a_changed_source_and_noreplace_holds(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    p.write_bytes(b"z" * 7)
    with pytest.raises(kit.Refused):
        kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    with pytest.raises(kit.Refused, match="NOREPLACE"):
        kit.rename_noreplace(str(a), str(b))
    assert b.read_text() == "b"


def test_save_hook_fails_every_rank_when_publication_fails(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    calls = []
    monkeypatch.setattr(pl.callbacks.ModelCheckpoint, "_save_checkpoint",
                        lambda self, trainer, fp: calls.append(fp))
    fake_kit = types.SimpleNamespace(mark_saved=lambda p: None,
                                     publish_saved=lambda *a, **k: (_ for _ in ()).throw(OSError("NAS down")),
                                     record_saved_local=lambda p: None)
    monkeypatch.setitem(sys.modules, "exp43_kit", fake_kit)
    ev = drv.Evidence(str(tmp_path), verbatim=True)
    ev.bind_rank(0)
    drv.install_publication(ev, "nas")
    trainer = types.SimpleNamespace(is_global_zero=True,
                                    strategy=types.SimpleNamespace(broadcast=lambda obj, src=0: obj))
    with pytest.raises(RuntimeError, match="not acknowledged"):
        pl.callbacks.ModelCheckpoint._save_checkpoint(object(), trainer, "/x/epoch=0-step=2500.ckpt")
    assert calls == ["/x/epoch=0-step=2500.ckpt"]
    peer = types.SimpleNamespace(is_global_zero=False,
                                 strategy=types.SimpleNamespace(broadcast=lambda obj, src=0: (False, "rank0 failed")))
    with pytest.raises(RuntimeError, match="rank0 failed"):
        pl.callbacks.ModelCheckpoint._save_checkpoint(object(), peer, "/x/epoch=0-step=2500.ckpt")


# --------------------------------------------------------------------------- #
# evidence driver: attribution, observer, init parity
# --------------------------------------------------------------------------- #
def _cls(module, name="C"):
    return type(name, (), {"__module__": module, "forward": lambda self, *a: a,
                           "__call__": lambda self, *a: a})


def test_checkpoint_owner_classification():
    dino = _cls("transformers.models.dinov3_vit.modeling_dinov3_vit", "DINOv3ViTLayer")()
    dit = _cls("src.models.transformer", "TransformerBlock")()
    assert drv.checkpoint_owner_class(dino.forward) == "dinov3"
    assert drv.checkpoint_owner_class(functools.partial(dino.forward, 1)) == "dinov3"
    assert drv.checkpoint_owner_class(dit) == "dit"
    assert drv.checkpoint_owner_class(lambda x: x) == "unclassified"
    assert drv.checkpoint_owner_class(_cls("other.mod")()) == "unclassified"


def test_counted_checkpoint_separates_reentrant_and_default(monkeypatch, tmp_path):
    import torch.utils.checkpoint as tcp
    seen = []
    monkeypatch.setattr(tcp, "checkpoint", lambda f, *a, use_reentrant=None, **k: seen.append(use_reentrant) or f(*a))
    ev = drv.Evidence(str(tmp_path), verbatim=True)
    import src.models.transformer as T
    import src.training.diffusion as D
    for name in ("flash_attn_func",):
        monkeypatch.setattr(T, name, None)
    snapshot = {k: getattr(D.DiffusionCondTrainingWrapper, k) for k in
                ("_apply_yaw_aug", "on_fit_start", "on_train_end", "training_step")}
    rot = D.rotate_scene_metadata
    try:
        drv.install(ev)
        dino = _cls("transformers.models.dinov3_vit.modeling_dinov3_vit")()
        tcp.checkpoint(dino.forward, 1, use_reentrant=False)
        tcp.checkpoint(dino.forward, 1, use_reentrant=True)
        tcp.checkpoint(dino.forward, 1)
        tcp.checkpoint(lambda x: x, 1, use_reentrant=False)
        assert ev.counts["ckpt_dinov3"] == 1 and ev.counts["ckpt_reentrant"] == 1
        assert ev.counts["ckpt_default"] == 1 and ev.counts["ckpt_unclassified"] == 1
    finally:
        for k, v in snapshot.items():
            setattr(D.DiffusionCondTrainingWrapper, k, v)
        D.rotate_scene_metadata = rot


def test_loss_observer_has_no_per_step_host_sync():
    import ast
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(drv.Evidence.record_loss))).body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        fn.body = fn.body[1:]                                   # drop the docstring
    code = ast.unparse(fn)
    for bad in (".item(", ".cpu(", "float(", ".tolist(", ".numpy("):
        assert bad not in code, bad


def test_loss_observer_returns_the_original_object_and_changes_nothing(tmp_path, monkeypatch):
    import src.models.transformer as T
    import src.training.diffusion as D
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()

    def run(observe):
        cap._mods()
        torch.manual_seed(5)
        w = cap.build_wrapper(m, cap.tiny_config(16))
        cap.attach_stub_trainer(w, 3, 0)
        ev = drv.Evidence(str(tmp_path / f"o{observe}"), verbatim=False)
        os.makedirs(ev.dir, exist_ok=True)
        g = torch.Generator().manual_seed(2)
        batch = (torch.randn(2, 4, 64, generator=g), [cap.make_md(m, i, 16) for i in range(2)])
        torch.manual_seed(9)
        out = D.DiffusionCondTrainingWrapper.training_step(w, batch, 0)
        if observe:
            ev.bind_rank(0)
            ev.record_loss(3, out)
        out.backward()
        return out, {n: p.grad.clone() for n, p in w.named_parameters() if p.grad is not None}, \
            torch.get_rng_state(), ev
    a, ga, ra, ev = run(True)
    b, gb, rb, _ = run(False)
    assert torch.equal(a, b) and torch.equal(ra, rb)
    assert ga.keys() == gb.keys() and all(torch.equal(ga[k], gb[k]) for k in ga)
    assert bool(ev.finite) is True and ev.counts["training_step"] == 1


def test_init_signature_parity_through_lightning(tmp_path, monkeypatch):
    """plan §14 H5: two real Lightning setups differing only in name / save dir / max steps
    (offline W&B logger + ModelCheckpoint at the production cadence) reach on_fit_start with
    identical state, parameter, RNG and Sobol signatures."""
    import pytorch_lightning as pl
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    m = cap._mods()
    sigs = []

    class Sig(pl.Callback):
        def on_fit_start(self, trainer, module):
            sigs.append((drv.state_signature(module), drv.param_signature(module), drv.rng_signature(module)))
            raise KeyboardInterrupt

    def batches():
        g = torch.Generator().manual_seed(3)
        return [(torch.randn(2, 4, 64, generator=g), [cap.make_md(m, i, 16) for i in range(2)])]

    for name, steps in (("a_smoke", 30), ("b_full", 40000)):
        pl.seed_everything(42, workers=True)
        w = cap.build_wrapper(m, cap.tiny_config(16, {"flip": True}))
        logger = pl.loggers.WandbLogger(project="p", name=name, save_dir=str(tmp_path / name))
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=steps, logger=logger,
                             enable_progress_bar=False, enable_model_summary=False,
                             callbacks=[pl.callbacks.ModelCheckpoint(dirpath=str(tmp_path / name / "ck"),
                                                                     every_n_train_steps=2500, save_top_k=-1),
                                        Sig()])
        try:
            trainer.fit(w, torch.utils.data.DataLoader(batches(), batch_size=None))
        except KeyboardInterrupt:
            pass
    assert len(sigs) == 2 and sigs[0] == sigs[1]


# --------------------------------------------------------------------------- #
# rank-private evidence verdicts (synthetic, real recomputation)
# --------------------------------------------------------------------------- #
def _evidence(steps=6, world=2, micro=32):
    exp = kit.expected_stream(str(_REPO), steps, range(world), micro, verbatim=True)
    fit = {"kind": "fit_start", "world_size": world, "yaw_aug": kit.YAW_BLOCK,
           "flash_attn_func_is_none": False, "global_step": 0, "syncbn_modules": 4 if world > 1 else 0,
           "gpu_name": "NVIDIA RTX A6000", "state_signature": "s", "param_signature": "p",
           "rng_signature": {"python": "r"}, "numerics": {"x": 1}}
    recs, sums = {}, {}
    for r in range(world):
        rows = [dict(fit, rank=r)] + [{"kind": "offsets", "rank": r, "step": s, "n": micro,
                                       "offsets": list(exp[r]["verbatim"][s][0]),
                                       "flips": list(exp[r]["verbatim"][s][1]), "t": float(s)} for s in range(steps)]
        recs[r] = rows
        sums[r] = {"rank": r, "counts": {"apply_yaw_aug": steps, "training_step": steps, "ckpt_reentrant": 0,
                                         "ckpt_default": 0, "ckpt_unclassified": 0, "flash_calls": 3,
                                         "ckpt_dinov3": 5, "ckpt_dit": 5, "checkpoints_published": 0,
                                         "mirror_calls": exp[r]["flips"], "flip_anomalies": 0,
                                         "stream_mismatch_steps": 0},
                   "stream_digest": exp[r]["digest"], "offset_histogram": exp[r]["hist"],
                   "n_offsets": exp[r]["n"], "n_steps": steps, "steps_contiguous_from_0": True,
                   "sampled_offsets": exp[r]["sampled"], "sampled_flips": exp[r]["sampled_flips"],
                   "flips_applied": exp[r]["flips"], "all_losses_finite": True, "global_step": steps}
        sums[r] = json.loads(json.dumps(sums[r]))          # no aliasing with the recomputation ``exp``
    log = "\n".join(([MEM.format(r=0, m=1), MEM.format(r=1, m=2)] if world > 1 else []) + [B, STEP1])
    return recs, sums, log, exp


def test_evidence_pass():
    recs, sums, log, exp = _evidence()
    assert kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda r, s: s[1]["counts"].update(ckpt_reentrant=1), "ckpt_reentrant"),
    (lambda r, s: s[0]["counts"].update(ckpt_unclassified=2), "ckpt_unclassified"),
    (lambda r, s: s[0]["counts"].update(ckpt_dinov3=0), "ckpt_dinov3"),
    (lambda r, s: s[0]["counts"].update(flash_calls=0), "flash_calls"),
    (lambda r, s: s[1].update(all_losses_finite=False), "all_losses_finite"),
    (lambda r, s: s[0].update(stream_digest="0" * 64), "stream_digest"),
    (lambda r, s: r[0][3]["flips"].__setitem__(0, 1 - r[0][3]["flips"][0]), "verbatim"),
    (lambda r, s: r[1][4].pop("flips"), "verbatim"),
    (lambda r, s: s[0].update(flips_applied=s[0]["flips_applied"] - 1), "flips_applied"),
    (lambda r, s: s[1]["counts"].update(mirror_calls=0), "mirror_calls"),
    (lambda r, s: s[0]["counts"].update(flip_anomalies=1), "flip_anomalies"),
    (lambda r, s: s[1]["counts"].update(stream_mismatch_steps=2), "stream_mismatch_steps"),
    (lambda r, s: s[0]["sampled_flips"]["0"].__setitem__(0, 1 - s[0]["sampled_flips"]["0"][0]), "sampled_flips"),
    (lambda r, s: s[1]["sampled_offsets"]["1"].__setitem__(5, (s[1]["sampled_offsets"]["1"][5] + 1) % 512),
     "sampled_offsets"),
    (lambda r, s: r[1][0].update(yaw_aug={"enabled": True, "img_w": 512, "seed": 42}), "yaw_aug"),
    (lambda r, s: r[1][2]["offsets"].__setitem__(0, 1), "verbatim"),
    (lambda r, s: r[0][0].update(state_signature="other"), "state_signature"),
    (lambda r, s: r[1][0].update(syncbn_modules=0), "SyncBatchNorm"),
    (lambda r, s: r[0][0].update(yaw_aug={"enabled": True}), "yaw_aug"),
    (lambda r, s: s.pop(1), "ranks"),
])
def test_evidence_failures(mutate, needle):
    recs, sums, log, exp = _evidence()
    mutate(recs, sums)
    assert any(needle in x for x in kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None))


def test_evidence_reference_and_banner():
    recs, sums, log, exp = _evidence()
    ref = {"kind": "ddp_smoke", "fit_start": {str(r): dict(recs[r][0]) for r in range(2)}}
    assert kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None, ref) == []
    ref["fit_start"]["1"]["rng_signature"] = {"python": "different"}
    assert any("rng_signature" in x for x in kit.evidence_reasons(recs, sums, log, "DDP_SMOKE", "2x32", 6, exp, None, ref))
    assert any("banner" in x for x in kit.evidence_reasons(recs, sums, log + "\n" + B, "DDP_SMOKE", "2x32", 6, exp, None))


# --------------------------------------------------------------------------- #
# full-state checkpoint audit (synthetic tiny checkpoint)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_ckpt(monkeypatch):
    import src.models.transformer as T
    monkeypatch.setattr(T, "flash_attn_func", None)
    m = cap._mods()
    cfg = cap.tiny_config(16, {"flip": True})
    cfg["training"]["use_ema"] = True
    w = kit.build_wrapper_cpu(str(_REPO), cfg)
    opt = torch.optim.AdamW(w.diffusion.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-3)
    named = list(w.diffusion.named_parameters())
    params = [p for _, p in named]
    state = {i: {"step": torch.tensor(40000.0), "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
             for i, (n, p) in enumerate(named) if p.requires_grad and n not in kit.NO_GRAD_PARAMS}
    sd = w.state_dict()
    sd["diffusion_ema.initted"] = torch.tensor([1.0])
    sd["diffusion_ema.step"] = torch.tensor([40000])
    key = ("ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2500, "
           "'every_n_epochs': 0, 'train_time_interval': None}")
    ck = {"global_step": 40000, "epoch": 8, "model_config": cfg, "state_dict": sd,
          "optimizer_states": [{"state": state, "param_groups": [{"lr": kit.scheduled_lr(kit.ENDPOINT),
                                                                  "initial_lr": 5e-05, "betas": (0.9, 0.999),
                                                                  "weight_decay": 0.001,
                                                                  "params": list(range(len(params)))}]}],
          "lr_schedulers": [dict(kit.SCHED)],
          "loops": {"fit_loop": {"epoch_loop.automatic_optimization.optim_progress":
                                 {"optimizer": {"step": {"total": {"completed": 40000}}}}}},
          "callbacks": {key: {"dirpath": "/ck"}}}
    return ck, cfg, w


def test_checkpoint_state_pass(tiny_ckpt):
    ck, cfg, w = tiny_ckpt
    assert kit.checkpoint_state_reasons(ck, cfg, "/ck", w) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda ck: ck.update(global_step=39999), "global_step"),
    (lambda ck: ck["model_config"]["training"].update(seed=1), "model_config"),
    (lambda ck: ck["lr_schedulers"][0].update(last_epoch=1), "InverseLR"),
    (lambda ck: ck["state_dict"].update({"diffusion_ema.initted": torch.tensor([0.0])}), "EMA state"),
    (lambda ck: ck["callbacks"].clear(), "callbacks"),
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].update(lr=1.0), "AdamW"),
    (lambda ck: ck["optimizer_states"][0]["state"].popitem(), "optimizer state"),
    # audit fix 2026-11-11: the group's lr is the SCHEDULED lr; the base lr lives in initial_lr
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].update(lr=5e-05), "scheduled InverseLR"),
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].update(lr=True), "scheduled InverseLR"),
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].update(initial_lr=1e-04), "initial_lr"),
    (lambda ck: ck["optimizer_states"][0]["param_groups"][0].pop("initial_lr"), "initial_lr"),
])
def test_checkpoint_state_failures(tiny_ckpt, mutate, needle):
    ck, cfg, w = tiny_ckpt
    import copy
    ck = copy.deepcopy(ck)
    mutate(ck)
    assert any(needle in x for x in kit.checkpoint_state_reasons(ck, cfg, "/ck", w))


# --------------------------------------------------------------------------- #
# audit fix 2026-11-11: scheduled lr + never-gradient parameters (the _r1 false FAIL)
# --------------------------------------------------------------------------- #
def test_scheduled_lr_is_the_real_inverselr_after_40000_steps():
    from src.training.utils import InverseLR
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=kit.ADAMW["lr"], betas=tuple(kit.ADAMW["betas"]),
                            weight_decay=kit.ADAMW["weight_decay"])
    s = kit.SCHED
    sch = InverseLR(opt, inv_gamma=s["inv_gamma"], power=s["power"], warmup=s["warmup"], final_lr=s["final_lr"])
    for _ in range(kit.ENDPOINT):
        opt.step()
        sch.step()
    pg = opt.param_groups[0]
    assert sch.last_epoch == kit.ENDPOINT and pg["initial_lr"] == kit.ADAMW["lr"]
    assert math.isclose(pg["lr"], kit.scheduled_lr(kit.ENDPOINT), rel_tol=1e-12, abs_tol=0.0)
    assert kit.scheduled_lr(kit.ENDPOINT) == 4.902903378454601e-05      # the _r1 40k checkpoint's lr
    assert not math.isclose(kit.scheduled_lr(kit.ENDPOINT), kit.ADAMW["lr"], rel_tol=1e-3)


def test_no_grad_params_are_the_seven_pinned_names():
    assert kit.NO_GRAD_PARAMS == {
        "conditioner.conditioners.source.proj_out.weight",
        "conditioner.conditioners.source.proj_out.bias",
        "conditioner.conditioners.source_vit.vit.embeddings.mask_token",
        "conditioner.conditioners.context_poses.proj_out.weight",
        "conditioner.conditioners.context_poses.proj_out.bias",
        "conditioner.conditioners.context_audio.net.cnn.fc_backup.weight",
        "conditioner.conditioners.context_audio.net.cnn.fc_backup.bias"}


def _a_trainable_name(w):
    return next(n for n, p in w.diffusion.named_parameters() if p.requires_grad and n not in kit.NO_GRAD_PARAMS)


def test_only_never_gradient_parameters_may_lack_optimizer_state(tiny_ckpt, monkeypatch):
    import copy
    ck, cfg, w = tiny_ckpt
    named = [n for n, _ in w.diffusion.named_parameters()]
    name = _a_trainable_name(w)
    idx = named.index(name)
    with_state = copy.deepcopy(ck)
    without = copy.deepcopy(ck)
    del without["optimizer_states"][0]["state"][idx]
    # not exempt: a missing state is a FAIL naming the parameter
    got = kit.checkpoint_state_reasons(without, cfg, "/ck", w)
    assert any("optimizer state" in r and name in r for r in got), got
    # exempt: the missing state is expected, and a PRESENT state is a FAIL (the set is exact)
    monkeypatch.setattr(kit, "NO_GRAD_PARAMS", kit.NO_GRAD_PARAMS | {name})
    assert kit.checkpoint_state_reasons(without, cfg, "/ck", w) == []
    got = kit.checkpoint_state_reasons(with_state, cfg, "/ck", w)
    assert any("optimizer state" in r and name in r for r in got), got


def test_frozen_param_reasons(tiny_ckpt, monkeypatch):
    import copy
    ck, cfg, w = tiny_ckpt
    name = _a_trainable_name(w)
    monkeypatch.setattr(kit, "NO_GRAD_PARAMS", frozenset({name}))
    last = copy.deepcopy(ck)
    assert kit.frozen_param_reasons(ck, last) == []
    last["state_dict"]["diffusion." + name] = last["state_dict"]["diffusion." + name] + 1e-7
    assert any("changed" in r for r in kit.frozen_param_reasons(ck, last))
    del last["state_dict"]["diffusion." + name]
    assert any("only one" in r for r in kit.frozen_param_reasons(ck, last))
    monkeypatch.setattr(kit, "NO_GRAD_PARAMS", frozenset({"not.a.parameter"}))
    assert kit.frozen_param_reasons(ck, ck) == []                    # absent from both: nothing to compare


def test_admission_state_subprocess_runs_the_live_kit_on_the_tree(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, env=None, **kw):
        seen.update(cmd=cmd, cwd=cwd)
        return 0, json.dumps({"tree": os.path.realpath(str(tmp_path)), "reasons": []}) + "\n", ""

    monkeypatch.setattr(kit, "run", fake_run)
    st = kit.admission_state_subprocess(str(tmp_path), "/x.ckpt", "/ck", hf_sealed=str(tmp_path / "hub"))
    cmd = seen["cmd"]
    assert st["tree"] == os.path.realpath(str(tmp_path)) and seen["cwd"] == str(tmp_path)
    assert cmd[1] == "-B" and cmd[2] == os.path.abspath(kit.__file__) and cmd[3] == "admission-state"
    assert cmd[cmd.index("--tree") + 1] == str(tmp_path)


# --------------------------------------------------------------------------- #
# sealed HF hub and the AR gates
# --------------------------------------------------------------------------- #
def _fake_hub(tmp_path, cfg=b"{}", weights=b"w"):
    src = tmp_path / "src" / kit.DINO_REPO_DIR / "snapshots" / kit.DINO_REVISION
    src.mkdir(parents=True)
    (src / "config.json").write_bytes(cfg)
    (src / "model.safetensors").write_bytes(weights)
    return tmp_path / "src"


def test_sealed_hub_build_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(kit, "DINO_CONFIG_SHA256", hashlib.sha256(b"{}").hexdigest())
    monkeypatch.setattr(kit, "DINO_SHA256", hashlib.sha256(b"w").hexdigest())
    dest = tmp_path / "sealed"
    kit.build_hf_sealed(str(_fake_hub(tmp_path)), str(dest))
    assert kit.hf_sealed_reasons(str(dest)) == []
    ref = dest / "hub" / kit.DINO_REPO_DIR / "refs" / "main"
    os.chmod(ref.parent, 0o755)
    os.chmod(ref, 0o644)
    ref.write_text("0" * 40)
    assert kit.hf_sealed_reasons(str(dest))
    with pytest.raises(kit.Refused, match="pinned"):
        kit.build_hf_sealed(str(_fake_hub(tmp_path / "bad", weights=b"corrupt")), str(tmp_path / "s2"))


def test_ar_counts_and_verify_receipt(tmp_path):
    ar = tmp_path / "AR"
    for sub in kit.AR_SUBDIRS:
        d = ar / sub / "Cafe" / "Cafe_idx_0"
        d.mkdir(parents=True)
        (d / "a.bin").write_bytes(b"1")
    man = kit.ar_manifest(str(ar), [("Cafe", "Cafe_idx_0")], content=True, workers=1)
    assert kit.ar_counts_reasons(str(ar), [("Cafe", "Cafe_idx_0")], man) == []
    (ar / "single_channel_ir_1" / "Cafe" / "Cafe_idx_0" / "b.bin").write_bytes(b"2")      # added file
    assert kit.ar_counts_reasons(str(ar), [("Cafe", "Cafe_idx_0")], man)
    (ar / "single_channel_ir_1" / "Cafe" / "Cafe_idx_0" / "b.bin").unlink()
    (ar / "metadata" / "Cafe" / "Cafe_idx_0" / "a.bin").write_bytes(b"9")                # same-size mutation
    again = kit.ar_manifest(str(ar), [("Cafe", "Cafe_idx_0")], content=True, workers=1)
    assert again != man, "a same-size content change must change the full content manifest"
    pinned = {"rooms": man, "digest": kit.digest_obj(man)}
    base = {"verdict": "PASS", "manifest_digest": kit.digest_obj(man), "got_digest": kit.digest_obj(man),
            "ar_root": os.path.realpath(str(ar)), "t": time.time(), "seconds": 1.0, "differing_rooms": []}
    r = str(tmp_path / "r.json")
    kit.atomic_write_json(r, base)
    assert kit.ar_verify_reasons(r, pinned, str(ar)) == []
    now = base["t"]
    mutations = [("t", now - 49 * 3600), ("t", now + 3600), ("t", True), ("t", "1"),
                 ("verdict", "FAIL"), ("differing_rooms", ["Cafe/Cafe_idx_0"]),
                 ("manifest_digest", "0" * 64), ("got_digest", "0" * 64), ("ar_root", "/elsewhere")]
    for key, val in mutations:
        kit.atomic_write_json(r, {**base, key: val})
        assert kit.ar_verify_reasons(r, pinned, str(ar), now=now), (key, val)
    kit.atomic_write_json(r, {**base, "extra": 1})
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)
    kit.atomic_write_json(r, {k: v for k, v in base.items() if k != "seconds"})
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)
    kit.atomic_write_json(r, base)
    assert kit.ar_verify_reasons(r, {"rooms": man, "digest": "0" * 64}, str(ar), now=now)
    (tmp_path / "r.json").write_text("{trunc")
    assert kit.ar_verify_reasons(r, pinned, str(ar), now=now)


def test_driver_pin_is_enforced_on_a6000_and_neuronic_is_refused():
    assert kit.driver_reasons("a6000", kit.DRIVER_A6000) == []
    assert kit.driver_reasons("a6000", "580.1")
    assert kit.driver_reasons("a6000", None)
    assert kit.driver_reasons("neuronic", "550.54.15")              # exp_43: no neuronic platform
    assert kit.driver_reasons("neuronic", kit.DRIVER_A6000)
    assert kit.driver_reasons("neuronic", "")


def _assets(hub="/sealed"):
    snap = os.path.join(hub, "hub", kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)
    return kit.current_assets(hub, [os.path.join(snap, "model.safetensors"), os.path.join(snap, "config.json")])


def _admission(tmp_path, wfx, **over):
    """An exact admission record over a scratch NAS: 16 cadence files + MANIFEST, and a copy of
    the module's REAL W&B archive (+ its real history CSV binding, plan §5.4)."""
    import shutil
    run_id = RID
    nas = tmp_path / "nas" / run_id
    nas.mkdir(parents=True)
    dest = kit.wandb_dest(run_id, str(tmp_path / "nas"))
    shutil.copytree(wfx["archive"]["dest_dir"], dest)
    ck = {}
    for name in kit.EXPECTED_CKPTS:
        (nas / name).write_bytes(name.encode())
        ck[name] = hashlib.sha256(name.encode()).hexdigest()
    kit.write_manifest(str(nas / "MANIFEST.sha256"), ck)
    final = kit.ckpt_name(kit.ENDPOINT)
    rec = {"kind": "exp43_admission", "admitted": True, "arm": "RF", "run_id": run_id,
           "platform": "a6000", "rung": "2x32", "train_sha": "a" * 40, "runtree_digest": "1" * 64,
           "env_digest": "2" * 64, "nas_dir": str(nas), "checkpoints": ck,
           "manifest_sha256": kit.sha256_file(str(nas / "MANIFEST.sha256")),
           "ckpt_path": str(nas / final), "sha256": ck[final], "bytes": 9, "global_step": 40000,
           "model_config": kit.ARM_CONFIG, "model_config_sha256": kit.PINS[kit.ARM_CONFIG],
           "embedded_config_sha256": "3" * 64, "ema_inventory_sha256": "4" * 64,
           "strict_load": True, "full_evidence_sha256": "5" * 64, "check_sha256": "6" * 64,
           "audit_sha256": "7" * 64, "historical": dict(kit.HISTORICAL), "current_assets": _assets(),
           "trajectory_equivalence_claim_allowed": False, "note": "", "t": 1.0,
           "wandb_archive": dict(wfx["binding"], dest_dir=dest), "train_history": dict(wfx["history"])}
    rec.update(over)
    path = tmp_path / "admission.json"
    kit.atomic_write_json(str(path), rec)
    return str(path), rec, nas


def test_ckpt_names_are_the_lightning_epoch_step_names():
    assert kit.ckpt_name(2500) == "epoch=0-step=2500.ckpt"
    assert kit.ckpt_name(5000) == "epoch=1-step=5000.ckpt"
    assert kit.ckpt_name(40000) == "epoch=8-step=40000.ckpt"
    assert len(kit.EXPECTED_CKPTS) == 16


def test_admission_validator_accepts_the_exact_record_and_rehashes(tmp_path, wandb_fx):
    path, rec, nas = _admission(tmp_path, wandb_fx)
    for mode in (None, "final", "all"):
        reasons, got, sha = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=mode)
        assert reasons == [] and got == rec and sha == kit.sha256_file(path), (mode, reasons)


@pytest.mark.parametrize("key,val", [
    ("admitted", 1), ("arm", "YAW"), ("kind", "admission"), ("run_id", "exp43_RF_neuronic_2x32"),
    ("platform", "neuronic"), ("rung", "8x8"), ("train_sha", "a" * 39), ("sha256", "0" * 64),
    ("ckpt_path", "/elsewhere/epoch=8-step=40000.ckpt"), ("global_step", 40000.0), ("global_step", True),
    ("bytes", 0), ("model_config", "x.json"), ("strict_load", False), ("audit_sha256", "Z" * 64),
    ("trajectory_equivalence_claim_allowed", None), ("note", None), ("t", True), ("arm", "C16"),
    ("kind", "exp40_admission"), ("wandb_archive", None), ("train_history", None)])
def test_admission_validator_rejects_every_field_mutation(tmp_path, wandb_fx, key, val):
    path, rec, nas = _admission(tmp_path, wandb_fx, **{key: val})
    reasons, got, _ = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=None)
    assert reasons and got is None


def test_admission_validator_rejects_schema_and_nas_mutations(tmp_path, wandb_fx):
    path, rec, nas = _admission(tmp_path, wandb_fx)
    root = str(tmp_path / "nas")
    kit.atomic_write_json(path, {**rec, "extra": 1})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, {k: v for k, v in rec.items() if k != "note"})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, {**rec, "checkpoints": {k: v for k, v in list(rec["checkpoints"].items())[1:]}})
    assert kit.admission_reasons(path, nas_root=root)[0]
    kit.atomic_write_json(path, rec)
    for name in kit.EXPECTED_CKPTS:                       # every cadence file, one at a time
        p = nas / name
        old = p.read_bytes()
        p.write_bytes(old + b"x")
        assert kit.admission_reasons(path, nas_root=root, rehash="all")[0], name
        if name == kit.ckpt_name(kit.ENDPOINT):
            assert kit.admission_reasons(path, nas_root=root, rehash="final")[0]
        p.write_bytes(old)
    assert kit.admission_reasons(path, nas_root=root, rehash="all")[0] == []
    (nas / "epoch=8-step=40000.ckpt.partial").write_bytes(b"")
    assert kit.admission_reasons(path, nas_root=root, rehash="all")[0]
    (nas / "epoch=8-step=40000.ckpt.partial").unlink()
    (tmp_path / "admission.json").write_text("{trunc")
    assert kit.admission_reasons(path, nas_root=root)[0]


# --------------------------------------------------------------------------- #
# code review 1: bounded group termination (B4), lock-record failure (H3)
# --------------------------------------------------------------------------- #
def test_terminate_group_escalates_to_sigkill_for_a_term_ignoring_group():
    proc = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 60 & wait"], start_new_session=True)
    try:
        time.sleep(0.3)
        assert kit.group_alive(proc.pid)
        assert kit.terminate_group(proc.pid, grace_s=0.5, kill_wait_s=5, poll_s=0.05) is True
        assert not kit.group_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    with pytest.raises(kit.Refused):
        kit.terminate_group(os.getpgrp())


def test_gpu_lock_record_failure_releases_the_locks(tmp_path, capsys):
    hs = tmp_path / "hs"
    hs.mkdir()
    rc = kit.main(["gpu-lock", "--uuids", U0, "--handshake-dir", str(hs), "--payload-json", "{}",
                   "--out", str(tmp_path / "missing_dir" / "locks.json")])
    assert rc == 2 and "released" in capsys.readouterr().out
    assert not (hs / f"GPU_LOCK_{U0}").exists()                                # nothing leaked
    assert kit.main(["gpu-lock", "--uuids", U0, "--handshake-dir", str(hs), "--payload-json", "{}",
                     "--out", str(tmp_path / "locks.json")]) == 0
    assert (hs / f"GPU_LOCK_{U0}").exists()
    assert kit.main(["gpu-unlock", "--lock-json", str(tmp_path / "locks.json")]) == 0


def test_launchers_install_signal_safe_cleanup_before_taking_locks():
    for name in ("exp43_train_local.sh",):                      # exp_43: the A6000 launcher only
        text = (_EXP / name).read_text()
        trap = text.index("trap cleanup EXIT")
        assert "trap 'on_signal INT' INT" in text and "trap 'on_signal TERM' TERM" in text
        assert "stop-group --pgid \"$TRAIN_PID\"" in text
        assert trap < text.index('"$PY" "$KIT" gpu-lock'), "the trap must precede lock acquisition"
        assert "gpu-unlock" in text[text.index("cleanup() {"):trap]
        assert 'bad=1' in text and "exit 12" in text
        assert trap < text.index("exec setsid")


def test_local_launcher_interrupt_releases_locks_and_kills_only_its_group(tmp_path):
    """B4 end-to-end on a stub: the launcher's cleanup block, extracted verbatim, run under a
    fake kit; SIGTERM while waiting -> the owned group is terminated and the lock released."""
    text = (_EXP / "exp43_train_local.sh").read_text()
    block = text[text.index("TRAIN_PID=\"\"; WATCH_PID=\"\""):text.index('"$PY" "$KIT" gpu-lock')]
    fake = tmp_path / "fakekit.py"
    fake.write_text(
        "import sys, os, json\n"
        "sys.path.insert(0, %r)\n" % str(_EXP) +
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('k', %r)\n" % str(_EXP / "exp43_kit.py") +
        "k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\n"
        "sys.exit(k.main(sys.argv[1:]))\n")
    hs = tmp_path / "hs"
    hs.mkdir()
    script = tmp_path / "launcher.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"PY={sys.executable}; KIT={fake}; MODE=FULL; TMPDIR={tmp_path / 'tmpdir'}; mkdir -p $TMPDIR\n"
        f"LOCK_JSON={tmp_path / 'locks.json'}\n"
        "die() { echo \"REFUSED: $*\"; exit 2; }\n" + block +
        f"\"$PY\" \"$KIT\" gpu-lock --uuids {U0} --handshake-dir {hs} --payload-json '{{}}' --out \"$LOCK_JSON\" || die lock\n"
        "( exec setsid sleep 300 ) &\nTRAIN_PID=$!\n"
        f"echo $TRAIN_PID > {tmp_path / 'pid'}\n"
        "wait \"$TRAIN_PID\"\n")
    proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        if (tmp_path / "pid").exists() and (hs / f"GPU_LOCK_{U0}").exists():
            break
        time.sleep(0.1)
    pid = int((tmp_path / "pid").read_text())
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 130, out
    assert "INTERRUPTED by SIGTERM" in out
    assert not kit.group_alive(pid)
    assert not (hs / f"GPU_LOCK_{U0}").exists() and (tmp_path / "locks_released.json").exists()
    assert not (tmp_path / "tmpdir").exists()


# --------------------------------------------------------------------------- #
# code review 1 H2: re-hash manifested destinations, orphan adoption, killable copy, NAS probe
# --------------------------------------------------------------------------- #
def test_an_already_manifested_but_mutated_nas_file_is_not_acknowledged(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))
    (nas / RID / p.name).write_bytes(b"rotted" * 50)
    with pytest.raises(kit.Refused, match="re-hash"):
        kit.publish_saved(str(p), RID, timeout_s=60, dest_root=str(nas))


def test_orphan_adoption_rehashes_the_source(tmp_path, monkeypatch):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    (nas / RID).mkdir(parents=True)
    p = _ckpt(src, 5000)
    (nas / RID / p.name).write_bytes(p.read_bytes())
    real, calls = kit.sha256_file, []

    def flaky(path, *a):
        calls.append(path)
        return "0" * 64 if (path == str(p) and len(calls) > 2) else real(path, *a)
    monkeypatch.setattr(kit, "sha256_file", flaky)
    with pytest.raises(kit.Refused, match="orphan was adopted"):
        kit.publish_one(str(src), p.name, str(nas / RID))


def test_publication_timeout_kills_the_copy_process_group(tmp_path, monkeypatch):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    slow = tmp_path / "slowpy"
    slow.write_text(f"#!/bin/bash\necho $$ > {tmp_path / 'copy.pid'}\nsleep 120\n")
    slow.chmod(0o755)
    t0 = time.time()
    with pytest.raises(kit.Refused, match="exceeded"):
        kit.publish_saved(str(p), RID, timeout_s=1.0, dest_root=str(nas), python=str(slow))
    assert time.time() - t0 < 60
    pid = int((tmp_path / "copy.pid").read_text())
    assert not kit.group_alive(pid), "the timed-out publication must not keep writing"


def test_nas_capability_probe_and_receipt_validation(tmp_path):
    root = os.path.realpath(str(tmp_path))
    rec = kit.nas_capability_probe(root, size=1 << 16)
    assert rec["verdict"] == "PASS" and len(rec["checks"]) == 5, rec
    path = str(tmp_path / "nas_capability_a6000.json")
    kit.atomic_write_json(path, rec)
    assert kit.nas_receipt_reasons(path, dest_root=root) == []
    for key, val, needle in [("verdict", "FAIL", "did not PASS"), ("dest_root", "/x", "tested"),
                             ("t", time.time() - 49 * 3600, "age"), ("t", time.time() + 600, "age"),
                             ("checks", {**rec["checks"], "o_excl_create": "FAILED"}, "each True"),
                             ("checks", {f"renamed_{i}": True for i in range(5)}, "not exactly"),
                             ("mount", {**rec["mount"], "findmnt_rc": 2}, "findmnt failed"),
                             ("mount", {**rec["mount"], "source": ""}, "empty or malformed"),
                             ("mount", {"source": rec["mount"]["source"]}, "exact schema")]:
        kit.atomic_write_json(path, {**rec, key: val})
        assert any(needle in x for x in kit.nas_receipt_reasons(path, dest_root=root)), (key, needle)
    kit.atomic_write_json(path, {**rec, "more": 1})
    assert any("exact schema" in x for x in kit.nas_receipt_reasons(path, dest_root=root))
    kit.atomic_write_json(path, rec)
    moved = {**rec["mount"], "source": "//10.0.0.9/other", "fstype": "cifs"}
    assert any("now on" in x for x in kit.nas_receipt_reasons(path, dest_root=root, current_mount=moved))


def test_mount_identity_takes_the_innermost_mount_of_an_autofs_path(monkeypatch, tmp_path):
    monkeypatch.setattr(kit, "run", lambda cmd, **k: (0, "/etc/auto.nas autofs\n//10.0.0.1/share cifs\n", ""))
    m = kit.mount_identity(str(tmp_path))
    assert (m["source"], m["fstype"]) == ("//10.0.0.1/share", "cifs")


def test_atomic_write_json_leaves_no_temp_on_failure(tmp_path):
    with pytest.raises(TypeError):
        kit.atomic_write_json(str(tmp_path / "x.json"), {"bad": object()})
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- #
# code review 1 B5 / B6 / M6: strict FULL evidence, audit, run result, derived admission
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def full_stream():
    return kit.expected_stream(kit.ROOT, kit.ENDPOINT, range(2), 32, verbatim=False)


def _fit(r, world=2):
    sig = hashlib.sha256(b"state").hexdigest()
    return {"kind": "fit_start", "rank": r, "world_size": world, "local_rank": r,
            "gpu_name": "NVIDIA RTX A6000", "gpu_uuid": [U0, U1][r][4:], "syncbn_modules": 20,
            "batchnorm_modules": 20, "yaw_aug": dict(kit.YAW_BLOCK), "flash_attn_func_is_none": False,
            "state_signature": sig, "param_signature": hashlib.sha256(b"param").hexdigest(),
            "rng_signature": {k: hashlib.sha256(k.encode()).hexdigest() for k in sorted(kit.RNG_KEYS)},
            "global_step": 0,
            "numerics": {"matmul_allow_tf32": True, "cudnn_allow_tf32": True, "cudnn_benchmark": False,
                         "cudnn_deterministic": False, "float32_matmul_precision": "medium"}}


def _summary(r, e, steps, published):
    keys = sorted(str(t) for t in kit.SAMPLED_STEPS if t < steps)
    return {"rank": r, "counts": {"apply_yaw_aug": steps, "training_step": steps, "ckpt_reentrant": 0,
                                  "ckpt_default": 0, "ckpt_unclassified": 0, "flash_calls": 9,
                                  "ckpt_dinov3": 9, "ckpt_dit": 9, "checkpoints_published": published,
                                  "mirror_calls": e["flips"], "flip_anomalies": 0, "stream_mismatch_steps": 0},
            "stream_digest": e["digest"], "offset_histogram": e["hist"], "n_offsets": e["n"],
            "sampled_offsets": {k: e["sampled"][k] for k in keys},
            "sampled_flips": {k: e["sampled_flips"][k] for k in keys}, "flips_applied": e["flips"],
            "sampled_losses": {k: (2.5).hex() for k in keys}, "n_steps": steps, "global_step": steps,
            "steps_contiguous_from_0": True, "all_losses_finite": True, "pid": 1000 + r}


_SMOKE = {}
_REAL_EXPECTED_STREAM = kit.expected_stream          # never the monkeypatched stand-in (no -k artefact)


def _real_smoke_stream():
    if "s" not in _SMOKE:
        _SMOKE["s"] = _REAL_EXPECTED_STREAM(kit.ROOT, kit.SMOKE_STEPS["DDP_SMOKE"], range(2), 32, verbatim=False)
    return _SMOKE["s"]


@pytest.fixture(scope="module")
def smoke_stream():
    return _real_smoke_stream()


def _streams(full_stream):
    """expected_stream stand-in: the real 40k stream for FULL, the real 30-step one otherwise."""
    return lambda tree, steps, *a, **k: full_stream if steps == kit.ENDPOINT else _real_smoke_stream()


def _full_records(tmp_path, stream, run_id=RID, smoke=None, preserved_sha="8" * 64):
    """The exact producer shapes: the check record, the DDP_SMOKE reference receipt and the
    FULL evidence of a 2x32 A6000 run, bound by sha256."""
    sha = "b" * 40
    argv = kit.train_argv("a6000", "2x32", "FULL", run_id, "/save")
    sargv = kit.train_argv("a6000", "2x32", "DDP_SMOKE", run_id, "/smoke", "2026-11-08_00-00-00")
    fits = {str(r): _fit(r) for r in range(2)}
    links = {"AcousticRooms": "/ar", kit.VAE: "/vae"}
    check = {"verdict": "PASS", "reasons": [], "sha": sha, "tree_digest": "1" * 64, "env_digest": "2" * 64,
             "argv": argv, "links": links}
    rec_dir = tmp_path / "receipts"
    rec_dir.mkdir(exist_ok=True)
    kit.atomic_write_json(str(rec_dir / "check.json"), check)
    kst = smoke if smoke is not None else _real_smoke_stream()
    ref = {"verdict": "PASS", "kind": "ddp_smoke", "platform": "a6000", "rung": "2x32", "mode": "DDP_SMOKE",
           "run_id": run_id, "stamp": "2026-11-08_00-00-00", "sha": sha, "tree_digest": "1" * 64,
           "env_digest": "2" * 64, "recipe_projection": kit.recipe_projection(sargv), "argv": sargv,
           "train_rc": 0, "watcher_rc": 0, "publisher_rc": 0, "reference_sha256": None,
           "evidence": {"verdict": "PASS", "reasons": []}, "fit_start": fits,
           "summaries": {str(r): _summary(r, kst[r], 30, 0) for r in range(2)}, "check_sha256": "3" * 64,
           "wandb_preserved_sha256": "9" * 64}
    kit.atomic_write_json(str(rec_dir / "ref.json"), ref)
    ev = {"verdict": "PASS", "kind": "full_evidence", "platform": "a6000", "rung": "2x32", "mode": "FULL",
          "run_id": run_id, "stamp": "2026-11-08_01-00-00", "sha": sha, "tree_digest": "1" * 64,
          "env_digest": "2" * 64, "recipe_projection": kit.recipe_projection(argv), "argv": argv,
          "train_rc": 0, "watcher_rc": 0, "publisher_rc": 0,
          "reference_sha256": kit.sha256_file(str(rec_dir / "ref.json")),
          "evidence": {"verdict": "PASS", "reasons": []}, "fit_start": fits,
          "summaries": {str(r): _summary(r, stream[r], 40000, 16 if r == 0 else 0) for r in range(2)},
          "check_sha256": kit.sha256_file(str(rec_dir / "check.json")), "wandb_preserved_sha256": preserved_sha}
    kit.atomic_write_json(str(rec_dir / "ev.json"), ev)
    return rec_dir, ev


def test_the_reference_receipt_validator_is_exact(tmp_path, full_stream, smoke_stream):
    """code-r2 B3: the DDP_SMOKE receipt -- every produced field, nested exactly."""
    d, ev = _full_records(tmp_path, full_stream, smoke=smoke_stream)
    ref = kit.read_json(str(d / "ref.json"))
    args = ("ddp_smoke", "a6000", "2x32", "b" * 40, "1" * 64, "2" * 64, ev["recipe_projection"])
    assert kit.receipt_reasons(ref, *args) == []
    import copy
    cases = [
        (lambda x: x.pop("publisher_rc"), "exact schema"), (lambda x: x.update(extra=1), "exact schema"),
        (lambda x: x.update(mode="FULL"), "mode"), (lambda x: x.update(run_id="exp43_RF_neuronic_2x32"), "run_id"),
        (lambda x: x.update(stamp=""), "stamp"), (lambda x: x.update(publisher_rc=1), "publisher_rc"),
        (lambda x: x.update(wandb_preserved_sha256=None), "wandb_preserved_sha256"),
        (lambda x: x.pop("wandb_preserved_sha256"), "exact schema"),
        (lambda x: x.update(reference_sha256="0" * 64), "reference_sha256"),
        (lambda x: x["argv"].__setitem__(x["argv"].index("--max-steps") + 1, "29"), "--max-steps"),
        (lambda x: x["fit_start"]["1"].pop("gpu_uuid"), "fit_start keys"),
        (lambda x: x["fit_start"]["1"].update(gpu_uuid=x["fit_start"]["0"]["gpu_uuid"]), "same GPU"),
        (lambda x: x["fit_start"]["0"].update(gpu_name="NVIDIA L40"), "gpu_name"),
        (lambda x: x["fit_start"]["0"].update(syncbn_modules=0), "syncbn_modules"),
        (lambda x: x["fit_start"]["0"].update(batchnorm_modules=21), "syncbn_modules"),
        (lambda x: x["fit_start"]["1"].update(local_rank=0), "local_rank"),
        (lambda x: x["fit_start"]["0"].update(rank=True), "rank"),
        (lambda x: x["fit_start"]["0"]["numerics"].pop("cudnn_benchmark"), "numerics"),
        (lambda x: x["fit_start"]["0"]["yaw_aug"].update(enabled=1), "yaw_aug"),
        (lambda x: x["summaries"]["1"].pop("pid"), "summary keys"),
        (lambda x: x["summaries"]["1"].update(rank=0), "summary of rank"),
        (lambda x: x["summaries"]["0"]["counts"].pop("ckpt_dit"), "counters"),
        (lambda x: x["summaries"]["0"]["counts"].update(ckpt_dit=True), "counters"),
        (lambda x: x["summaries"]["0"]["offset_histogram"].__setitem__(0, 0), "histogram"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"].pop("3"), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"]["0"].__setitem__(0, 512), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"]["0"].__setitem__(0, -1), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_offsets"]["0"].__setitem__(0, True), "sampled offsets"),
        (lambda x: x["summaries"]["0"]["sampled_flips"]["0"].__setitem__(0, 2), "sampled flips"),
        (lambda x: x["summaries"]["0"]["sampled_flips"]["1"].__setitem__(0, True), "sampled flips"),
        (lambda x: x["summaries"]["1"]["sampled_flips"].pop("3"), "sampled flips"),
        (lambda x: x["summaries"]["1"]["sampled_flips"]["2"].pop(), "sampled flips"),
        (lambda x: x["summaries"]["0"].update(flips_applied=0), "flips_applied"),
        (lambda x: x["summaries"]["0"].update(flips_applied=True), "flips_applied"),
        (lambda x: x["summaries"]["0"].update(flips_applied=30 * 32 + 1), "flips_applied"),
        (lambda x: x["summaries"]["0"]["counts"].update(mirror_calls=1), "flips_applied"),
        (lambda x: x["summaries"]["0"]["counts"].update(flip_anomalies=1), "flip_anomalies"),
        (lambda x: x["summaries"]["1"]["counts"].update(stream_mismatch_steps=1), "stream_mismatch_steps"),
        (lambda x: x["summaries"]["1"]["counts"].pop("mirror_calls"), "counters"),
        (lambda x: x["summaries"]["0"].pop("sampled_flips"), "summary keys"),
        (lambda x: x["summaries"]["0"].pop("flips_applied"), "summary keys"),
        (lambda x: x["summaries"]["0"].update(stream_digest="x"), "stream_digest"),
        (lambda x: x["summaries"]["0"]["sampled_losses"].update({"0": "nan"}), "sampled losses"),
        (lambda x: x["summaries"]["0"]["sampled_losses"].update({"0": (float("inf")).hex()}), "sampled losses"),
    ]
    for mutate, needle in cases:
        bad = copy.deepcopy(ref)
        mutate(bad)
        got = kit.receipt_reasons(bad, *args)
        assert any(needle in x for x in got), (needle, got)
    # exp_43: the neuronic PILOT receipt kind is gone; a pilot-shaped receipt is refused by kind
    pilot = dict(copy.deepcopy(ref), kind="pilot", mode="PILOT", platform="neuronic", run_id="exp43_RF_neuronic_2x32",
                 sec_per_step=2.0)
    for kind in ("pilot", "smoke", "full_evidence"):
        assert any("receipt kind" in x for x in kit.receipt_reasons(pilot, kind, *args[1:])), kind
    assert kit.receipt_reasons(pilot, *args)


def test_full_evidence_validator_accepts_the_bound_record(tmp_path, full_stream, monkeypatch):
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    d, ev = _full_records(tmp_path, full_stream)
    r, sha, rec = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"),
                                            str(d / "ref.json"), recompute=True)
    assert r == [] and rec == ev and sha == kit.sha256_file(str(d / "ev.json"))


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: e.update(verdict="FAIL"), "verdict"), (lambda e: e.update(kind="ddp_smoke"), "kind"),
    (lambda e: e.update(mode="PILOT"), "mode"), (lambda e: e.update(platform="neuronic"), "platform"),
    (lambda e: e.update(rung="8x8"), "rung"), (lambda e: e.update(sha="z"), "40-hex"),
    (lambda e: e.update(tree_digest="x"), "tree_digest"), (lambda e: e.update(train_rc=1), "train_rc"),
    (lambda e: e.update(watcher_rc=False), "watcher_rc"), (lambda e: e.update(publisher_rc=3), "publisher_rc"),
    (lambda e: e.update(evidence={"verdict": "PASS", "reasons": ["x"]}), "evidence"),
    (lambda e: e["argv"].__setitem__(e["argv"].index("--seed") + 1, "43"), "recipe projection"),
    (lambda e: e["summaries"]["0"]["counts"].update(checkpoints_published=15), "checkpoints_published"),
    (lambda e: e["summaries"]["1"]["counts"].update(checkpoints_published=1), "checkpoints_published"),
    (lambda e: e["summaries"]["1"]["counts"].update(ckpt_reentrant=1), "ckpt_reentrant"),
    (lambda e: e["summaries"]["0"]["counts"].update(flash_calls=0), "flash_calls"),
    (lambda e: e["summaries"]["0"]["counts"].update(apply_yaw_aug=39999), "apply_yaw_aug"),
    (lambda e: e["summaries"]["1"].update(all_losses_finite=None), "all_losses_finite"),
    (lambda e: e["summaries"]["0"].update(steps_contiguous_from_0=False), "contiguous"),
    (lambda e: e["summaries"]["1"].update(stream_digest="0" * 64), "recomputation"),
    (lambda e: e["summaries"]["1"].update(flips_applied=e["summaries"]["1"]["flips_applied"] + 1), "flips_applied"),
    (lambda e: [e["summaries"]["0"].update(flips_applied=e["summaries"]["0"]["flips_applied"] + 1),
                e["summaries"]["0"]["counts"].update(mirror_calls=e["summaries"]["0"]["flips_applied"])],
     "RF recomputation"),
    (lambda e: e["summaries"]["0"]["sampled_flips"]["2500"].__setitem__(3, 1 - e["summaries"]["0"]["sampled_flips"]["2500"][3]),
     "recomputation"),
    (lambda e: e["summaries"]["1"]["sampled_offsets"]["39999"].__setitem__(0, (e["summaries"]["1"]["sampled_offsets"]["39999"][0] + 1) % 512),
     "recomputation"),
    (lambda e: e["summaries"]["0"]["counts"].update(stream_mismatch_steps=1), "stream_mismatch_steps"),
    (lambda e: e["summaries"]["1"]["counts"].update(flip_anomalies=3), "flip_anomalies"),
    (lambda e: e["summaries"]["0"].update(flips_applied=0), "flips_applied"),
    (lambda e: e["summaries"]["0"].pop("sampled_flips"), "summary keys"),
    (lambda e: e["summaries"].pop("1"), "exactly ranks"),
    (lambda e: e["fit_start"]["0"].update(yaw_aug={"enabled": True}), "yaw_aug"),
    (lambda e: e["fit_start"]["1"].update(rng_signature={"python": "other"}), "rng_signature"),
    (lambda e: e.update(reference_sha256="0" * 64), "reference receipt is not the one"),
    (lambda e: e.update(check_sha256="0" * 64), "check record is not the one"),
    (lambda e: e.update(extra=1), "exact schema"), (lambda e: e.pop("stamp"), "exact schema"),
    (lambda e: e.update(wandb_preserved_sha256="x" * 64), "wandb_preserved_sha256"),
    (lambda e: e.pop("wandb_preserved_sha256"), "exact schema"),
    (lambda e: e.update(stamp="x"), "stamp"),
    (lambda e: e["fit_start"]["0"].pop("gpu_uuid"), "fit_start keys"),
    (lambda e: e["fit_start"]["1"].pop("batchnorm_modules"), "fit_start keys"),
    (lambda e: e["fit_start"]["0"].update(gpu_name="NVIDIA L40"), "gpu_name"),
    (lambda e: e["fit_start"]["1"].update(rank=0), "fit_start rank"),
    (lambda e: e["summaries"]["0"].pop("sampled_losses"), "summary keys"),
    (lambda e: e["summaries"]["1"].pop("rank"), "summary keys"),
    (lambda e: e["summaries"]["0"]["counts"].update(extra_counter=0), "counters"),
])
def test_full_evidence_validator_rejects_every_critical_mutation(tmp_path, full_stream, monkeypatch, mutate, needle):
    import copy
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    d, ev = _full_records(tmp_path, full_stream)
    bad = copy.deepcopy(ev)
    mutate(bad)
    kit.atomic_write_json(str(d / "ev.json"), bad)
    r, _, rec = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"),
                                          str(d / "ref.json"), recompute=True)
    assert rec is None and any(needle in x for x in r), (needle, r)


def test_full_evidence_binding_to_check_and_reference(tmp_path, full_stream):
    d, ev = _full_records(tmp_path, full_stream)
    chk = kit.read_json(str(d / "check.json"))
    kit.atomic_write_json(str(d / "check.json"), {**chk, "env_digest": "9" * 64})
    assert kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"))[0]
    kit.atomic_write_json(str(d / "check.json"), chk)
    ref = kit.read_json(str(d / "ref.json"))
    kit.atomic_write_json(str(d / "ref.json"), {**ref, "tree_digest": "9" * 64})
    assert kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], None, str(d / "ref.json"))[0]
    assert kit.full_evidence_reasons(str(d / "ev.json"), "exp43_RF_a6000_2x32_r1")[0]


@pytest.fixture(scope="module")
def sealed_hub(tmp_path_factory):
    """A real sealed hub built from this box's HF cache (code-r3 B1 tests)."""
    src = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(os.path.join(src, kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)):
        pytest.skip("the pinned DINOv3 snapshot is not in this box's HF cache")
    return kit.build_hf_sealed(src, str(tmp_path_factory.mktemp("sealed") / "hf_sealed"))


def _inject(tree, w, sealed):
    """In-process stand-ins for the admission tree and its subprocess (the real ones are tested
    separately): the tiny wrapper, interpreted from ``tree``; DINOv3 'resolved' in ``sealed``."""
    def builder(repo, sha, check_rec, ev, parent):
        assert check_rec["links"] and sha == ev["sha"]
        return str(tree)

    def state(t, ckpt, dirpath, strict_load=True, hf_sealed=None):
        assert t == str(tree) and hf_sealed == sealed
        st = kit.checkpoint_admission_state(ckpt, dirpath, t, strict_load=strict_load)
        snap = os.path.join(os.path.realpath(sealed), "hub", kit.DINO_REPO_DIR, "snapshots", kit.DINO_REVISION)
        st.update(hf_sealed=os.path.realpath(sealed),
                  dino_resolved=[os.path.join(snap, "model.safetensors"), os.path.join(snap, "config.json")])
        return st
    return {"tree_builder": builder, "state_fn": state, "hf_sealed": sealed}


def _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream):
    """16 cadence checkpoints (the 40k one a real tiny full-state checkpoint), save markers,
    NAS publication through the real protocol, and the bound evidence records."""
    ck, cfg, w = tiny_ckpt
    tree = tmp_path / "tree"
    (tree / kit.EXPREL).mkdir(parents=True)
    (tree / kit.ARM_CONFIG).write_text(json.dumps(cfg))
    (tree / kit.AR_MANIFEST).write_bytes((_REPO / kit.AR_MANIFEST).read_bytes())
    monkeypatch.setattr(kit, "build_wrapper_cpu", lambda t, c: w)
    monkeypatch.setattr(kit, "expected_stream", _streams(full_stream))
    run_id = RID
    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    for n in kit.EXPECTED_CKPTS:
        if n in (kit.ckpt_name(kit.ENDPOINT), kit.ckpt_name(kit.CADENCE)):   # the audit reads both
            torch.save(ck, str(ckdir / n))
        else:
            (ckdir / n).write_bytes(n.encode() * 10)
        kit.mark_saved(str(ckdir / n))
    nas_root = os.path.realpath(str(tmp_path / "nas"))
    for n in kit.EXPECTED_CKPTS:
        kit.publish_pass(str(ckdir), run_id, dest_root=nas_root, only=[n])
    # plan §5.4: the launch's offline W&B run, preserved from $TMPDIR and archived to the NAS
    stage = tmp_path / "tmpdir" / "wandb"
    stage.mkdir(parents=True)
    _write_wandb_run(stage)
    (tmp_path / "train.log").write_text(f"wandb: Starting a new run with run id {W_ID}.\n")
    (tmp_path / "receipts").mkdir(exist_ok=True)
    pre = str(tmp_path / "receipts" / "wandb_preserved.json")
    kit.atomic_write_json(pre, kit.wandb_preserve([str(stage)], str(tmp_path / "keep"), str(tmp_path / "train.log")))
    arc = str(tmp_path / "receipts" / "wandb_archive.json")
    kit.atomic_write_json(arc, kit.wandb_archive(run_id, pre, nas_root=nas_root))
    d, ev = _full_records(tmp_path, full_stream, run_id, preserved_sha=kit.sha256_file(pre))
    return tree, ckdir, nas_root, d, ev


def _rr_files(d):
    return {"check": str(d / "check.json"), "reference": str(d / "ref.json"), "evidence": str(d / "ev.json"),
            "audit": str(d / "audit.json"), "wandb_preserved": str(d / "wandb_preserved.json"),
            "wandb_archive": str(d / "wandb_archive.json")}


def test_full_audit_and_derived_admission_end_to_end(tmp_path, tiny_ckpt, monkeypatch, full_stream, sealed_hub):
    tree, ckdir, nas_root, d, ev = _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream)
    nas = os.path.join(nas_root, ev["run_id"])
    v = kit.audit(str(ckdir), str(tree), nas_dir=nas, evidence=str(d / "ev.json"), run_id=ev["run_id"],
                  expected_dirpath="/ck", check_path=str(d / "check.json"), reference_path=str(d / "ref.json"))
    assert v["verdict"] == "PASS", v["reasons"]
    kit.atomic_write_json(str(d / "audit.json"), v)
    rr = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", _rr_files(d),
                        "/ck", str(ckdir), str(tree), nas_root=nas_root)
    kit.atomic_write_json(str(d / "run_result.json"), rr)
    rec = kit.admit(str(d / "run_result.json"), str(d), str(_REPO), note="test", nas_root=nas_root,
                    **_inject(tree, tiny_ckpt[2], sealed_hub))
    assert set(rec) == kit.ADMISSION_KEYS
    assert (rec["kind"], rec["arm"]) == ("exp43_admission", "RF")
    assert (rec["train_sha"], rec["runtree_digest"], rec["env_digest"]) == (ev["sha"], "1" * 64, "2" * 64)
    wa = kit.read_json(str(d / "wandb_archive.json"))
    assert rec["wandb_archive"]["dest_dir"] == kit.wandb_dest(RID, nas_root) == wa["dest_dir"]
    assert rec["wandb_archive"]["run_file_sha256"] == wa["run_file_sha256"] == rec["train_history"]["run_file_sha256"]
    assert rec["wandb_archive"]["preserved_receipt_sha256"] == ev["wandb_preserved_sha256"]
    h = rec["train_history"]
    assert h["rows"] == 400 and os.path.dirname(h["csv"]).startswith(str(d / f"train_history_{RID}__"))
    assert open(h["csv"]).read().splitlines()[:2] == ["step,train/loss,lr", f"99,{2.0 / (1 + 99 / 1000)!r},5e-05"]
    kit.atomic_write_json(str(tmp_path / "admission.json"), rec)
    assert kit.admission_reasons(str(tmp_path / "admission.json"), nas_root=nas_root, rehash="all")[0] == []
    # every bound W&B / history byte, mutated one at a time, is refused by the "all" re-hash
    for p in sorted(Path(wa["dest_dir"]).rglob("*")) + [Path(h["csv"]), Path(h["sidecar"])]:
        if not p.is_file():
            continue
        os.chmod(p, 0o644)
        old = p.read_bytes()
        p.write_bytes(old + b" ")
        got = kit.admission_reasons(str(tmp_path / "admission.json"), nas_root=nas_root, rehash="all")[0]
        assert got, p
        assert kit.admission_reasons(str(tmp_path / "admission.json"), nas_root=nas_root, rehash="final")[0] == []
        p.write_bytes(old)
    assert kit.admission_reasons(str(tmp_path / "admission.json"), nas_root=nas_root, rehash="all")[0] == []
    # every NAS checkpoint, mutated one at a time, blocks admission
    for n in kit.EXPECTED_CKPTS:
        p = os.path.join(nas, n)
        os.chmod(p, 0o644)
        old = open(p, "rb").read()
        open(p, "wb").write(old + b"!")
        with pytest.raises(kit.Refused):
            kit.admit(str(d / "run_result.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
        open(p, "wb").write(old)
    open(os.path.join(nas, ".epoch=8-step=40000.ckpt.partial-1"), "wb").write(b"")
    with pytest.raises(kit.Refused, match="beyond"):
        kit.admit(str(d / "run_result.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))


def test_admission_refuses_unbound_or_altered_records(tmp_path, tiny_ckpt, monkeypatch, full_stream, sealed_hub):
    tree, ckdir, nas_root, d, ev = _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream)
    nas = os.path.join(nas_root, ev["run_id"])
    v = kit.audit(str(ckdir), str(tree), nas_dir=nas, evidence=str(d / "ev.json"), run_id=ev["run_id"],
                  expected_dirpath="/ck", check_path=str(d / "check.json"), reference_path=str(d / "ref.json"))
    kit.atomic_write_json(str(d / "audit.json"), v)
    files = _rr_files(d)
    rr = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", files, "/ck",
                        str(ckdir), str(tree), nas_root=nas_root)
    kit.atomic_write_json(str(d / "rr.json"), rr)
    kit.atomic_write_json(str(d / "audit.json"), {**v, "full_evidence_sha256": "0" * 64})
    with pytest.raises(kit.Refused, match="not the recorded bytes"):           # run result binds the audit
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    kit.atomic_write_json(str(d / "rr.json"), {**rr, "verdict": "FAIL"})
    with pytest.raises(kit.Refused, match="verdict"):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    kit.atomic_write_json(str(d / "rr.json"), {**rr, "platform": "neuronic"})
    with pytest.raises(kit.Refused):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    with pytest.raises(kit.Refused, match="note"):
        kit.admit(str(d / "rr.json"), str(d), str(_REPO), note={"admitted": False}, nas_root=nas_root,
                  **_inject(tree, tiny_ckpt[2], sealed_hub))


def test_audit_refuses_wrong_names_markers_and_strays(tmp_path):
    for s in kit.CKPT_STEPS:
        (tmp_path / kit.ckpt_name(s)).write_bytes(b"x")
        kit.mark_saved(str(tmp_path / kit.ckpt_name(s)))
    (tmp_path / "epoch=0-step=2500.ckpt.saved").unlink()
    (tmp_path / "stray.txt").write_text("x")
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False)
    assert v["verdict"] == "FAIL"
    assert any("save marker" in r for r in v["reasons"]) and any("unexpected" in r for r in v["reasons"])
    os.rename(tmp_path / kit.ckpt_name(5000), tmp_path / "epoch=0-step=5000.ckpt")   # wrong epoch
    v = kit.audit(str(tmp_path), str(_REPO), strict_load=False)
    assert any("missing" in r for r in v["reasons"])


# --------------------------------------------------------------------------- #
# code review 2: admission interprets the run in an immutable tree at the training commit (H2)
# --------------------------------------------------------------------------- #
def _head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_admission_tree_reproduces_the_training_tree_digest_or_refuses(tmp_path):
    sha = _head()
    links = {"AcousticRooms": "/ar/at/training/time", kit.VAE: "/vae/at/training/time"}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": RID, "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    assert kit.verify_runtree(str(_REPO), sha, tree, {r: os.path.realpath(os.path.join(_REPO, r))
                                                       for r in kit.LINKS}) == []
    assert open(os.path.join(tree, kit.EXPREL, "exp43_kit.py"), "rb").read() == subprocess.run(
        ["git", "show", f"{sha}:{kit.EXPREL}/exp43_kit.py"], cwd=_REPO, capture_output=True, check=True).stdout
    with pytest.raises(kit.Refused, match="run-tree digest"):
        kit.build_admission_tree(str(_REPO), sha, {"links": links}, dict(ev, tree_digest="0" * 64), str(tmp_path))
    with pytest.raises(kit.Refused, match="training-time links"):
        kit.build_admission_tree(str(_REPO), sha, {}, ev, str(tmp_path))


def _require_head_kit(sha):
    """The admission tree runs HEAD's COMMITTED kit: these tests need it to be this revision."""
    if b"def dino_resolved_paths" not in subprocess.run(["git", "show", f"{sha}:{kit.EXPREL}/exp43_kit.py"],
                                                         cwd=_REPO, capture_output=True, check=True).stdout:
        pytest.skip("HEAD's committed kit predates the sealed-hub admission-state (run after committing)")


def test_admission_state_runs_in_the_admission_trees_own_kit(tmp_path, tiny_ckpt, sealed_hub):
    sha = _head()
    _require_head_kit(sha)
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": RID, "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    ck, cfg, w = tiny_ckpt
    torch.save(ck, str(tmp_path / "tiny.ckpt"))
    st = kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False,
                                        hf_sealed=sealed_hub)
    assert st["tree"] == os.path.realpath(tree) and st["sha256"] == kit.sha256_file(str(tmp_path / "tiny.ckpt"))
    assert any("model_config" in r for r in st["reasons"])     # the tiny config is not the RF arm config


def test_an_already_manifested_name_rehashes_the_source_too(tmp_path):
    src, nas = tmp_path / "ckpts", tmp_path / "nas"
    src.mkdir()
    p = _ckpt(src, 2500)
    kit.mark_saved(str(p))
    kit.publish_pass(str(src), RID, dest_root=str(nas), only=[p.name])
    p.write_bytes(b"y" * len(p.read_bytes()))                          # same-size source mutation
    with pytest.raises(kit.Refused):
        kit.publish_pass(str(src), RID, dest_root=str(nas), only=[p.name])


# --------------------------------------------------------------------------- #
# code review 3: RNG schema, balanced histogram corruption, recomputed smoke stream (H1)
# --------------------------------------------------------------------------- #
def test_reference_receipt_rng_schema_bins_and_recomputed_stream(tmp_path, full_stream, smoke_stream):
    import copy
    d, ev = _full_records(tmp_path, full_stream, smoke=smoke_stream)
    ref = kit.read_json(str(d / "ref.json"))
    args = ("ddp_smoke", "a6000", "2x32", "b" * 40, "1" * 64, "2" * 64, ev["recipe_projection"])
    assert kit.receipt_reasons(ref, *args) == [] and kit.receipt_stream_reasons(ref, kit.ROOT, "DDP_SMOKE", "2x32") == []
    for mutate in (lambda f: f.pop("sobol"), lambda f: f.update(extra="0" * 64), lambda f: f.update(numpy="x")):
        bad = copy.deepcopy(ref)
        mutate(bad["fit_start"]["0"]["rng_signature"])
        assert any("rng_signature" in x for x in kit.receipt_reasons(bad, *args))
    bad = copy.deepcopy(ref)
    h = bad["summaries"]["0"]["offset_histogram"]
    h[0], h[1] = -1, h[1] + h[0] + 1                                         # balanced: the sum is unchanged
    assert any("histogram" in x for x in kit.receipt_reasons(bad, *args))
    bad = copy.deepcopy(ref)                                                  # a different VALID stream: rank 1's
    bad["summaries"]["0"] = dict(bad["summaries"]["1"], rank=0)
    assert kit.receipt_reasons(bad, *args) == []
    assert any("recomputation" in x for x in kit.receipt_stream_reasons(bad, kit.ROOT, "DDP_SMOKE", "2x32"))
    bad = copy.deepcopy(ref)
    bad["summaries"]["1"]["sampled_offsets"]["2"] = list(reversed(bad["summaries"]["1"]["sampled_offsets"]["2"]))
    assert any("recomputation" in x for x in kit.receipt_stream_reasons(bad, kit.ROOT, "DDP_SMOKE", "2x32"))
    flipped = copy.deepcopy(ref)                                  # exp_43: the flip bits are recomputed too
    flipped["summaries"]["0"]["sampled_flips"]["3"] = [1 - b for b in flipped["summaries"]["0"]["sampled_flips"]["3"]]
    assert kit.receipt_reasons(flipped, *args) == []
    got = kit.receipt_stream_reasons(flipped, kit.ROOT, "DDP_SMOKE", "2x32")
    assert any("rank 0" in x and "sampled_flips" in x for x in got), got
    # and the FULL validator recomputes the reference's stream when asked to
    kit.atomic_write_json(str(d / "ref.json"), bad)
    ev2 = kit.read_json(str(d / "ev.json"))
    kit.atomic_write_json(str(d / "ev.json"), {**ev2, "reference_sha256": kit.sha256_file(str(d / "ref.json"))})
    r, _, _ = kit.full_evidence_reasons(str(d / "ev.json"), ev["run_id"], str(d / "check.json"), str(d / "ref.json"),
                                        recompute=True, tree=kit.ROOT)
    assert any("DDP_SMOKE receipt rank 1" in x for x in r), r


# --------------------------------------------------------------------------- #
# code review 3: admission resolves DINOv3 ONLY from the sealed hub (B1); provenance (M2)
# --------------------------------------------------------------------------- #
def test_sealed_env_removes_every_competing_hf_variable():
    base = {"HF_HOME": "/home/x/.cache/hf", "HF_HUB_CACHE": "/c", "HUGGINGFACE_HUB_CACHE": "/d",
            "TRANSFORMERS_CACHE": "/e", "HF_DATASETS_CACHE": "/f", "PYTHONPATH": "/p", "PATH": "/bin"}
    e = kit.sealed_env("/sealed", base)
    assert e["HF_HOME"] == "/sealed" and e["HF_HUB_CACHE"] == "/sealed/hub" and e["HF_HUB_OFFLINE"] == "1"
    assert e["TRANSFORMERS_OFFLINE"] == "1" and e["PATH"] == "/bin" and "PYTHONPATH" not in e
    assert not {"HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"} & set(e)


def test_admission_state_resolves_dinov3_only_under_the_sealed_hub(tmp_path, tiny_ckpt, sealed_hub, monkeypatch):
    sha = _head()
    _require_head_kit(sha)
    links = {r: os.path.realpath(os.path.join(_REPO, r)) for r in kit.LINKS}
    ev = {"tree_digest": kit.tree_digest(str(_REPO), sha, links), "run_id": RID, "sha": sha}
    tree = kit.build_admission_tree(str(_REPO), sha, {"links": links}, ev, str(tmp_path))
    ck, cfg, w = tiny_ckpt
    torch.save(ck, str(tmp_path / "tiny.ckpt"))
    decoy = tmp_path / "decoy_hf"                                              # a competing ordinary cache
    decoy.mkdir()
    monkeypatch.setenv("HF_HOME", str(decoy))
    monkeypatch.setenv("HF_HUB_CACHE", str(decoy / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(decoy / "hub"))
    st = kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False, hf_sealed=sealed_hub)
    hub = os.path.join(os.path.realpath(sealed_hub), "hub") + os.sep
    assert len(st["dino_resolved"]) == 2 and all(p.startswith(hub) for p in st["dino_resolved"]), st
    assert st["hf_sealed"] == os.path.realpath(sealed_hub) and st["sha256"] == kit.sha256_file(str(tmp_path / "tiny.ckpt"))
    assert kit.current_assets_reasons(kit.current_assets(sealed_hub, st["dino_resolved"])) == []
    # in a process whose environment is NOT the sealed hub, admission-state refuses before loading
    got = kit.admission_state(str(tmp_path / "tiny.ckpt"), "/ck", tree=tree, strict_load=False, hf_sealed=sealed_hub)
    assert got["sha256"] is None and any("sealed hub" in x or "competing" in x for x in got["reasons"]), got
    with pytest.raises(kit.Refused, match="sealed HF hub"):
        kit.admission_state_subprocess(tree, str(tmp_path / "tiny.ckpt"), "/ck", strict_load=False)


def test_admit_requires_a_valid_sealed_hub(tmp_path):
    with pytest.raises(kit.Refused, match="--hf-sealed"):
        kit.admit("rr.json", str(tmp_path), str(_REPO), hf_sealed=None)
    (tmp_path / "not_sealed" / "hub").mkdir(parents=True)
    with pytest.raises(kit.Refused, match="sealed HF hub"):
        kit.admit("rr.json", str(tmp_path), str(_REPO), hf_sealed=str(tmp_path / "not_sealed"))
    with pytest.raises(SystemExit):
        kit.main(["admit", "--run-result", "x", "--receipts-dir", "y", "--out", "z"])     # flag is mandatory


@pytest.mark.parametrize("key,val,needle", [
    ("historical", {}, "historical"), ("historical", {**kit.HISTORICAL, "exp17_launch_commit": "41aa31dc"}, "historical"),
    ("current_assets", {}, "current_assets keys"),
    ("dinov3_revision", "0" * 40, "dinov3_revision"), ("dinov3_config_sha256", "0" * 64, "dinov3_config_sha256"),
    ("dinov3_weights_sha256", "0" * 64, "dinov3_weights_sha256"), ("vae_sha256", "0" * 64, "vae_sha256"),
    ("acousticrooms_manifest_digest", "0" * 64, "acousticrooms_manifest_digest"),
    ("hf_sealed_hub", "relative/hub", "hf_sealed_hub"), ("dinov3_resolved", ["/elsewhere/model.safetensors"], "dinov3_resolved"),
    ("t", time.time() + 3600, "non-future"), ("t", float("nan"), "non-future"), ("t", 0, "non-future"),
])
def test_admission_validator_checks_provenance_fields(tmp_path, wandb_fx, key, val, needle):
    if key in kit.CURRENT_ASSET_KEYS:
        path, rec, nas = _admission(tmp_path, wandb_fx, current_assets={**_assets(), key: val})
    else:
        path, rec, nas = _admission(tmp_path, wandb_fx, **{key: val})
    reasons, got, _ = kit.admission_reasons(path, nas_root=str(tmp_path / "nas"), rehash=None)
    assert got is None and any(needle in x for x in reasons), (needle, reasons)


# --------------------------------------------------------------------------- #
# exp_43 6d: the offline W&B run (preserve on every exit path, NAS archive), the history CSV,
# the RF admission, the leftover gate (plan §5.1, §5.4, §6 "additional red-first tests")
# --------------------------------------------------------------------------- #
W_ID = "abc12345"


def _write_wandb_run(parent, steps=range(99, 40000, 100), run_id=W_ID, exit_record=True,
                     stamp="20261113_010000"):
    """A small offline W&B run written by the INSTALLED wandb's own datastore writer, shaped like
    PL's WandbLogger of the 40k recipe (``trainer/global_step`` 99, 199, ..., 39999), plus the
    usual files/ logs/ tmp/code/ layout."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore
    d = Path(parent) / f"offline-run-{stamp}-{run_id}"
    (d / "files").mkdir(parents=True)
    (d / "logs").mkdir()
    (d / "tmp" / "code").mkdir(parents=True)
    (d / "files" / "wandb-summary.json").write_text('{"train/loss": 0.4}')
    (d / "logs" / "debug.log").write_text("debug\n")
    ds = datastore.DataStore()
    ds.open_for_write(str(d / f"run-{run_id}.wandb"))
    r = pb.Record()
    r.header.SetInParent()
    ds.write(r)
    r = pb.Record()
    r.run.run_id = run_id
    ds.write(r)
    for i, step in enumerate(steps):
        r = pb.Record()
        for k, v in (("train/loss", 2.0 / (1 + step / 1000)), ("train/lr", 5e-5), ("trainer/global_step", step),
                     ("_step", i)):
            it = r.history.item.add()
            it.key, it.value_json = k, json.dumps(v)
        r.history.step.num = i
        ds.write(r)
    if exit_record:
        r = pb.Record()
        r.exit.exit_code, r.exit.runtime = 0, 10
        ds.write(r)
    r = pb.Record()
    r.footer.SetInParent()
    ds.write(r)
    ds.close()
    return d


@pytest.fixture(scope="module")
def wandb_fx(tmp_path_factory):
    """ONE real chain, built once: a 400-row offline run in a fake ``$TMPDIR/wandb``, preserved,
    archived to a scratch NAS root, and extracted by the PINNED history tool (a subprocess)."""
    base = tmp_path_factory.mktemp("wandb_fx")
    stage = base / "tmpdir" / "wandb"
    stage.mkdir(parents=True)
    _write_wandb_run(stage)
    (base / "train.log").write_text(f"wandb: Starting a new run with run id {W_ID}.\n")
    pre_path, arc_path = str(base / "preserved.json"), str(base / "archive.json")
    pre = kit.wandb_preserve([str(stage)], str(base / "keep"), str(base / "train.log"))
    kit.atomic_write_json(pre_path, pre)
    arc = kit.wandb_archive(RID, pre_path, nas_root=str(base / "nas"))
    kit.atomic_write_json(arc_path, arc)
    (base / "history").mkdir()
    hist = kit.train_history_subprocess(os.path.join(arc["dest_dir"], arc["run_dir_name"]), str(base / "history"), RID)
    binding = {"receipt_sha256": kit.sha256_file(arc_path), "preserved_receipt_sha256": kit.sha256_file(pre_path),
               "preserved_dir": pre["dest_dir"], "dest_dir": arc["dest_dir"], "run_dir_name": arc["run_dir_name"],
               "files_digest": arc["files_digest"], "n_files": len(arc["files"]),
               "manifest_sha256": arc["manifest_sha256"], "run_file_sha256": arc["run_file_sha256"]}
    return {"base": base, "stage": stage, "preserved": pre, "preserved_path": pre_path, "archive": arc,
            "archive_path": arc_path, "history": hist, "binding": binding, "nas_root": str(base / "nas")}


def _stage(tmp_path, n=1, run_ids=(W_ID, "zzz99999"), real=False):
    """A fake ``$TMPDIR/wandb`` with ``n`` offline runs (plain bytes unless ``real``)."""
    stage = tmp_path / "tmpdir" / "wandb"
    stage.mkdir(parents=True)
    for i in range(n):
        if real:
            _write_wandb_run(stage, run_id=run_ids[i], steps=range(99, 400, 100))
            continue
        d = stage / f"offline-run-20261113_01000{i}-{run_ids[i]}"
        (d / "files").mkdir(parents=True)
        (d / "tmp" / "code").mkdir(parents=True)
        (d / f"run-{run_ids[i]}.wandb").write_bytes(b"W&B" + bytes(range(256)) * 9)
        (d / "files" / "config.yaml").write_text("a: 1\n")
    log = tmp_path / "train.log"
    log.write_text(f"wandb: Starting a new run with run id {run_ids[0]}.\n")
    return stage, log


def test_wandb_preserve_locates_exactly_one_run_and_copies_it_verified(tmp_path):
    stage, log = _stage(tmp_path)
    src = stage / f"offline-run-20261113_010000-{W_ID}"
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    rec = kit.wandb_preserve([str(stage), str(tmp_path / "absent")], str(tmp_path / "keep"), str(log))
    assert set(rec) == kit.WANDB_PRESERVE_KEYS and rec["kind"] == "exp43_wandb_preserved"
    assert rec["dest_dir"] == os.path.realpath(str(tmp_path / "keep" / src.name)) and rec["log_run_ids"] == [W_ID]
    assert rec["run_file"] == f"run-{W_ID}.wandb" and rec["dirs"] == ["files", "tmp", "tmp/code"]
    assert rec["run_file_sha256"] == hashlib.sha256((src / rec["run_file"]).read_bytes()).hexdigest()
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before          # the source is untouched
    assert Path(rec["dest_dir"], "tmp", "code").is_dir()
    kit.atomic_write_json(str(tmp_path / "pre.json"), rec)
    assert kit.wandb_preserved_reasons(str(tmp_path / "pre.json")) == ([], rec)
    with pytest.raises(kit.Refused, match="preserved once"):
        kit.wandb_preserve([str(stage)], str(tmp_path / "keep"), str(log))
    # a second search path that aliases the same run counts once
    os.symlink(str(stage), str(tmp_path / "alias"))
    assert kit.wandb_candidates([str(stage), str(tmp_path / "alias")]) == [str(src)]


@pytest.mark.parametrize("case,needle", [
    ("zero", "0 offline W&B run"), ("two", "2 offline W&B run"), ("two_dirs", "2 offline W&B run"),
    ("log_names_another", "names W&B run"), ("symlink_inside", "symlink"), ("two_run_files", "run files"),
    ("no_run_file", "run files"), ("run_dir_is_a_symlink", "not a real directory")])
def test_wandb_preserve_is_fail_closed(tmp_path, case, needle):
    stage, log = _stage(tmp_path, n=2 if case == "two" else (0 if case == "zero" else 1))
    src = stage / f"offline-run-20261113_010000-{W_ID}"
    search = [str(stage)]
    if case == "two_dirs":                                         # one in $TMPDIR/wandb, one in $WANDB_DIR
        other = tmp_path / "wandb_dir"
        other.mkdir()
        os.rename(str(stage / src.name), str(other / src.name))
        (stage / "offline-run-20261113_020000-zzz99999").mkdir()
        search.append(str(other))
    elif case == "log_names_another":
        log.write_text("wandb: wandb sync /tmp/exp43.x/wandb/offline-run-20261113_020000-zzz99999\n")
    elif case == "symlink_inside":
        os.symlink("/etc/hostname", str(src / "files" / "link"))
    elif case == "two_run_files":
        (src / "run-other.wandb").write_bytes(b"x")
    elif case == "no_run_file":
        (src / f"run-{W_ID}.wandb").unlink()
    elif case == "run_dir_is_a_symlink":
        real = tmp_path / "real_run"
        os.rename(str(src), str(real))
        os.symlink(str(real), str(src))
    with pytest.raises(kit.Refused, match=needle):
        kit.wandb_preserve(search, str(tmp_path / "keep"), str(log))
    assert not (tmp_path / "keep").exists() or not any((tmp_path / "keep").iterdir())


def test_wandb_preserved_receipt_and_copy_mutations_are_refused(tmp_path):
    stage, log = _stage(tmp_path)
    rec = kit.wandb_preserve([str(stage)], str(tmp_path / "keep"), str(log))
    path = str(tmp_path / "pre.json")
    for key, val in [("kind", "exp40_wandb"), ("run_file_sha256", "0" * 64), ("files_digest", "0" * 64),
                     ("run_dir_name", "offline-run-x"), ("dirs", ["tmp", "files"]), ("log_run_ids", ["zzz99999"]),
                     ("dest_dir", "relative/dir"), ("t", True), ("files", {}), ("run_file", "run-zzz.wandb")]:
        kit.atomic_write_json(path, {**rec, key: val})
        assert kit.wandb_preserved_reasons(path)[0], key
    kit.atomic_write_json(path, {**rec, "extra": 1})
    assert any("exact schema" in x for x in kit.wandb_preserved_reasons(path)[0])
    kit.atomic_write_json(path, rec)
    copy = Path(rec["dest_dir"])
    for mutate in (lambda: (copy / "files" / "config.yaml").write_text("a: 2\n"),
                   lambda: (copy / "files" / "extra.txt").write_text("x"),
                   lambda: (copy / f"run-{W_ID}.wandb").unlink()):
        snapshot = {p: p.read_bytes() for p in copy.rglob("*") if p.is_file()}
        mutate()
        assert kit.wandb_preserved_reasons(path)[0]
        for p in copy.rglob("*"):
            if p.is_file() and p not in snapshot:
                p.unlink()
        for p, b in snapshot.items():
            p.write_bytes(b)
        assert kit.wandb_preserved_reasons(path)[0] == []


def test_wandb_archive_rehashes_and_every_mutation_is_refused(tmp_path):
    stage, log = _stage(tmp_path)
    pre = str(tmp_path / "pre.json")
    kit.atomic_write_json(pre, kit.wandb_preserve([str(stage)], str(tmp_path / "keep"), str(log)))
    nas = str(tmp_path / "nas")
    rec = kit.wandb_archive(RID, pre, nas_root=nas)
    assert set(rec) == kit.WANDB_ARCHIVE_KEYS and rec["dest_dir"] == os.path.join(nas, f"{RID}__wandb")
    assert sorted(os.listdir(rec["dest_dir"])) == sorted([kit.ARCHIVE_MANIFEST, rec["run_dir_name"]])
    assert rec["preserved_receipt_sha256"] == kit.sha256_file(pre)
    path = str(tmp_path / "arc.json")
    kit.atomic_write_json(path, rec)
    assert kit.wandb_archive_reasons(path, RID, nas) == ([], rec)
    assert kit.wandb_archive(RID, pre, nas_root=nas)["files"] == rec["files"]        # resumable / idempotent
    assert kit.wandb_archive_reasons(path, RID + "_r1", nas)[0]                      # another RUN_ID
    for key, val in [("dest_dir", "/elsewhere"), ("manifest_sha256", "0" * 64), ("run_file_sha256", "1" * 64),
                     ("preserved_receipt_sha256", "x"), ("src_dir", "rel"), ("kind", "wandb")]:
        kit.atomic_write_json(path, {**rec, key: val})
        assert kit.wandb_archive_reasons(path, RID, nas)[0], key
    kit.atomic_write_json(path, rec)
    run = Path(rec["dest_dir"], rec["run_dir_name"])
    man = Path(rec["dest_dir"], kit.ARCHIVE_MANIFEST)
    local = Path(rec["src_dir"])
    for target in (run / f"run-{W_ID}.wandb", run / "files" / "config.yaml", man, local / f"run-{W_ID}.wandb"):
        os.chmod(target, 0o644)
        old = target.read_bytes()
        target.write_bytes(old + b"!")
        assert kit.wandb_archive_reasons(path, RID, nas)[0], target
        target.write_bytes(old)
    for extra in (run / "files" / ".config.yaml.partial-1-x", Path(rec["dest_dir"], "stray")):
        extra.write_text("x")
        assert kit.wandb_archive_reasons(path, RID, nas)[0], extra
        extra.unlink()
    assert kit.wandb_archive_reasons(path, RID, nas)[0] == []
    # a destination that records another run is never overwritten
    stage2, log2 = _stage(tmp_path / "b", run_ids=("zzz99999", W_ID))
    pre2 = str(tmp_path / "pre2.json")
    kit.atomic_write_json(pre2, kit.wandb_preserve([str(stage2)], str(tmp_path / "keep2"), str(log2)))
    with pytest.raises(kit.Refused, match="not this run's archive|records another"):
        kit.wandb_archive(RID, pre2, nas_root=nas)


def test_wandb_cli_is_write_once(tmp_path):
    stage, log = _stage(tmp_path)
    out = str(tmp_path / "pre.json")
    args = ["wandb-preserve", "--search", str(stage), "--keep-parent", str(tmp_path / "keep"), "--log", str(log),
            "--out", out]
    assert kit.main(args) == 0 and kit.wandb_preserved_reasons(out)[0] == []
    assert kit.main(args) == 2                                                   # receipt exists
    arc = str(tmp_path / "arc.json")
    aargs = ["wandb-archive", "--run-id", RID, "--preserved", out, "--dest-root", str(tmp_path / "nas"), "--out", arc]
    assert kit.main(aargs) == 0 and kit.wandb_archive_reasons(arc, RID, str(tmp_path / "nas"))[0] == []
    assert kit.main(aargs) == 2
    assert kit.main(["wandb-archive", "--run-id", "exp40_C16_a6000_2x32", "--preserved", out, "--dest-root",
                     str(tmp_path / "nas2"), "--out", str(tmp_path / "x.json")]) == 2


def test_the_history_tool_is_pinned_and_bound_by_its_sidecar(wandb_fx):
    tool = _REPO / kit.TRAIN_HISTORY_TOOL
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == kit.TRAIN_HISTORY_TOOL_SHA256
    h = wandb_fx["history"]
    assert set(h) == kit.TRAIN_HISTORY_KEYS and h["rows"] == 400
    assert h["run_file_sha256"] == wandb_fx["archive"]["run_file_sha256"]
    assert kit.train_history_reasons(h, wandb_fx["archive"]["run_file_sha256"]) == []
    side = kit.read_json(h["sidecar"])
    assert side["run_dir"] == os.path.realpath(os.path.join(wandb_fx["archive"]["dest_dir"], wandb_fx["archive"]["run_dir_name"]))
    assert side["cadence"]["ok"] is True and side["tool"]["sha256"] == kit.TRAIN_HISTORY_TOOL_SHA256
    assert kit.train_history_reasons(h, "0" * 64)                                     # another run file
    for key, val in [("tool", "x.py"), ("tool_sha256", "0" * 64), ("rows", 399), ("rows", True),
                     ("csv", "rel.csv"), ("csv_sha256", "z")]:
        assert kit.train_history_reasons({**h, key: val}, h["run_file_sha256"], rehash=False), key
    assert kit.train_history_reasons({**h, "extra": 1}, h["run_file_sha256"], rehash=False)


def test_history_csv_and_sidecar_mutations_are_refused(tmp_path, wandb_fx):
    import shutil
    d = tmp_path / "h"
    shutil.copytree(os.path.dirname(wandb_fx["history"]["csv"]), str(d))
    h = dict(wandb_fx["history"], csv=str(d / "exp43_train_history.csv"), sidecar=str(d / "exp43_train_history.json"))
    side = kit.read_json(h["sidecar"])
    side["csv"]["path"] = h["csv"]
    kit.atomic_write_json(h["sidecar"], side)
    h["sidecar_sha256"] = kit.sha256_file(h["sidecar"])
    assert kit.train_history_reasons(h, h["run_file_sha256"]) == []
    for target in (h["csv"], h["sidecar"]):
        os.chmod(target, 0o644)
        old = open(target, "rb").read()
        open(target, "wb").write(old.replace(b"99,", b"98,", 1) if target == h["csv"] else old + b" ")
        assert any("re-hash" in x for x in kit.train_history_reasons(h, h["run_file_sha256"])), target
        open(target, "wb").write(old)
    for key, val in [("cadence", {"ok": False}), ("row_count", 399), ("nonfinite", {"train/loss": [99], "lr": []}),
                     ("expected_max_steps", None), ("schema", "x"), ("run_file_sha256", "0" * 64)]:
        kit.atomic_write_json(h["sidecar"], {**side, key: val})
        bad = dict(h, sidecar_sha256=kit.sha256_file(h["sidecar"]))
        assert kit.train_history_reasons(bad, h["run_file_sha256"]), key
    kit.atomic_write_json(h["sidecar"], side)
    open(h["csv"], "a").write("40099,1.0,5e-05\n")                         # a row beyond the cadence
    side2 = dict(side, csv=dict(side["csv"], sha256=kit.sha256_file(h["csv"])))
    kit.atomic_write_json(h["sidecar"], side2)
    bad = dict(h, csv_sha256=kit.sha256_file(h["csv"]), sidecar_sha256=kit.sha256_file(h["sidecar"]))
    assert any("CSV" in x for x in kit.train_history_reasons(bad, h["run_file_sha256"]))


@pytest.mark.parametrize("case,needle", [
    ("pin", "pinned history extractor"), ("cadence_gap", "exited 3"), ("no_exit_record", "exited 2"),
    ("stub_rc0_bogus", None)])
def test_train_history_subprocess_is_fail_closed(tmp_path, monkeypatch, case, needle):
    parent = tmp_path / "run"
    parent.mkdir()
    steps = [s for s in range(99, 40000, 100) if s != 20099] if case == "cadence_gap" else range(99, 40000, 100)
    run_dir = _write_wandb_run(parent, steps=steps, exit_record=case != "no_exit_record")
    (tmp_path / "out").mkdir()
    if case == "pin":
        monkeypatch.setattr(kit, "TRAIN_HISTORY_TOOL_SHA256", "0" * 64)
    if case == "stub_rc0_bogus":                                   # a stub tool (pinned to ITS bytes) exit 0
        root = tmp_path / "stubroot"
        stub = root / kit.TRAIN_HISTORY_TOOL
        stub.parent.mkdir(parents=True)
        stub.write_text("import sys, json\na = sys.argv\n"
                        "open(a[a.index('--out-csv') + 1], 'w').write('step,train/loss,lr\\n99,1.0,5e-05\\n')\n"
                        "json.dump({'run_file_sha256': 'x', 'row_count': 400}, open(a[a.index('--out-json') + 1], 'w'))\n")
        monkeypatch.setattr(kit, "TRAIN_HISTORY_TOOL_SHA256", hashlib.sha256(stub.read_bytes()).hexdigest())
        h = kit.train_history_subprocess(str(run_dir), str(tmp_path / "out"), RID, tool_root=str(root))
        assert kit.train_history_reasons(h, "0" * 64), h                       # the binding catches the stub
        return
    with pytest.raises(kit.Refused, match=needle):
        kit.train_history_subprocess(str(run_dir), str(tmp_path / "out"), RID)


def test_admission_refuses_every_unbound_wandb_or_history_artefact(tmp_path, tiny_ckpt, monkeypatch, full_stream,
                                                                   sealed_hub):
    tree, ckdir, nas_root, d, ev = _tiny_full_run(tmp_path, tiny_ckpt, monkeypatch, full_stream)
    nas = os.path.join(nas_root, ev["run_id"])
    v = kit.audit(str(ckdir), str(tree), nas_dir=nas, evidence=str(d / "ev.json"), run_id=ev["run_id"],
                  expected_dirpath="/ck", check_path=str(d / "check.json"), reference_path=str(d / "ref.json"))
    kit.atomic_write_json(str(d / "audit.json"), v)
    rr = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", _rr_files(d), "/ck",
                        str(ckdir), str(tree), nas_root=nas_root)
    assert rr["kind"] == "exp43_run_result" and sorted(rr["files"]) == sorted(kit.RUN_RESULT_FILES)
    kit.atomic_write_json(str(d / "rr.json"), rr)

    def admit():
        return kit.admit(str(d / "rr.json"), str(d), str(_REPO), nas_root=nas_root, **_inject(tree, tiny_ckpt[2], sealed_hub))
    wa = kit.read_json(str(d / "wandb_archive.json"))
    pre = kit.read_json(str(d / "wandb_preserved.json"))
    # the NAS archive and the LOCAL preserved copy: every file, mutated one at a time
    for root in (Path(wa["dest_dir"]), Path(pre["dest_dir"])):
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            os.chmod(p, 0o644)
            old = p.read_bytes()
            p.write_bytes(old + b"#")
            with pytest.raises(kit.Refused):
                admit()
            p.write_bytes(old)
    # the receipts bound by the run result (sha256)
    for name in ("wandb_preserved.json", "wandb_archive.json"):
        old = (d / name).read_bytes()
        (d / name).write_bytes(old.replace(b'"t": ', b'"t":  '))
        with pytest.raises(kit.Refused, match="not the recorded bytes"):
            admit()
        (d / name).write_bytes(old)
    # the FULL evidence must name the same preserve receipt
    evr, aur = kit.read_json(str(d / "ev.json")), kit.read_json(str(d / "audit.json"))
    kit.atomic_write_json(str(d / "ev.json"), {**evr, "wandb_preserved_sha256": "7" * 64})
    kit.atomic_write_json(str(d / "audit.json"), {**aur, "full_evidence_sha256": kit.sha256_file(str(d / "ev.json"))})
    rr2 = kit.run_result(ev["run_id"], ev["sha"], "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", _rr_files(d), "/ck",
                         str(ckdir), str(tree), nas_root=nas_root)
    kit.atomic_write_json(str(d / "rr.json"), rr2)
    with pytest.raises(kit.Refused, match="do not name one run"):
        admit()
    kit.atomic_write_json(str(d / "ev.json"), evr)
    kit.atomic_write_json(str(d / "audit.json"), aur)
    kit.atomic_write_json(str(d / "rr.json"), rr)
    # the pinned extractor
    monkeypatch.setattr(kit, "TRAIN_HISTORY_TOOL_SHA256", "0" * 64)
    with pytest.raises(kit.Refused, match="pinned history extractor"):
        admit()
    monkeypatch.undo()


def test_run_result_names_the_wandb_receipts_and_rejects_exp40_kinds(tmp_path):
    files = {k: str(tmp_path / f"{k}.json") for k in kit.RUN_RESULT_FILES}
    for p in files.values():
        Path(p).write_text("{}")
    assert kit.RUN_RESULT_FILES == ("check", "reference", "evidence", "audit", "wandb_preserved", "wandb_archive")
    rr = kit.run_result(RID, "b" * 40, "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", files, "/ck", "/ck", "/t")
    kit.atomic_write_json(str(tmp_path / "rr.json"), rr)
    assert kit.resolve_run_result(str(tmp_path / "rr.json"), str(tmp_path))[1] == files
    kit.atomic_write_json(str(tmp_path / "rr.json"), {**rr, "kind": "exp40_run_result"})
    with pytest.raises(kit.Refused, match="exp_43 FULL run-result"):
        kit.resolve_run_result(str(tmp_path / "rr.json"), str(tmp_path))
    with pytest.raises(kit.Refused):
        kit.run_result(RID, "b" * 40, "x", {k: v for k, v in files.items() if k != "wandb_archive"}, "/ck", "/ck", "/t")
    assert kit.main(["run-result", "--run-id", RID, "--sha", "b" * 40, "--verdict",
                     "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", "--expected-dirpath", "/ck", "--ckpt-dir", "/ck",
                     "--runtree", "/t", "--out", str(tmp_path / "rr2.json"),
                     *sum(([f"--{k.replace('_', '-')}", v] for k, v in files.items()), [])]) == 0


def test_fresh_destination_reasons(tmp_path):
    save, rec, nas = str(tmp_path / "save"), str(tmp_path / "receipts"), str(tmp_path / "nas")
    assert kit.fresh_destination_reasons("FULL", RID, save, "", rec, nas) == []
    targets = [os.path.join(save, kit.PROJECT, RID), os.path.join(save, "runtime", RID, "wandb_preserved"),
               os.path.join(nas, RID), kit.wandb_dest(RID, nas),
               os.path.join(rec, f"wandb_preserved_{RID}.json"), os.path.join(rec, f"wandb_archive_{RID}.json")]
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        Path(t).write_text("x")
        assert any(t in x for x in kit.fresh_destination_reasons("FULL", RID, save, "", rec, nas)), t
        os.unlink(t)
    ns = kit.namespace(RID, "SMOKE", STAMP)
    os.makedirs(os.path.join(save, "runtime", ns, "wandb_preserved"))
    assert kit.fresh_destination_reasons("SMOKE", RID, save, STAMP, rec, nas)


def _smoke_cli_inputs(tmp_path, preserved):
    recs, sums, log, exp = _evidence(steps=kit.SMOKE_STEPS["SMOKE"], world=1)
    evdir = tmp_path / "ev"
    evdir.mkdir()
    with open(evdir / "rank0.jsonl", "w") as f:
        for row in recs[0]:
            f.write(json.dumps(row) + "\n")
    kit.atomic_write_json(str(evdir / "rank0.summary.json"), sums[0])
    (tmp_path / "train.log").write_text(log)
    check = {"sha": "b" * 40, "tree_digest": "1" * 64, "env_digest": "2" * 64,
             "argv": kit.train_argv("a6000", "2x32", "SMOKE", RID, "/s", STAMP)}
    kit.atomic_write_json(str(tmp_path / "check.json"), check)
    return ["evidence", "--platform", "a6000", "--rung", "2x32", "--mode", "SMOKE", "--run-id", RID,
            "--save-root", "/s", "--stamp", STAMP, "--evidence-dir", str(evdir), "--log", str(tmp_path / "train.log"),
            "--check-json", str(tmp_path / "check.json"), "--receipts", str(tmp_path / "receipts"),
            "--train-rc", "0", "--watcher-rc", "0", "--wandb-preserved", preserved, "--out", str(tmp_path / "out.json")]


def test_smoke_evidence_cli_requires_the_preserved_wandb_run(tmp_path, wandb_fx):
    """plan §8 L4: a SMOKE / DDP_SMOKE proves the W&B preservation path before the FULL run."""
    args = _smoke_cli_inputs(tmp_path, wandb_fx["preserved_path"])
    assert kit.main(args) == 0
    out = kit.read_json(str(tmp_path / "out.json"))
    assert out["verdict"] == "PASS" and out["wandb_preserved_sha256"] == kit.sha256_file(wandb_fx["preserved_path"])
    os.unlink(str(tmp_path / "out.json"))
    args[args.index("--wandb-preserved") + 1] = str(tmp_path / "missing.json")
    assert kit.main(args) == 1
    out = kit.read_json(str(tmp_path / "out.json"))
    assert out["verdict"] == "FAIL" and out["wandb_preserved_sha256"] is None
    assert any("W&B preserve receipt" in x for x in out["evidence"]["reasons"])
    with pytest.raises(SystemExit):
        kit.main([a for a in args if a not in ("--wandb-preserved", str(tmp_path / "missing.json"))])


def _launcher_cleanup_script(tmp_path, n_runs, log_id=W_ID):
    """The launcher's cleanup block, extracted VERBATIM, under a real kit, after a 'spawned'
    training whose offline W&B run sits in $TMPDIR/wandb (as PL's save_dir='.' fallback puts it)."""
    text = (_EXP / "exp43_train_local.sh").read_text()
    block = text[text.index("TRAIN_PID=\"\"; WATCH_PID=\"\""):text.index('"$PY" "$KIT" gpu-lock')]
    fake = tmp_path / "fakekit.py"
    fake.write_text(
        "import sys\nimport importlib.util\n"
        "spec = importlib.util.spec_from_file_location('k', %r)\n" % str(_EXP / "exp43_kit.py") +
        "k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)\n"
        "sys.exit(k.main(sys.argv[1:]))\n")
    tmpdir = tmp_path / "tmpdir"
    stage = tmpdir / "wandb"
    stage.mkdir(parents=True)
    for i in range(n_runs):
        _write_wandb_run(stage, run_id=[W_ID, "zzz99999"][i], steps=range(99, 400, 100), stamp=f"2026111{i}_010000")
    (tmp_path / "train.log").write_text(f"wandb: Starting a new run with run id {log_id}.\n")
    hs = tmp_path / "hs"
    hs.mkdir()
    script = tmp_path / "launcher.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"PY={sys.executable}; KIT={fake}; MODE=FULL; TMPDIR={tmpdir}\n"
        f"LOCK_JSON={tmp_path / 'locks.json'}; TRAIN_LOG={tmp_path / 'train.log'}\n"
        f"WANDB_DIR={tmp_path / 'rt' / 'wandb'}; WANDB_KEEP={tmp_path / 'rt' / 'wandb_preserved'}\n"
        f"WANDB_PRESERVED_JSON={tmp_path / 'wandb_preserved.json'}\n"
        "die() { echo \"REFUSED: $*\"; exit 2; }\n" + block +
        f"\"$PY\" \"$KIT\" gpu-lock --uuids {U0} --handshake-dir {hs} --payload-json '{{}}' --out \"$LOCK_JSON\" || die lock\n"
        "TRAIN_SPAWNED=1\n( exec setsid sleep 300 ) &\nTRAIN_PID=$!\n"
        f"echo $TRAIN_PID > {tmp_path / 'pid'}\n"
        "wait \"$TRAIN_PID\"\n")
    proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(200):
        if (tmp_path / "pid").exists() and (hs / f"GPU_LOCK_{U0}").exists():
            break
        time.sleep(0.1)
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    return proc, out, tmpdir, hs


def test_launcher_preserves_the_wandb_run_on_an_interrupt_before_removing_tmpdir(tmp_path):
    proc, out, tmpdir, hs = _launcher_cleanup_script(tmp_path, n_runs=1)
    assert proc.returncode == 130 and "INTERRUPTED by SIGTERM" in out, out
    rec = kit.read_json(str(tmp_path / "wandb_preserved.json"))
    assert kit.wandb_preserved_reasons(str(tmp_path / "wandb_preserved.json"))[0] == []
    assert rec["dest_dir"].startswith(str(tmp_path / "rt" / "wandb_preserved")) and rec["log_run_ids"] == [W_ID]
    assert not tmpdir.exists()                                   # removed only AFTER the verified copy
    assert not (hs / f"GPU_LOCK_{U0}").exists()


@pytest.mark.parametrize("n_runs,log_id", [(0, W_ID), (2, W_ID), (1, "zzz99999")])
def test_launcher_keeps_tmpdir_when_the_wandb_run_cannot_be_preserved(tmp_path, n_runs, log_id):
    proc, out, tmpdir, hs = _launcher_cleanup_script(tmp_path, n_runs=n_runs, log_id=log_id)
    assert proc.returncode == 130, out
    assert "W&B PRESERVATION FAILED" in out and f"CLEANUP: kept {tmpdir}" in out, out
    assert tmpdir.is_dir() and len(list((tmpdir / "wandb").iterdir())) == n_runs   # nothing deleted
    assert not (tmp_path / "wandb_preserved.json").exists()
    assert not (hs / f"GPU_LOCK_{U0}").exists()


def test_launcher_wires_the_wandb_preservation_into_every_mode():
    text = _LAUNCHER.read_text()
    main = text[text.index("# --- spawn:"):]
    assert main.index("TRAIN_SPAWNED=1") < main.index("exec setsid")
    stop = main.index('"$PY" "$KIT" stop-group --pgid "$TRAIN_PID" --grace 120 || die')
    assert stop < main.index("\npreserve_wandb") < main.index("# --- verdicts")
    assert '[ "$WANDB_OK" = 1 ] || FINAL=1' in main
    assert main.count('--wandb-preserved "$WANDB_PRESERVED_JSON"') == 3            # 2 evidence calls + run result
    assert '"$PY" "$KIT" wandb-archive --run-id "$RUN_ID" --preserved "$WANDB_PRESERVED_JSON"' in main
    assert '--wandb-archive "$WANDB_ARCHIVE_JSON"' in main
    assert main.index("wandb-archive --run-id") < main.index('"$PY" "$KIT" run-result')
    cleanup = text[text.index("cleanup() {"):text.index("on_signal() {")]
    assert cleanup.index("stop-group") < cleanup.index("preserve_wandb") < cleanup.index("rm -rf")
    assert 'if [ "$KEEP_TMPDIR" = 1 ]' in cleanup
    pres = text[text.index("preserve_wandb() {"):text.index("cleanup() {")]
    for s_ in ('--search "$TMPDIR/wandb"', '--search "$WANDB_DIR/wandb"', '--search "$WANDB_DIR"',
               '--keep-parent "$WANDB_KEEP"', '--log "$TRAIN_LOG"', "KEEP_TMPDIR=1"):
        assert s_ in pres, s_
    assert 'WANDB_KEEP="$RT/wandb_preserved"' in text
    assert 'WANDB_PRESERVED_JSON="$RECEIPTS/wandb_preserved_${RUN_ID}.json"' in text


# --- the leftover gate (plan §5.1): no exp_40 / neuronic / quantisation semantics survive ---------- #
LEFTOVER_TOKENS = ("exp40", "C16", "offset_step", "quantize", "neuronic", "sbatch")
# (file, exact string literal) -> why it may name exp_40; every other occurrence fails the gate
LEFTOVER_ALLOWLIST = {
    ("exp43_kit.py", "src/tests/fixtures/exp40_yaw_aug_enabled_golden.json"):
        "the E1 golden of the reference YAW draws (exp_15 / exp_17 pins): RF's offsets must EQUAL it (§2.3)",
    ("exp43_kit.py", "~/.exp40_cards.lock"):
        "the A6000 card flock, SHARED ON PURPOSE across exp_40 / exp_42 / exp_43 (plan §5.1)",
    ("exp43_train_local.sh", '"$HOME/.exp40_cards.lock"'):
        "the same shared card flock, taken by the launcher (plan §5.1)",
}


def _python_semantic_strings(path):
    """Every NAME and every non-docstring STRING token of a Python file (comments dropped)."""
    import ast
    import io
    import tokenize
    src = path.read_text()
    docs = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(first.value.lineno)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME or (tok.type == tokenize.STRING and tok.start[0] not in docs):
            out.append(tok.string)
        elif tok.type not in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER, tokenize.OP, tokenize.NUMBER):
            out.append(tok.string)
    return out


def _bash_semantic_words(path):
    """Every whitespace-separated word of a bash script with comments dropped (full-line and
    trailing ``# ...`` outside quotes)."""
    words = []
    for line in path.read_text().splitlines():
        code, quote = [], None
        for i, ch in enumerate(line):
            if quote:
                quote = None if ch == quote else quote
            elif ch in "\"'":
                quote = ch
            elif ch == "#" and (i == 0 or line[i - 1] in " \t") and not line[max(0, i - 1):i + 2].startswith("$#"):
                break
            code.append(ch)
        words += "".join(code).split()
    return words


def _leftovers(name, units):
    bad = []
    for u in units:
        for t in LEFTOVER_TOKENS:
            hit = t in u if t == "C16" else t in u.lower()
            if hit and not any(f == name and lit in u for (f, lit) in LEFTOVER_ALLOWLIST):
                bad.append((t, u))
    return bad


def test_leftover_gate_finds_no_exp40_semantics_outside_the_allowlist():
    assert _leftovers("exp43_kit.py", _python_semantic_strings(_EXP / "exp43_kit.py")) == []
    assert _leftovers("exp43_evidence_driver.py", _python_semantic_strings(_EXP / "exp43_evidence_driver.py")) == []
    assert _leftovers("exp43_train_local.sh", _bash_semantic_words(_EXP / "exp43_train_local.sh")) == []
    for (name, lit), why in LEFTOVER_ALLOWLIST.items():
        text = (_EXP / name).read_text()
        assert lit in text and why, (name, lit)                     # every allowance is still needed


def test_leftover_gate_catches_a_planted_leftover(tmp_path):
    py = tmp_path / "exp43_kit.py"
    py.write_text('"""exp40 in a docstring is prose."""\n# exp40 in a comment too\nX = "FLAC_exp40_C16"\n'
                  'def f():\n    """offset_step prose"""\n    return quantize_yaw_offsets\n')
    got = {t for t, _ in _leftovers("exp43_kit.py", _python_semantic_strings(py))}
    assert got == {"exp40", "C16", "quantize"}
    sh = tmp_path / "exp43_train_local.sh"
    sh.write_text('# exp40 header comment\nKIT="$RUNTREE/exp40_kit.py"  # neuronic note\necho "sbatch" # x\n'
                  'CARDS_LOCK="$HOME/.exp40_cards.lock"\n')
    got = {t for t, _ in _leftovers("exp43_train_local.sh", _bash_semantic_words(sh))}
    assert got == {"exp40", "sbatch"}
