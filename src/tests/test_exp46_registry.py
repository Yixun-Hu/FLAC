"""exp_45 — the run registry (worklog/.../exp_45_rf_data_curve_claude/exp45_registry.py), forked from exp_38's
test_exp38_registry.py (round B); every edit is named in its commit (plan_rf_data_curve.md r3 §3.3).

Executable statement of plan_rf_data_curve.md r3 §2-§3.2 (exp_38: plan_data_efficiency.md §6-§7, §10 item 4):
the three RF runs' configs,
dataset configs, argv, chunk plan, checkpoint names, run-dir states, checkpoint audit and
log gates. CPU-only; synthetic checkpoints and logs.
"""
import hashlib
import importlib.util
import json
import os
import shlex

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
EXP38 = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")   # read-only reuse


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reg = _load("exp45_registry")

RF = {t: f"exp45_RFf{t}_l40_2x32" for t in ("025", "050", "075")}
# plan r3 fact F2: exp_38's de_yaw_f025 FULL argv on neuronic, verbatim (cmd of
# at_launch_de_yaw_f025_neuronic_FULL_2026-11-09_18-01-10.json without the torchrun prefix)
DE_YAW_F025_RD = "/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/de_yaw_f025"
DE_YAW_F025_ARGV = [
    "train.py", "--model-config", "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
    "--dataset-config", "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json",
    "--pretransform-ckpt-path", "weights/FLAC/VAE.safetensors", "--max-steps", "40000", "--batch-size", "32",
    "--accum-batches", "1", "--num-workers", "6", "--seed", "42", "--num-gpus", "2", "--num-nodes", "1",
    "--strategy", "ddp_find_unused_parameters_true", "--sync-batchnorm", "true", "--precision", "bf16-mixed",
    "--val-every", "-1", "--gradient-clip-val", "0.0", "--logger", "none", "--checkpoint-every", "2500",
    "--name", "de_yaw_f025", "--experiment-name", "de_yaw_f025", "--save-dir", DE_YAW_F025_RD,
    "--evidence-json", DE_YAW_F025_RD + "/evidence/run_evidence.json"]


def _opt(argv, flag):
    return argv[argv.index(flag) + 1]


# ------------------------------------------------------------------ runs and rungs
def test_the_registry_imports_and_registers_exactly_the_three_rf_runs():
    """plan r3 §3.2: exactly three runs; no P1 bridge, no 8x8 rung, no _r suffix (import-time construction)."""
    assert reg.required_runs() == reg.all_runs() == reg.campaign_runs() == [RF["025"], RF["050"], RF["075"]]
    assert reg.REGISTERED_RUNS == frozenset(RF.values()) and reg.OPTIONAL_RUNS == ()
    assert sorted(reg.ARMS) == ["rf"] and sorted(reg.RUNGS) == ["2x32"]
    for t, run in RF.items():
        assert reg.parse_run_id(run) == ("rf", t, "2x32") and reg.run_id("rf", t) == run
    for bad in ("de_yaw_f025", "de_bf_f025", "de_p1_f025", "exp45_RFf100_l40_2x32", "exp45_RFf025_l40_8x8",
                "exp45_RFf025_l40_2x32_r8x8", "exp45_YAWf025_l40_2x32", "exp45_RFf025_a6000_2x32", "", None):
        with pytest.raises(reg.GateError):
            reg.parse_run_id(bad)


def test_the_one_rung_keeps_eff_and_syncbn_batch_64():
    assert reg.RUNGS == {"2x32": {"ngpu": 2, "micro": 32, "default": True}}
    assert reg.check_rung("a6000", "2x32")["micro"] == reg.check_rung("neuronic", "2x32")["micro"] == 32
    for platform in ("a6000", "neuronic"):
        with pytest.raises(reg.GateError):
            reg.check_rung(platform, "8x8", allow_nonmatched=True)
        assert reg.PLATFORMS[platform]["rungs"] == ("2x32",)
    assert reg.PLATFORMS["neuronic"]["save_root"] == "/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve"
    assert reg.PLATFORMS["a6000"]["save_root"] == "/media/diskstation/yixunhu/FLAC/checkpoints/exp45_rf_data_curve"
    assert reg.CAMPAIGN_ROOT == {"a6000": "/home/yixunhu/codespace/exp-45-campaign"}


# ------------------------------------------------------------------ configs
def test_the_rf_config_is_exp15s_yaw_config_plus_only_the_flip():
    """plan r3 F1: the RF config (sha d6a02dde…) differs from exp_15's YAW config only in training.yaw_aug.flip."""
    rf = json.load(open(os.path.join(REPO, reg.ARMS["rf"]["config"])))
    yaw = json.load(open(os.path.join(REPO, reg.E0_CONFIGS["yaw"]["config"])))
    assert reg.sha256_file(os.path.join(REPO, reg.ARMS["rf"]["config"])) == \
        "d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d"
    assert rf["training"]["yaw_aug"].pop("flip") is True and rf == yaw
    assert reg.check_config("rf", "2x32")["training"]["yaw_aug"] == reg.RF_YAW_AUG
    assert reg.chunk_plan("rf", "2x32") is None                 # no frame averaging: no chunk plan


def test_the_e0_reference_configs_are_pinned_and_not_launchable():
    for name, c in reg.E0_CONFIGS.items():
        assert reg.sha256_file(os.path.join(REPO, c["config"])) == c["config_sha256"], name
        assert c["config"] in reg.EXTERNAL_CONFIGS
    assert not set(reg.E0_CONFIGS) & set(reg.ARMS)
    yaw = json.load(open(os.path.join(REPO, reg.E0_CONFIGS["yaw"]["config"])))
    p1 = json.load(open(os.path.join(REPO, reg.E0_CONFIGS["bvp1"]["config"])))
    assert yaw["training"].pop("yaw_aug") == {"enabled": True, "img_w": 512, "seed": 42} and yaw == p1


@pytest.mark.parametrize("name,mutate", [
    ("flip off", lambda tr: tr["yaw_aug"].update(flip=False)), ("flip int", lambda tr: tr["yaw_aug"].update(flip=1)),
    ("no flip", lambda tr: tr["yaw_aug"].pop("flip")), ("seed", lambda tr: tr["yaw_aug"].update(seed=7)),
    ("img_w", lambda tr: tr["yaw_aug"].update(img_w=256)), ("disabled", lambda tr: tr["yaw_aug"].update(enabled=False)),
    ("c16 grid", lambda tr: tr["yaw_aug"].update(offset_step=32)), ("no yaw_aug", lambda tr: tr.pop("yaw_aug")),
    ("fa cond", lambda tr: tr.update(cond_method="fa_invariant")), ("fa angles", lambda tr: tr.update(frame_avg_angles=[0.0])),
    ("canonicalize", lambda tr: tr.update(canonicalize={"enabled": True})), ("ema", lambda tr: tr.update(use_ema=False)),
])
def test_check_config_refuses_every_rf_block_mutation(tmp_path, monkeypatch, name, mutate):
    rel = reg.ARMS["rf"]["config"]
    root = tmp_path / "repo"
    (root / os.path.dirname(rel)).mkdir(parents=True)
    cfg = json.load(open(os.path.join(REPO, rel)))
    mutate(cfg["training"])
    (root / rel).write_text(json.dumps(cfg, indent=4))
    with pytest.raises(reg.GateError):                          # the sha pin first ...
        reg.check_config("rf", "2x32", repo=str(root))
    monkeypatch.setitem(reg.ARMS["rf"], "config_sha256", reg.sha256_file(str(root / rel)))
    with pytest.raises(reg.GateError):                          # ... and the block check behind it
        reg.check_config("rf", "2x32", repo=str(root))


# ------------------------------------------------------------------ subsets
@pytest.mark.parametrize("tag,n", [("025", 73473), ("050", 145665), ("075", 218463)])
def test_check_subset_reads_the_registered_split(tag, n):
    out = reg.check_subset(tag)
    assert out["n_items"] == n and out["rooms"] == 243 and out["restrict_to_split"] is True


def test_check_subset_refuses_a_tampered_split(tmp_path):
    f = reg.FRACS["025"]
    root = tmp_path / "repo"
    for rel in (f["split"], f["dataset_config"]):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(open(os.path.join(REPO, rel), "rb").read())
    reg.check_subset("025", repo=str(root))
    (root / f["split"]).write_bytes(open(os.path.join(REPO, f["split"]), "rb").read() + b" ")
    with pytest.raises(reg.GateError):
        reg.check_subset("025", repo=str(root))


# ------------------------------------------------------------------ checkpoint names, legs, run dirs
def test_checkpoint_names_match_exp14s_dc_van_run_dir():
    # /media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/dc_van_f025/ (listed 2026-11-07)
    seen = {2500: "epoch=2-step=2500.ckpt", 12500: "epoch=10-step=12500.ckpt",
            15000: "epoch=13-step=15000.ckpt", 30000: "epoch=26-step=30000.ckpt",
            40000: "epoch=34-step=40000.ckpt"}
    for step, name in seen.items():
        assert reg.ckpt_name(step, "025") == name
    assert reg.ckpt_name(40000, "050") == "epoch=17-step=40000.ckpt"
    assert reg.ckpt_name(40000, "075") == "epoch=11-step=40000.ckpt"
    assert len(reg.cadence_steps()) == 16


def _touch(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")


def test_run_dir_fresh_and_complete_states_only(tmp_path):
    """No resume state exists (Codex code r1 HIGH 2): a run dir is fresh (0) or complete (40,000)."""
    run = RF["025"]
    assert reg.check_run_dir(str(tmp_path / "absent"), run, 0)["state"] == "fresh"
    d = tmp_path / "rd"
    _touch(d, [f"at_launch_{run}_x.json"])
    assert reg.check_run_dir(str(d), run, 0)["state"] == "fresh"
    _touch(d, ["epoch=2-step=2500.ckpt"])
    with pytest.raises(reg.GateError):
        reg.check_run_dir(str(d), run, 0)                       # fresh start refuses a checkpoint
    for bad in (2500, 10000, 37500, -1):
        with pytest.raises(reg.GateError):
            reg.check_run_dir(str(d), run, bad)                 # there is no resume boundary
    full = tmp_path / "full"
    _touch(full, [reg.ckpt_name(s, "025") for s in reg.cadence_steps()])
    assert reg.check_run_dir(str(full), run, 40000)["state"] == "complete"
    (full / ".exp38_lock").mkdir()
    with pytest.raises(reg.GateError):
        reg.check_run_dir(str(full), run, 40000)                # a live / crashed run
    assert reg.check_run_dir(str(full), run, 40000, allow_lock=True)["state"] == "complete"
    _touch(full, [reg.ckpt_name(42500, "025")])
    with pytest.raises(reg.GateError):
        reg.check_run_dir(str(full), run, 40000, allow_lock=True)   # a checkpoint beyond the endpoint
    d2 = tmp_path / "foreign"
    _touch(d2, ["somebody_else.txt"])
    with pytest.raises(reg.GateError):
        reg.check_run_dir(str(d2), run, 0)
    for marker in ("TRAINING_COMPLETE.json", "CHECKPOINTS.sha256"):
        d3 = tmp_path / f"done_{marker}"
        _touch(d3, [marker])
        with pytest.raises(reg.GateError):
            reg.check_run_dir(str(d3), run, 0)                  # a finished run is never reused
    d4 = tmp_path / "tmpfile"
    _touch(d4, [".TRAINING_COMPLETE.json.tmp.123"])
    with pytest.raises(reg.GateError):
        reg.check_run_dir(str(d4), run, 0)                      # a torn write is foreign


def test_d2_campaign_is_the_three_rf_runs(monkeypatch):
    assert reg.TRAINING_PLATFORM == "neuronic"
    assert reg.campaign_runs() == list(RF.values()) and all(r in reg.REGISTERED_RUNS for r in reg.campaign_runs())
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")
    assert reg.campaign_runs() == reg.required_runs()            # no bridge run on any platform


def test_only_the_registered_matrix_is_launchable():
    assert sorted(reg.REGISTERED_RUNS) == sorted(reg.all_runs("2x32"))
    for bad in ("de_p1_f025", "de_yaw_f025", "exp45_RFf025_l40_2x32 ", "exp45_RFf100_l40_2x32", "", None):
        with pytest.raises(reg.GateError):
            reg.parse_run_id(bad)
        with pytest.raises(reg.GateError):
            reg.build_train_argv(bad, "neuronic", allow_nonmatched=True)


# ------------------------------------------------------------------ argv
@pytest.mark.parametrize("tag", ["025", "050", "075"])
def test_the_argv_is_exp38s_de_yaw_argv_with_exactly_the_registered_substitutions(tag):
    """plan r3 §2 argv rule: F2 with exactly --model-config (RF), the fraction's dataset config, --name /
    --experiment-name = run id, --save-dir / --evidence-json under the exp_45 save root -- nothing else."""
    run = RF[tag]
    rd = f"/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve/{run}"
    want = list(DE_YAW_F025_ARGV)
    for flag, value in (("--model-config", reg.ARMS["rf"]["config"]), ("--dataset-config", reg.FRACS[tag]["dataset_config"]),
                        ("--name", run), ("--experiment-name", run), ("--save-dir", rd),
                        ("--evidence-json", rd + "/evidence/run_evidence.json")):
        want[want.index(flag) + 1] = value
    got = reg.build_train_argv(run, "neuronic")
    assert got == want
    diff = {got[i - 1] for i in range(1, len(got)) if got[i] != DE_YAW_F025_ARGV[i]}
    assert diff <= {"--model-config", "--dataset-config", "--name", "--experiment-name", "--save-dir", "--evidence-json"}
    proj, proj38 = reg.argv_projection(got), reg.argv_projection(DE_YAW_F025_ARGV)
    assert len(proj) == len(proj38) and proj[:2] == ["train.py", "--model-config"]
    assert [i for i, (x, y) in enumerate(zip(proj, proj38)) if x != y] == ([2] if tag == "025" else [2, 4])


@pytest.mark.parametrize("run", sorted(RF.values()))
def test_argv_per_run(run):
    arm, tag, _ = reg.parse_run_id(run)
    argv = reg.build_train_argv(run, "a6000")
    assert _opt(argv, "--model-config") == reg.ARMS["rf"]["config"]
    assert _opt(argv, "--dataset-config") == reg.FRACS[tag]["dataset_config"]
    assert (_opt(argv, "--batch-size"), _opt(argv, "--num-gpus"), _opt(argv, "--accum-batches")) == ("32", "2", "1")
    assert (_opt(argv, "--sync-batchnorm"), _opt(argv, "--precision"), _opt(argv, "--seed")) == ("true", "bf16-mixed", "42")
    assert (_opt(argv, "--max-steps"), _opt(argv, "--checkpoint-every"), _opt(argv, "--logger")) == ("40000", "2500", "none")
    assert "--ckpt-path" not in argv and not any("exp14_" in a for a in argv)
    n = reg.build_train_argv(run, "neuronic")
    assert _opt(n, "--save-dir") == f"/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve/{run}"
    assert [a for a in n if not a.startswith("/n/")] == [a for a in argv if not a.startswith("/media/")]


def test_there_is_no_resume_argv(capsys):
    """Codex code r1 HIGH 2: no resume parameter, no --ckpt-path, no CLI flag."""
    import inspect
    assert "resume" not in "".join(inspect.signature(reg.build_train_argv).parameters)
    for run in reg.all_runs():
        assert "--ckpt-path" not in reg.build_train_argv(run, "neuronic")
    with pytest.raises(SystemExit):
        reg._cli(["argv", RF["050"], "--platform", "neuronic", "--resume-step", "10000"])
    with pytest.raises(SystemExit):
        reg._cli(["run-dir-check", "/x", "--run", RF["050"], "--state", "10000"])


def test_smoke_argv_is_ddp_only():
    """Codex code r1 finding 11 (plan r3 §5 L5: every SMOKE is 2 x L40): the single-rank SMOKE argv is refused."""
    for kw in ({}, {"ddp": False}):
        with pytest.raises(reg.GateError, match="2 x 32"):
            reg.build_smoke_argv(RF["025"], 20, "/tmp/x", **kw)
    ddp = reg.build_smoke_argv(RF["025"], 30, f"/tmp/smk/{RF['025']}_x", ddp=True)
    full = reg.build_train_argv(RF["025"], "a6000")
    diff = {i for i, (a, b) in enumerate(zip(ddp, full)) if a != b}
    assert len(ddp) == len(full) and {ddp[i - 1] for i in diff} == {"--max-steps", "--checkpoint-every",
                                                                     "--save-dir", "--name", "--experiment-name",
                                                                     "--evidence-json"}
    assert (_opt(ddp, "--max-steps"), _opt(ddp, "--checkpoint-every")) == ("30", "1000000")
    with pytest.raises(reg.GateError):
        reg.build_smoke_argv(RF["025"], 500, "/tmp/x", ddp=True)


# ------------------------------------------------------------------ checkpoint audit (exact schemas, Codex code r1 HIGH 1)
def _ckpt_dict(run, step, run_dir, n_state=None, shapes=None):
    """A checkpoint with EXACTLY the PL 2.1 schema of the real 40k endpoints (synthetic tensors)."""
    import torch
    arm, tag, rung = reg.parse_run_id(run)
    cfg = json.load(open(os.path.join(REPO, reg.model_config_rel(arm, rung))))
    lr = reg.inverse_lr(step)
    n = reg.N_OPT_STATE if n_state is None else n_state
    ep = reg.epoch_of_step(step, tag)
    prog = {"ready": step, "completed": step, "started": step, "processed": step}
    shp = (lambda i: shapes[i]) if shapes else (lambda i: (1,))
    z2 = {"ready": 0, "completed": 0}
    z4 = {"ready": 0, "completed": 0, "started": 0, "processed": 0}
    p2 = {"ready": step, "completed": step}
    p3 = {"ready": step, "completed": step, "started": step}
    batch = {"total": dict(prog, completed=step - 1), "current": dict(prog, completed=step - 1), "is_last_batch": False}
    eprog = {"ready": ep + 1, "completed": ep, "started": ep + 1, "processed": ep}
    loops = {   # the exact PL 2.1 structure of the real endpoints, with this step's counters
        "fit_loop": {"epoch_loop.state_dict": {"_batches_that_stepped": step - 1}, "epoch_loop.batch_progress": batch,
                     "epoch_loop.scheduler_progress": {"total": p2, "current": p2},
                     "epoch_loop.automatic_optimization.state_dict": {},
                     "epoch_loop.automatic_optimization.optim_progress": {"optimizer": {
                         "step": {"total": p2, "current": p2}, "zero_grad": {"total": p3, "current": p3}}},
                     "epoch_loop.manual_optimization.state_dict": {},
                     "epoch_loop.manual_optimization.optim_step_progress": {"total": z2, "current": z2},
                     "epoch_loop.val_loop.state_dict": {},
                     "epoch_loop.val_loop.batch_progress": {"total": z4, "current": z4, "is_last_batch": False},
                     "epoch_progress": {"total": eprog, "current": eprog}, "state_dict": {}},
        "predict_loop": {"state_dict": {}, "batch_progress": {"total": z4, "current": z4}},
        "test_loop": {"state_dict": {}, "batch_progress": {"total": z4, "current": z4, "is_last_batch": False}},
        "validate_loop": {"state_dict": {}, "batch_progress": {"total": z4, "current": z4, "is_last_batch": False}}}
    mc = {"monitor": None, "best_model_score": None, "best_model_path": os.path.join(run_dir, reg.ckpt_name(step, tag)),
          "current_score": None, "dirpath": run_dir, "best_k_models": {}, "kth_best_model_path": "",
          "kth_value": torch.tensor(float("inf")), "last_model_path": ""}
    return {
        "global_step": step, "epoch": ep, "model_config": cfg, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion_ema.ema_model.w": torch.zeros(2), "diffusion.model.w": torch.zeros(2),
                       "diffusion_ema.step": torch.tensor([step]), "diffusion_ema.initted": torch.tensor([1.0]),
                       "diffusion.conditioner.c": torch.zeros(1),
                       **{f"diffusion.pretransform.f{i}": torch.zeros(1) for i in range(reg.N_STATE_TENSORS - 5)}},
        "optimizer_states": [{"state": {i: {"step": torch.tensor(float(step)), "exp_avg": torch.zeros(shp(i)),
                                            "exp_avg_sq": torch.zeros(shp(i))} for i in reg.OPT_STATE_IDS[:n]},
                              "param_groups": [{"lr": lr, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.001,
                                                "amsgrad": False, "maximize": False, "foreach": None,
                                                "capturable": False, "differentiable": False, "fused": None,
                                                "decoupled_weight_decay": True, "initial_lr": 5e-05,
                                                "params": list(range(753))}]}],
        "lr_schedulers": [{"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0, "base_lrs": [5e-05],
                           "last_epoch": step, "_step_count": step + 1, "_last_lr": [lr],
                           "_get_lr_called_within_step": False}],
        "loops": loops, "callbacks": {reg.MODEL_CHECKPOINT_KEY: mc}}


def _fake_ckpt(path, run, step, run_dir=None, mutate=None):
    import torch
    ck = _ckpt_dict(run, step, run_dir or os.path.dirname(path))
    if mutate:
        mutate(ck)
    torch.save(ck, path)


def test_inverse_lr_matches_the_recorded_100_percent_value():
    assert abs(reg.inverse_lr(40000) - 4.902903378454601e-05) < 1e-15


def _set(path, value):
    def m(ck):
        d = ck
        for k in path[:-1]:
            d = d[k]
        if value is _DEL:
            del d[path[-1]]
        else:
            d[path[-1]] = value(d[path[-1]]) if callable(value) else value
    return m


_DEL = object()
_MC = reg.MODEL_CHECKPOINT_KEY
_OS = ("optimizer_states", 0)
_G = ("optimizer_states", 0, "param_groups", 0)
_SC = ("lr_schedulers", 0)
_FL = ("loops", "fit_loop")


def _mutations():
    import torch
    step = 40000
    return {
        "global_step": _set(("global_step",), 37500), "global_step type": _set(("global_step",), 40000.0),
        "epoch": _set(("epoch",), 33), "pl version": _set(("pytorch-lightning_version",), "2.2.0"),
        "extra top key": _set(("hyper_parameters",), {}), "missing top key": _set(("loops",), _DEL),
        "model_config": _set(("model_config",), json.load(open(os.path.join(REPO, reg.E0_CONFIGS["yaw"]["config"])))),
        "EMA missing": _set(("state_dict", "diffusion_ema.ema_model.w"), _DEL),
        "EMA shape": _set(("state_dict", "diffusion_ema.ema_model.w"), torch.zeros(3)),
        "EMA step": _set(("state_dict", "diffusion_ema.step"), torch.tensor([1])),
        "non-finite": _set(("state_dict", "diffusion.model.w"), torch.tensor([float("nan"), 0.0])),
        "no optimizer": _set(("optimizer_states",), []),
        "two optimizers": _set(("optimizer_states",), lambda v: v + v),
        "opt extra key": _set(_OS + ("extra",), 1),
        "group extra key": _set(_G + ("extra",), 1), "group missing key": _set(_G + ("fused",), _DEL),
        "group betas": _set(_G + ("betas",), [0.9, 0.99]), "group wd": _set(_G + ("weight_decay",), 0.01),
        "group foreach": _set(_G + ("foreach",), True), "group lr": _set(_G + ("lr",), 1e-4),
        "state count": _set(_OS + ("state",), lambda st: {k: v for k, v in st.items() if k != 0}),
        "state id swapped": _set(_OS + ("state",), lambda st: {(500 if k == 0 else k): v for k, v in st.items()}),
        "state id out of range": _set(_OS + ("state",), lambda st: {(9999 if k == 0 else k): v for k, v in st.items()}),
        "params list": _set(_G + ("params",), list(range(752))),
        "loop extra nested key": _set(_FL + ("epoch_loop.state_dict", "extra"), 1),
        "loop counter type": _set(_FL + ("epoch_loop.state_dict", "_batches_that_stepped"), 39999.0),
        "loop batches stepped": _set(_FL + ("epoch_loop.state_dict", "_batches_that_stepped"), 1),
        "loop scheduler progress": _set(_FL + ("epoch_loop.scheduler_progress", "total", "ready"), 1),
        "loop manual steps": _set(_FL + ("epoch_loop.manual_optimization.optim_step_progress", "total", "ready"), 3),
        "loop validation ran": _set(_FL + ("epoch_loop.val_loop.batch_progress", "total", "ready"), 3),
        "test loop ran": _set(("loops", "test_loop", "batch_progress", "total", "completed"), 1),
        "callback best_k_models": _set(("callbacks", _MC, "best_k_models"), {"a.ckpt": 1.0}),
        "callback kth_value": _set(("callbacks", _MC, "kth_value"), None),
        "callback last path": _set(("callbacks", _MC, "last_model_path"), "x.ckpt"),
        "callback best path": _set(("callbacks", _MC, "best_model_path"), "/elsewhere/x.ckpt"),
        "callback score": _set(("callbacks", _MC, "best_model_score"), 0.5),
        "state key": _set(_OS + ("state", 3, "max_exp_avg_sq"), torch.zeros(1)),
        "state step": _set(_OS + ("state", 5, "step"), torch.tensor(1.0)),
        "state nan": _set(_OS + ("state", 7, "exp_avg"), torch.tensor([float("inf")])),
        "sched extra key": _set(_SC + ("extra",), 1), "sched missing key": _set(_SC + ("_get_lr_called_within_step",), _DEL),
        "sched last lr": _set(_SC + ("_last_lr",), lambda v: [v[0] * 2]),
        "sched step count": _set(_SC + ("_step_count",), step), "sched base lr": _set(_SC + ("base_lrs",), [1e-4]),
        "two schedulers": _set(("lr_schedulers",), lambda v: v + v),
        "loops extra": _set(("loops", "extra_loop"), {}), "fit loop missing key": _set(_FL + ("epoch_progress",), _DEL),
        "fit loop progress": _set(_FL + ("epoch_loop.batch_progress",), {"total": {"ready": 1, "processed": 1}}),
        "callbacks empty": _set(("callbacks",), {}),
        "callback cadence": _set(("callbacks",), lambda v: {_MC.replace("2500", "5000"): v[_MC]}),
        "callback extra": _set(("callbacks", "EarlyStopping{}"), {}),
        "callback payload key": _set(("callbacks", _MC, "extra"), 1),
        "callback monitor": _set(("callbacks", _MC, "monitor"), "val/loss"),
        "callback dirpath": _set(("callbacks", _MC, "dirpath"), "/elsewhere"),
    }


def test_audit_ckpt_pass_and_every_schema_mutation(tmp_path):
    run, step = RF["025"], 40000
    good = tmp_path / reg.ckpt_name(step, "025")
    _fake_ckpt(str(good), run, step)
    out = reg.audit_ckpt(str(good), run, step, run_dir=str(tmp_path), inventory=False)
    assert out["global_step"] == step and out["sha256"] == reg.sha256_file(str(good))
    assert out["bytes"] == os.path.getsize(str(good))
    for i, (name, mutate) in enumerate(_mutations().items()):
        d = tmp_path / f"c{i}"
        d.mkdir()
        p = d / reg.ckpt_name(step, "025")
        _fake_ckpt(str(p), run, step, run_dir=str(d), mutate=mutate)
        with pytest.raises(reg.GateError):
            reg.audit_ckpt(str(p), run, step, run_dir=str(d), inventory=False)
            pytest.fail(f"mutation {name!r} was admitted")
    with pytest.raises(reg.GateError):
        reg.audit_ckpt(str(good), run, step, run_dir=str(tmp_path / "elsewhere"), inventory=False)
    with pytest.raises(reg.GateError):
        reg.audit_ckpt(str(good), run, 37500, inventory=False)
    with pytest.raises(reg.GateError):                          # only the dc_van readback may carry run_contract
        _fake_ckpt(str(good), run, step, mutate=_set(("run_contract",), {}))
        reg.audit_ckpt(str(good), run, step, run_dir=str(tmp_path), inventory=False)


@pytest.mark.parametrize("arm,rung", [("rf", "2x32")])
def test_the_real_inventory_path_for_every_arm_and_rung(arm, rung):
    """A fresh wrapper's own state_dict + optimizer shapes pass; one dropped key, one reshaped
    tensor and one wrong optimizer-state shape are refused (the path audit_ckpt runs)."""
    import torch
    reg._FRESH.clear()
    inv, shapes, wrapper = reg.fresh_inventory(arm, rung)
    sd = wrapper.state_dict()
    assert len(sd) == reg.N_STATE_TENSORS and len(shapes) == 753
    assert reg.inventory_problems(sd, arm, rung)[0] == []
    k0 = sorted(sd)[0]
    assert reg.inventory_problems({k: v for k, v in sd.items() if k != k0}, arm, rung)[0]
    k1 = next(k for k in sorted(sd) if sd[k].ndim >= 1 and sd[k].shape[0] > 1)
    assert reg.inventory_problems({**sd, k1: sd[k1][:1].clone()}, arm, rung)[0]
    run = reg.run_id(arm, "025", rung)
    ck = _ckpt_dict(run, 40000, "/rd", shapes=shapes)
    assert reg.ckpt_schema_problems(ck, 40000, "025", "/rd", shapes) == []
    ck["optimizer_states"][0]["state"][0]["exp_avg"] = torch.zeros(1, 1, 1, 7)
    assert reg.ckpt_schema_problems(ck, 40000, "025", "/rd", shapes)
    reg._FRESH.clear()


# exp_43's RF 100 % endpoint (plan F12; train.json, 4,551 steps / epoch) -- only where the NAS is mounted
RF_100 = "/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug/exp43_RF_a6000_2x32/epoch=8-step=40000.ckpt"


@pytest.mark.skipif(not os.path.exists(RF_100), reason="NAS not mounted")
def test_the_real_rf_100_percent_endpoint_satisfies_the_exact_schema():
    """The RF 100 % checkpoint embeds the RF config and the RF wrapper's exact inventory; the only schema
    differences from a fraction run are the epoch counters (8, not f025's 34)."""
    import torch
    ck = torch.load(RF_100, map_location="cpu", weights_only=False)
    assert reg.canonical(ck["model_config"]) == reg.canonical(json.load(open(os.path.join(REPO, reg.ARMS["rf"]["config"]))))
    probs = reg.ckpt_schema_problems(ck, 40000, "025")
    assert len(probs) == 3 and probs[0] == "epoch 8 != 34" and probs[1].startswith("epoch progress")
    assert reg.inventory_problems(ck["state_dict"], "rf", "2x32")[0] == []
    reg._FRESH.clear()


def test_ckpt_manifest_needs_the_full_cadence(tmp_path):
    run = RF["075"]
    d = tmp_path / run
    _touch(d, [reg.ckpt_name(s, "075") for s in reg.cadence_steps(37500)])
    with pytest.raises(reg.GateError):
        reg.write_ckpt_manifest(str(d), run)
    _touch(d, [reg.ckpt_name(40000, "075")])
    out = reg.write_ckpt_manifest(str(d), run)
    lines = open(out).read().strip().split("\n")
    assert len(lines) == 16 and lines[-1].endswith("epoch=11-step=40000.ckpt")
    assert lines[0].split("  ")[0] == hashlib.sha256(b"x").hexdigest()
    with pytest.raises(reg.GateError):
        reg.write_ckpt_manifest(str(d), run)                    # never overwritten


# ------------------------------------------------------------------ log gates
def _log(run=RF["025"], ngpu=2, n=73473, banner=None, flash_missing=0, step=True, end=False,
         loss="0.512", members=None, spe=None):
    arm, tag, _ = reg.parse_run_id(run)
    spe = reg.FRACS[tag]["steps_per_epoch"] if spe is None else spe
    parts = ["Seed set to 42"]
    parts += ["No module named 'flash_attn'", reg.FLASH_MISSING + ", disabling Flash Attention"] * flash_missing
    parts += [f"Found {n} files in 243 subfolders"] * ngpu
    if ngpu > 1:
        for r in (members if members is not None else range(ngpu)):
            parts.append(f"Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {r + 1}/{ngpu}")
    parts += [reg.RF_BANNER] if banner is None else list(banner)
    if step:
        parts.append(f"Epoch 0:   0%|          | 1/{spe} [00:05<1:00:00,  0.30it/s, train/loss={loss}]")
    text = "\n".join(parts)
    if end:
        text += f"\rEpoch 34:  84%|########4 | 968/{spe} [59:53<11:08,  0.27it/s] " + reg.FIT_END.format(n=40000)
    return text


def test_log_gate_start_pass():
    assert reg.log_gate(_log(), RF["025"], "start", "flash", 2) == []
    assert reg.log_gate(_log(flash_missing=2), RF["025"], "start", "sdpa", 2) == []
    assert reg.log_gate(_log(RF["075"], n=218463), RF["075"], "start", "flash", 2) == []


@pytest.mark.parametrize("ngpu,spe", [(1, 2296), (1, 1148), (4, 574), (0, 1148)])
def test_log_gate_refuses_every_world_size_but_2(ngpu, spe):
    """Codex code r1 finding 11: every exp_45 launch path is 2 x 32 (plan r3 §5 L5, RUNGS); a log of any other world
    size is refused explicitly -- whatever its loader length (a single rank's own 2,296 batches included)."""
    probs = reg.log_gate(_log(ngpu=max(ngpu, 1), spe=spe), RF["025"], "start", "flash", ngpu)
    assert probs and "2 x 32 only" in probs[0] and f"{ngpu}-rank" in probs[0]
    assert reg.RUNGS == {"2x32": {"ngpu": 2, "micro": 32, "default": True}}


C16_BANNER = "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"


@pytest.mark.parametrize("kw,backend", [
    ({"banner": []}, "flash"),                                   # no banner
    ({"banner": [reg.YAW_BANNER]}, "flash"),                     # exp_15's YAW line instead of RF's
    ({"banner": [C16_BANNER]}, "flash"),                         # exp_40's C16 line
    ({"banner": [reg.RF_BANNER, reg.RF_BANNER]}, "flash"),       # two RF banners (one per rank is impossible: F1)
    ({"banner": [reg.RF_BANNER, reg.YAW_BANNER]}, "flash"),      # RF + another arm's line
    ({"spe": 4551}, "flash"),                                    # the 100 % loader length, not f025's
    ({"n": 291210}, "flash"),
    ({"members": [0]}, "flash"),
    ({"flash_missing": 2}, "flash"),
    ({"flash_missing": 1}, "sdpa"),
    ({"step": False}, "flash"),
    ({"loss": "nan"}, "flash"),
    ({"loss": "-inf"}, "flash"),
])
def test_log_gate_start_failures(kw, backend):
    assert reg.log_gate(_log(**kw), RF["025"], "start", backend, 2) != []


def test_log_gate_banner_after_the_first_step_fails():
    text = _log(banner=[]) + "\n" + reg.RF_BANNER
    assert any("banner" in p for p in reg.log_gate(text, RF["025"], "start", "flash", 2))


def test_log_gate_end_needs_lightnings_marker_and_enough_finite_losses():
    assert reg.log_gate(_log(end=True), RF["025"], "end", "flash", 2) == []
    assert reg.log_gate(_log(), RF["025"], "end", "flash", 2) != []
    assert reg.log_gate(_log(end=True), RF["025"], "end", "flash", 2, min_losses=2) != []


def test_parse_losses_is_strict():
    good, bad = reg.parse_losses("train/loss=0.5, x train/loss=nan] train/loss=inf, train/mse_loss=7 train/loss=abc]")
    assert good == [0.5] and bad == ["nan", "inf", "abc"]


# ------------------------------------------------------------------ pins and backend
def test_vit_pin_check_on_a_fake_cache(tmp_path):
    snap = tmp_path / reg.VIT_REPO_DIR / "snapshots" / reg.VIT_REV
    snap.mkdir(parents=True)
    for name in reg.VIT_FILES:
        (snap / name).write_bytes(b"not the weights")
    with pytest.raises(reg.GateError):
        reg.vit_pin_check(str(tmp_path))
    with pytest.raises(reg.GateError):
        reg.vit_pin_check(str(tmp_path / "empty"))


def test_backend_probe_semantics(monkeypatch):
    import importlib.util as iu
    real = iu.find_spec

    class Spec:
        def __init__(self, origin):
            self.origin = origin

    overlay = os.path.join(REPO, "overlays/no_flash/flash_attn.py")
    for origin, flash_ok, sdpa_ok in ((None, False, True), (overlay, False, True),
                                      ("/site-packages/flash_attn/__init__.py", True, False)):
        monkeypatch.setattr(iu, "find_spec", lambda name, o=origin: (Spec(o) if o else None)
                            if name == "flash_attn" else real(name))
        for backend, ok in (("flash", flash_ok), ("sdpa", sdpa_ok)):
            if ok:
                assert reg.backend_probe(backend)["attention_backend"] == backend
            else:
                with pytest.raises(reg.GateError):
                    reg.backend_probe(backend)


# ------------------------------------------------------------------ subset-audit binding, snapshot
def _fake_audit(root):
    bound = {"src/configs/dataset_configs/custom_metadata/AR_md.py": None, "src/data/dataset.py": None,
             "data/AR/train_frac_manifest_s2026.json": None, "data/AR/train.json": None,
             f"{reg.EXP38_DIR}/exp38_subset_audit.py": None}
    for t in reg.FRACS:
        bound[reg.FRACS[t]["dataset_config"]] = None
    for rel in list(bound) + [reg.FRACS[t]["split"] for t in reg.FRACS]:
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(open(os.path.join(REPO, rel), "rb").read())
    for t in reg.FRACS:
        bound[reg.FRACS[t]["split"]] = None
    bound = {rel: reg.sha256_file(str(root / rel)) for rel in bound}
    res = {"verdict": "PASS", "problems": [], "bound": bound, "root_fingerprint": "f" * 64,
           "fractions": {t: {"split_sha256": reg.FRACS[t]["split_sha256"]} for t in reg.FRACS}}
    path = root / reg.SUBSET_AUDIT
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res).encode()
    path.write_bytes(text)
    (root / (reg.SUBSET_AUDIT + ".sha256")).write_text(hashlib.sha256(text).hexdigest() + "  x\n")
    return path


def test_check_subset_audit_binding(tmp_path):
    path = _fake_audit(tmp_path)
    assert reg.check_subset_audit(str(tmp_path))["sha256"]
    (tmp_path / "src/data/dataset.py").write_text("# changed\n")
    with pytest.raises(reg.GateError):
        reg.check_subset_audit(str(tmp_path))                   # bound file changed
    _fake_audit(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"PASS"', b'"FAIL"'))
    with pytest.raises(reg.GateError):
        reg.check_subset_audit(str(tmp_path))                   # sidecar mismatch
    with pytest.raises(reg.GateError):
        reg.check_subset_audit(str(tmp_path / "nothing"))


def test_the_fork_runs_from_its_own_folder_and_reads_exp38_read_only():
    """exp_45 (plan r3 §3.2): the records live in exp_45's folder under ``rf_data_curve_`` names; exp_38's tools and
    artifacts are read at exp_38's own paths (their bound paths keep naming exp_38's folder), all tracked, all
    entering the closure through EXTERNAL_CONFIGS."""
    import subprocess
    assert reg.EXP_DIR == "worklog/worklog_yixun/exp_45_rf_data_curve_claude"
    assert reg.EXP38_DIR == "worklog/worklog_yixun/exp_38_data_efficiency_claude"
    for rel in (reg.SUBSET_AUDIT, reg.AR_CONTENT_MANIFEST, reg.REF_DIR, reg.REF_MANIFEST, reg.E0A_DECISION):
        assert rel.startswith(reg.EXP38_DIR + "/"), rel
    assert f"{reg.EXP38_DIR}/exp38_subset_audit.py" in reg.SUBSET_AUDIT_BOUND
    assert not any(r.startswith(reg.EXP_DIR + "/") for r in reg.SUBSET_AUDIT_BOUND)
    assert reg.CLOSURE_EXP_PREFIXES == ("exp45_",)
    for rel in (reg.E0_RECEIPT, reg.env_receipt_rel("neuronic"), reg.smoke_receipt_rel("x", "neuronic", "t")):
        assert rel.startswith(reg.EXP_DIR + "/rf_data_curve_"), rel
    tracked = set(subprocess.run(["git", "-C", REPO, "ls-files", "--", reg.EXP38_DIR], capture_output=True,
                                 text=True, check=True).stdout.split("\n"))
    ro = reg.EXP38_READONLY
    assert len(ro) == len(set(ro)) == 3 + 3 + 1 + 68 + 2 and set(ro) <= tracked and set(ro) <= set(reg.EXTERNAL_CONFIGS)
    lines = open(os.path.join(REPO, reg.REF_MANIFEST)).read().strip().split("\n")
    assert sorted(ln.split("  ", 1)[1] for ln in lines) == reg.ref_set() and len(reg.ref_set()) == 68
    spec = importlib.util.spec_from_file_location("exp38_eval_ro", os.path.join(EXP38, "exp38_eval.py"))
    ev38 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev38)
    assert reg.ref_set() == ev38.ref_set()                       # exp_38's own reference enumeration


def test_the_snapshot_carries_every_read_only_exp38_dependency_with_its_hash(tmp_path):
    files = reg.snapshot_files(reg.required_runs()[0])
    if f"{reg.EXP_DIR}/exp45_supervise.py" not in files:
        pytest.skip("the exp_45 kit is not committed yet (the closure lists tracked files only)")
    reg.snapshot_tools(reg.required_runs()[0], str(tmp_path / "snap"))
    manifest = dict(reversed(ln.split("  ", 1)) for ln in (tmp_path / "snap" / "SNAPSHOT.sha256").read_text().strip().split("\n"))
    for rel in reg.EXP38_READONLY:
        assert manifest.get(rel) == reg.sha256_file(os.path.join(REPO, rel)), rel
        assert reg.sha256_file(str(tmp_path / "snap" / rel)) == manifest[rel]
    kit = [r for r in manifest if r.startswith(reg.EXP_DIR + "/")]
    assert kit and all(os.path.basename(r).startswith("exp45_") and "/" not in r[len(reg.EXP_DIR) + 1:] for r in kit)
    exp38 = {r for r in manifest if r.startswith(reg.EXP38_DIR + "/")}
    assert exp38 == set(reg.EXP38_READONLY)                     # nothing else of exp_38's folder
    assert f"{reg.EXP38_DIR}/exp38_eval.py" not in manifest and f"{reg.EXP38_DIR}/exp38_supervise.py" not in manifest


def test_snapshot_is_complete_immutable_and_usable(tmp_path):
    """Codex code r1 HIGH 5: training is launched FROM the snapshot -- the closure (same digest as
    the tree), the subset-audit sidecar, and the two cwd-relative data links."""
    files = reg.snapshot_files(RF["050"])
    if reg.ARMS["rf"]["config"] not in files or f"{reg.EXP_DIR}/exp45_supervise.py" not in files:
        pytest.skip("the exp_45 kit is not committed yet (the closure lists tracked files only)")
    assert "train.py" in files and "src/training/run_evidence.py" in files and reg.SUBSET_AUDIT in files
    assert not any(f.startswith("src/tests/") for f in files)
    dest = tmp_path / "snap"
    out = reg.snapshot_tools(RF["050"], str(dest))
    assert reg.verify_snapshot(str(dest), out["snapshot_sha256"])["verified"]
    assert reg.closure_digest(str(dest)) == reg.closure_digest()
    for rel in reg.SNAPSHOT_LINKS:
        assert os.path.islink(dest / rel) and os.readlink(dest / rel) == os.path.realpath(os.path.join(REPO, rel))
    reg.check_config("rf", "2x32", repo=str(dest))
    reg.check_subset("050", repo=str(dest))
    assert reg.check_subset_audit(str(dest))["sha256"]
    with pytest.raises(reg.GateError):
        reg.snapshot_tools(RF["050"], str(dest))                # never overwritten
    with pytest.raises(reg.GateError):
        reg.verify_snapshot(str(dest), "0" * 64)                # not the launch record's snapshot
    link = dest / "AcousticRooms"
    link.unlink()
    link.symlink_to(tmp_path)
    with pytest.raises(reg.GateError):
        reg.verify_snapshot(str(dest))                          # a re-pointed data link
    link.unlink()
    link.symlink_to(os.path.realpath(os.path.join(REPO, "AcousticRooms")))
    reg.verify_snapshot(str(dest))
    assert reg.closure_of_manifest((dest / "SNAPSHOT.sha256").read_text()) == reg.closure_digest()["closure_sha256"]
    # the tree is CLOSED (Codex code r3 HIGH 1): bytecode of a listed module is the only unlisted entry allowed
    pyc = dest / "src" / "models" / "__pycache__" / "transformer.cpython-310.pyc"
    pyc.parent.mkdir(exist_ok=True)
    pyc.write_bytes(b"pyc")
    reg.verify_snapshot(str(dest))
    for make in (lambda: (dest / "src" / "models" / "evil.py").write_text("x = 1\n"),
                 lambda: (dest / "src" / "flash_attn.cpython-310-x86_64-linux-gnu.so").write_bytes(b"so"),
                 lambda: (dest / "extra_link").symlink_to(tmp_path),
                 lambda: (dest / "newpkg").mkdir(),
                 lambda: (dest / "src" / "models" / "__pycache__" / "evil.cpython-310.pyc").write_bytes(b"x")):
        make()
        with pytest.raises(reg.GateError, match="unmanifested"):
            reg.verify_snapshot(str(dest))
        for extra in reg.snapshot_extras(str(dest), {l.split("  ", 1)[1] for l in (dest / "SNAPSHOT.sha256").read_text().split("\n") if l},
                                         {l.split("  ", 1)[1] for l in (dest / "SNAPSHOT.links").read_text().split("\n") if l}):
            q = dest / extra.rstrip("/")
            q.unlink() if (q.is_symlink() or q.is_file()) else q.rmdir()
        reg.verify_snapshot(str(dest))
    cfg = dest / reg.ARMS["rf"]["config"]
    cfg.write_text(cfg.read_text() + " ")
    with pytest.raises(reg.GateError):
        reg.verify_snapshot(str(dest))


# ------------------------------------------------------------------ dry run
@pytest.mark.parametrize("platform", ["a6000", "neuronic"])
def test_describe_every_required_run(platform, tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "check_subset_audit", lambda repo=None: {"sha256": "0" * 64})
    for run in reg.required_runs():
        d = reg.describe_run(run, platform, save_root=str(tmp_path))
        assert d["run_dir_state"] == "fresh" and d["subset"]["restrict_to_split"] is True
        assert d["final_ckpt"].endswith("-step=40000.ckpt")


def test_cli_dry_run_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(reg, "check_subset_audit", lambda repo=None: {"sha256": "0" * 64})
    assert reg._cli(["dry-run", "--platform", "a6000", "--save-root", str(tmp_path)]) == 0
    assert "DRY-RUN OK: 3 run(s) on a6000" in capsys.readouterr().out
    _touch(tmp_path / RF["050"], ["epoch=2-step=2500.ckpt"])
    assert reg._cli(["dry-run", "--platform", "a6000", "--save-root", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):                              # the 8x8 rung does not exist
        reg._cli(["dry-run", "--platform", "neuronic", "--rung", "8x8", "--save-root", str(tmp_path)])
