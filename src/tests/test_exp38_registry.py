"""exp_38 round B — the run registry (worklog/.../exp_38_data_efficiency_claude/exp38_registry.py).

Executable statement of plan_data_efficiency.md §6-§7 and §10 item 4: the six runs' configs,
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
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reg = _load("exp38_registry")

# dc_van_f025's at-launch argv (exp-14-data-curve:worklog/worklog_yixun/exp_14_data_curve/
# at_launch_dc_van_f025_20260916T190832-1836418.txt), with its kit-local config path.
DC_VAN_F025_ARGV = (
    "python train.py --model-config /home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/"
    "exp_14_data_curve_claude/configs/FLAC_AR_exp14_vanS.json --dataset-config "
    "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json --pretransform-ckpt-path "
    "weights/FLAC/VAE.safetensors --max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 "
    "--seed 42 --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true --logger none "
    "--checkpoint-every 2500 --name dc_van_f025 --experiment-name dc_van_f025 --save-dir "
    "/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/dc_van_f025 --run-contract-json "
    "/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/dc_van_f025/run_contract.json")


def _opt(argv, flag):
    return argv[argv.index(flag) + 1]


# ------------------------------------------------------------------ runs and rungs
def test_required_runs_are_the_six_and_the_bridge_is_optional():
    assert reg.required_runs() == ["de_yaw_f025", "de_yaw_f050", "de_yaw_f075",
                                   "de_bf_f025", "de_bf_f050", "de_bf_f075"]
    assert reg.all_runs()[-1] == "de_p1_f025"
    assert reg.run_id("bf", "050", "8x8") == "de_bf_f050_r8x8"
    assert reg.parse_run_id("de_bf_f050_r8x8") == ("bf", "050", "8x8")
    for bad in ("dc_van_f025", "de_bf_f100", "de_bf_f025_r2x32", "de_cyl_f025", ""):
        with pytest.raises(reg.GateError):
            reg.parse_run_id(bad)


def test_rungs_keep_eff_and_syncbn_batch_64_and_8x8_is_opt_in():
    for r in reg.RUNGS.values():
        assert r["ngpu"] * r["micro"] == 64
    assert reg.check_rung("a6000", "2x32")["micro"] == 32
    with pytest.raises(reg.GateError):
        reg.check_rung("a6000", "8x8", allow_nonmatched=True)      # not on this box
    with pytest.raises(reg.GateError):
        reg.check_rung("neuronic", "8x8")                          # opt-in only
    assert reg.check_rung("neuronic", "8x8", allow_nonmatched=True)["micro"] == 8


# ------------------------------------------------------------------ configs
@pytest.mark.parametrize("micro", [32, 8])
def test_committed_bf_configs_are_the_base_plus_exactly_the_cap(micro):
    base_text = open(os.path.join(REPO, reg.BF_BASE_CONFIG)).read()
    path = os.path.join(REPO, reg.bf_config_rel(micro))
    assert open(path).read() == reg.derive_bf_config_text(base_text, micro)
    base, got = json.loads(base_text), json.load(open(path))
    assert got["training"].pop("frame_avg_max_fwd_samples") == micro
    assert got == base


def test_bf_base_is_the_b_f_100_percent_config():
    assert reg.sha256_file(os.path.join(REPO, reg.BF_BASE_CONFIG)) == reg.BF_BASE_SHA256


@pytest.mark.parametrize("arm,rung", [("yaw", "2x32"), ("bf", "2x32"), ("bf", "8x8"), ("p1", "2x32")])
def test_check_config_accepts_the_registered_arm_configs(arm, rung):
    cfg = reg.check_config(arm, rung)
    assert cfg["training"].get("cond_method", "vanilla") == reg.ARMS[arm]["cond_method"]


def test_yaw_config_is_bvp1_plus_only_the_yaw_aug_block():
    yaw = json.load(open(os.path.join(REPO, reg.ARMS["yaw"]["config"])))
    p1 = json.load(open(os.path.join(REPO, reg.ARMS["p1"]["config"])))
    assert yaw["training"].pop("yaw_aug") == {"enabled": True, "img_w": 512, "seed": 42}
    assert yaw == p1


def test_check_config_refuses_a_tampered_bf_config(tmp_path):
    root = tmp_path / "repo"
    for rel in (reg.BF_BASE_CONFIG, reg.bf_config_rel(32)):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(open(os.path.join(REPO, rel)).read())
    reg.check_config("bf", "2x32", repo=str(root))
    bad = json.loads((root / reg.bf_config_rel(32)).read_text())
    bad["training"]["frame_avg_max_fwd_samples"] = 64
    (root / reg.bf_config_rel(32)).write_text(json.dumps(bad, indent=4) + "\n")
    with pytest.raises(reg.GateError):
        reg.check_config("bf", "2x32", repo=str(root))


def test_the_factory_hands_the_cap_to_the_wrapper():
    from src.training.factory import _parse_frame_avg_cap_config
    cfg = json.load(open(os.path.join(REPO, reg.bf_config_rel(32))))
    assert _parse_frame_avg_cap_config(cfg["training"]) == {"frame_avg_max_fwd_samples": 32}


def test_derive_refuses_bad_caps_and_a_base_that_already_has_one():
    base_text = open(os.path.join(REPO, reg.BF_BASE_CONFIG)).read()
    for cap in (0, -1, True, 32.0, "32"):
        with pytest.raises(reg.GateError):
            reg.derive_bf_config_text(base_text, cap)
    with pytest.raises(reg.GateError):
        reg.derive_bf_config_text(reg.derive_bf_config_text(base_text, 32), 32)


# ------------------------------------------------------------------ chunk plan (announcement 06)
@pytest.mark.parametrize("rung", ["2x32", "8x8"])
def test_bf_chunk_plan_is_one_angle_per_forward(rung):
    plan = reg.chunk_plan("bf", rung)
    assert plan["frame_avg_max_fwd_samples"] == reg.RUNGS[rung]["micro"]
    assert plan["angles_per_chunk"] == 1 and plan["shared_rope_draws"] == 0
    assert reg.chunk_plan("yaw", rung) is None


@pytest.mark.parametrize("batch", [32, 8])
def test_batched_orbit_at_cap_equal_batch_issues_the_per_angle_loops_calls(batch):
    """With cap == per-rank batch the batched orbit makes exactly the legacy loop's conditioner
    calls (one per non-zero angle, B samples each, ascending angle order) and the same sum."""
    import torch
    from src.data import yaw_rotation as yr

    class Spy:
        def __init__(self):
            self.calls = []

        def __call__(self, mds, device, only_ids=None):
            self.calls.append([float(m["source_vit"][0]) for m in mds])
            out = torch.stack([m["source_vit"] for m in mds]).sum(-1, keepdim=True)
            return {"source_vit": [out, None]}

    def md(i):
        depth = torch.zeros(3, 4, 512)
        depth[0, :, i % 512] = 1.0
        return {"depth": depth, "source_vit": torch.tensor([1.0 + i, 2.0, 0.5])}

    mds = [md(i) for i in range(batch)]
    base = {"source_vit": [torch.zeros(batch, 1), None]}
    angles = (0.0, 90.0, 180.0, 270.0)
    spy_b, spy_l = Spy(), Spy()
    got = yr._orbit_average_batched(spy_b, mds, base, ["source_vit"], angles, 512, "cpu",
                                    cap=batch, cap_is_explicit=True)
    ref = yr._orbit_average_loop(spy_l, mds, base, ["source_vit"], angles, 512, "cpu")
    assert len(spy_b.calls) == 3 and all(len(c) == batch for c in spy_b.calls)
    assert spy_b.calls == spy_l.calls
    assert torch.equal(got["source_vit"], ref["source_vit"])
    # the default cap (64) at micro 32 would share two angles in one forward -- the reason for the key
    if batch == 32:
        spy_d = Spy()
        yr._orbit_average_batched(spy_d, mds, base, ["source_vit"], angles, 512, "cpu")
        assert [len(c) for c in spy_d.calls] == [64, 32]


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
    run = "de_bf_f025"
    assert reg.check_run_dir(str(tmp_path / "absent"), run, 0)["state"] == "fresh"
    d = tmp_path / "rd"
    _touch(d, ["at_launch_de_bf_f025_x.json"])
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


def test_only_the_registered_matrix_is_launchable():
    """Codex code r1 HIGH 3: P1 only at f025 / 2x32; the 8x8 rung only for the six primary runs."""
    assert sorted(reg.REGISTERED_RUNS) == sorted(reg.all_runs("2x32") + reg.required_runs("8x8"))
    for bad in ("de_p1_f050", "de_p1_f075", "de_p1_f025_r8x8", "de_p1_f050_r8x8", "de_yaw_f100",
                "de_bf_f025_r4x16", "de_bf_f025 ", "", None):
        with pytest.raises(reg.GateError):
            reg.parse_run_id(bad)
        with pytest.raises(reg.GateError):
            reg.build_train_argv(bad, "neuronic", allow_nonmatched=True)


# ------------------------------------------------------------------ argv
def test_a6000_argv_is_dc_vans_argv_with_only_the_registered_differences():
    got = reg.build_train_argv("de_p1_f025", "a6000")
    ref = shlex.split(DC_VAN_F025_ARGV)[1:]
    ref = ref[:ref.index("--run-contract-json")]
    swap = {"--model-config": reg.ARMS["p1"]["config"],
            "--name": "de_p1_f025", "--experiment-name": "de_p1_f025",
            "--save-dir": "/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency/de_p1_f025"}
    for flag, value in swap.items():
        ref[ref.index(flag) + 1] = value
    ref += ["--evidence-json", swap["--save-dir"] + "/evidence/run_evidence.json"]
    i = ref.index("--strategy")
    ref[i:i] = ["--num-nodes", "1"]
    i = ref.index("--logger")
    ref[i:i] = ["--precision", "bf16-mixed", "--val-every", "-1", "--gradient-clip-val", "0.0"]
    assert got == ref


@pytest.mark.parametrize("run", ["de_yaw_f025", "de_yaw_f050", "de_yaw_f075",
                                 "de_bf_f025", "de_bf_f050", "de_bf_f075"])
def test_argv_per_run(run):
    arm, tag, _ = reg.parse_run_id(run)
    argv = reg.build_train_argv(run, "a6000")
    assert _opt(argv, "--model-config") == (reg.ARMS["yaw"]["config"] if arm == "yaw"
                                            else reg.bf_config_rel(32))
    assert _opt(argv, "--dataset-config") == reg.FRACS[tag]["dataset_config"]
    assert (_opt(argv, "--batch-size"), _opt(argv, "--num-gpus"), _opt(argv, "--accum-batches")) == ("32", "2", "1")
    assert (_opt(argv, "--sync-batchnorm"), _opt(argv, "--precision"), _opt(argv, "--seed")) == ("true", "bf16-mixed", "42")
    assert (_opt(argv, "--max-steps"), _opt(argv, "--checkpoint-every")) == ("40000", "2500")
    assert "--ckpt-path" not in argv and not any("exp14_" in a for a in argv)
    n = reg.build_train_argv(run, "neuronic")
    assert _opt(n, "--save-dir") == f"/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency/{run}"
    assert [a for a in n if not a.startswith("/n/")] == [a for a in argv if not a.startswith("/media/")]


def test_8x8_argv_needs_the_opt_in_and_uses_cap_8():
    with pytest.raises(reg.GateError):
        reg.build_train_argv("de_bf_f025_r8x8", "neuronic")
    argv = reg.build_train_argv("de_bf_f025_r8x8", "neuronic", allow_nonmatched=True)
    assert (_opt(argv, "--batch-size"), _opt(argv, "--num-gpus")) == ("8", "8")
    assert _opt(argv, "--model-config") == reg.bf_config_rel(8)


def test_there_is_no_resume_argv(capsys):
    """Codex code r1 HIGH 2: no resume parameter, no --ckpt-path, no CLI flag."""
    import inspect
    assert "resume" not in "".join(inspect.signature(reg.build_train_argv).parameters)
    for run in reg.all_runs():
        assert "--ckpt-path" not in reg.build_train_argv(run, "neuronic")
    with pytest.raises(SystemExit):
        reg._cli(["argv", "de_bf_f050", "--platform", "neuronic", "--resume-step", "10000"])
    with pytest.raises(SystemExit):
        reg._cli(["run-dir-check", "/x", "--run", "de_bf_f050", "--state", "10000"])


def test_smoke_argv_single_rank_and_ddp():
    argv = reg.build_smoke_argv("de_bf_f025", 20, "/tmp/x")
    assert (_opt(argv, "--num-gpus"), _opt(argv, "--batch-size"), _opt(argv, "--max-steps")) == ("1", "32", "20")
    assert "--sync-batchnorm" not in argv and _opt(argv, "--checkpoint-every") == "1000000"
    ddp = reg.build_smoke_argv("de_bf_f025", 30, "/tmp/smk/de_bf_f025_x", ddp=True)
    full = reg.build_train_argv("de_bf_f025", "a6000")
    diff = {i for i, (a, b) in enumerate(zip(ddp, full)) if a != b}
    assert len(ddp) == len(full) and {ddp[i - 1] for i in diff} == {"--max-steps", "--checkpoint-every",
                                                                     "--save-dir", "--name", "--experiment-name",
                                                                     "--evidence-json"}
    assert (_opt(ddp, "--max-steps"), _opt(ddp, "--checkpoint-every")) == ("30", "1000000")
    with pytest.raises(reg.GateError):
        reg.build_smoke_argv("de_bf_f025", 500, "/tmp/x")


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
    return {
        "global_step": step, "epoch": ep, "model_config": cfg, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion_ema.ema_model.w": torch.zeros(2), "diffusion.model.w": torch.zeros(2),
                       "diffusion_ema.step": torch.tensor([step]), "diffusion_ema.initted": torch.tensor([1.0]),
                       "diffusion.conditioner.c": torch.zeros(1),
                       **{f"diffusion.pretransform.f{i}": torch.zeros(1) for i in range(reg.N_STATE_TENSORS - 5)}},
        "optimizer_states": [{"state": {i: {"step": torch.tensor(float(step)), "exp_avg": torch.zeros(shp(i)),
                                            "exp_avg_sq": torch.zeros(shp(i))} for i in range(n)},
                              "param_groups": [{"lr": lr, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.001,
                                                "amsgrad": False, "maximize": False, "foreach": None,
                                                "capturable": False, "differentiable": False, "fused": None,
                                                "decoupled_weight_decay": True, "initial_lr": 5e-05,
                                                "params": list(range(753))}]}],
        "lr_schedulers": [{"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0, "base_lrs": [5e-05],
                           "last_epoch": step, "_step_count": step + 1, "_last_lr": [lr],
                           "_get_lr_called_within_step": False}],
        "loops": {"fit_loop": {**{k: {} for k in reg.FIT_LOOP_KEYS},
                               "epoch_loop.automatic_optimization.optim_progress":
                                   {"optimizer": {"step": {"total": {"completed": step}}}},
                               "epoch_loop.batch_progress": {"total": prog},
                               "epoch_progress": {"total": {"processed": ep}}},
                  "predict_loop": {}, "test_loop": {}, "validate_loop": {}},
        "callbacks": {reg.MODEL_CHECKPOINT_KEY: {k: None for k in reg.MODEL_CHECKPOINT_STATE_KEYS} | {"dirpath": run_dir}}}


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
        "model_config": _set(("model_config",), json.load(open(os.path.join(REPO, reg.ARMS["p1"]["config"])))),
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
    run, step = "de_yaw_f025", 40000
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


@pytest.mark.parametrize("arm,rung", [("yaw", "2x32"), ("bf", "2x32"), ("bf", "8x8"), ("p1", "2x32")])
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


DC_VAN_F025_40K = "/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/dc_van_f025/epoch=34-step=40000.ckpt"
REAL_100 = {"YAW": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/YAW/epoch=8-step=40000.ckpt",
            "BF": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/BF/epoch=8-step=40000.ckpt"}


@pytest.mark.skipif(not os.path.exists(DC_VAN_F025_40K), reason="NAS not mounted")
def test_audit_ckpt_admits_exp14s_real_dc_van_f025_endpoint():
    """Real-data readback: dc_van_f025 was trained with P1's config at our fraction 025 (the
    full exact-schema audit incl. the inventory; dc_van carries exp_14's run_contract key)."""
    rd = os.path.dirname(DC_VAN_F025_40K)
    out = reg.audit_ckpt(DC_VAN_F025_40K, "de_p1_f025", 40000, run_dir=rd, extra_top_keys={"run_contract"})
    assert out["sha256"] == "aa1a2cd624333a18eb2f9f019f6b46beaf2d94c1b2d7dc455f238ad0cec9433c"
    assert out["bytes"] == 723923435
    with pytest.raises(reg.GateError):                          # without the exp_14 allowance
        reg.audit_ckpt(DC_VAN_F025_40K, "de_p1_f025", 40000, run_dir=rd)
    with pytest.raises(reg.GateError):
        reg.audit_ckpt(DC_VAN_F025_40K, "de_p1_f025", 40000, run_dir="/elsewhere", extra_top_keys={"run_contract"})


@pytest.mark.skipif(not all(os.path.exists(p) for p in REAL_100.values()), reason="NAS not mounted")
@pytest.mark.parametrize("name", sorted(REAL_100))
def test_the_real_100_percent_endpoints_satisfy_the_exact_schema(name):
    """YAW / B-F 100 % (train.json, 4,551 steps / epoch): the only schema differences from a
    fraction run are the epoch counters -- every key set and recipe value is the registered one."""
    import torch
    ck = torch.load(REAL_100[name], map_location="cpu", weights_only=False)
    probs = reg.ckpt_schema_problems(ck, 40000, "025")
    assert len(probs) == 2 and probs[0] == "epoch 8 != 34" and probs[1].startswith("epoch progress")


def test_ckpt_manifest_needs_the_full_cadence(tmp_path):
    run = "de_bf_f075"
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
def _log(run="de_yaw_f025", ngpu=2, n=73473, banner=True, flash_missing=0, step=True, end=False,
         loss="0.512", members=None):
    parts = ["Seed set to 42"]
    parts += ["No module named 'flash_attn'", reg.FLASH_MISSING + ", disabling Flash Attention"] * flash_missing
    parts += [f"Found {n} files in 243 subfolders"] * ngpu
    if ngpu > 1:
        for r in (members if members is not None else range(ngpu)):
            parts.append(f"Initializing distributed: GLOBAL_RANK: {r}, MEMBER: {r + 1}/{ngpu}")
    if banner:
        parts.append(reg.YAW_BANNER)
    if step:
        parts.append(f"Epoch 0:   0%|          | 1/1148 [00:05<1:00:00,  0.30it/s, train/loss={loss}]")
    text = "\n".join(parts)
    if end:
        text += "\rEpoch 34:  84%|########4 | 968/1148 [59:53<11:08,  0.27it/s] " + reg.FIT_END.format(n=40000)
    return text


def test_log_gate_start_pass():
    assert reg.log_gate(_log(), "de_yaw_f025", "start", "flash", 2) == []
    assert reg.log_gate(_log(run="de_bf_f025", banner=False), "de_bf_f025", "start", "flash", 2) == []
    assert reg.log_gate(_log(flash_missing=2), "de_yaw_f025", "start", "sdpa", 2) == []
    assert reg.log_gate(_log(ngpu=1), "de_yaw_f025", "start", "flash", 1) == []


@pytest.mark.parametrize("kw,run,backend", [
    ({"banner": False}, "de_yaw_f025", "flash"),
    ({"n": 291210}, "de_yaw_f025", "flash"),
    ({"members": [0]}, "de_yaw_f025", "flash"),
    ({"flash_missing": 2}, "de_yaw_f025", "flash"),
    ({"flash_missing": 1}, "de_yaw_f025", "sdpa"),
    ({"step": False}, "de_yaw_f025", "flash"),
    ({"loss": "nan"}, "de_yaw_f025", "flash"),
    ({"loss": "-inf"}, "de_yaw_f025", "flash"),
    ({}, "de_bf_f025", "flash"),                                 # a B-F run printing the yaw banner
])
def test_log_gate_start_failures(kw, run, backend):
    assert reg.log_gate(_log(**kw), run, "start", backend, 2) != []


def test_log_gate_banner_after_the_first_step_fails():
    text = _log(banner=False) + "\n" + reg.YAW_BANNER
    assert any("banner" in p for p in reg.log_gate(text, "de_yaw_f025", "start", "flash", 2))


def test_log_gate_end_needs_lightnings_marker_and_enough_finite_losses():
    assert reg.log_gate(_log(end=True), "de_yaw_f025", "end", "flash", 2) == []
    assert reg.log_gate(_log(), "de_yaw_f025", "end", "flash", 2) != []
    assert reg.log_gate(_log(end=True), "de_yaw_f025", "end", "flash", 2, min_losses=2) != []


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
             f"{reg.EXP_DIR}/exp38_subset_audit.py": None}
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


def test_snapshot_is_complete_immutable_and_usable(tmp_path):
    """Codex code r1 HIGH 5: training is launched FROM the snapshot -- the closure (same digest as
    the tree), the subset-audit sidecar, and the two cwd-relative data links."""
    files = reg.snapshot_files("de_bf_f050")
    if reg.bf_config_rel(32) not in files or "worklog/worklog_yixun/exp_38_data_efficiency_claude/exp38_supervise.py" not in files:
        pytest.skip("the exp_38 kit is not committed yet (the closure lists tracked files only)")
    assert "train.py" in files and "src/training/run_evidence.py" in files and reg.SUBSET_AUDIT in files
    assert not any(f.startswith("src/tests/") for f in files)
    dest = tmp_path / "snap"
    out = reg.snapshot_tools("de_bf_f050", str(dest))
    assert reg.verify_snapshot(str(dest), out["snapshot_sha256"])["verified"]
    assert reg.closure_digest(str(dest)) == reg.closure_digest()
    for rel in reg.SNAPSHOT_LINKS:
        assert os.path.islink(dest / rel) and os.readlink(dest / rel) == os.path.realpath(os.path.join(REPO, rel))
    reg.check_config("bf", "2x32", repo=str(dest))
    reg.check_subset("050", repo=str(dest))
    assert reg.check_subset_audit(str(dest))["sha256"]
    with pytest.raises(reg.GateError):
        reg.snapshot_tools("de_bf_f050", str(dest))             # never overwritten
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
    cfg = dest / reg.bf_config_rel(32)
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
    assert "DRY-RUN OK: 6 run(s) on a6000" in capsys.readouterr().out
    _touch(tmp_path / "de_yaw_f050", ["epoch=2-step=2500.ckpt"])
    assert reg._cli(["dry-run", "--platform", "a6000", "--save-root", str(tmp_path)]) == 2
    assert reg._cli(["dry-run", "--platform", "neuronic", "--rung", "8x8", "--save-root", str(tmp_path)]) == 2
