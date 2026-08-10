"""Tests for the exp_11 Q10 restart record and its per-leg producer manifests.

Two of the re-pin review's blocking findings live here.

*Fix 3 — the recorder was fail-OPEN.* It re-hashed the resume checkpoint only
``if os.path.isfile(resume_path)``; a manifest naming a file that could not be
resolved was recorded on the strength of its own claimed hash, and no other field
of the manifest was checked at all. A leg recorded that way is what the >40k gate
then trusts, so the whole trajectory lineage rested on a string in a mutable file
under gitignored ``outputs_FLAC``.

*Fix 2 — the >40k gate was EXISTENTIAL.* One valid RESTART row admitted ANY later
same-config checkpoint in the arm's directory. The producer manifest binds
``step -> sha256 -> leg``: the screen re-hashes the file it is about to evaluate
and must find exactly that hash published for exactly that step by a fully
validated leg.

The fixtures are byte-level (small files, plain hashing) — no torch, no GPU.
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_EXPDIR = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude")


def _load(name):
    if _EXPDIR not in sys.path:
        sys.path.insert(0, _EXPDIR)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EXPDIR, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PM = _load("fa_orbit_producer_manifest")
REC = _load("fa_orbit_record_restart")

ARM = "C8"
INITIAL_JOB = "3648695"
LEG_JOB = "3662829"
LEG_UUID = "leg-uuid-c8"
LEG_COMMIT = "c" * 40
LAUNCH_COMMIT = "2" * 40
RESTART_TIME = "51:00:00"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


@pytest.fixture
def world(tmp_path):
    """A miniature repo: launcher pins, audited registry, save-dir, checkpoints."""
    root = str(tmp_path)
    expdir = os.path.join(root, "exp")
    os.makedirs(expdir)
    launcher = write(os.path.join(expdir, "fa_orbit_train.sbatch"), "\n".join([
        'PIN_PLACEHOLDER="TO-PIN-AFTER-P0"',
        'PINNED_RUNG="8x8"',
        'PINNED_MB="8"',
        'PINNED_NGPU="8"',
        "PINNED_MAXSTEPS=100000                     # Q10 budget",
        'PINNED_TIME_LIMIT_C8="35:00:00"',
        f'PINNED_TIME_LIMIT_RESTART_C8="{RESTART_TIME}"',
    ]) + "\n")
    cfg = write(os.path.join(expdir, "FLAC_AR_BF_C8.json"), json.dumps({"training": {"a": 8}}))
    cfg_sha = sha(open(cfg, "rb").read())

    save_dir = os.path.join("outputs", f"exp11_{ARM}")
    ckpt_dir = os.path.join(root, save_dir, f"FLAC_exp11_{ARM}", f"exp11_{ARM}", "checkpoints")
    os.makedirs(ckpt_dir)
    steps = {}
    for step in (40000, 42500, 45000):
        p = os.path.join(ckpt_dir, f"epoch={step // 5000}-step={step}.ckpt")
        write(p, f"weights for step {step}\n")
        steps[step] = (p, sha(open(p, "rb").read()))

    registry = os.path.join(expdir, "arm_launch_registry.json")
    with open(registry, "w") as fh:
        json.dump({"arms": {ARM: {
            "manifest_path": "outputs/exp11_C8/launch_manifest.txt",
            "manifest_sha256": "d" * 64, "job": INITIAL_JOB, "mode": "INITIAL",
            "launch_uuid": "initial-uuid", "commit": LAUNCH_COMMIT, "rung": "8x8",
            "micro": "8", "ngpu": "8", "max_steps": "40000", "config_sha256": cfg_sha,
            "vae_sha256": "b" * 64, "p0_manifest_sha256": "a" * 64, "save_dir": save_dir,
            "training_seed": 42, "final_ckpt_sha256": steps[40000][1], "final_step": 40000,
        }}, "restarts": {}}, fh, indent=2)
    return {"root": root, "expdir": expdir, "launcher": launcher, "config": cfg,
            "config_sha": cfg_sha, "registry": registry, "save_dir": save_dir,
            "ckpt_dir": ckpt_dir, "steps": steps}


def leg_manifest(world, **over):
    """A RESTART launch manifest exactly as fa_orbit_train.sbatch publishes one."""
    f = {"job": LEG_JOB, "mode": "RESTART", "launch_uuid": LEG_UUID, "commit": LEG_COMMIT,
         "arm": ARM, "rung": "8x8", "micro": "8", "ngpu": "8", "max_steps": "100000",
         "config_sha256": world["config_sha"], "vae_sha256": "b" * 64,
         "p0_manifest_sha256": "a" * 64, "save_dir": world["save_dir"],
         "resume_ckpt": world["steps"][40000][0], "expected_step": "40000",
         "resume_ckpt_sha256": world["steps"][40000][1], "time_limit": RESTART_TIME,
         "model_config": world["config"]}
    f.update(over)
    # "" drops the token from the line entirely, which is how a field goes
    # missing in a whitespace-paired manifest — an empty VALUE is unwriteable.
    head = (f"job {f['job']} " if f["job"] else "") + f"host neu000 mode {f['mode']}"
    if f["launch_uuid"]:
        head += f" launch_uuid {f['launch_uuid']}"
    text = "\n".join([
        "# exp_11 arm launch manifest",
        head,
        f"arm {f['arm']} rung {f['rung']} micro {f['micro']} ngpu {f['ngpu']} "
        f"max_steps {f['max_steps']} ckpt_every 2500",
        f"commit {f['commit']}",
        f"p0_manifest_sha256 {f['p0_manifest_sha256']}",
        f"model_config {f['model_config']}",
        f"config_sha256 {f['config_sha256']}",
        f"vae_sha256 {f['vae_sha256']}",
        f"time_limit {f['time_limit']} min_free_mb 36500",
        f"resume_ckpt {f['resume_ckpt']} expected_step {f['expected_step']} "
        f"resume_ckpt_sha256 {f['resume_ckpt_sha256']}",
        f"save_dir {f['save_dir']}",
        "",
    ])
    return write(os.path.join(world["root"], f"restart_manifest_{f['job']}.txt"), text)


def run_record(world, manifest, *extra):
    return REC.main([ARM, manifest, "--registry", world["registry"],
                     "--launcher", world["launcher"], "--producer-dir", world["expdir"],
                     "--repo-root", world["root"], *extra])


def registry_of(world):
    with open(world["registry"]) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# fix 3 — the recorder
# --------------------------------------------------------------------------
def test_records_a_clean_leg(world):
    assert run_record(world, leg_manifest(world)) == 0
    legs = registry_of(world)["restarts"][ARM]
    assert len(legs) == 1
    leg = legs[0]
    assert leg["job"] == LEG_JOB and leg["mode"] == "RESTART"
    assert leg["resume_ckpt_sha256"] == world["steps"][40000][1]
    assert leg["producer_manifest"] == PM.manifest_name(ARM, LEG_JOB)
    assert leg["time_limit"] == RESTART_TIME and leg["max_steps"] == "100000"


def test_missing_resume_file_is_refused_not_trusted(world, capsys):
    """RED for the fail-open finding: the manifest claims the right hash for a
    file that is not there. The old recorder skipped the re-hash and recorded it."""
    gone = os.path.join(world["ckpt_dir"], "epoch=8-step=40000.ckpt")
    os.unlink(gone)
    assert run_record(world, leg_manifest(world)) == 2
    assert "does not exist" in capsys.readouterr().out
    assert registry_of(world)["restarts"] == {}


def test_resume_file_is_always_rehashed(world, capsys):
    """The file exists but its bytes are not the audited ones; the manifest still
    claims the audited hash."""
    with open(world["steps"][40000][0], "w") as fh:
        fh.write("a different 40k checkpoint\n")
    assert run_record(world, leg_manifest(world)) == 2
    assert "does not continue that run" in capsys.readouterr().out


def test_non_canonical_resume_path_is_refused(world, capsys):
    stray = write(os.path.join(world["root"], "elsewhere", "epoch=8-step=40000.ckpt"),
                  open(world["steps"][40000][0]).read())
    assert run_record(world, leg_manifest(world, resume_ckpt=stray)) == 2
    assert "canonical directory" in capsys.readouterr().out


@pytest.mark.parametrize("over,needle", [
    ({"mode": "INITIAL"}, "not RESTART"),
    ({"job": ""}, "records no job"),
    ({"job": INITIAL_JOB}, "IS the INITIAL job"),
    ({"launch_uuid": ""}, "records no launch_uuid"),
    ({"commit": ""}, "records no commit"),
    ({"arm": "C16"}, "manifest arm"),
    ({"rung": "16x4"}, "manifest rung"),
    ({"micro": "16"}, "manifest micro"),
    ({"max_steps": "40000"}, "Q10 budget pin"),
    ({"expected_step": "37500"}, "audited final step"),
    ({"time_limit": "35:00:00"}, "RESTART wall pin"),
    ({"config_sha256": "0" * 64}, "manifest config_sha256"),
    ({"vae_sha256": "0" * 64}, "manifest vae_sha256"),
    ({"p0_manifest_sha256": "0" * 64}, "manifest p0_manifest_sha256"),
    ({"save_dir": "outputs/exp11_C16"}, "manifest save_dir"),
    ({"resume_ckpt_sha256": "0" * 64}, "!= the file's actual"),
])
def test_every_identity_field_is_validated(world, capsys, over, needle):
    assert run_record(world, leg_manifest(world, **over)) == 2
    assert needle in capsys.readouterr().out
    assert registry_of(world)["restarts"] == {}


def test_arm_without_an_audited_anchor_is_refused(world, capsys):
    reg = registry_of(world)
    del reg["arms"][ARM]["final_ckpt_sha256"]
    with open(world["registry"], "w") as fh:
        json.dump(reg, fh)
    assert run_record(world, leg_manifest(world)) == 2
    assert "no audited final_ckpt_sha256" in capsys.readouterr().out


def test_duplicate_legs_are_refused(world):
    assert run_record(world, leg_manifest(world)) == 0
    with pytest.raises(SystemExit) as exc:
        run_record(world, leg_manifest(world))
    assert "ALREADY recorded" in str(exc.value)
    assert len(registry_of(world)["restarts"][ARM]) == 1


def test_publication_is_atomic_and_leaves_no_debris(world):
    assert run_record(world, leg_manifest(world)) == 0
    leftovers = [f for f in os.listdir(world["expdir"]) if f.startswith(".")]
    assert leftovers == [], f"tmp files survived publication: {leftovers}"
    json.load(open(world["registry"]))               # still parseable
    json.load(open(os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB))))


def test_a_refused_record_writes_nothing(world):
    before = open(world["registry"]).read()
    assert run_record(world, leg_manifest(world, time_limit="35:00:00")) == 2
    assert open(world["registry"]).read() == before
    assert not os.path.exists(os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB)))


def test_dry_run_publishes_nothing(world):
    before = open(world["registry"]).read()
    assert run_record(world, leg_manifest(world), "--dry-run") == 0
    assert open(world["registry"]).read() == before
    assert not os.path.exists(os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB)))


# --------------------------------------------------------------------------
# fix 2 — the producer manifest
# --------------------------------------------------------------------------
def producer(world):
    return PM.load(os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB)))


def test_producer_manifest_holds_the_legs_own_checkpoints(world):
    assert run_record(world, leg_manifest(world)) == 0
    man = producer(world)
    assert sorted(man["checkpoints"], key=int) == ["42500", "45000"]  # 40000 is the INITIAL run's
    for step in (42500, 45000):
        assert man["checkpoints"][str(step)]["sha256"] == world["steps"][step][1]
    assert man["leg_manifest_sha256"] and man["chains_to"] == world["steps"][40000][1]


def test_extend_appends_new_steps_only(world):
    assert run_record(world, leg_manifest(world)) == 0
    new = os.path.join(world["ckpt_dir"], "epoch=9-step=47500.ckpt")
    write(new, "weights for step 47500\n")
    assert run_record(world, leg_manifest(world), "--extend") == 0
    man = producer(world)
    assert sorted(man["checkpoints"], key=int) == ["42500", "45000", "47500"]
    assert len(registry_of(world)["restarts"][ARM]) == 1, "extending must not add a second row"


def test_a_published_step_may_never_change(world, capsys):
    assert run_record(world, leg_manifest(world)) == 0
    with open(world["steps"][42500][0], "w") as fh:
        fh.write("swapped after publication\n")
    assert run_record(world, leg_manifest(world), "--extend", "--rehash-all") == 2
    assert "may never change" in capsys.readouterr().out


def test_steps_past_the_budget_are_not_the_legs_output(world):
    write(os.path.join(world["ckpt_dir"], "epoch=30-step=102500.ckpt"), "beyond the budget\n")
    assert run_record(world, leg_manifest(world)) == 0
    assert "102500" not in producer(world)["checkpoints"]


# --------------------------------------------------------------------------
# fix 2 — what the screen asks: this checkpoint, from this leg
# --------------------------------------------------------------------------
def chain(world, step, path=None, ckpt_sha=None):
    path = path or world["steps"][step][0]
    ckpt_sha = ckpt_sha or PM.sha256_file(path)
    return PM.verify_chain(registry_of(world), ARM, step, path, ckpt_sha,
                           world["expdir"], world["root"])


def test_the_produced_checkpoint_is_admissible(world):
    assert run_record(world, leg_manifest(world)) == 0
    problems, note = chain(world, 45000)
    assert problems == []
    assert "producer binding OK" in note and LEG_JOB in note


def test_a_foreign_checkpoint_at_a_produced_step_is_refused(world):
    """The existential loophole, closed: a valid leg exists, the step is one the
    leg produced, the config matches — but these are not that leg's bytes."""
    assert run_record(world, leg_manifest(world)) == 0
    with open(world["steps"][45000][0], "w") as fh:
        fh.write("a same-config checkpoint from a WRONG restart\n")
    problems, _ = chain(world, 45000)
    assert problems and "NOT the checkpoint that leg produced" in problems[0]


def test_a_step_the_leg_never_produced_is_refused(world):
    assert run_record(world, leg_manifest(world)) == 0
    stray = write(os.path.join(world["ckpt_dir"], "epoch=20-step=60000.ckpt"), "unpublished\n")
    problems, _ = chain(world, 60000, path=stray)
    assert problems and "produced no checkpoint at step 60000" in problems[0]


def test_a_checkpoint_moved_out_of_the_published_path_is_refused(world):
    assert run_record(world, leg_manifest(world)) == 0
    copy = write(os.path.join(world["root"], "elsewhere", "epoch=9-step=45000.ckpt"),
                 open(world["steps"][45000][0]).read())
    problems, _ = chain(world, 45000, path=copy)
    assert problems and "was published at" in problems[0]


def test_an_arm_with_no_recorded_leg_admits_nothing(world):
    problems, _ = chain(world, 45000)
    assert problems and "has no RESTART entry" in problems[0]


def test_a_missing_producer_manifest_admits_nothing(world):
    """A registry row alone is no longer sufficient evidence."""
    assert run_record(world, leg_manifest(world)) == 0
    os.unlink(os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB)))
    problems, _ = chain(world, 45000)
    assert problems and "producer manifest" in problems[0]


@pytest.mark.parametrize("field,value,needle", [
    ("mode", "INITIAL", "leg mode"),
    ("resume_ckpt_sha256", "0" * 64, "resume_ckpt_sha256"),
    ("chains_to", "0" * 64, "chains_to"),
    ("save_dir", "outputs/exp11_C16", "save_dir"),
    ("config_sha256", "0" * 64, "config_sha256"),
    ("expected_step", "30000", "resumed at step"),
    ("commit", "", "records no commit"),
    ("producer_manifest", "", "records no producer_manifest"),
])
def test_the_screen_revalidates_every_leg_field(world, field, value, needle):
    """The old gate checked mode and the resume hash. Tamper with any other field
    of the recorded row and the checkpoint must stop being admissible."""
    assert run_record(world, leg_manifest(world)) == 0
    reg = registry_of(world)
    reg["restarts"][ARM][0][field] = value
    with open(world["registry"], "w") as fh:
        json.dump(reg, fh)
    problems, _ = chain(world, 45000)
    assert problems and needle in problems[0]


def test_a_producer_manifest_from_another_leg_is_refused(world):
    """Swapping in a different leg's published file must not launder a checkpoint."""
    assert run_record(world, leg_manifest(world)) == 0
    man_path = os.path.join(world["expdir"], PM.manifest_name(ARM, LEG_JOB))
    man = PM.load(man_path)
    man["job"] = "9999999"
    PM.write_atomic(man_path, man)
    problems, _ = chain(world, 45000)
    assert problems and "!= the registry leg's" in problems[0]
