"""exp_43 plan §5.4 — the offline W&B training-history extractor, test first.

The tool (``worklog/worklog_yixun/exp_43_std_aug_claude/exp43_train_history.py``) is loaded by
path. Every fixture run is a tiny ``.wandb`` file written inside ``tmp_path`` by the installed
wandb's OWN datastore writer (``wandb.sdk.internal.datastore.DataStore.open_for_write`` + the
``wandb_internal_pb2.Record`` protobuf), shaped like a PL ``WandbLogger`` offline run of the 40k
recipe (``trainer/global_step`` 99, 199, ...). The single real-data test is opt-in and skips when
exp_40's FULL run directory is absent.
"""
import ast
import csv
import glob
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude/exp43_train_history.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


th = _load("exp43_train_history", _TOOL)

from wandb.proto import wandb_internal_pb2 as pb  # noqa: E402
from wandb.sdk.internal import datastore  # noqa: E402

RUN_ID = "abc12345"
DIRNAME = f"offline-run-20261113_010000-{RUN_ID}"


# --------------------------------------------------------------------------- #
# fixture writer (the installed wandb's own writer)
# --------------------------------------------------------------------------- #
def _rec(kind, **kw):
    r = pb.Record()
    if kind == "header":
        r.header.SetInParent()
    elif kind == "run":
        r.run.run_id = RUN_ID
    elif kind == "exit":
        r.exit.exit_code = kw.get("code", 0)
        r.exit.runtime = 321
    elif kind == "footer":
        r.footer.SetInParent()
    elif kind == "output":
        r.output_raw.line = kw["line"]
    elif kind == "history":
        for k, v in kw["items"]:
            it = r.history.item.add()
            it.key = k
            it.value_json = v if isinstance(v, str) and kw.get("raw") else json.dumps(v)
        r.history.step.num = kw.get("num", 0)
    elif kind == "partial":
        it = r.request.partial_history.item.add()
        it.key = "train/loss"
        it.value_json = "1.0"
    return r


def row(step, loss=None, lr=5e-5, num=0, extra=(), drop=()):
    loss = 2.0 / (1 + step / 1000) if loss is None else loss
    items = [("_timestamp", 1.79e9 + step), ("train/loss", loss), ("train/std_data", 1.2), ("train/lr", lr),
             ("train/mse_loss", loss), ("epoch", step // 4550), ("trainer/global_step", step),
             ("_runtime", 2.0 * step), ("_step", num), *extra]
    return _rec("history", items=[(k, v) for k, v in items if k not in drop], num=num)


def default_rows(steps=range(99, 1000, 100)):
    rows = [row(s, num=i) for i, s in enumerate(steps)]
    # one big row (gradient histograms) that spans three 32 KiB blocks: FIRST / MIDDLE / LAST records
    big = row(199, num=99, extra=[("gradients/w", {"_type": "histogram", "values": [0.5] * 14000})])
    rows[1] = big
    return rows


def write_run(root, body=None, *, head=True, tail=True, dirname=DIRNAME, fname=None):
    d = Path(root) / "wandb" / dirname
    d.mkdir(parents=True)
    path = d / (fname or f"run-{RUN_ID}.wandb")
    ds = datastore.DataStore()
    ds.open_for_write(str(path))
    recs = ([_rec("header"), _rec("run"), _rec("output", line="Epoch 0 ...")] if head else [])
    recs += list(default_rows() if body is None else body)
    recs += ([_rec("output", line="done"), _rec("exit"), _rec("footer")] if tail else [])
    for r in recs:
        if isinstance(r, bytes):
            ds._write_data(r)
        else:
            ds.write(r)
    ds.close()
    return d, path


def _run(d, tmp_path, *extra, csv_name="h.csv", json_name="h.json"):
    oc, oj = tmp_path / csv_name, tmp_path / json_name
    rc = th.main(["extract", "--run-dir", str(d), "--out-csv", str(oc), "--out-json", str(oj), *extra])
    return rc, oc, oj


def _refused(d, tmp_path, *extra):
    rc, oc, oj = _run(d, tmp_path, *extra)
    assert rc == 2 and not oc.exists() and not oj.exists()


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extracts_every_loss_row_with_the_logged_trainer_global_step(tmp_path):
    body = default_rows() + [_rec("history", items=[("gradients/x", {"_type": "histogram"}), ("_step", 50)])]
    d, f = write_run(tmp_path, body)
    rc, oc, oj = _run(d, tmp_path)
    assert rc == 0
    with open(oc, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["step", "train/loss", "lr"]
    steps = list(range(99, 1000, 100))
    assert [int(r[0]) for r in rows[1:]] == steps
    assert rows[1] == ["99", repr(2.0 / (1 + 99 / 1000)), repr(5e-5)]
    assert oc.read_text().startswith("step,train/loss,lr\n99,")
    side = json.loads(oj.read_text())
    assert side["schema"] == th.SCHEMA
    assert side["run_file_sha256"] == _sha(f) and side["run_file_bytes"] == f.stat().st_size
    assert side["run_id"] == RUN_ID and side["run_dir"] == str(d.resolve())
    assert side["row_count"] == 10 and side["first_step"] == 99 and side["last_step"] == 999
    assert side["history_rows_total"] == 11 and side["history_rows_without_loss"] == 1
    assert side["columns"] == {"step": "trainer/global_step", "train/loss": "train/loss", "lr": "train/lr"}
    assert {"train/loss", "train/lr", "train/mse_loss", "trainer/global_step", "_step", "gradients/w",
            "gradients/x", "epoch"} <= set(side["keys_seen"])
    assert side["keys_seen"] == sorted(side["keys_seen"])
    assert side["cadence"]["ok"] is True and side["cadence"]["period"] == 100
    assert side["cadence"]["gaps"] == [] and side["cadence"]["duplicates"] == {}
    assert side["parse"]["complete"] is True and side["parse"]["records_by_type"]["history"] == 11
    assert side["parse"]["exit_record"] == {"present": True, "exit_code": 0, "runtime": 321}
    assert side["parse"]["multi_block_records"] >= 1
    assert side["tool"]["sha256"] == _sha(_TOOL)
    assert side["csv"]["sha256"] == _sha(oc) and side["csv"]["rows"] == 10
    assert side["wandb_version"] == "0.15.4"
    assert "_batches_that_stepped" in side["step_semantics"]


def test_cadence_problems_are_reported_not_fixed(tmp_path):
    steps = [99, 199, 399, 399, 450, 499, 299]
    d, _ = write_run(tmp_path, [row(s, num=i) for i, s in enumerate(steps)])
    rc, oc, oj = _run(d, tmp_path)
    assert rc == 3
    with open(oc, newline="") as fh:
        assert [int(r[0]) for r in list(csv.reader(fh))[1:]] == steps     # as logged, in file order
    cad = json.loads(oj.read_text())["cadence"]
    assert cad["ok"] is False
    assert cad["duplicates"] == {"399": 2}
    assert cad["off_cadence"] == [450]
    assert cad["non_monotonic"] == [{"index": 3, "previous": 399, "step": 399},
                                    {"index": 6, "previous": 499, "step": 299}]
    assert cad["gaps"] == []          # 299 is present (out of order), so no expected step is missing


def test_gaps_against_expected_max_steps(tmp_path):
    d, _ = write_run(tmp_path, [row(s, num=i) for i, s in enumerate([99, 199, 399, 499])])
    rc, _oc, oj = _run(d, tmp_path, "--expected-max-steps", "700")
    assert rc == 3
    cad = json.loads(oj.read_text())["cadence"]
    assert cad["gaps"] == [299, 599, 699] and cad["expected_last_step"] == 699
    d2, _ = write_run(tmp_path / "b", [row(s, num=i) for i, s in enumerate(range(99, 700, 100))])
    rc2, _oc2, oj2 = _run(d2, tmp_path, "--expected-max-steps", "700", csv_name="b.csv", json_name="b.json")
    assert rc2 == 0 and json.loads(oj2.read_text())["cadence"]["ok"] is True


def test_first_row_must_be_step_99(tmp_path):
    d, _ = write_run(tmp_path, [row(s, num=i) for i, s in enumerate([199, 299])])
    rc, _oc, oj = _run(d, tmp_path)
    cad = json.loads(oj.read_text())["cadence"]
    assert rc == 3 and cad["first_step_is_99"] is False and cad["gaps"] == [99]


def test_nonfinite_values_are_kept_and_reported(tmp_path):
    d, _ = write_run(tmp_path, [row(99, loss=float("nan")), row(199, num=1)])
    rc, oc, oj = _run(d, tmp_path)
    assert rc == 0 and oc.read_text().split("\n")[1] == "99,nan,5e-05"
    assert json.loads(oj.read_text())["nonfinite"] == {"train/loss": [99], "lr": []}


def test_record_ending_within_six_bytes_of_a_block_end_parses(tmp_path):
    """wandb's naive ``while scan_data()`` would read the next 'padding' and assert at a clean EOF."""
    pre = [_rec("header"), _rec("run")] + [row(s, num=i) for i, s in enumerate(range(99, 600, 100))]
    tail = [_rec("exit"), _rec("footer")]
    used = 7 + sum(7 + r.ByteSize() for r in pre + tail)
    target = th.BLOCK - 3 - used - 7
    filler = next(r for r in (_rec("output", line="x" * n) for n in range(target - 8, target + 1))
                  if r.ByteSize() == target)
    d, f = write_run(tmp_path, pre + [filler] + tail, head=False, tail=False)
    assert f.stat().st_size == th.BLOCK - 3
    rc, _oc, oj = _run(d, tmp_path)
    assert rc == 0 and json.loads(oj.read_text())["row_count"] == 6


# --------------------------------------------------------------------------- #
# refusals: the run file must parse completely
# --------------------------------------------------------------------------- #
def _cut(path, n):
    data = path.read_bytes()
    path.write_bytes(data[:n])


@pytest.mark.parametrize("where", ["last_byte", "inside_last_record", "inside_record_header", "block_boundary"])
def test_truncated_run_file_is_refused(tmp_path, where):
    d, f = write_run(tmp_path)
    size = f.stat().st_size
    cut = {"last_byte": size - 1, "inside_last_record": size - 4, "inside_record_header": 7 + 3,
           "block_boundary": th.BLOCK}[where]
    assert cut < size
    _cut(f, cut)
    _refused(d, tmp_path)


def test_corrupted_byte_is_refused(tmp_path):
    d, f = write_run(tmp_path)
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    _refused(d, tmp_path)


def test_bad_file_header_is_refused(tmp_path):
    d, f = write_run(tmp_path)
    data = bytearray(f.read_bytes())
    data[0:4] = b"XXXX"
    f.write_bytes(bytes(data))
    _refused(d, tmp_path)


@pytest.mark.parametrize("payload", [b"\xff\xff\xff\xff\xff", b""])
def test_unparseable_or_typeless_record_is_refused(tmp_path, payload):
    d, _ = write_run(tmp_path, default_rows() + [payload])
    _refused(d, tmp_path)


def test_unflushed_partial_history_request_is_refused(tmp_path):
    d, _ = write_run(tmp_path, default_rows() + [_rec("partial")])
    _refused(d, tmp_path)


def test_missing_exit_record_is_refused_unless_allowed(tmp_path):
    d, _ = write_run(tmp_path, tail=False)
    _refused(d, tmp_path)
    rc, _oc, oj = _run(d, tmp_path, "--allow-no-exit-record")
    side = json.loads(oj.read_text())
    assert rc == 0 and side["parse"]["exit_record"]["present"] is False and side["allow_no_exit_record"] is True


@pytest.mark.parametrize("bad", [
    {"drop": ("trainer/global_step",)},          # no step key
    {"drop": ("train/lr",)},                     # no lr
    {"step_json": '"99"'},                       # step is a string
    {"step_json": "99.0"},                       # step is a float
    {"step_json": "true"},                       # step is a bool
    {"step_json": "-1"},                         # negative step
    {"loss_json": '"x"'},                        # loss not a number
    {"loss_json": "true"},                       # loss is a bool
    {"dup": True},                               # duplicate key in one row
])
def test_malformed_loss_rows_are_refused(tmp_path, bad):
    items = [("train/loss", bad.get("loss_json", "1.5")), ("train/lr", "5e-05"),
             ("trainer/global_step", bad.get("step_json", "99")), ("_step", "0")]
    items = [(k, v) for k, v in items if k not in bad.get("drop", ())]
    if bad.get("dup"):
        items.append(("train/loss", "1.6"))
    d, _ = write_run(tmp_path, [_rec("history", items=items, raw=True)])
    _refused(d, tmp_path)


def test_existing_outputs_are_never_overwritten(tmp_path):
    d, _ = write_run(tmp_path)
    for name in ("h.csv", "h.json"):
        p = tmp_path / name
        p.write_text("keep")
        rc, oc, oj = _run(d, tmp_path)
        assert rc == 2 and p.read_text() == "keep"
        other = oj if p == oc else oc
        assert not other.exists()
        p.unlink()


def test_outputs_inside_the_run_dir_are_refused(tmp_path):
    d, _ = write_run(tmp_path)
    rc = th.main(["extract", "--run-dir", str(d), "--out-csv", str(d / "h.csv"), "--out-json", str(tmp_path / "h.json")])
    assert rc == 2 and not (d / "h.csv").exists() and not (tmp_path / "h.json").exists()


@pytest.mark.parametrize("layout", ["online_name", "no_file", "two_files", "id_mismatch"])
def test_run_dir_layout_is_checked(tmp_path, layout):
    if layout == "online_name":
        d, _ = write_run(tmp_path, dirname=f"run-20261113_010000-{RUN_ID}")
    elif layout == "id_mismatch":
        d, _ = write_run(tmp_path, fname="run-zzzz9999.wandb")
    else:
        d, f = write_run(tmp_path)
        if layout == "no_file":
            f.unlink()
        else:
            (d / "run-other123.wandb").write_bytes(f.read_bytes())
    _refused(d, tmp_path)


# --------------------------------------------------------------------------- #
# read-only, no network, installed reader
# --------------------------------------------------------------------------- #
def test_run_directory_is_only_read(tmp_path):
    d, f = write_run(tmp_path)
    before = (f.read_bytes(), f.stat().st_mtime_ns, sorted(os.listdir(d)))
    os.chmod(f, 0o444)
    os.chmod(d, 0o555)
    try:
        rc, _oc, _oj = _run(d, tmp_path)
    finally:
        os.chmod(d, 0o755)
        os.chmod(f, 0o644)
    assert rc == 0
    assert (f.read_bytes(), f.stat().st_mtime_ns, sorted(os.listdir(d))) == before


def test_tool_uses_no_wandb_run_api(monkeypatch, tmp_path):
    import wandb
    tree = ast.parse(_TOOL.read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden = {"init", "sync", "login", "Api", "_attach", "open_for_scan", "open_for_write", "open_for_append"}
    assert not (attrs | names) & forbidden
    for name in ("init", "login"):
        monkeypatch.setattr(wandb, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError(name)))
    d, _ = write_run(tmp_path)
    assert _run(d, tmp_path)[0] == 0


def test_python_O_is_refused_because_the_reader_checks_by_assert(tmp_path):
    d, _ = write_run(tmp_path)
    out = subprocess.run([sys.executable, "-O", str(_TOOL), "extract", "--run-dir", str(d), "--out-csv",
                          str(tmp_path / "o.csv"), "--out-json", str(tmp_path / "o.json")],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 2 and "assert" in out.stderr and not (tmp_path / "o.csv").exists()


def test_cli_end_to_end_subprocess(tmp_path):
    d, _ = write_run(tmp_path)
    out = subprocess.run([sys.executable, "-I", str(_TOOL), "extract", "--run-dir", str(d), "--out-csv",
                          str(tmp_path / "o.csv"), "--out-json", str(tmp_path / "o.json")],
                         capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr
    assert json.loads((tmp_path / "o.json").read_text())["row_count"] == 10


# --------------------------------------------------------------------------- #
# opt-in: exp_40's FULL run (skipped when absent)
# --------------------------------------------------------------------------- #
# Fixed, NON-recursive patterns: the exp_40 run trees hold an AcousticRooms symlink that a
# recursive glob would walk. WANDB_DIR of the exp_40 launcher was <save root>/runtime/<run id>/wandb.
_EXP40_GLOBS = [
    "/home/yixunhu/codespace/exp-40-yaw-aug-c16/wandb/offline-run-*",
    "/home/yixunhu/codespace/exp-40-yaw-aug-c16/outputs_FLAC/exp40_yaw_aug_c16/wandb/offline-run-*",
    "/home/yixunhu/codespace/exp-40-yaw-aug-c16/outputs_FLAC/exp40_yaw_aug_c16/runtime/*/wandb/offline-run-*",
    "/home/yixunhu/codespace/exp-40-yaw-aug-c16/outputs_FLAC/exp40_yaw_aug_c16/runtime/*/wandb/wandb/offline-run-*",
    "/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/wandb/offline-run-*",
    "/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/*/wandb/offline-run-*",
    "/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/*/offline-run-*",
]


def _real_run_dirs():
    env = os.environ.get("EXP43_REAL_WANDB_RUN_DIR")
    if env:
        return [env] if os.path.isdir(env) else []
    found = []
    for g in _EXP40_GLOBS:
        found += [p for p in glob.glob(g) if os.path.isdir(p) and glob.glob(os.path.join(p, "run-*.wandb"))]
    return sorted(set(found))


def test_real_40k_run_opt_in(tmp_path):
    dirs = _real_run_dirs()
    if not dirs:
        pytest.skip("exp_40 FULL offline W&B run absent (searched the exp_40 worktree wandb/ + outputs_FLAC/ and "
                    "the NAS exp_40 root); set EXP43_REAL_WANDB_RUN_DIR to a 40k offline-run directory")
    for i, d in enumerate(dirs):
        rc, oc, oj = _run(d, tmp_path, "--expected-max-steps", "40000",
                          csv_name=f"r{i}.csv", json_name=f"r{i}.json")
        side = json.loads(oj.read_text())
        if side["last_step"] < 39999:
            continue                                   # a smoke / partial run in the same tree
        assert rc == 0, side["cadence"]
        assert side["row_count"] == 400 and side["first_step"] == 99 and side["last_step"] == 39999
        assert {"train/loss", "train/lr", "trainer/global_step"} <= set(side["keys_seen"])
        return
    pytest.skip(f"no 40k run among {dirs}")
