#!/usr/bin/env python3
"""exp_43 plan §5.4 / F11 — training-loss history of one OFFLINE W&B run, read from its ``.wandb`` file.

    extract --run-dir <.../wandb/offline-run-YYYYMMDD_HHMMSS-<id>> --out-csv H.csv --out-json H.json
            [--expected-max-steps N] [--allow-no-exit-record]

Exit codes: 0 = written, cadence check passed; 3 = written, cadence check FAILED (gaps / duplicates /
off-cadence / out-of-order rows are REPORTED in the sidecar and the CSV keeps every row exactly as
logged — nothing is fixed); 2 = refused, nothing written.

Reader. The run file ``run-<id>.wandb`` is read with the INSTALLED wandb's own leveldb-log reader
(``wandb.sdk.internal.datastore.DataStore.scan_data`` / ``_read_header``, wandb 0.15.4 pinned in
the ``flac`` env — any other version is refused because private attributes are used) and decoded
with its protobuf ``wandb.proto.wandb_internal_pb2.Record``. The file is opened ``rb`` by this tool
(``DataStore.open_for_scan`` would open it ``r+b``) and fed through a hashing reader, so the sha256
recorded is the sha256 of exactly the bytes parsed; size / mtime / inode must not change while it
is read. Nothing calls ``wandb.init``, ``wandb sync``, the public API or the network; nothing is
written into the run directory. The reader checks checksums and padding with ``assert``, so the
tool refuses to run under ``python -O``.

Complete parse, or refusal: bad file header; any record header / checksum / padding failure; a
multi-block record cut short (EOF after FIRST/MIDDLE — wandb's own reader returns a silent "EOF"
there); bytes left over; a payload protobuf cannot decode; a record with no known type; an
unflushed ``request.partial_history``; a run with no ``exit`` record (unless
``--allow-no-exit-record``). The legitimate end-of-file within 6 bytes of a 32 KiB block end (where
wandb's naive loop would assert on missing padding) is accepted.

Columns. One CSV row per history record that has ``train/loss``, in file order:
  step        <- ``trainer/global_step`` (JSON integer >= 0, verbatim). PL 2.1's ``WandbLogger.log_metrics``
                 adds this key with ``trainer.fit_loop.epoch_loop._batches_that_stepped``, read in
                 ``advance()`` after the optimiser step and before the counter's increment: with
                 ``--accum-batches 1`` the row logged for the n-th optimisation step carries n - 1, and
                 ``log_every_n_steps = 100`` logs when (step + 1) % 100 == 0 -> 99, 199, ..., 39999.
  train/loss  <- ``train/loss`` (logging rank 0's single-batch loss of that step, random timesteps; F11)
  lr          <- ``train/lr`` (``optimizers[0].param_groups[0]['lr']`` read inside ``training_step``,
                 i.e. the lr that step's update used)
A loss row without either key, a non-integer / bool / negative step, a non-numeric loss or lr, or a
key repeated inside one row is refused. Floats are written with ``repr`` (shortest round-trip;
``nan`` / ``inf`` kept and listed under ``nonfinite``).

Cadence (period 100, phase 99): expected steps 99, 199, ... up to ``--expected-max-steps`` - 1 (else
up to the largest on-cadence step seen). Reported: ``gaps`` (expected steps absent), ``duplicates``,
``off_cadence`` (step + 1 not a multiple of 100), ``non_monotonic`` (a step not larger than the
previous row's), ``first_step_is_99``; ``ok`` iff all are clean.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

SCHEMA = "exp43-train-history/1"
BLOCK = 32768
PERIOD = 100
PINNED_WANDB = "0.15.4"
_DIR = re.compile(r"^offline-run-\d{8}_\d{6}-([0-9a-z]+)$")
_FIRST, _MIDDLE = 2, 3
COLUMNS = {"step": "trainer/global_step", "train/loss": "train/loss", "lr": "train/lr"}
STEP_SEMANTICS = (
    "step = the logged 'trainer/global_step' value, verbatim. PL 2.1 WandbLogger.log_metrics adds it with "
    "trainer.fit_loop.epoch_loop._batches_that_stepped, read in advance() after the optimiser step and before "
    "the counter is incremented: with --accum-batches 1 the row of the n-th optimisation step carries n-1 "
    "(the loss of the 100th step is at step 99; the checkpoint 'step=2500' follows the row at 2499).")
LOSS_SEMANTICS = ("train/loss = the logging rank's (rank 0) single-batch loss of that optimisation step at "
                  "random timesteps (F11); lr = train/lr, the lr that step's update used.")


class Refusal(Exception):
    """A precondition failed; nothing is written."""


class _HashingReader:
    """Read-only, sequential: every byte handed to the parser is hashed and counted."""

    def __init__(self, fp):
        self._fp, self.h, self.n = fp, hashlib.sha256(), 0

    def read(self, k=-1):
        b = self._fp.read(k)
        self.h.update(b)
        self.n += len(b)
        return b

    def tell(self):
        return self.n


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_run_file(run_dir):
    d = Path(run_dir)
    if not d.is_dir():
        raise Refusal(f"run directory {d} does not exist")
    m = _DIR.match(d.name)
    if not m:
        raise Refusal(f"{d.name!r} is not an offline run directory (offline-run-YYYYMMDD_HHMMSS-<id>)")
    files = sorted(p for p in d.iterdir() if p.name.endswith(".wandb"))
    if len(files) != 1:
        raise Refusal(f"expected exactly one .wandb file in {d}, found {[p.name for p in files]}")
    if files[0].name != f"run-{m.group(1)}.wandb" or not files[0].is_file():
        raise Refusal(f"run file {files[0].name} does not match the directory's run id {m.group(1)}")
    return d.resolve(), files[0].resolve(), m.group(1)


def read_run(path):
    """Parse every record of one .wandb file; return counts, the exit record and the history rows."""
    if not __debug__:
        raise Refusal("python -O strips the assert statements wandb's datastore reader uses for its "
                      "header / checksum / padding checks; run without -O")
    import wandb
    if wandb.__version__ != PINNED_WANDB:
        raise Refusal(f"wandb {wandb.__version__} != the pinned {PINNED_WANDB} (private reader API)")
    from google.protobuf.message import DecodeError
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    with open(path, "rb") as fp:
        st0 = os.fstat(fp.fileno())
        size = st0.st_size
        reader = _HashingReader(fp)
        ds = datastore.DataStore()
        ds._fp, ds._fname, ds._index, ds._size_bytes, ds._opened_for_scan = reader, str(path), 0, size, True
        seen = []
        scan_record = ds.scan_record

        def spy():
            r = scan_record()
            seen.append(None if r is None else r[0])
            return r
        ds.scan_record = spy
        by_type, rows, exit_rec, multi, n = {}, [], None, 0, 0
        try:
            ds._read_header()
            while ds._index < size:
                seen.clear()
                data = ds.scan_data()
                if data is None:
                    if any(t is not None for t in seen):
                        raise Refusal(f"truncated multi-block record at byte {ds._index} (types {seen})")
                    break
                n += 1
                multi += sum(t is not None for t in seen) > 1
                rec = pb.Record()
                try:
                    rec.ParseFromString(data)
                except DecodeError as e:
                    raise Refusal(f"record {n} does not decode as wandb Record: {e}")
                kind = rec.WhichOneof("record_type")
                if kind is None:
                    raise Refusal(f"record {n} has no known record type")
                by_type[kind] = by_type.get(kind, 0) + 1
                if kind == "history":
                    rows.append(_history_items(rec.history, n))
                elif kind == "exit":
                    exit_rec = {"present": True, "exit_code": rec.exit.exit_code, "runtime": rec.exit.runtime}
                elif kind == "request" and rec.request.WhichOneof("request_type") == "partial_history":
                    raise Refusal(f"record {n} is an unflushed partial_history request")
        except Refusal:
            raise
        except Exception as e:      # the installed reader signals corruption by assert / Exception
            raise Refusal(f"run file does not parse at byte {ds._index}: {type(e).__name__}: {e}")
        if ds._index != size or reader.n != size or fp.read(1) != b"":
            raise Refusal(f"parse stopped at byte {ds._index} of {size}")
        st1 = os.fstat(fp.fileno())
    if (st1.st_size, st1.st_mtime_ns, st1.st_ino) != (st0.st_size, st0.st_mtime_ns, st0.st_ino):
        raise Refusal("the run file changed while it was read")
    if n == 0:
        raise Refusal("the run file holds no record")
    return {"sha256": reader.h.hexdigest(), "bytes": size, "records": n, "records_by_type": by_type,
            "multi_block_records": multi, "exit_record": exit_rec or {"present": False}, "rows": rows}


def _history_items(hist, n):
    items = {}
    for it in hist.item:
        key = it.key if it.key else "nested:" + ".".join(it.nested_key)
        if key in items:
            raise Refusal(f"history record {n} repeats key {key!r}")
        try:
            items[key] = json.loads(it.value_json)
        except ValueError as e:
            raise Refusal(f"history record {n} key {key!r}: value_json does not parse: {e}")
    return items


def _number(v, what, step):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise Refusal(f"{what} at step {step} is not a number: {v!r}")
    return float(v)


def loss_rows(rows):
    out = []
    for i, r in enumerate(rows):
        if "train/loss" not in r:
            continue
        for k in ("trainer/global_step", "train/lr"):
            if k not in r:
                raise Refusal(f"history row {i} has train/loss but no {k}")
        s = r["trainer/global_step"]
        if isinstance(s, bool) or not isinstance(s, int) or s < 0:
            raise Refusal(f"history row {i}: trainer/global_step {s!r} is not a non-negative integer")
        out.append((s, _number(r["train/loss"], "train/loss", s), _number(r["train/lr"], "train/lr", s)))
    return out


def cadence(steps, expected_max_steps=None):
    counts = {}
    for s in steps:
        counts[s] = counts.get(s, 0) + 1
    off = [s for s in steps if (s + 1) % PERIOD]
    on = [s for s in steps if not (s + 1) % PERIOD]
    if expected_max_steps is not None:
        last = (expected_max_steps // PERIOD) * PERIOD - 1
    else:
        last = max(on) if on else -1
    expected = list(range(PERIOD - 1, last + 1, PERIOD))
    present = set(steps)
    rep = {
        "period": PERIOD, "phase": PERIOD - 1, "expected_first_step": PERIOD - 1,
        "expected_last_step": last if expected else None, "expected_rows": len(expected),
        "gaps": [s for s in expected if s not in present],
        "duplicates": {str(s): c for s, c in sorted(counts.items()) if c > 1},
        "off_cadence": off,
        "non_monotonic": [{"index": i, "previous": steps[i - 1], "step": steps[i]}
                          for i in range(1, len(steps)) if steps[i] <= steps[i - 1]],
        "first_step_is_99": bool(steps) and steps[0] == PERIOD - 1,
    }
    if expected_max_steps is not None:
        rep["last_step_is_expected"] = bool(steps) and steps[-1] == last
    rep["ok"] = (bool(steps) and not rep["gaps"] and not rep["duplicates"] and not off
                 and not rep["non_monotonic"] and rep["first_step_is_99"]
                 and rep.get("last_step_is_expected", True))
    return rep


def extract(a):
    run_dir, run_file, run_id = find_run_file(a.run_dir)
    outs = [os.path.abspath(a.out_csv), os.path.abspath(a.out_json)]
    if outs[0] == outs[1]:
        raise Refusal("--out-csv and --out-json are the same path")
    for o in outs:
        if os.path.lexists(o):
            raise Refusal(f"refusing to overwrite {o}")
        if os.path.commonpath([os.path.realpath(os.path.dirname(o)), str(run_dir)]) == str(run_dir):
            raise Refusal(f"refusing to write into the run directory: {o}")
    if a.expected_max_steps is not None and a.expected_max_steps < PERIOD:
        raise Refusal("--expected-max-steps must be >= 100")
    run = read_run(run_file)
    if not run["exit_record"]["present"] and not a.allow_no_exit_record:
        raise Refusal("the run has no exit record (unfinished or crashed); pass --allow-no-exit-record to accept")
    rows = loss_rows(run["rows"])
    steps = [s for s, _l, _r in rows]
    cad = cadence(steps, a.expected_max_steps)
    keys = sorted({k for r in run["rows"] for k in r})
    with open(outs[0], "x", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["step", "train/loss", "lr"])
        for s, loss, lr in rows:
            w.writerow([s, repr(loss), repr(lr)])
    side = {
        "schema": SCHEMA,
        "tool": {"path": os.path.abspath(__file__), "sha256": _sha256(__file__)},
        "run_dir": str(run_dir), "run_file": str(run_file), "run_id": run_id,
        "run_file_sha256": run["sha256"], "run_file_bytes": run["bytes"],
        "wandb_version": PINNED_WANDB,
        "reader": "wandb.sdk.internal.datastore.DataStore._read_header/scan_data over the file opened 'rb'; "
                  "records decoded with wandb.proto.wandb_internal_pb2.Record",
        "parse": {"complete": True, "records": run["records"], "records_by_type": run["records_by_type"],
                  "multi_block_records": run["multi_block_records"], "exit_record": run["exit_record"]},
        "allow_no_exit_record": bool(a.allow_no_exit_record),
        "history_rows_total": len(run["rows"]), "row_count": len(rows),
        "history_rows_without_loss": len(run["rows"]) - len(rows),
        "first_step": steps[0] if steps else None, "last_step": steps[-1] if steps else None,
        "keys_seen": keys, "keys_seen_count": len(keys),
        "columns": COLUMNS, "step_semantics": STEP_SEMANTICS, "loss_semantics": LOSS_SEMANTICS,
        "expected_max_steps": a.expected_max_steps, "cadence": cad,
        "nonfinite": {"train/loss": [s for s, l_, _r in rows if not math.isfinite(l_)],
                      "lr": [s for s, _l, r_ in rows if not math.isfinite(r_)]},
        "csv": {"path": outs[0], "sha256": _sha256(outs[0]), "rows": len(rows)},
    }
    try:
        with open(outs[1], "x") as fh:
            json.dump(side, fh, indent=1, sort_keys=True)
            fh.write("\n")
    except OSError as e:
        os.unlink(outs[0])
        raise Refusal(f"could not write {outs[1]}: {e}")
    print(f"{len(rows)} rows, steps {side['first_step']}..{side['last_step']}, cadence ok={cad['ok']} -> {outs[0]}")
    return 0 if cad["ok"] else 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--out-csv", required=True)
    e.add_argument("--out-json", required=True)
    e.add_argument("--expected-max-steps", type=int, default=None)
    e.add_argument("--allow-no-exit-record", action="store_true")
    a = p.parse_args(argv)
    try:
        return extract(a)
    except Refusal as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
