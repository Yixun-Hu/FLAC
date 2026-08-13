#!/usr/bin/env python3
"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.

    python3 fa_orbit_record_restart.py C4L outputs_FLAC/exp11_C4L/<manifest>.txt
    python3 fa_orbit_record_restart.py C4L <manifest> --extend   # later, as the leg saves more

A restart is only admissible if it provably continues the audited INITIAL run, so
this refuses unless the resume checkpoint ON DISK -- always re-hashed, never
trusted from the manifest -- equals that arm's recorded final_ckpt_sha256.

Re-pin review, required fix 3. The previous version was fail-OPEN: it re-hashed
only `if os.path.isfile(resume_path)`, so a manifest naming a file that could not
be resolved was recorded on the strength of its own claimed hash, and nothing
else in the manifest was checked at all. Now:

  * the canonical resume file MUST exist, sit in the audited launch's own
    checkpoint directory, and is ALWAYS re-hashed;
  * every identity field is validated against the INITIAL registry row (arm, job,
    uuid, commit, rung, config sha, VAE and P0 manifest shas, save-dir, seed) and
    against the Q10 pins read out of the launcher itself (budget 100000, resume
    step = the audited final step, and the arm's RESTART wall pin), so recorder
    and launcher cannot disagree;
  * publication is atomic (tmp + rename) under the store lock;
  * duplicates are refused -- one leg, one row.

It also publishes the leg's PRODUCER MANIFEST (fix 2): every checkpoint this leg
produced, re-hashed from disk, into an append-only per-leg file the screen
verifies each >40k checkpoint against. Re-run with --extend as the leg saves more.

ROUND 5 — CHUNKED legs. The cluster never backfills a 34-160 h allocation, so a
leg now trains to the next 2500-step boundary and exits (`chunk_end` in its
manifest). Such a leg is recorded as a CHAIN LINK in `arms.<ARM>.chain`:

    INITIAL anchor (40000) <- link(40000 -> 42500) <- link(42500 -> 45000) <- ...

Each link is admissible only if it resumed the TIP of the chain (or the audited
INITIAL anchor, for the first link) with the file that is on disk NOW, re-hashed.
The endpoint checkpoint the leg produced is likewise located in the audited
canonical directory and hashed by this recorder, which is what makes it the next
link's anchor: fa_orbit_ckpt_preflight.py --chain refuses a chunk whose resume
file is not the last link's `final_ckpt_sha256`, so chunk N+1 cannot start until
chunk N is recorded here. Appending a link never touches the INITIAL fields, is
atomic (tmp+rename under the store lock), and is idempotent per job.

ROUND-5 REVIEW B6 — ATTRIBUTION. Location + re-hashing prove which bytes are on
disk, not WHICH JOB WROTE THEM: a failed leg's (pre-published) manifest plus a
pre-existing checkpoint of the right name was enough to mint a link. So a chunk
is now recorded only on the producing job's own post-classification attestation
(`endpoint_ckpt … endpoint_step … endpoint_sha256 …`, appended by
fa_orbit_train.sbatch to its own manifest), checked against this recorder's
independent re-hash. And a chunk that HAS a predecessor is never re-parented
onto the INITIAL anchor when that predecessor is incomplete — it is refused.

ROUND-5 r2 REVIEW. Two further requirements on a chunk link:

  * THE SCHEDULER MUST AGREE (blocking 2). The attestation is the job's own word.
    `sacct -X -n -P -j <job> -o State` must report exactly COMPLETED; an empty,
    failing or unavailable sacct refuses. `--skip-sacct` (off by default, never
    used by the watchdog) exists for documented manual recovery.
  * THE ATTESTED PATH IS THE RECORD (blocking 3). Lightning's ModelCheckpoint
    version counter means a retry at an already-written boundary saves
    `epoch=E-step=N-v1.ckpt`. Globbing `*-step=N.ckpt` would bind the failed
    attempt's stale bytes to the retry (or find two files and refuse forever), so
    the recorder follows the attested path — verified to exist, to sit in the
    canonical directory, and to carry this leg's step in its name — and stores it
    in the link as `final_ckpt_path` for the next chunk to resume from.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa_orbit_producer_manifest as pm            # noqa: E402
from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402

PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')
# The pinned checkpoint cadence: a chunk may only end ON a saved checkpoint,
# otherwise the next chunk has nothing to resume.
CHUNK_STEP = 2500
# Lightning's ModelCheckpoint version counter: a retry at a boundary whose
# unversioned name already exists writes `epoch=E-step=N-v1.ckpt` (then -v2...).
# An attested endpoint may therefore carry EITHER name shape (round-5 r2 B3).
ENDPOINT_NAME_RE = re.compile(r"-step=(\d+)(?:-v\d+)?\.ckpt$")


def sacct_state(job, sacct_bin="sacct"):
    """(state, problem): what the SCHEDULER says about ``job``.

    Round-5 r2 review, blocking 2. The producing job's own attestation says what
    it wrote; it cannot say whether Slurm agrees the job finished — a leg killed
    after it appended its attestation, or one whose node died during epilogue,
    still leaves a positive attestation on disk. So a chunk link additionally
    requires ``sacct`` to report exactly COMPLETED for the manifest's job.

    Fail-CLOSED in every ambiguous direction: a missing/unrunnable ``sacct``, a
    nonzero exit, and an EMPTY answer (the job is unknown to the accounting
    database, or accounting is lagging) are all refusals, never "probably fine".
    """
    argv = [sacct_bin, "-X", "-n", "-P", "-j", str(job), "-o", "State"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, (f"could not ask the scheduler about job {job} "
                      f"({' '.join(argv)}): {type(exc).__name__}: {exc} — a chunk link "
                      "requires scheduler confirmation that the job COMPLETED")
    if proc.returncode != 0:
        return None, (f"`{' '.join(argv)}` exited {proc.returncode} "
                      f"({proc.stderr.strip()[:200]}) — the scheduler's verdict on job {job} is "
                      "UNKNOWN, and an unknown verdict is never read as a successful one")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None, (f"sacct reports nothing for job {job} — the scheduler cannot confirm it "
                      "COMPLETED (accounting may be lagging, or this job never ran); a chunk "
                      "link is never recorded on an unconfirmed job")
    return lines[0], None


def read_pins(launcher):
    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
    pins = {}
    with open(launcher) as fh:
        for line in fh:
            m = PIN_RE.match(line)
            if m:
                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return pins


def parse_manifest(path):
    raw = open(path, "rb").read()
    man = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, _, rest = line.partition(" ")
            man[k] = rest.strip()
    return raw, man


def kvs(man, key):
    f = (f"{key} " + man.get(key, "")).split()
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def check_identity(arm, man, initial, pins, repo_root,
                   want_anchor=None, want_step=None, want_time=None):
    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins.

    A CHUNK leg proves the same identity but resumes the tip of the recorded
    chain rather than the audited 40k anchor, and is walled by the arm's CHUNK
    pin rather than its RESTART pin, so the caller may override those three
    expectations. Everything else is identical for both leg kinds."""
    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
    tk = kvs(man, "time_limit")
    problems = []
    anchor = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
    final_step = want_step if want_step is not None else initial.get("final_step")
    if not anchor:
        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
                        "INITIAL run's final checkpoint before recording a leg")
    if jk.get("mode") != "RESTART":
        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
                       ("commit", man.get("commit"))):
        if not got:
            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
    if jk.get("job") and initial.get("job") == jk.get("job"):
        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
                        "already registered, not a restart leg")
    for label, got, want in (("arm", ak.get("arm"), arm),
                             ("rung", ak.get("rung"), initial.get("rung")),
                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
                              initial.get("p0_manifest_sha256")),
                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
        if got != want:
            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
    if ak.get("rung") != pins.get("PINNED_RUNG"):
        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
                        f"{pins.get('PINNED_MAXSTEPS')!r}")
    step_label = "audited final step" if want_step is None else "recorded chain tip's final step"
    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the {step_label} "
                        f"{final_step!r} — a leg resumes where the run it continues ended")
    pin_label = "RESTART wall pin"
    if want_time is None:
        want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
    else:
        pin_label = "CHUNK wall pin"
    if tk.get("time_limit") != want_time:
        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's {pin_label} "
                        f"{want_time!r}")
    if int(initial.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
    # the config the leg names must still hash to the audited value
    cfg_path = man.get("model_config", "")
    if not cfg_path or not os.path.isfile(cfg_path):
        problems.append(f"manifest model_config {cfg_path!r} does not exist")
    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
    resume_real = ""
    if not resume_path or resume_path == "<none>":
        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
                        "a continuation of the audited run")
    elif not os.path.isfile(resolve(repo_root, resume_path)):
        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
                        "NOT accept the manifest's claimed hash in its place")
    else:
        resume_real = resolve(repo_root, resume_path)
        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
        if os.path.realpath(os.path.dirname(resume_real)) != canon:
            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
                            f"directory {canon}")
        got = pm.sha256_file(resume_real)
        if anchor and got != anchor:
            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
                            f"{anchor[:12]} — this leg does not continue that run")
        if rk.get("resume_ckpt_sha256") != got:
            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
                            f"!= the file's actual {got[:12]}")
    return problems, resume_real


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def chunk_end_of(man):
    """The leg's per-job stop step as written by the launcher, or None.

    A manifest with no `chunk_end` line (or the literal `<none>`) is a
    whole-budget RESTART leg and takes the original recording path untouched."""
    v = (man.get("chunk_end", "").split() or [""])[0]
    return None if not v or v == "<none>" else v


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_attested_endpoint(man, chunk_end, ckpt_dir, repo_root):
    """(endpoint_path, endpoint_sha, problems) for the ATTESTED endpoint file.

    Round-5 r2 review, blocking 3(b). This used to glob
    ``<canonical>/*-step=<chunk_end>.ckpt`` and require exactly one hit. That is
    wrong twice over once Lightning's version counter is in play: a failed
    attempt that saved, followed by a successful retry at the same boundary,
    leaves BOTH ``epoch=E-step=N.ckpt`` (the stale attempt) and
    ``epoch=E-step=N-v1.ckpt`` (the retry) — the glob then either binds the
    predecessor's bytes to the retry, or finds two hits and refuses forever.

    The producing job already told us which file it wrote. So the recorder now
    follows the ATTESTED PATH and verifies it, rather than re-deriving a path
    from a name pattern:

      * the attested file must EXIST;
      * it must sit in the audited launch's canonical checkpoint directory;
      * its NAME must encode this leg's chunk_end (either name shape);
      * it is re-hashed here, from disk, by this recorder.
    """
    attested = kvs(man, "endpoint_ckpt").get("endpoint_ckpt", "")
    if not attested or attested == "<none>":
        return None, None, []           # check_endpoint_attestation reports this
    cand = resolve(repo_root, attested)
    if not os.path.isfile(cand):
        return None, None, [f"the attested endpoint checkpoint {attested} does not exist — the "
                            "recorder does NOT accept the manifest's claimed hash in its place"]
    if os.path.realpath(os.path.dirname(cand)) != ckpt_dir:
        return None, None, [f"the attested endpoint checkpoint {attested} is not in the audited "
                            f"launch's canonical directory {ckpt_dir}"]
    m = ENDPOINT_NAME_RE.search(os.path.basename(cand))
    if not m or (chunk_end is not None and int(m.group(1)) != int(chunk_end)):
        return None, None, [f"the attested endpoint checkpoint {os.path.basename(cand)} does not "
                            f"carry this leg's chunk end step {chunk_end} in its name (expected "
                            f"`*-step={chunk_end}.ckpt` or `*-step={chunk_end}-v<N>.ckpt`)"]
    return cand, pm.sha256_file(cand), []


def check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha):
    """The producing job's own statement of what it wrote (round-5 review B6).

    The launcher publishes its manifest BEFORE training, so the manifest alone
    proves intent, never authorship: a FAILED leg's manifest plus a pre-existing
    checkpoint of the right name would otherwise become a chain link, and
    re-hashing proves only which bytes are on disk now. After its exit class is
    SETTLED the job appends to its own manifest

        endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>

    (or `endpoint_ckpt <none> ...` for every non-success class), and this
    function makes that attestation MANDATORY and fail-closed:

      * no attestation at all           -> refuse (name the launcher append);
      * `<none>`                        -> refuse (the leg says it produced none);
      * endpoint_step != chunk_end      -> refuse (it attests another boundary);
      * attested sha != our own re-hash -> refuse (it is not that file).

    The attested PATH is resolved and located by resolve_attested_endpoint; the
    ``endpoint``/``endpoint_sha`` passed here are that function's findings.
    """
    ak = kvs(man, "endpoint_ckpt")
    if "endpoint_ckpt" not in man:
        return ["the manifest carries no endpoint attestation — a chunk is recorded only on the "
                "PRODUCING JOB's own statement of what it wrote. fa_orbit_train.sbatch appends "
                "`endpoint_ckpt <path> endpoint_step <n> endpoint_sha256 <sha>` to this manifest "
                "once its exit class is settled; a manifest without that line was either written "
                "by an older launcher or belongs to a leg that never finished"]
    attested = ak.get("endpoint_ckpt", "")
    if not attested or attested == "<none>":
        return [f"the producing job attested `endpoint_ckpt <none>` (class "
                f"{ak.get('endpoint_class', '?')}) — it did not reach the success class, or wrote no "
                "checkpoint at its chunk end; a leg that attests nothing is never a chain link"]
    problems = []
    if str(ak.get("endpoint_step")) != str(chunk_end):
        problems.append(f"the attested endpoint_step {ak.get('endpoint_step')!r} != this leg's "
                        f"chunk_end {chunk_end!r} — the job attests a different boundary")
    if endpoint is None:
        return problems      # the unresolvable endpoint is already a problem upstream
    if ak.get("endpoint_sha256") != endpoint_sha:
        problems.append(f"the attested endpoint_sha256 "
                        f"{str(ak.get('endpoint_sha256'))[:12]} != this recorder's own re-hash of "
                        f"{os.path.basename(endpoint)} ({str(endpoint_sha)[:12]}) — the file on disk "
                        "is not the file that job attested writing")
    return problems


def record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw):
    """Append ONE chain link for a chunked leg: <resume_step> -> <chunk_end>.

    The link is what makes the NEXT chunk admissible (preflight --chain), so it
    is written only when this leg provably resumed the tip of the chain and its
    endpoint checkpoint is on disk, unambiguous, and hashed here from that disk.
    """
    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
    job = jk.get("job")
    problems = []
    if args.extend:
        problems.append("--extend extends a producer manifest and has no meaning for a CHUNK leg; "
                        "each chunk is recorded once, as one immutable chain link")
    if not job:
        problems.append("manifest records no job — a leg with no identity is not a record")

    # --- the SCHEDULER's verdict (round-5 r2 review, blocking 2) -------------
    # The attestation is the job's own word; this is Slurm's. Both are required:
    # the attestation says WHICH bytes, sacct says the job actually COMPLETED.
    # --skip-sacct exists for documented MANUAL recovery only (accounting purged,
    # a hand-audited leg) and is off by default — it is never used by the watchdog.
    if job and not args.skip_sacct:
        state, sacct_problem = sacct_state(job, args.sacct_bin)
        if sacct_problem:
            problems.append(sacct_problem)
        elif state != "COMPLETED":
            problems.append(f"the scheduler reports job {job} as {state!r}, not COMPLETED — a chunk "
                            "link records a leg that the SCHEDULER agrees finished successfully; "
                            "if this is a documented manual recovery, re-run with --skip-sacct and "
                            "say so in the worklog")

    # --- the chunk boundary itself ------------------------------------------
    chunk_end = resume_step = None
    try:
        chunk_end = int(chunk_raw)
    except (TypeError, ValueError):
        problems.append(f"manifest chunk_end {chunk_raw!r} is not an integer")
    try:
        resume_step = int(rk.get("expected_step"))
    except (TypeError, ValueError):
        problems.append(f"manifest expected_step {rk.get('expected_step')!r} is not an integer")
    budget = None
    try:
        budget = int(pins.get("PINNED_MAXSTEPS"))
    except (TypeError, ValueError):
        problems.append(f"the launcher's PINNED_MAXSTEPS {pins.get('PINNED_MAXSTEPS')!r} is not an "
                        "integer — the chunk boundary cannot be checked against the budget")
    if chunk_end is not None:
        if chunk_end % CHUNK_STEP:
            problems.append(f"manifest chunk_end {chunk_end} is not a multiple of {CHUNK_STEP} (the "
                            "pinned checkpoint cadence: a chunk must end ON a checkpoint)")
        if resume_step is not None and chunk_end <= resume_step:
            problems.append(f"manifest chunk_end {chunk_end} does not exceed the resume step "
                            f"{resume_step} — the leg would produce no new checkpoint")
        if budget is not None and chunk_end > budget:
            problems.append(f"manifest chunk_end {chunk_end} exceeds the pinned budget {budget}")

    # --- which link (or the audited anchor) this chunk continues ------------
    chain = initial.get("chain") or []
    mine = [i for i, link in enumerate(chain) if link.get("job") == job]
    if len(mine) > 1:
        raise SystemExit(f"{arm} has {len(mine)} chain links claiming job {job} — the registry is "
                         "inconsistent; fix it before recording")
    at = mine[0] if mine else len(chain)
    # Round-5 review NON-BLOCKING: one leg, one link — by job AND by identity. A
    # different job id carrying an already-recorded launch uuid or manifest is an
    # inconsistent registry, not a second chunk.
    if not mine:
        for i, link in enumerate(chain):
            if jk.get("launch_uuid") and link.get("launch_uuid") == jk.get("launch_uuid"):
                problems.append(f"chain link {i} (job {link.get('job')!r}) already carries launch_uuid "
                                f"{jk.get('launch_uuid')!r} — a different job with the same launch uuid "
                                "means the registry is inconsistent; fix it before recording")
            if link.get("manifest_sha256") == man_sha:
                problems.append(f"chain link {i} (job {link.get('job')!r}) was recorded from a manifest "
                                f"with this exact sha256 {man_sha[:12]} — the same manifest cannot be "
                                "two chunks")
    prev = chain[at - 1] if at > 0 else None
    if prev is None:
        want_anchor = want_step = None                 # the audited INITIAL anchor
        prev_desc = f"the audited INITIAL anchor at step {initial.get('final_step')}"
    else:
        want_anchor, want_step = prev.get("final_ckpt_sha256"), prev.get("final_step")
        prev_desc = f"chain link job {prev.get('job')} ending at step {prev.get('final_step')}"
        # Round-5 review B6: a predecessor that records no endpoint is NOT
        # evidence, and falling back to the INITIAL 40k anchor here would silently
        # re-parent this chunk onto the anchor — the chain's whole point is that
        # every link continues the one before it. Refuse; the fallback exists
        # only for the FIRST link (at == 0).
        if not want_anchor or want_step is None:
            problems.append(f"the predecessor chain link (index {at - 1}, job {prev.get('job')!r}) "
                            "carries no final_ckpt_sha256/final_step — it is not evidence of a "
                            "checkpoint, and a chunk with a predecessor is NEVER re-parented onto the "
                            "audited INITIAL anchor; repair or re-record that link first")
            print("RECORD REFUSED:")
            for p in problems:
                print(f"  !! {p}")
            return 2
    want_time = pins.get(f"PINNED_TIME_LIMIT_CHUNK_{arm}")
    if not want_time:
        problems.append(f"the launcher carries no PINNED_TIME_LIMIT_CHUNK_{arm} pin, so this "
                        "chunk leg's wall time cannot be bound to a pin")
    ident, resume_real = check_identity(arm, man, initial, pins, args.repo_root,
                                        want_anchor=want_anchor, want_step=want_step,
                                        want_time=want_time)
    problems += ident

    # --- the endpoint checkpoint this leg produced --------------------------
    # Located by the job's OWN attestation, not by a name glob (blocking 3(b)):
    # with Lightning's version counter a same-boundary retry writes `-v1`, so a
    # glob would either bind the failed attempt's stale bytes or refuse forever.
    ckpt_dir = canonical_ckpt_dir(initial.get("save_dir", ""), arm, args.repo_root)
    endpoint, endpoint_sha, endpoint_problems = resolve_attested_endpoint(
        man, chunk_end, ckpt_dir, args.repo_root)
    problems += endpoint_problems
    problems += check_endpoint_attestation(man, chunk_end, endpoint, endpoint_sha)
    if problems:
        print("RECORD REFUSED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    # check_identity already re-hashed the resume file and proved it IS this hash
    resume_sha = want_anchor if want_anchor is not None else initial.get("final_ckpt_sha256")
    # final_ckpt_path (blocking 3(b)): the ACTUAL endpoint file, versioned name
    # and all, so the next chunk resumes the recorded tip by path instead of
    # re-deriving one from a glob that cannot tell `-v1` from its stale twin.
    link = {"job": job, "launch_uuid": jk.get("launch_uuid"),
            "manifest_path": os.path.abspath(args.manifest), "manifest_sha256": man_sha,
            "resume_step": resume_step, "resume_ckpt_sha256": resume_sha,
            "final_step": chunk_end, "final_ckpt_sha256": endpoint_sha,
            "final_ckpt_path": pm.rel_to(args.repo_root, endpoint),
            "recorded_utc": utc_now()}

    if mine:
        old = chain[at]
        differing = sorted(k for k in link if k != "recorded_utc"
                           and str(old.get(k)) != str(link[k]))
        if differing:
            print("RECORD REFUSED:")
            print(f"  !! {arm} job {job} is ALREADY a chain link whose content differs "
                  f"({', '.join(differing)}) — a recorded link is immutable, and rewriting one "
                  "would re-parent every chunk recorded after it")
            return 2
        print(f"{arm} chunk link job {job} ({resume_step} -> {chunk_end}) is already recorded, "
              "byte-identical — no-op")
        return 0

    initial.setdefault("chain", []).append(link)
    if not args.dry_run:
        pm.write_atomic(args.registry, reg)
    print(f"recorded {arm} chunk link job {job}: {resume_step} -> {chunk_end}, continuing "
          f"{prev_desc} " + ("(dry run, nothing written)" if args.dry_run else "(published)"))
    print(f"  resume {str(resume_sha)[:12]} -> endpoint {link['final_ckpt_sha256'][:12]} "
          f"({pm.rel_to(args.repo_root, endpoint)})")
    print(f"  {arm} chain is now {len(initial['chain'])} link(s), tip at step {chunk_end}"
          f" of the {budget} budget")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
    ap.add_argument("arm")
    ap.add_argument("manifest")
    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
                    help="where the Q10 pins are read from")
    ap.add_argument("--producer-dir", default=HERE,
                    help="where the per-leg producer manifests are published")
    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    help="root the manifest's relative paths resolve against")
    ap.add_argument("--extend", action="store_true",
                    help="this leg is already recorded: extend its producer manifest only")
    ap.add_argument("--rehash-all", action="store_true",
                    help="re-hash published checkpoints too (full audit, expensive)")
    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
    ap.add_argument("--skip-sacct", action="store_true",
                    help="DOCUMENTED MANUAL RECOVERY ONLY: skip the scheduler's COMPLETED "
                         "confirmation for a chunk link. Off by default; the watchdog never "
                         "passes it. Use only when accounting cannot answer for a leg you have "
                         "audited by hand, and record why in the worklog.")
    ap.add_argument("--sacct-bin", default=os.environ.get("SACCT_BIN", "sacct"),
                    help="the sacct executable (test hook; changes no decision)")
    args = ap.parse_args(argv)

    arm = args.arm
    pins = read_pins(args.launcher)
    if not pins.get("PINNED_MAXSTEPS"):
        raise SystemExit(f"no PINNED_* values found in {args.launcher}")

    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
    store = os.path.dirname(os.path.abspath(args.registry)) or "."
    lock_fd = os.open(store, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return record(args, arm, pins)
    finally:
        os.close(lock_fd)


def record(args, arm, pins):
    reg = json.load(open(args.registry))
    initial = reg.get("arms", {}).get(arm)
    if initial is None:
        raise SystemExit(f"{arm} has no INITIAL registry entry")

    raw, man = parse_manifest(args.manifest)
    man_sha = hashlib.sha256(raw).hexdigest()
    # Round 5: a leg that carries a `chunk_end` line stopped at a chunk boundary
    # and is recorded as a CHAIN LINK, not as a whole-budget RESTART row.
    chunk_raw = chunk_end_of(man)
    if chunk_raw is not None:
        return record_chunk(args, arm, reg, initial, man, man_sha, pins, chunk_raw)
    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
    job = jk.get("job")

    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
            or l.get("manifest_sha256") == man_sha]
    if same and not args.extend:
        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
                         "one leg, one row; use --extend to extend its producer manifest")
    if len(same) > 1:
        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
                        "inconsistent; fix it before recording")
    if args.extend and not same:
        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
    if problems:
        print("RECORD REFUSED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    anchor = initial["final_ckpt_sha256"]
    producer = pm.manifest_name(arm, job)
    row = {
        "manifest_path": args.manifest, "manifest_sha256": man_sha,
        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
        "time_limit": kvs(man, "time_limit").get("time_limit"),
        "producer_manifest": producer, "chains_to": anchor,
        "recorded_at": pm._now(),
    }
    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
              "chains_to": anchor, "leg_manifest_sha256": man_sha}

    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
    prod_path = os.path.join(args.producer_dir, producer)
    known = (pm.load(prod_path) or {}).get("checkpoints", {})
    found, scan_problems = pm.scan_checkpoints(
        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
        rehash_all=args.rehash_all, repo_root=args.repo_root)
    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
    if scan_problems or pub_problems:
        print("RECORD REFUSED:")
        for p in scan_problems + pub_problems:
            print(f"  !! {p}")
        return 2

    if args.extend:
        for i, leg in enumerate(legs):
            if leg.get("job") == job:
                legs[i] = {**leg, "producer_manifest": producer}
    else:
        legs.append(row)
    if not args.dry_run:
        pm.write_atomic(args.registry, reg)
    verb = "extended" if args.extend else "recorded"
    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
          f"({'dry run, nothing written' if args.dry_run else 'published'})")
    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
          f"{len(kept)} already published"
          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
