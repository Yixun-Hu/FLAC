#!/usr/bin/env python3
"""exp_11 RESTART checkpoint preflight (round-3 review B2).

exp_10's `bf_resume_launch.sh` proved a restart with ONE CPU-side ``torch.load``
before spending an allocation; round 3 shipped only a path check, which would
have accepted a zero-byte file or a renamed C4 checkpoint under the C16 root
(orbit size does not change the module tree, so it would even have loaded). This
restores the exp_10 depth for the sweep:

  - the checkpoint's embedded ``global_step`` equals EXPECTED_STEP exactly;
  - its embedded ``model_config`` deep-equals this arm's config file, so the
    orbit, conditioning method and architecture it was trained under are the
    ones this run would continue;
  - full warm optimizer state (non-empty ``state``), ``lr_schedulers``, and EMA
    weights are present — a stripped/weights-only file is the wrong file, since
    exp_11 has no optimizer-reset lineage;
  - the target budget still lies ahead (``global_step < max_steps``), so a
    "restart" cannot terminate immediately on Lightning's ``>=`` stop rule while
    printing the completion literal;
  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
    bound to the same rung, commit and budget.

Three resume contracts share those structural checks and differ only in what the
resume file must BE:
  * default   — a crash restart of the same launch (same budget, same commit);
  * --extension — the Q10 40k -> 100k leg: the audited INITIAL launch identity,
    resuming that launch's audited final checkpoint;
  * --chain    — a CHUNK (round 5): the same INITIAL identity, resuming the TIP
    of ``arms.<ARM>.chain``, the per-chunk links fa_orbit_record_restart.py
    writes. An empty chain refuses, so chunk N+1 cannot start until chunk N is
    recorded.

Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
Exit 0 = admissible; nonzero = refuse to launch.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt_config(path):
    """The model_config embedded in a Lightning checkpoint, on CPU.

    Shared with the screen driver (fa_orbit_screen.sbatch), which asserts the
    checkpoint's own orbit before spending an evaluation on it: a screen that
    silently evaluated the wrong arm's checkpoint would poison a futility gate."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    cfg = ck.get("model_config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint carries no embedded model_config: {path}")
    return cfg, ck.get("global_step")


def load_ckpt_state_keys(path):
    """The checkpoint's state_dict KEYS (used to prove EMA weights exist before a
    screen spends a GPU: eval_FLAC silently evaluates online weights when the EMA
    entries are absent)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    return list((ck.get("state_dict") or {}).keys())


def parse_manifest(path):
    """The launcher's own manifest format: whitespace-separated `key value...`."""
    out = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            out[key] = rest.strip()
    return out


def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
    man = parse_manifest(manifest_path)
    problems = []
    fields = man.get("arm", "")
    # `arm <ARM> rung <RUNG> micro <MB> ngpu <N> max_steps <S> ...`
    tokens = ("arm " + fields).split()
    kv = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}
    if kv.get("arm") != arm:
        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
    if kv.get("rung") != rung:
        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
                        "(a restart may not change the rung: it would change rank count, "
                        "sampler partitioning and worker seeding mid-lineage)")
    if kv.get("max_steps") != str(maxsteps):
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
    # "no opinion", it is missing provenance — the restart must not proceed on it.
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
                        "to the lineage that produced this checkpoint")
    elif not commit:
        problems.append("no running commit supplied to compare against the manifest commit")
    elif man_commit != commit:
        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
    return problems, man


def kv_line(man, key):
    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
    f = (f"{key} " + man.get(key, "")).split()
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def canonical_ckpt_dir(save_dir, arm, repo_root):
    """<save_dir>/FLAC_exp11_<ARM>/exp11_<ARM>/checkpoints, as the launcher builds it.

    save_dir is recorded relative to the repo root, so it is resolved against it."""
    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))


def _load_registry_row(registry_path, arm):
    """(row, problems) for the arm's INITIAL registry row."""
    if not os.path.isfile(registry_path):
        return None, [f"audited launch registry not found: {registry_path}"]
    row = json.load(open(registry_path)).get("arms", {}).get(arm)
    if row is None:
        return None, [f"{arm} is not in the audited launch registry {registry_path}"]
    return row, []


def check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps):
    """Everything an EXTENSION and a CHAIN leg must BOTH prove about the INITIAL launch.

    The two contracts differ only in their RESUME ANCHOR (the extension chains to
    the audited 40k checkpoint; a chain leg chains to the last recorded chunk),
    so the launch-identity half lives here and is shared verbatim.
    """
    problems = []
    kv, jkv = kv_line(man, "arm"), kv_line(man, "job")

    got_sha = sha256_file(manifest_path)
    if got_sha != reg.get("manifest_sha256"):
        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
                        "was registered")
    for label, got_v, want_v in (("arm", kv.get("arm"), arm),
                                 ("job", jkv.get("job"), reg.get("job")),
                                 ("launch mode", jkv.get("mode"), "INITIAL"),
                                 ("launch_uuid", jkv.get("launch_uuid"), reg.get("launch_uuid")),
                                 ("rung", kv.get("rung"), reg.get("rung")),
                                 ("rung (this run)", rung, reg.get("rung")),
                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
                                 ("save_dir", man.get("save_dir"), reg.get("save_dir"))):
        if got_v != want_v:
            problems.append(f"{label} {got_v!r} != registered {want_v!r}")
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
                        "the lineage that produced this checkpoint")
    elif man_commit != reg.get("commit"):
        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
                        f"{str(reg.get('commit'))[:12]}")
    if int(reg.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {reg.get('training_seed')!r} != 42")
    # The INITIAL budget is the manifest's and the registry's; the extension's is
    # this run's, and it must strictly cover the resume point without shrinking.
    initial_budget = reg.get("max_steps")
    if kv.get("max_steps") != initial_budget:
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
                        f"{initial_budget!r} (the INITIAL budget, which an extension preserves)")
    try:
        if max_steps < int(initial_budget):
            problems.append(f"extension budget {max_steps} does not extend the registered "
                            f"{initial_budget} — an extension may only raise the budget")
    except (TypeError, ValueError):
        problems.append(f"registered max_steps {initial_budget!r} is not an integer")
    if sha256_file(config_path) != reg.get("config_sha256"):
        problems.append(f"{config_path} sha256 != the registered config_sha256 "
                        f"{str(reg.get('config_sha256'))[:12]}")
    return problems


def check_canonical_dir(man, arm, ckpt_path, repo_root):
    """The resume file sits in the REGISTERED launch's own run directory."""
    save_dir = man.get("save_dir", "")
    if not save_dir:
        return ["manifest records no save_dir"]
    canon = canonical_ckpt_dir(save_dir, arm, repo_root)
    if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
        return [f"resume checkpoint {ckpt_path} does not live in the registered "
                f"launch's canonical run directory {canon}"]
    return []


def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
                            ckpt_sha, expected_step, max_steps, repo_root="."):
    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).

    A crash restart continues the SAME launch: same budget, same reviewed commit,
    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
    — it raises the budget from 40000 to 100000 and runs later reviewed code —
    and demanding equality there is exactly what gave jobs 3662828-30 their third
    hard-abort path.

    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
    proves it against the COMMITTED registry rather than the mutable manifest
    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
    that IS that launch's audited final checkpoint, sitting in that launch's own
    canonical run directory. Budget and running commit may move; nothing that
    identifies the run may.
    """
    reg, problems = _load_registry_row(registry_path, arm)
    if problems:
        return problems, {}
    man = parse_manifest(manifest_path)
    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
    # the resumed checkpoint IS the audited anchor, in the audited run directory
    anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
    if not anchor:
        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
                        "has nothing to chain to (audit the arm's final checkpoint first)")
    elif ckpt_sha != anchor:
        problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the audited final checkpoint "
                        f"{anchor[:12]} — this leg does not continue that run")
    if final_step is not None and int(final_step) != int(expected_step):
        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
    return problems, man


def check_link_manifest(i, link):
    """Link ``i`` must agree with the MANIFEST it cites (round-5 r2, blocking 4).

    Continuity alone proves only that the numbers in the registry line up, and
    the registry is a mutable, uncommitted record (round-5 B1 deliberately took
    it out of both drift gates). A single structurally valid forged link —
    resume hash/step copied from the audited anchor, an increasing final step,
    and an arbitrary final hash — therefore passed every ancestry check. So each
    link is now checked against the artefact it names:

      * its manifest_path exists and its BYTES hash to the recorded manifest_sha256;
      * the manifest's job / launch_uuid are the link's;
      * the manifest's endpoint attestation (step, sha256) is the link's
        final_step / final_ckpt_sha256;
      * the manifest's chunk_end is the link's final_step.

    THREAT MODEL, honestly stated: this is COOPERATIVE INTEGRITY, not
    cryptographic provenance. It defends against accidents (a stale glob, a
    mis-recorded link, a manifest that drifted after recording) and casual
    tampering (editing the registry by hand). It cannot defend against an author
    who edits registry and manifest together: both are uncommitted files under
    the same user between publication and commit, and nothing here is signed.
    The audit trail is the git history of the committed records, not this check.
    """
    problems = []
    job, uuid = link.get("job"), link.get("launch_uuid")
    man_path, man_sha = link.get("manifest_path"), link.get("manifest_sha256")
    if not man_path:
        return [f"chain link {i} (job {job!r}) cites no manifest_path — a link with no manifest "
                "is a bare assertion, not a record"]
    if not os.path.isfile(man_path):
        return [f"chain link {i} (job {job!r}) cites manifest {man_path}, which does not exist — "
                "the record cannot be checked against the artefact it names"]
    got = sha256_file(man_path)
    if got != man_sha:
        problems.append(f"chain link {i} (job {job!r}): its manifest {man_path} now hashes "
                        f"{got[:12]}, not the recorded {str(man_sha)[:12]} — the manifest changed "
                        "after the link was recorded")
    man = parse_manifest(man_path)
    jkv = kv_line(man, "job")
    if jkv.get("job") != str(job):
        problems.append(f"chain link {i}: its manifest records job {jkv.get('job')!r}, not the "
                        f"link's {job!r}")
    if jkv.get("launch_uuid") != uuid:
        problems.append(f"chain link {i} (job {job!r}): its manifest records launch_uuid "
                        f"{jkv.get('launch_uuid')!r}, not the link's {uuid!r}")
    ckv = kv_line(man, "chunk_end")
    if str(ckv.get("chunk_end")) != str(link.get("final_step")):
        problems.append(f"chain link {i} (job {job!r}): its manifest declares chunk_end "
                        f"{ckv.get('chunk_end')!r}, not the link's final_step "
                        f"{link.get('final_step')!r}")
    akv = kv_line(man, "endpoint_ckpt")
    if "endpoint_ckpt" not in man:
        problems.append(f"chain link {i} (job {job!r}): its manifest carries no endpoint "
                        "attestation, so nothing in it says that job produced this checkpoint")
        return problems
    if str(akv.get("endpoint_step")) != str(link.get("final_step")):
        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_step "
                        f"{akv.get('endpoint_step')!r} != the link's final_step "
                        f"{link.get('final_step')!r}")
    if akv.get("endpoint_sha256") != link.get("final_ckpt_sha256"):
        problems.append(f"chain link {i} (job {job!r}): its manifest attests endpoint_sha256 "
                        f"{str(akv.get('endpoint_sha256'))[:12]} != the link's final_ckpt_sha256 "
                        f"{str(link.get('final_ckpt_sha256'))[:12]} — the record does not match "
                        "the manifest it cites")
    # Round-5 r3 blocking 1: the RESUME half of the link must also match the
    # manifest it cites, or a registry-only edit can re-parent a genuine later
    # manifest onto the anchor (endpoint checks all still pass). Same
    # cooperative-integrity scope as above.
    rkv = kv_line(man, "resume_ckpt")
    if str(rkv.get("expected_step")) != str(link.get("resume_step")):
        problems.append(f"chain link {i} (job {job!r}): its manifest resumed at expected_step "
                        f"{rkv.get('expected_step')!r}, not the link's resume_step "
                        f"{link.get('resume_step')!r} — the link re-parents that manifest")
    if rkv.get("resume_ckpt_sha256") != link.get("resume_ckpt_sha256"):
        problems.append(f"chain link {i} (job {job!r}): its manifest resumed checkpoint "
                        f"{str(rkv.get('resume_ckpt_sha256'))[:12]}, not the link's resume sha "
                        f"{str(link.get('resume_ckpt_sha256'))[:12]} — the link re-parents that "
                        "manifest")
    # ...and the scheduler's verdict on the link's job is rechecked, so a
    # positive manifest from a job that later NODE_FAILed cannot be inserted
    # into the mutable registry around the recorder's COMPLETED gate. sacct
    # history AGES OUT on this cluster, so an EMPTY answer is accepted with a
    # loud warning (fail-closed here would brick every chain older than the
    # accounting retention window); an explicit non-COMPLETED verdict refuses.
    state = _link_sacct_state(job)
    if state == "EXPIRED":
        # rc==0 with EMPTY output is the one case where "no answer" is a
        # plausible truth (accounting retention aged the job out); everything
        # else — a broken SACCT_BIN, a timeout, a nonzero exit — is a FAILED
        # QUERY and stays fail-closed (round-5 r4).
        print(f"WARNING: sacct has no record of chain link {i}'s job {job!r} (accounting "
              "retention expired) — accepting the link on its manifest alone")
    elif state.startswith("QUERY_FAILED"):
        problems.append(f"chain link {i} (job {job!r}): the scheduler verdict could not be "
                        f"obtained ({state}) — refusing rather than skipping the "
                        "scheduler-integrity check")
    elif state != "COMPLETED":
        problems.append(f"chain link {i} (job {job!r}): sacct says {state}, not COMPLETED — a "
                        "link recorded for an unsuccessful job is not lineage")
    return problems


def _link_sacct_state(job):
    """The scheduler's verdict for a job: a State string, 'EXPIRED' for a
    successful-but-empty answer, or 'QUERY_FAILED: <why>' for a failed query."""
    import subprocess
    try:
        out = subprocess.run([os.environ.get("SACCT_BIN", "sacct"), "-X", "-n", "-P",
                              "-j", str(job), "-o", "State"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"QUERY_FAILED: {type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return f"QUERY_FAILED: sacct rc={out.returncode}: {out.stderr.strip()[:120]}"
    first = out.stdout.strip().splitlines()
    return first[0].split()[0] if first and first[0].strip() else "EXPIRED"


def check_chain_ancestry(reg, chain):
    """EVERY link, from the audited INITIAL anchor to the tip (round-5 B6).

    Checking only ``chain[-1]`` accepted a crafted registry: append a link whose
    ``final_ckpt_sha256`` is whatever file you want to run, and the tip check
    passes while nothing connects it to the audited 40k anchor. The chain is a
    lineage, so it is verified as one — link 0 must resume the INITIAL anchor,
    every later link must resume its predecessor's endpoint, and the steps must
    strictly increase. Any break names the offending link INDEX.

    Round-5 r2 (blocking 4): continuity is necessary but not sufficient — every
    link is ALSO checked against the manifest it cites (check_link_manifest),
    because a lone structurally valid forged link satisfied continuity by
    construction.
    """
    problems = []
    prev_sha, prev_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
    prev_label = "the audited INITIAL anchor"
    for i, link in enumerate(chain):
        job = link.get("job")
        r_sha, r_step = link.get("resume_ckpt_sha256"), link.get("resume_step")
        f_sha, f_step = link.get("final_ckpt_sha256"), link.get("final_step")
        problems += check_link_manifest(i, link)
        if not f_sha or f_step is None:
            problems.append(f"chain link {i} (job {job!r}) carries no final_ckpt_sha256/final_step — "
                            "it is not evidence of a checkpoint")
        if prev_sha and r_sha != prev_sha:
            problems.append(f"chain link {i} (job {job!r}) resume_ckpt_sha256 {str(r_sha)[:12]} != "
                            f"{prev_label}'s final_ckpt_sha256 {str(prev_sha)[:12]} — the chain is "
                            f"BROKEN at link {i}: it does not continue what precedes it")
        if prev_step is not None and str(r_step) != str(prev_step):
            problems.append(f"chain link {i} (job {job!r}) resume_step {r_step!r} != {prev_label}'s "
                            f"final_step {prev_step!r} — the chain is BROKEN at link {i}")
        try:
            if prev_step is not None and f_step is not None and int(f_step) <= int(prev_step):
                problems.append(f"chain link {i} (job {job!r}) final_step {f_step!r} does not exceed "
                                f"{prev_label}'s {prev_step!r} — chunk steps only ever increase")
        except (TypeError, ValueError):
            problems.append(f"chain link {i} (job {job!r}) has a non-integer step "
                            f"(resume {r_step!r}, final {f_step!r})")
        prev_sha, prev_step, prev_label = f_sha, f_step, f"chain link {i}"
    return problems


def check_chain_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
                        ckpt_sha, expected_step, max_steps, repo_root="."):
    """The CHUNKED-LEG contract: resume the TIP of this arm's recorded chain.

    A chunk leg is an extension leg that stops early, so it must prove exactly
    the same INITIAL launch identity — but its resume point is no longer the
    audited 40k anchor: it is the endpoint of the previous chunk, recorded as the
    last link of ``arms.<ARM>.chain`` by fa_orbit_record_restart.py.

    Fail-closed by construction: an EMPTY or ABSENT chain refuses, so chunk N+1
    cannot run until chunk N has been recorded, and every link's final_step /
    final_ckpt_sha256 was written by the recorder from the file on disk. The
    resume file is re-hashed by the caller (``ckpt_sha``); a manifest's claimed
    hash is never trusted here or anywhere else in this chain.

    Round-5 review B6: the WHOLE ancestry is validated (check_chain_ancestry),
    not merely ``chain[-1]`` — otherwise a crafted registry could append a tip
    with an arbitrary hash and no link back to the audited 40k anchor.

    Round-5 r2 (blocking 3(d)): nothing here assumes a checkpoint FILENAME shape.
    The resume file arrives as an explicit ``--ckpt`` path and is identified by
    its sha256 and its directory, so a Lightning-versioned endpoint
    (``epoch=E-step=N-v1.ckpt``, written when a retry finds the unversioned name
    taken) is admissible exactly like an unversioned one.
    """
    reg, problems = _load_registry_row(registry_path, arm)
    if problems:
        return problems, {}
    man = parse_manifest(manifest_path)
    problems = check_initial_identity(man, manifest_path, reg, arm, rung, config_path, max_steps)
    if not reg.get("final_ckpt_sha256"):
        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — a chain of "
                        "chunks must still descend from the audited INITIAL run")
    chain = reg.get("chain") or []
    if not chain:
        problems.append(f"{arm} has no recorded chain link in {registry_path}: a chunk may only "
                        "resume the tip of a RECORDED chain, so chunk N+1 is inadmissible until "
                        "chunk N is recorded — run fa_orbit_record_restart.py on the previous "
                        "chunk's launcher manifest first")
    else:
        # B6: the FULL ancestry, not just the tip — a tip alone can be crafted.
        problems += check_chain_ancestry(reg, chain)
        tip = chain[-1]
        tip_step, tip_sha = tip.get("final_step"), tip.get("final_ckpt_sha256")
        if str(tip_step) != str(expected_step):
            problems.append(f"EXPECTED_STEP {expected_step} != the last recorded chain link's "
                            f"final_step {tip_step!r} (link job {tip.get('job')!r}) — a chunk "
                            "resumes the TIP of the chain, never an earlier link")
        if not tip_sha:
            problems.append(f"the last recorded chain link (job {tip.get('job')!r}) carries no "
                            "final_ckpt_sha256 — it is not evidence of a checkpoint")
        elif ckpt_sha != tip_sha:
            problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the last recorded chain "
                            f"link's final_ckpt_sha256 {str(tip_sha)[:12]} — this file is not the "
                            "checkpoint that chunk produced")
    problems += check_canonical_dir(man, arm, ckpt_path, repo_root)
    return problems, man


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expected-step", type=int, required=True)
    ap.add_argument("--config", required=True, help="this arm's model config json")
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--launch-manifest", default="",
                    help="the arm's original launch manifest (binds rung/commit/budget)")
    ap.add_argument("--extension", action="store_true",
                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
                         "INITIAL launch identity instead of requiring an equal budget/commit")
    ap.add_argument("--chain", action="store_true",
                    help="this restart is a CHUNK continuing an already-recorded chain: same "
                         "INITIAL identity as --extension, but the resume anchor is the LAST "
                         "recorded arms.<ARM>.chain link instead of the audited 40k checkpoint")
    ap.add_argument("--chunk-end", type=int, default=None,
                    help="this leg's stop step (a chunk boundary): EXPECTED_STEP < N <= max-steps "
                         "and a multiple of 2500")
    ap.add_argument("--launch-registry", default="",
                    help="the committed arm launch registry (required with --extension/--chain)")
    ap.add_argument("--repo-root", default=".",
                    help="root the registry's relative save_dir is resolved against")
    args = ap.parse_args(argv)
    if args.extension and args.chain:
        ap.error("--extension and --chain are mutually exclusive: a leg either resumes the audited "
                 "40k anchor or the tip of the recorded chunk chain, never both")
    if args.extension and not args.launch_registry:
        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")
    if args.chain and not args.launch_registry:
        ap.error("--chain requires --launch-registry (the recorded chunk chain lives in it)")

    if not os.path.isfile(args.ckpt):
        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
        return 2

    import torch  # deferred: keeps --help and unit imports cheap

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:                      # truncated, empty or not a torch file
        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
              f"{type(exc).__name__}: {exc}")
        return 2
    if not isinstance(ck, dict):
        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
        return 2

    problems = []
    gs = ck.get("global_step")
    if gs != args.expected_step:
        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
    if isinstance(gs, int) and gs >= args.max_steps:
        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
                        "immediately and still print the completion literal")

    mc = ck.get("model_config")
    if not isinstance(mc, dict):
        problems.append("checkpoint carries no embedded 'model_config' dict")
    else:
        want = json.load(open(args.config))
        if mc != want:
            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
            problems.append(
                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")

    opts = ck.get("optimizer_states") or []
    if len(opts) != 1:
        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
    elif not opts[0].get("state"):
        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
                        "WARM continuations and have no optimizer-reset lineage")
    if not ck.get("lr_schedulers"):
        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
    sd = ck.get("state_dict") or {}
    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
    if not n_ema:
        problems.append("no EMA weights in state_dict")

    # The per-job stop step is checked against the BUDGET (--max-steps), which the
    # launcher still pins at 100000: a chunk narrows where this leg stops, never
    # what the campaign is allowed to reach.
    if args.chunk_end is not None:
        if args.chunk_end % 2500 != 0:
            problems.append(f"--chunk-end {args.chunk_end} is not a multiple of 2500 (the pinned "
                            "checkpoint cadence: a chunk must end ON a checkpoint)")
        if not args.expected_step < args.chunk_end <= args.max_steps:
            problems.append(f"--chunk-end {args.chunk_end} must satisfy EXPECTED_STEP "
                            f"{args.expected_step} < chunk_end <= max_steps {args.max_steps}")

    digest = sha256_file(args.ckpt)     # needed by the extension/chain anchor checks
    man = {}
    if args.launch_manifest:
        if not os.path.isfile(args.launch_manifest):
            problems.append(f"launch manifest not found: {args.launch_manifest}")
        elif args.extension:
            more, man = check_extension_binding(
                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
            problems += more
        elif args.chain:
            more, man = check_chain_binding(
                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
            problems += more
        else:
            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
                                               args.commit, args.max_steps)
            problems += more
    elif args.extension:
        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")
    elif args.chain:
        problems.append("--chain requires --launch-manifest (the audited INITIAL manifest)")

    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    tr = mc.get("training", {})
    print(f"restart lineage OK: {args.ckpt}")
    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
          f"angles={tr.get('frame_avg_angles')}")
    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
          f"lr={opts[0]['param_groups'][0].get('lr')} "
          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
    if man and args.extension:
        print(f"extension lineage OK: {args.arm} {args.expected_step} -> "
              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} continues "
              f"the audited launch job {kv_line(man, 'job').get('job')} "
              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
        if args.chunk_end is not None:
            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
    elif man and args.chain:
        print(f"chain lineage OK: {args.arm} {args.expected_step} -> "
              f"{args.chunk_end if args.chunk_end is not None else args.max_steps} resumes the tip "
              f"of the recorded chunk chain, under the audited launch job "
              f"{kv_line(man, 'job').get('job')} "
              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
        if args.chunk_end is not None:
            print(f"  chunk leg: stops at {args.chunk_end} of the {args.max_steps} budget")
    elif man:
        print(f"  bound to launch manifest: {args.launch_manifest}")
    print(f"CKPT_SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
