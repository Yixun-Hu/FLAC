#!/usr/bin/env python3
"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.

    python3 fa_orbit_record_restart.py C4L outputs_FLAC/exp11_C4L/<manifest>.txt

A restart is only admissible if it provably continues the audited INITIAL run, so
this refuses unless the manifest's resume_ckpt_sha256 equals that arm's recorded
final_ckpt_sha256 (the audited 40k checkpoint). An sbatch return, or a manifest on
its own, does not establish the chain -- the sha does.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "arm_launch_registry.json")


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


def main(argv):
    if len(argv) != 3:
        raise SystemExit(f"usage: {os.path.basename(argv[0])} <ARM> <manifest path>")
    arm, man_path = argv[1], argv[2]
    reg = json.load(open(REG))
    initial = reg["arms"].get(arm)
    if initial is None:
        raise SystemExit(f"{arm} has no INITIAL registry entry")
    anchor = initial.get("final_ckpt_sha256")
    if not anchor:
        raise SystemExit(f"{arm} has no audited final_ckpt_sha256 to chain from")

    raw, man = parse_manifest(man_path)
    jk, rk = kvs(man, "job"), kvs(man, "resume_ckpt")
    if jk.get("mode") != "RESTART":
        raise SystemExit(f"manifest mode is {jk.get('mode')!r}, not RESTART")
    resume_sha = rk.get("resume_ckpt_sha256")
    if resume_sha != anchor:
        raise SystemExit(f"resume_ckpt_sha256 {str(resume_sha)[:12]} != the audited INITIAL "
                         f"final checkpoint {anchor[:12]} — this leg does not continue that run")
    resume_path = man.get("resume_ckpt", "").split()[0] if man.get("resume_ckpt") else ""
    if resume_path and os.path.isfile(resume_path):
        h = hashlib.sha256()
        with open(resume_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != anchor:
            raise SystemExit(f"the resume file on disk hashes {h.hexdigest()[:12]}, not {anchor[:12]}")

    reg.setdefault("restarts", {}).setdefault(arm, []).append({
        "manifest_path": man_path,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "job": jk.get("job"), "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
        "commit": man.get("commit"),
        "resume_ckpt": resume_path, "resume_ckpt_sha256": resume_sha,
        "expected_step": rk.get("expected_step"),
        "max_steps": kvs(man, "arm").get("max_steps"),
        "time_limit": kvs(man, "time_limit").get("time_limit"),
        "chains_to": anchor,
    })
    json.dump(reg, open(REG, "w"), indent=2)
    open(REG, "a").write("\n")
    print(f"recorded {arm} RESTART job {jk.get('job')} chaining to {anchor[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
