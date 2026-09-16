"""The fail-closed launch contract of exp_14 (plan §4 Round D, ``verify_exp14.py``).

Run by the launcher before anything is started, and again by hand whenever a number from
this experiment is quoted. It re-checks, on the machine that trains, every premise the
six runs rest on:

* the two arm configs are the sha-pinned kit files, and they differ **only** in the
  geometry backbone (so the curve measures data efficiency, not two recipes);
* the three fraction dataset configs are the base config plus exactly two fields;
* the committed split files are the ones the detached checksum file and the manifest
  describe, and -- recomputed from the files themselves, never trusted from the manifest --
  they are nested, have zero starved targets, and have the sizes and eligible-context
  histograms the manifest claims;
* ``train.json`` / ``unseen_eval.json`` / the VAE are the pinned bytes;
* the receiver-level anchor audit (plan §3 D3): no on-disk RIR outside ``train.json``
  sits at a receiver that ``train.json`` uses, which is what makes the historical 100 %
  runs exact anchors under the restricted-context definition;
* both repositories are at the sha the launch record names.

Every check prints ``PASS``/``FAIL`` with the numbers behind it; any FAIL exits 3. The
expected git shas are **passed in** from the immutable launch record and never hard-coded
(finding r2-7); the file shas are pinned here because they are the frozen artifacts.
"""
import hashlib
import json
import os
import subprocess
from collections import namedtuple

from src.tools.data_curve.names import FRACTIONS

#: Pinned by the plan (§2, verified 2026-09-16). The kit configs are byte copies of
#: ``FLAC_AR_exp09.json`` (cyl) and ``FLAC_AR_BVp1.json`` (van).
ARM_CONFIG_SHAS = {
    "FLAC_AR_exp14_cylS.json":
        "8df8a8117449ced6c9e7068891e119327f69d89bbeff11532fe1fa98066fab90",
    "FLAC_AR_exp14_vanS.json":
        "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
}

#: The ONLY keys the van config may lack relative to the cyl config: the geometry backbone
#: on both ViT blocks (source_vit, context_poses_vit) and the two training-side flags that
#: follow from it. Anything else -- a different lr, a different sampler -- would make the
#: two arms two experiments.
ARM_CONFIG_CYL_ONLY_KEYS = (
    "/model/conditioning/configs/1/config/ViT/implementation",
    "/model/conditioning/configs/1/config/ViT/gauge",
    "/model/conditioning/configs/2/config/ViT/implementation",
    "/model/conditioning/configs/2/config/ViT/gauge",
    "/training/cond_method",
    "/training/frame_avg_angles",
)

#: Files whose bytes the whole experiment is pinned to. ``weights/`` is a symlink on this
#: box; the sha is of the file it resolves to.
PINNED_FILE_SHAS = {
    "data/AR/train.json":
        "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03",
    "data/AR/unseen_eval.json":
        "9a9d817abc3e19f41351e07325ffa929c1f2846d0c77e80d538d9e4a21342ba8",
    "weights/FLAC/VAE.safetensors":
        "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
}

BASE_TRAIN_CONFIG = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
#: tag -> fraction, from the single source of truth for this experiment's names.
FRACTION_TAGS = tuple(sorted(FRACTIONS))
SPLIT_SEED = 2026
EXIT_OK = 0
EXIT_FAILED = 3


class CheckResult(namedtuple("CheckResult", "name ok detail")):
    """One line of the verifier's report."""

    def line(self):
        return f"{'PASS' if self.ok else 'FAIL'} {self.name}: {self.detail}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(obj, prefix=""):
    """Flatten a parsed JSON document to ``{key path: leaf}``.

    Dicts and lists *of containers* are descended (so the two ViT conditioner blocks keep
    their index), while a list of scalars stays one leaf -- ``frame_avg_angles`` is a
    single key that either exists or does not, not a set of indexed keys.
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}/{key}"))
        return out
    if isinstance(obj, list) and any(isinstance(x, (dict, list)) for x in obj):
        out = {}
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}/{index}"))
        return out
    return {prefix: obj}


def _load(path):
    with open(path) as fin:
        return json.load(fin)


def check_arm_configs(kit_dir, expected_shas=None):
    """The two arm configs: pinned bytes, and a difference confined to the backbone."""
    expected_shas = ARM_CONFIG_SHAS if expected_shas is None else expected_shas
    results = []
    parsed = {}
    for name, expected in sorted(expected_shas.items()):
        path = os.path.join(kit_dir, "configs", name)
        try:
            found = sha256_file(path)
            parsed[name] = _load(path)
        except (OSError, ValueError) as err:
            results.append(CheckResult(f"arm config sha {name}", False,
                                       f"{path} unusable ({type(err).__name__}: {err})"))
            continue
        results.append(CheckResult(
            f"arm config sha {name}", found == expected,
            f"{found[:16]}… == {expected[:16]}…" if found == expected
            else f"{found} != pinned {expected} ({path})",
        ))

    name = "arm configs differ only in the geometry backbone"
    if len(parsed) != 2:
        results.append(CheckResult(name, False, "one of the two configs could not be read"))
        return results
    cyl = flatten(parsed["FLAC_AR_exp14_cylS.json"])
    van = flatten(parsed["FLAC_AR_exp14_vanS.json"])
    cyl_only = sorted(set(cyl) - set(van))
    van_only = sorted(set(van) - set(cyl))
    differing = sorted(k for k in set(cyl) & set(van) if cyl[k] != van[k])
    ok = (tuple(cyl_only) == tuple(sorted(ARM_CONFIG_CYL_ONLY_KEYS))
          and not van_only and not differing)
    results.append(CheckResult(
        name, ok,
        f"cyl-only {len(cyl_only)} key(s) == the {len(ARM_CONFIG_CYL_ONLY_KEYS)} backbone "
        f"keys, van-only 0, differing values 0" if ok else
        f"cyl-only {cyl_only}, van-only {van_only}, differing {differing}",
    ))
    return results


def check_dataset_configs(flac_wt):
    """Each fraction config == the base config + its split + ``restrict_to_split``."""
    results = []
    base_path = os.path.join(flac_wt, BASE_TRAIN_CONFIG)
    try:
        base = _load(base_path)
    except (OSError, ValueError) as err:
        return [CheckResult("base training dataset config", False,
                            f"{base_path} unusable ({type(err).__name__}: {err})")]
    for tag in FRACTION_TAGS:
        rel = f"src/configs/dataset_configs/AR/train/acousticroom_train_frac{tag}.json"
        name = f"dataset config {os.path.basename(rel)}"
        try:
            variant = _load(os.path.join(flac_wt, rel))
        except (OSError, ValueError) as err:
            results.append(CheckResult(name, False, f"unusable ({type(err).__name__}: {err})"))
            continue
        expected = json.loads(json.dumps(base))
        expected["datasets"][0]["json_file_path"] = f"data/AR/train_frac{tag}_s{SPLIT_SEED}.json"
        expected["modalities"]["acoustic_context"]["restrict_to_split"] = True
        flat_expected, flat_variant = flatten(expected), flatten(variant)
        differing = sorted(
            key for key in set(flat_expected) | set(flat_variant)
            if flat_expected.get(key, "<absent>") != flat_variant.get(key, "<absent>")
        )
        results.append(CheckResult(
            name, not differing,
            f"== base + json_file_path(train_frac{tag}_s{SPLIT_SEED}.json) + "
            "restrict_to_split" if not differing else f"unexpected difference(s): {differing}",
        ))
    return results


def check_pinned_files(flac_wt, expected=None):
    """``train.json``, ``unseen_eval.json`` and the VAE are the pinned bytes."""
    expected = PINNED_FILE_SHAS if expected is None else expected
    results = []
    for rel, want in sorted(expected.items()):
        path = os.path.join(flac_wt, rel)
        try:
            found = sha256_file(path)
        except OSError as err:
            results.append(CheckResult(f"pinned file {rel}", False, f"unreadable ({err})"))
            continue
        results.append(CheckResult(
            f"pinned file {rel}", found == want,
            f"{found[:16]}… ({os.path.getsize(path)} bytes)" if found == want
            else f"{found} != pinned {want}",
        ))
    return results


def check_git_head(repo_dir, expected_sha, label):
    """``git -C <repo> rev-parse HEAD`` equals the sha the launch record names."""
    try:
        head = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as err:
        return CheckResult(f"{label} HEAD", False, f"git failed on {repo_dir}: {err}")
    if head.returncode != 0:
        return CheckResult(f"{label} HEAD", False,
                           f"{repo_dir} is not a git repository ({head.stderr.strip()})")
    found = head.stdout.strip()
    dirty = subprocess.run(["git", "-C", repo_dir, "diff", "--quiet", "HEAD"],
                           capture_output=True, text=True)
    state = "clean" if dirty.returncode == 0 else "TRACKED FILES MODIFIED"
    return CheckResult(
        f"{label} HEAD", found == expected_sha,
        f"{found} ({repo_dir}, tracked-file state: {state})" if found == expected_sha
        else f"{found} != expected {expected_sha} ({repo_dir})",
    )


# ======================================================================================
# The committed split artifacts: their checksums, and their contents recomputed
# ======================================================================================
def _split_name(tag):
    return f"train_frac{tag}_s{SPLIT_SEED}.json"


def _manifest_names():
    return (f"train_frac_manifest_s{SPLIT_SEED}.json",
            f"train_frac_manifest_s{SPLIT_SEED}.sha256")


def check_split_checksums(data_dir):
    """The split files and the manifest are the bytes the detached checksum file names.

    The manifest hashes every split it emitted but never itself (finding r2-7), so the
    ``.sha256`` sidecar is the only thing that pins the manifest -- and the manifest's own
    ``files`` map is checked against the same recomputed digests, so a tampered manifest
    cannot agree with a tampered split.
    """
    manifest_name, checksums_name = _manifest_names()
    results = []
    try:
        with open(os.path.join(data_dir, checksums_name)) as fin:
            listed = [line.split(maxsplit=1) for line in fin.read().splitlines() if line.strip()]
        expected = {name.strip(): sha for sha, name in listed}
    except (OSError, ValueError) as err:
        return [CheckResult("split checksum file", False,
                            f"{checksums_name} unusable ({type(err).__name__}: {err})")]

    wanted = {_split_name(tag) for tag in FRACTION_TAGS} | {manifest_name}
    results.append(CheckResult(
        "split checksum file covers every artifact", set(expected) == wanted,
        f"{checksums_name} lists {len(expected)} file(s): {sorted(expected)}"
        if set(expected) == wanted else
        f"{checksums_name} lists {sorted(expected)}, expected {sorted(wanted)}",
    ))

    digests = {}
    for name in sorted(wanted):
        path = os.path.join(data_dir, name)
        try:
            digests[name] = sha256_file(path)
        except OSError as err:
            results.append(CheckResult(f"split file sha {name}", False, f"unreadable ({err})"))
            continue
        want = expected.get(name)
        results.append(CheckResult(
            f"split file sha {name}", digests[name] == want,
            f"{digests[name][:16]}… ({os.path.getsize(path)} bytes)"
            if digests[name] == want else f"{digests[name]} != {want} in {checksums_name}",
        ))

    try:
        manifest = _load(os.path.join(data_dir, manifest_name))
    except (OSError, ValueError) as err:
        results.append(CheckResult("manifest files map", False,
                                   f"manifest unusable ({type(err).__name__}: {err})"))
        return results
    claimed = manifest.get("files", {})
    mismatched = sorted(
        name for name in (_split_name(tag) for tag in FRACTION_TAGS)
        if claimed.get(name) != digests.get(name)
    )
    results.append(CheckResult(
        "manifest files map matches the split files", not mismatched,
        f"{len(claimed)} split digest(s) agree with the files on disk" if not mismatched
        else f"manifest disagrees for {mismatched}",
    ))
    train_sha = None
    try:
        train_sha = sha256_file(os.path.join(data_dir, "train.json"))
    except OSError as err:
        results.append(CheckResult("manifest names its source train.json", False,
                                   f"train.json unreadable ({err})"))
    if train_sha is not None:
        claimed_source = manifest.get("source_train_json_sha256")
        results.append(CheckResult(
            "manifest names its source train.json", claimed_source == train_sha,
            f"source_train_json_sha256 {train_sha[:16]}… == train.json"
            if claimed_source == train_sha
            else f"manifest claims {claimed_source}, train.json is {train_sha}",
        ))
    return results


def _rooms(split):
    """``{(scene, room): set(basenames)}`` for a parsed split file."""
    return {(scene, room): set(files)
            for scene, rooms in split.items() for room, files in rooms.items()}


def check_split_contents(data_dir):
    """Recompute, from the committed split files alone, everything the manifest claims.

    Nothing here trusts the manifest's own numbers: the sizes, the histograms, the nesting
    and the absence of starved targets are recomputed with the round-A primitives and then
    compared against what the manifest says.
    """
    from src.tools.make_ar_train_subsets import context_histogram

    manifest_name, _ = _manifest_names()
    try:
        train = _rooms(_load(os.path.join(data_dir, "train.json")))
        manifest = _load(os.path.join(data_dir, manifest_name))
        loaded = {tag: _rooms(_load(os.path.join(data_dir, _split_name(tag))))
                  for tag in FRACTION_TAGS}
    except (OSError, ValueError, AttributeError) as err:
        return [CheckResult("split contents", False,
                            f"cannot read the split artifacts ({type(err).__name__}: {err})")]

    results = []
    escapes = []
    for tag, rooms in loaded.items():
        for key, files in rooms.items():
            if key not in train:
                escapes.append(f"{tag}:{key[0]}/{key[1]} (room absent from train.json)")
            else:
                escapes += [f"{tag}:{key[1]}/{f}" for f in sorted(files - train[key])]
    results.append(CheckResult(
        "splits are subsets of train.json", not escapes,
        f"{sum(len(f) for r in loaded.values() for f in r.values())} retained entries, all "
        f"in train.json ({sum(len(f) for f in train.values())} entries)" if not escapes
        else f"{len(escapes)} entry/entries outside train.json: {escapes[:5]}",
    ))

    breaks = []
    for smaller, larger in zip(FRACTION_TAGS, FRACTION_TAGS[1:]):
        for key, files in loaded[smaller].items():
            missing = sorted(files - loaded[larger].get(key, set()))
            breaks += [f"{smaller}⊄{larger} {key[1]}/{f}" for f in missing]
    results.append(CheckResult(
        "splits are nested", not breaks,
        " ⊆ ".join(FRACTION_TAGS) + " holds entry-wise" if not breaks
        else f"{len(breaks)} entry/entries lost by a larger fraction: {breaks[:5]}",
    ))

    histograms = {tag: {key: context_histogram(files) for key, files in rooms.items()}
                  for tag, rooms in loaded.items()}
    starved = {tag: sum(h["0"] for h in per_room.values())
               for tag, per_room in histograms.items()}
    results.append(CheckResult(
        "no starved targets", not any(starved.values()),
        f"0 targets with an empty context pool in {'/'.join(FRACTION_TAGS)}"
        if not any(starved.values()) else f"starved targets per fraction: {starved}",
    ))

    size_problems, hist_problems = [], []
    for tag in FRACTION_TAGS:
        entry = manifest.get("fractions", {}).get(str(FRACTIONS[tag]), {})
        total = sum(len(files) for files in loaded[tag].values())
        if entry.get("final") != total:
            size_problems.append(f"{tag}: {total} on disk vs {entry.get('final')} claimed")
        recomputed = {"0": 0, "1-7": 0, ">=8": 0}
        for per_room in histograms[tag].values():
            for key, value in per_room.items():
                recomputed[key] += value
        if entry.get("context_histogram") != recomputed:
            hist_problems.append(f"{tag}: {recomputed} vs {entry.get('context_histogram')}")
        for (scene, room), files in loaded[tag].items():
            claimed = entry.get("per_room", {}).get(room, {})
            if claimed.get("final") != len(files):
                size_problems.append(f"{tag}/{room}: {len(files)} vs {claimed.get('final')}")
            if claimed.get("context_histogram") != histograms[tag][(scene, room)]:
                hist_problems.append(f"{tag}/{room}")
    results.append(CheckResult(
        "split sizes match the manifest", not size_problems,
        ", ".join(f"{tag}={sum(len(f) for f in loaded[tag].values())}"
                  for tag in FRACTION_TAGS) + f" over {len(loaded[FRACTION_TAGS[0]])} rooms"
        if not size_problems else f"{len(size_problems)} mismatch(es): {size_problems[:5]}",
    ))
    results.append(CheckResult(
        "eligible-context histograms match the manifest", not hist_problems,
        "; ".join(
            f"{tag} [0/1-7/>=8]=" + "/".join(
                str(manifest["fractions"][str(FRACTIONS[tag])]["context_histogram"][b])
                for b in ("0", "1-7", ">=8"))
            for tag in FRACTION_TAGS) if not hist_problems
        else f"{len(hist_problems)} mismatch(es): {hist_problems[:5]}",
    ))
    return results


# ======================================================================================
# The receiver-level anchor audit (plan §3 D3)
# ======================================================================================
def check_anchor_audit(dataset_root, train_json_path, folder_name="single_channel_ir_1"):
    """No RIR outside ``train.json`` sits at a receiver ``train.json`` trains on.

    This is what makes exp07_P1@40k and exp09_cylNoSSL@40k exact 100 % anchors under the
    restricted-context definition: FLAC's sampler fixes the context receiver to the
    target's receiver, so if every out-of-split file lives at a receiver the split never
    uses, the historical unrestricted runs could not have drawn one. Re-run here, on the
    machine that trains, because the whole curve is anchored on it.
    """
    from src.tools.make_ar_train_subsets import parse_nodes

    try:
        split = _load(train_json_path)
    except (OSError, ValueError) as err:
        return CheckResult("receiver-level anchor audit", False,
                           f"{train_json_path} unusable ({type(err).__name__}: {err})")

    rooms = listed_total = extra_total = 0
    receivers = set()
    violations, missing = [], []
    for scene in sorted(split):
        for room in sorted(split[scene]):
            rooms += 1
            room_dir = os.path.join(dataset_root, folder_name, scene, room)
            try:
                on_disk = {e for e in os.listdir(room_dir) if e.endswith(".wav")}
            except OSError as err:
                return CheckResult("receiver-level anchor audit", False,
                                   f"{room_dir} cannot be listed ({err})")
            listed = set(split[scene][room])
            listed_total += len(listed)
            room_receivers = set()
            for fname in listed:
                room_receivers.add(parse_nodes(fname)[1])
            receivers |= {(room, rec) for rec in room_receivers}
            missing += [f"{scene}/{room}/{f}" for f in sorted(listed - on_disk)]
            for fname in sorted(on_disk - listed):
                extra_total += 1
                try:
                    receiver = parse_nodes(fname)[1]
                except ValueError:
                    violations.append(f"{scene}/{room}/{fname} (malformed basename)")
                    continue
                if receiver in room_receivers:
                    violations.append(f"{scene}/{room}/{fname} at training receiver {receiver}")

    ok = not violations and not missing
    detail = (f"{rooms} rooms, {listed_total} listed files, {len(receivers)} training "
              f"receivers, {extra_total} extra on-disk files, {len(violations)} at a "
              f"training receiver")
    if missing:
        detail += f"; {len(missing)} listed file(s) absent on disk: {missing[:3]}"
    if violations:
        detail += f": {violations[:5]}"
    return CheckResult("receiver-level anchor audit", ok, detail)


# ======================================================================================
# The whole contract, and the CLI the launcher runs
# ======================================================================================
def _guarded(label, fn, *args, **kwargs):
    """Run one check; an exception becomes a FAIL line, never a traceback.

    The verifier stands between a mistake and ~4.5 GPU-days, so it must produce a verdict
    for every check even when a check itself breaks.
    """
    try:
        result = fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001 -- fail-closed by design
        return [CheckResult(label, False, f"check raised {type(err).__name__}: {err}")]
    # CheckResult is itself a tuple, so it is tested for before any sequence.
    return [result] if isinstance(result, CheckResult) else list(result)


def run_all(flac_wt, kit_dir, pkg_dir, expect_flac_sha, expect_pkg_sha, data_dir=None,
            dataset_root=None, train_json=None, expected_arm_shas=None):
    """Every check of the launch contract, in report order. ``dataset_root=None`` skips
    only the anchor audit (the one check that needs the RIR tree on disk)."""
    data_dir = data_dir or os.path.join(flac_wt, "data", "AR")
    train_json = train_json or os.path.join(data_dir, "train.json")
    results = []
    results += _guarded("arm configs", check_arm_configs, kit_dir,
                        expected_shas=expected_arm_shas)
    results += _guarded("dataset configs", check_dataset_configs, flac_wt)
    results += _guarded("split checksums", check_split_checksums, data_dir)
    results += _guarded("split contents", check_split_contents, data_dir)
    results += _guarded("pinned files", check_pinned_files, flac_wt)
    if dataset_root:
        results += _guarded("receiver-level anchor audit", check_anchor_audit,
                            dataset_root, train_json)
    results += _guarded("FLAC worktree HEAD", check_git_head, flac_wt, expect_flac_sha,
                        "FLAC worktree")
    results += _guarded("package HEAD", check_git_head, pkg_dir, expect_pkg_sha, "package")
    return results


def report(results):
    """Print one PASS/FAIL line per check plus a summary; return the process exit code."""
    for result in results:
        print(result.line())
    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks PASSED"
          + (f"; FAILED: {[r.name for r in failed]}" if failed else ""))
    return EXIT_FAILED if failed else EXIT_OK


def main(argv=None):
    import argparse

    default_wt = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))  # data_curve/ -> tools/ -> src/ -> wt
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.data_curve.verify",
        description="Fail-closed launch contract of exp_14 (exit 3 if any check fails).")
    parser.add_argument("--expect-flac-sha", required=True,
                        help="FLAC worktree HEAD recorded in the launch record")
    parser.add_argument("--expect-pkg-sha", required=True,
                        help="cylindrical-dinov3 package HEAD recorded in the launch record")
    parser.add_argument("--pkg-dir", required=True,
                        help="the package worktree reached via PYTHONPATH")
    parser.add_argument("--kit-dir", required=True,
                        help="the experiment kit (its configs/ holds the two arm configs)")
    parser.add_argument("--flac-wt", default=default_wt)
    parser.add_argument("--data-dir", default=None, help="default: <flac-wt>/data/AR")
    parser.add_argument("--dataset-root", default=None,
                        help="default: <flac-wt>/AcousticRooms (the RIR tree, read-only)")
    parser.add_argument("--skip-anchor-audit", action="store_true",
                        help="skip the only check that walks the RIR tree")
    args = parser.parse_args(argv)

    dataset_root = args.dataset_root or os.path.join(args.flac_wt, "AcousticRooms")
    return report(run_all(
        flac_wt=args.flac_wt, kit_dir=args.kit_dir, pkg_dir=args.pkg_dir,
        expect_flac_sha=args.expect_flac_sha, expect_pkg_sha=args.expect_pkg_sha,
        data_dir=args.data_dir,
        dataset_root=None if args.skip_anchor_audit else dataset_root,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
