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
FRACTION_TAGS = ("025", "050", "075")
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
