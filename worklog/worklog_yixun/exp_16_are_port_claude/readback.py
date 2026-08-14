#!/usr/bin/env python3
"""exp_16 (are_port): the arm's PINNED schedule, and the post-run artifact verdict.

Seat: Opus 5 Coder (SOP §Roles). Added in the r1 code-review round to close
finding 1 ("FULL is not pinned to 40,000 steps").

WHY THIS IS A MODULE AND NOT SHELL. Round 1 expressed the endpoint as a shell
DEFAULT (``DEFAULT_MAXSTEPS=40000``) that ``MAXSTEPS=1000`` silently replaced, and
expressed the readback as an inline heredoc that no test could reach. Both defects
have the same shape: a rule that only exists inside a shell string is a rule
nobody can exercise. So the endpoint, the cadence and the readback verdict live
here, are imported by ``are_launch.sh``, and are covered by
``src/tests/test_are_launch_schedule.py``.

THE PIN. exp_16's comparison is ARE-V@40,000 against P1@40,000. A run that stops
anywhere else is not that arm, and a checkpoint series written on a different
cadence is not comparable to the 2,500-step screens the plan pre-registers
(AR1b). FULL and RESTART are therefore pinned to **exactly** 40,000 steps and a
**2,500**-step cadence, and the post-run readback requires the artifact to prove
``global_step == 40000``. PROBE is deliberately free: it is a fit probe in its own
namespace and its whole purpose is to be short.
"""
import json
import os

# The pre-registered endpoint and cadence. Not defaults -- requirements.
ENDPOINT_STEPS = 40000
PRODUCTION_CHECKPOINT_EVERY = 2500

PROBE_DEFAULT_STEPS = 15
PROBE_DEFAULT_CHECKPOINT_EVERY = 5

MODES = ("PROBE", "FULL", "RESTART")
PINNED_MODES = ("FULL", "RESTART")

LAM_KEY = "are_lambda"
ANCHOR_KEY = "are_anchor"


def pinned_schedule(mode):
    """``(max_steps, checkpoint_every)`` a mode is REQUIRED to run, or ``None``.

    ``None`` means "free" and is returned only for PROBE.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if mode in PINNED_MODES:
        return ENDPOINT_STEPS, PRODUCTION_CHECKPOINT_EVERY
    return None


def schedule_problems(mode, max_steps, checkpoint_every):
    """Everything wrong with a requested schedule, as operator-facing strings.

    Empty list == admissible. Values are checked as INTEGERS the caller has
    already parsed; the shell validates the shape, this validates the policy.
    """
    problems = []
    pinned = pinned_schedule(mode)
    if pinned is None:
        return problems
    want_steps, want_every = pinned
    if max_steps != want_steps:
        problems.append(
            f"MODE={mode} is pinned to the pre-registered endpoint MAXSTEPS={want_steps} "
            f"(got {max_steps}): exp_16 compares ARE-V@{want_steps} against P1@{want_steps}, "
            "so a run that stops anywhere else is not this arm. PROBE is the mode for "
            "short runs.")
    if checkpoint_every != want_every:
        problems.append(
            f"MODE={mode} is pinned to CHECKPOINT_EVERY={want_every} (got {checkpoint_every}): "
            "the plan's trajectory readout (AR1b) screens every 2,500 steps against P1's "
            "same-step screens, and a different cadence produces an incomparable series.")
    return problems


def type_strict_diff(a, b, path="model_config", label_a="the first",
                     label_b="the second"):
    """First TYPE-STRICT difference between two parsed-JSON objects, or ``None``.

    ``1 == 1.0 == True`` in Python, so plain equality cannot tell a float lambda
    from an int or boolean one -- exactly the values the factory rejects. Types
    are compared before values, everywhere, recursively.

    ONE implementation, three call sites (the launcher's resume gate, the post-run
    readback, and eval's checkpoint binding); ``label_a``/``label_b`` only name the
    two operands in the message, so the rule itself cannot drift between them.
    """
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"
    if isinstance(a, dict):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        if only_a:
            return f"{path}: key(s) {only_a} present in {label_a}, absent from {label_b}"
        if only_b:
            return f"{path}: key(s) {only_b} present in {label_b}, absent from {label_a}"
        for k in a:
            d = type_strict_diff(a[k], b[k], f"{path}.{k}", label_a, label_b)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = type_strict_diff(x, y, f"{path}[{i}]", label_a, label_b)
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


_type_strict_diff = type_strict_diff


def readback_problems(mode, embedded_model_config, global_step, arm_model_config,
                      want_lambda, expected_step=0, max_steps=None):
    """Everything wrong with the checkpoint THIS invocation wrote. Empty == OK.

    Two questions, both mandatory:

    * **Did the run reach its pre-registered endpoint?** For FULL and RESTART the
      answer must be ``global_step == 40000`` exactly -- an rc=0 exit at any other
      step is a run that stopped early, and admitting it would put a truncated arm
      into the comparison. For PROBE the step only has to lie inside the window
      the invocation asked for.
    * **Did the treatment reach the artifact?** The embedded ``model_config`` must
      carry a typed-float ``are_lambda`` equal to the arm's, a populated
      ``are_anchor`` block, and must equal the arm's JSON type-strictly.
    """
    problems = []
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")

    if mode in PINNED_MODES:
        if global_step != ENDPOINT_STEPS:
            problems.append(
                f"global_step {global_step!r} != the pinned endpoint {ENDPOINT_STEPS}: "
                f"a MODE={mode} run that exits 0 anywhere else stopped early, and a "
                "truncated arm may not enter the comparison")
    else:
        if max_steps is None:
            raise ValueError("PROBE readback needs the invocation's max_steps")
        if not (expected_step < global_step <= max_steps):
            problems.append(
                f"global_step {global_step!r} outside the run window "
                f"({expected_step}, {max_steps}]")

    if not isinstance(embedded_model_config, dict):
        problems.append("checkpoint carries no embedded 'model_config' dict")
        return problems

    training = embedded_model_config.get("training") or {}
    got = training.get(LAM_KEY, "<absent>")
    if isinstance(got, bool) or not isinstance(got, float) or got != want_lambda:
        problems.append(
            f"embedded training.{LAM_KEY} {got!r} != {want_lambda!r} (a typed float) -> "
            "the treatment did NOT reach the artifact")
    anchor = training.get(ANCHOR_KEY, "<absent>")
    if not isinstance(anchor, dict) or not anchor:
        problems.append(
            f"embedded training.{ANCHOR_KEY} {anchor!r} is not a populated object -> the "
            "calibrated constants did NOT reach the artifact")
    diff = _type_strict_diff(embedded_model_config, arm_model_config)
    if diff is not None:
        problems.append(f"embedded model_config != the arm's JSON (type-strict): {diff}")
    return problems


def load_json(path):
    with open(path) as f:
        return json.load(f)


def repo_root(start=None):
    p = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p
