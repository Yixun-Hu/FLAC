"""exp_46 D1 (open item 13 / Codex code r3 finding 3) -- PREPARED, NOT on the campaign branch until Yixun approves E7 + E8.

Both upstream sampler copies (src/tests/test_exp45_sampler.py, E7, and src/tests/test_exp38_sampler.py, E8 -- the same
X45f blob) call LightningEnvironment.set_global_rank(1), which also sets lightning's GLOBAL rank_zero_only.rank and
leaves it at 1 for every later test of the process: a rank-0 banner is then silently suppressed. E7 / E8 insert the
same four lines (test_exp46_provenance.rank_restore) as the first body lines of
test_both_launch_environments_give_the_same_rank_and_world: monkeypatch records the incoming rank on the shared
callable and restores it at teardown.

* test_the_upstream_sampler_followed_by_a_real_banner_assertion -- RED until E7 + E8 land: one pytest session runs the
  upstream sampler node, then the shared-reference probe, then a real banner assertion
  (test_yaw_aug_training.py::test_banner_printed_once_at_fit_start, which pins no rank).
* test_the_four_line_restore_fixes_the_session_on_a_copy -- green now: the same session on a COPY of the X45f file in
  tmp (the frozen files untouched) fails unpatched and passes with exactly the E7 insertion (the teardown regression
  fails without the edit).

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_exp46_provenance as prov  # noqa: E402  (rank_restore, RANK_ANCHOR, X45F: one source)

SAMPLERS = {"E7": "src/tests/test_exp45_sampler.py", "E8": "src/tests/test_exp38_sampler.py"}
NODE = "test_both_launch_environments_give_the_same_rank_and_world"
PROBE = "src/tests/exp46_rank_probe.py"
BANNER = "src/tests/test_yaw_aug_training.py::test_banner_printed_once_at_fit_start"


def _session(*nodes):
    return subprocess.run([sys.executable, "-B", "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", *nodes],
                          cwd=REPO, capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("tag", sorted(SAMPLERS))
def test_the_upstream_sampler_followed_by_a_real_banner_assertion(tag):
    res = _session(f"{SAMPLERS[tag]}::{NODE}", PROBE, BANNER)
    assert res.returncode == 0 and "3 passed" in res.stdout, (tag, res.stdout[-3000:])


def test_the_four_line_restore_fixes_the_session_on_a_copy(tmp_path):
    x45f = prov._show(prov.X45F, SAMPLERS["E7"])
    assert prov._show(prov.X45F, SAMPLERS["E8"]) == x45f                 # the two copies are one X45f blob
    patched = x45f.replace(prov.RANK_ANCHOR, prov.RANK_ANCHOR + prov.rank_restore("E7"), 1)
    assert prov.insertion_problems(x45f, patched, "E7") == []
    for name, text, ok in (("test_unpatched_sampler_copy.py", x45f, False), ("test_patched_sampler_copy.py", patched, True)):
        path = tmp_path / name
        path.write_text(text)
        res = _session(f"{path}::{NODE}", PROBE, BANNER)
        assert (res.returncode == 0) is ok, (name, res.stdout[-3000:])
