"""exp_37 round B-a item 4: the UNINSTRUMENTED headline timer mode (plan_loc_efficiency.md §10, Codex BLOCKING 1).

The B1 headline ``core_forward_total`` must come from a pass whose timed region is the latency bundle's own --
no CUDA events, no allocator resets, no extra syncs inside the ``perf_counter`` windows. This guard pins, against
the bundle's reference copy (``NeurIPS_2026_Workshop_Latency_Reproducibility/code/reference``):

  1. the timed region of ``runner.execute_query`` (the first ``measure_core_forward`` sync through the candidate
     batch loop), statement by statement (``ast.dump``: formatting-blind, semantics-exact);
  2. every function the timed region executes in ``runner`` and ``engine``;
  3. the model / sampler / rotation / AGREE-scoring files, byte for byte.

Any exp_37 instrumentation (the B-b stage split, the B-c rotation variant) lives OUTSIDE these definitions. Round
B-c adds exactly TWO documented plumbing deltas (no work, no instrumentation), undone before the comparison:
  (1) ``frame_angles``: a keyword-only parameter of ``_cache_branch`` (default ``DEFAULT_FRAME_ANGLES``) passed by
      the three timed ``_cache_branch`` calls (B3 orbit sizes);
  (2) ``candidate_indices[batch_start:batch_stop]`` for ``range(batch_start, batch_stop)`` in the seed loop
      (``candidate_indices`` defaults to ``range(len(candidates))``, whose slice IS that range; a nested-1.0 m
      subset passes its full 0.5 m indices -- common random numbers).
Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-a; deltas registered in round B-c.
"""
import ast
import copy
import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility" / "code" / "reference"
TIMED_FUNCTIONS = {
    "src/localization/runner.py": ("_cache_branch", "_repeat_branch"),
    "src/localization/engine.py": ("cache_conditioning_branch", "cache_invariant_conditioning_branch",
                                   "_expand_cached", "merge_cached_conditioning", "encode_audio_features",
                                   "prepare_generated_audio", "candidate_seed", "deterministic_noise",
                                   "candidate_metadata", "generate_rir_batch", "generate_and_score_batch"),
}
TIMED_FILES = ("src/models/conditioners.py", "src/models/dit.py", "src/models/diffusion.py",
               "src/models/transformer.py", "src/models/pretransforms.py", "src/models/autoencoders.py",
               "src/models/blocks.py", "src/models/bottleneck.py", "src/models/utils.py",
               "src/inference/sampling.py", "src/data/yaw_rotation.py", "src/metrics/modules/Retrieval.py",
               "src/configs/dataset_configs/custom_metadata/AR_md.py")
INSTRUMENTATION = ("Event", "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved",
                   "elapsed_time", "record_function", "profiler", "nvtx")


def _functions(path: Path) -> dict:
    tree = ast.parse(path.read_text(), str(path))
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class _UndoDocumentedDeltas(ast.NodeTransformer):
    """Undo exactly the two registered plumbing deltas (and nothing else)."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "_cache_branch":
            node.keywords = [k for k in node.keywords if not (k.arg == "frame_angles" and isinstance(k.value, ast.Name)
                                                              and k.value.id == "frame_angles")]
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "candidate_indices"
                and ast.unparse(node.slice) == "batch_start:batch_stop"):
            return ast.Call(func=ast.Name(id="range", ctx=ast.Load()), keywords=[],
                            args=[ast.Name(id="batch_start", ctx=ast.Load()), ast.Name(id="batch_stop", ctx=ast.Load())])
        return node


def undo_deltas(node):
    node = copy.deepcopy(node)
    if isinstance(node, ast.FunctionDef) and node.name == "_cache_branch":
        kw = [a.arg for a in node.args.kwonlyargs]
        if kw == ["frame_angles"]:
            node.args.kwonlyargs, node.args.kw_defaults = [], []
            for name in ast.walk(node):
                if isinstance(name, ast.Name) and name.id == "frame_angles":
                    name.id = "DEFAULT_FRAME_ANGLES"
    return _UndoDocumentedDeltas().visit(node)


def timed_region(function: ast.FunctionDef) -> list:
    """``execute_query``'s statements from the first ``measure_core_forward`` sync through the batch loop."""
    body = function.body
    start = next(i for i, s in enumerate(body) if isinstance(s, ast.If)
                 and ast.unparse(s.test) == "measure_core_forward and device_type == 'cuda'")
    stop = next(i for i, s in enumerate(body) if isinstance(s, ast.For) and ast.unparse(s.target) == "batch_start")
    assert start < stop
    return body[start:stop + 1]


def test_the_reference_bundle_is_present():
    assert (REFERENCE / "src" / "localization" / "runner.py").is_file()


def test_the_execute_query_timed_region_is_the_bundle_s():
    ours = timed_region(_functions(REPO / "src/localization/runner.py")["execute_query"])
    theirs = timed_region(_functions(REFERENCE / "src/localization/runner.py")["execute_query"])
    assert [ast.dump(undo_deltas(s)) for s in ours] == [ast.dump(s) for s in theirs]
    text = "\n".join(ast.unparse(s) for s in ours)
    assert "context_started = time.perf_counter()" in text and "candidate_generation_seconds +=" in text
    assert not [word for word in INSTRUMENTATION if word in text]


@pytest.mark.parametrize("relative, name", [(r, n) for r, names in TIMED_FUNCTIONS.items() for n in names])
def test_every_function_the_timed_region_executes_is_the_bundle_s(relative, name):
    ours = _functions(REPO / relative)[name]
    theirs = _functions(REFERENCE / relative)[name]
    assert ast.dump(undo_deltas(ours)) == ast.dump(theirs), f"{relative}::{name} differs from the latency bundle"


@pytest.mark.parametrize("relative", TIMED_FILES)
def test_the_timed_model_stack_is_byte_identical(relative):
    digest = [hashlib.sha256((root / relative).read_bytes()).hexdigest() for root in (REPO, REFERENCE)]
    assert digest[0] == digest[1], relative


def test_the_guard_bites_on_an_event_inside_the_timer():
    source = (REPO / "src/localization/runner.py").read_text().replace(
        "    context_started = time.perf_counter()\n",
        "    context_started = time.perf_counter()\n    torch.cuda.Event(enable_timing=True).record()\n", 1)
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_query")
    theirs = timed_region(_functions(REFERENCE / "src/localization/runner.py")["execute_query"])
    assert [ast.dump(s) for s in timed_region(function)] != [ast.dump(s) for s in theirs]


def test_the_two_documented_deltas_are_present_and_nothing_else_differs():
    """Round B-c: the timed region passes frame_angles to every _cache_branch call and iterates candidate_indices;
    undoing exactly those yields the bundle (the comparisons above); a third, undocumented change would not."""
    region = timed_region(_functions(REPO / "src/localization/runner.py")["execute_query"])
    calls = [n for s in region for n in ast.walk(s) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_cache_branch"]
    assert len(calls) == 3 and all([k.arg for k in c.keywords] == ["frame_angles"] for c in calls)
    text = "\n".join(ast.unparse(s) for s in region)
    assert "candidate_indices[batch_start:batch_stop]" in text and "range(batch_start, batch_stop)" not in text
    branch = _functions(REPO / "src/localization/runner.py")["_cache_branch"]
    assert [a.arg for a in branch.args.kwonlyargs] == ["frame_angles"]
    assert ast.unparse(branch.args.kw_defaults[0]) == "DEFAULT_FRAME_ANGLES"
