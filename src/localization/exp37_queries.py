"""exp_37: ONE query-versus-selection rule for every source (Codex code review 2, HIGH 4).

A FULL-grid query record must be this run's record of the selection's query, with the selection's candidate count
and candidate hash and no parent / subset fields. A SUBSET record (a nested-1.0 m or c2f pass, or the replay / accuracy
run of one) must carry the expected ACTIVE candidate hash and count, and the selection's candidate hash as its PARENT.
The selection record must itself carry a candidate hash. Used by every loader: full / nested / c2f timing passes,
accuracy sources and their replays, and the offline readout. No heavy imports.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""

from __future__ import annotations

import math


def finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def query_reasons(query: dict, record: dict, *, run_sha256: str, active_sha256: str | None = None,
                  count: int | None = None) -> list[str]:
    bad = []
    if query.get("run_manifest_sha256") != run_sha256:
        bad.append("not this run's record")
    if query.get("query_index") != int(record["index"]) or query.get("query_id") != record["query_id"]:
        bad.append("another query than the selection's")
    selection_hash = record.get("candidate_indices_sha256")
    if not selection_hash:
        bad.append("the selection record carries no candidate hash")
    if active_sha256 is None:
        if query.get("candidate_count") != int(record["candidate_count"]) or \
                query.get("candidate_indices_sha256") != selection_hash:
            bad.append("not the selection's full candidate set (count / candidate hash)")
        if "parent_candidate_indices_sha256" in query or "candidate_subset" in query or "c2f" in query:
            bad.append("a full-grid record names a parent / subset")
    else:
        if query.get("candidate_indices_sha256") != active_sha256 or (count is not None and query.get(
                "candidate_count") != int(count)):
            bad.append("not the expected subset of candidates (active hash / count)")
        if query.get("parent_candidate_indices_sha256") != selection_hash:
            bad.append("its parent candidate set is not the selection's")
    return bad


def check_query(query: dict, record: dict, *, label: str = "", **kw) -> None:
    bad = query_reasons(query, record, **kw)
    if bad:
        raise RuntimeError(f"{label}query {record['index']}: {bad}")
