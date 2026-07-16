`.claude/hooks/model_change_handoff.py`

- Marker gating: RESOLVED — lines 163–165.
- Nonempty string `session_id`: RESOLVED — lines 109–112.
- Blocking `LOCK_EX`: RESOLVED — line 130.
- Model/type guards: REMAINING — line 75 calls `.get()` on a potentially truthy non-dict `message`.
- Verdict: REQUEST-CHANGES

`p1a_fit_probe.sh`

- OOM regex: RESOLVED — line 71.
- NaN/Inf regex: RESOLVED — line 72.
- Verdict: SHIP