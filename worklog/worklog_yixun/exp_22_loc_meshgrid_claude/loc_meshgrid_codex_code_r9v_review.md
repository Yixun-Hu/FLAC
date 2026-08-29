# exp_22 Codex verify — round r9v: REJECT (residuals #1 reporting join, #4 launch-record enforcement)

1. **PARTIALLY** — The matched-path measurement is valid, but v4’s `tie_evidence` mixes retired `0.0066687/8,064` evidence with the matched-path `85.393×`; the Markdown also misattributes the new margin to 8,064 pairs. [source](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1715), [report](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_report.md:62)

2. **RESOLVED** — Every cosine is gated against its own cell tolerance, with round-trip exactness and aggregate exactness independently required; v4 passes all 16. [meshgrid_offgrid_probe.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1079)

3. **RESOLVED** — Missing, empty, failed, partial, and count-inconsistent continuity records now refuse under the named gate. [meshgrid_offgrid_probe.py](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:1596)

4. **PARTIALLY** — V4’s tracked record says `git_status_dirty:true`, yet canonical admission accepts it and does not compare the recorded SHA, host, dirty state, or GPU UUIDs with the executing environment. [launch record](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_v4_launch_record.json:1), [validator](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:246)

5. **RESOLVED** — The required handoff supplies the retrieval run’s canonical status and matching `65d735…` report digest; v4 renders both. [offgrid_probe_report.md](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/offgrid_probe_report.md:85)

**Verdict — REJECT.** Minimal blocking set: **#1 and #4**. V4’s ranks and calibration remain unchanged from v3; rejection is limited to canonical evidence/reporting and launch provenance. Static review only; nothing was modified or executed.