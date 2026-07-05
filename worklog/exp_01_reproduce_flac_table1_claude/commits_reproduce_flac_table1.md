# Commits — exp_01_reproduce_flac_table1

Base: `0bd5da0` ("add FLAC and Frame papers") — the revert point all development builds from.

| SHA | Description | Code lines |
|---|---|---|
| `ef55438` | worklog: add experiment SOP and announcement 01 (SOP.md, announcement/01, .gitignore log rule) | 0 (docs + 1-line .gitignore) |
| `aa6b399` | worklog: archive pre-revert equivariance-probe code (frozen patch + 2 scripts; not live code) | 0 live (1146 archived) |
| `328283f` | exp_01: reproduce FLAC Table 1 K=1/K=8 (G) on unseen AR — PASS (plan, query, params, command, run script, logs, 10 metrics JSONs, results, analysis, review-N/A) | 24 (run_exp01.sh) |
| *(child of `328283f`)* | exp_01: log commit SHAs (this file; a commit cannot contain its own SHA) | 0 |

Note: `CLAUDE.md` (SOP binding section) is gitignored in this repo by upstream convention and remains local-only; the committed `worklog/SOP.md` is the canonical transferable SOP.
