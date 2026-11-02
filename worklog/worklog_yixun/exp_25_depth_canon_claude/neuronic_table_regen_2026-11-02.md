# Regenerating `model_comparison.md` on neuronic — commands for Yixun (written 2026-11-02 by the exp_25 Planner session)

**Why neuronic:** the living table has only ever been regenerated on neuronic (`/n/fs/gatrdp/codespace/FLAC`; commits `84990306` 2026-08-12, `ed048609` 2026-08-18). 23 published rows resolve only there: the 10 `exp_14 Z` rows (raw metric records never force-added to git), the 12 exp_11 orbit rows (`exp11_validate_rows.py` re-hashes the exp_11 checkpoints, which exist only on neuronic), and the `fa scratch @62.5k` K=8 row (its seed-42 screen file was never committed). On the A6000 box the generator therefore refuses (rc 4 row regression, then rc 3 exp_11 both-K gate). The 6 exp_25 rows need nothing local: their evidence (metrics, receipts, stream/canon sidecars) is in git and the exp25 contract runs offline (`dump_check="receipt"`, `check_link=False`) — proven in a fresh clone on 2026-10-31.

**Decision (Yixun, 2026-11-02):** option 1 — regenerate on neuronic; additionally force-add the 10 exp_14 Z cells and the exp_10 62.5k seed-42 screen file so the Z and 62.5k rows can be rebuilt off-cluster. (The 12 exp_11 rows stay neuronic-only: their validator needs the checkpoints themselves.)

Run with the `flac` env active, on `localization-exp`, from a tracked-clean tree. Every step prints what it will touch before touching it.

```bash
cd /n/fs/gatrdp/codespace/FLAC
git status --short | grep -v '^??'              # must print NOTHING (no local edits); stash or commit them first
git fetch origin && git checkout localization-exp && git pull --rebase origin localization-exp
git log --oneline -1                            # expect 4bc772c7 or newer

# --- 1. the 10 exp_14 Z cells = 50 metric records + their .screenmeta.json + .stream.json sidecars
#        (exp14_validate_cell.py reads all three per cell; the checkpoint hash is READ from exp14_ckpt_expect.json, never recomputed)
ls outputs_FLAC/exp11_*/FLAC_exp11_*/exp11_*/checkpoints/*exp14_*_zref_S40000_s4[2-6]_K[18]*.json | wc -l
#        expect 150 (= 5 arms x 2 K x 5 seeds x 3 files); if not 150, stop and tell me what is listed
git add -f outputs_FLAC/exp11_*/FLAC_exp11_*/exp11_*/checkpoints/*exp14_*_zref_S40000_s4[2-6]_K[18]*.json

# --- 2. the exp_10 62.5k seed-42 screen file (the fifth seed of `fa scratch @62.5k`, K=8)
find outputs_FLAC/exp10_BF -name '*exp10_BF_screen_S62500*.json' -not -name '*.stream.json'     # expect exactly 1 file
git add -f $(find outputs_FLAC/exp10_BF -name '*exp10_BF_screen_S62500*.json' -not -name '*.stream.json')

# --- 3. sha256 manifest for the imported evidence (same format as NEURONIC_EXP11_METRICS_SHA256SUMS.txt)
git diff --cached --name-only | sort | xargs sha256sum > worklog/worklog_yixun/NEURONIC_EXP14_Z_AND_EXP10_SCREEN_SHA256SUMS.txt
wc -l worklog/worklog_yixun/NEURONIC_EXP14_Z_AND_EXP10_SCREEN_SHA256SUMS.txt                      # expect 151
git add worklog/worklog_yixun/NEURONIC_EXP14_Z_AND_EXP10_SCREEN_SHA256SUMS.txt
git commit -m "evidence: force-add the 10 exp_14 Z cells (50 records + screenmeta + stream sidecars) and the exp_10 62.5k seed-42 screen file past the outputs_FLAC/ ignore, with a sha256 manifest — so model_comparison.md can be rebuilt off-cluster (Yixun 2026-11-02)"

# --- 4. regenerate the table (CPU only) — expect `wrote .../model_comparison.md` and 91 data rows (85 published + 6 exp_25)
CUDA_VISIBLE_DEVICES="" python worklog/worklog_yixun/gen_model_comparison.py
grep -c '^| ' worklog/worklog_yixun/model_comparison.md        # expect 92 (91 rows + the header row)
grep -n -E 'BLOCKED|\*pending' worklog/worklog_yixun/model_comparison.md   # expect only the pre-existing `fa scratch @67.5k` K=8 pending line
grep -n 'exp_25' worklog/worklog_yixun/model_comparison.md      # expect 6 numeric rows (CANON / P1-pin / YAW-pin, K=1 and K=8)
git add worklog/worklog_yixun/model_comparison.md
git commit -m "model_comparison: exp_25 depth_canon rows (CANON, P1-pin, YAW-pin x K=1/8) — announcement 04 regen on neuronic at 4bc772c7+"
git push origin localization-exp
```

If step 4 refuses instead of writing, paste its reason list to me: the generator names every row that would lose evidence, which tells us exactly what neuronic no longer holds. Do not add `--allow-row-regression` or `--allow-partial-exp11` — those retract rows.
