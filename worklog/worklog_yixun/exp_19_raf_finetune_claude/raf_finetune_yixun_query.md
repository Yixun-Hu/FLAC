# exp_19 raf_finetune — Yixun's driving query

## Query 1 (2026-08-19, second session on mae-cab-lab-server, verbatim)

> plan review go, and I think we will need to run FLAC fintune on RAF dataset, same as that what we did @FLAC_pdf.md on HAA dataset

Context: the "plan review go" half addressed exp_18 and was already executed by the peer session (`localization-exp [453dd7]`: Codex review `e71df84` → Rev 2 → Opus supplementary `20586ad` → Rev 3 → Yixun approval `ab20700`). The RAF half is the new directive and is what exp_19 exists for. Preceded in this session (2026-08-18) by Yixun's question "Is https://github.com/facebookresearch/real-acoustic-fields the RAF datasets been able to run FLAC on?" and this session's feasibility assessment (answer: no support in repo or paper; adaptable in the HAA role).

## Summary

Adapt the RAF dataset (Real Acoustic Fields, facebookresearch — 2 real rooms, dense measured RIRs with 6DoF-tracked source/listener poses, textured OBJ room mesh, multi-view images via Eyeful Tower) into FLAC's pipeline and finetune the AR-pretrained FLAC on it, mirroring the paper's HAA protocol (per-room few-shot finetune, README recipe: `FLAC_HAA_finetune.json`-style config, `--max-steps 1000`), then evaluate with the standard metric suite (T60/C50/EDT/l1/FD/Recall).

## Assumption / hypothesis

RAF plays the HAA role (real-measurement few-shot finetune/eval target), NOT the AR role — with only 2 rooms it cannot serve as a multi-room train set or an unseen-room generalization benchmark. Known adaptation gaps from this session's 2026-08-18 feasibility pass: RAF ships no depth panoramas — they must be rendered from the OBJ mesh (equirectangular 256×512, HAA convention renders at the SOURCE position, note the deliberate sign flip at `HAA_md.py:70`); RIRs need resampling (48 kHz per the RAF paper → 22,050 Hz) and crop/pad to 10,240 samples; new `data/RAF/` split JSONs + `RAF_md.py` metadata module + dataset configs; `data/HAA/prepare_data.py` is the preparation template. Working hypothesis: the HAA finetune recipe transfers to RAF and extends the paper's real-world evidence to a second real dataset.

## Why the experiment needs to run

Extends FLAC's real-world transfer claim beyond HAA to a second, denser real dataset; potentially grounds the localization workstream (exp_18 lineage) in real measured acoustics later (RAF's dense capture grid + tracked poses would make a strong real-data candidate set). Scope questions (sequencing vs exp_18, dataset acquisition route, finetune-only vs localization-oriented eval) surfaced to Yixun before planning.
