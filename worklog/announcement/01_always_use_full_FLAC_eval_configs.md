# 01 — Always compare against FLAC on FLAC's full data configurations

**Logged:** 2026-07-04 (from Yixun)

## Original instruction (verbatim)

> always compare our method with FLAC using all FLAC's data configuration @src/configs/dataset_configs/. For example, if we are going to eval our method and FLAC on unseen environment, we have 6337 items in 17 unseen room @data/AR/unseen_eval.json. Do not create new data eval configuration (like using a small amount of selected IRs for comparison)

## Rule

- Every comparison between our method and FLAC must run on the **existing, complete** dataset configurations under `src/configs/dataset_configs/` (AR and HAA), with their referenced split files under `data/`.
- Unseen-environment evaluation means the **full** `data/AR/unseen_eval.json` split: **6337 items in 17 unseen rooms** (10 scene types) — verified on 2026-07-04.
- **Do NOT create new eval dataset configurations** — no subsampled item lists, no hand-selected IR subsets, no reduced-room splits "for quick comparison". Headline numbers reported in any `_results.md` must come from the full split.
- Changing K (context size) is done only by switching between the existing `acousticroom_*eval_<K>.json` configs, never by authoring new ones.

## Why

Subsampled or custom splits produce numbers that are not comparable to FLAC's paper Table 1 (which averages per-scene over the full unseen split) and invite silent selection bias. All experiment conclusions must be apples-to-apples against the published FLAC evaluation protocol.
