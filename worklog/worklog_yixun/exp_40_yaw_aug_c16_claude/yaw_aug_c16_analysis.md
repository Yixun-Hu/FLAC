# exp_40 yaw_aug_c16 — analysis (Planner, 2026-11-12)

## Outcome

1. **A 16-angle discrete yaw augmentation is as flat in yaw as the continuous one, also between its training angles.** C16's T60 error stays within 9.69–9.80 % (K = 1) and 8.22–8.33 % (K = 8) over all 12 tested headings (θ = 0, the C4 angles, the 45° grid, the four half-step angles) and under uniform random yaw (9.758 / 8.271 %, vs 9.742 / 8.249 % at θ = 0). The hypothesis that a model trained on 16 orientations degrades halfway between them is not supported: the registered off-grid readout is "within YAW's C4 spread" for 13 of 14 (K, metric) pairs.
2. **C16 is not the same model as YAW at θ = 0.** T60 error is higher by 0.35 (K = 1) and 0.28 (K = 8) points; EDT is lower by 0.25 / 0.37 ms; R@1 is higher by 0.15 / 0.22 points; C50 and FD are equal to the third decimal or nearly so. The same offsets hold at every angle and under random yaw. The comparison is mixed, not a win for either.
3. **Both augmented arms beat vanilla P1 under rotation by a wide margin.** Under random yaw P1's T60 error rises by 0.75 points and its R@1 falls by 0.7 points; C16 and YAW move by at most 0.03 T60 points.

## Reliability

- **Evaluation integrity: high.** Every cell passed the driver's gates (clean eval worktree at the pin, sealed HF hub, AR re-hash receipt under 48 h, admission, per-cell receipt); the E0 replay reproduced three registered YAW cells EQUAL at the pin after the move to the local AcousticRooms copy; all 85 receipts re-verified after the run. The 92 comparator cells are read from the registry (sha `7dec3d76…`), not re-run.
- **One trained model per arm.** The SDs in the tables are over evaluation seeds only. Against the eval-seed SD, the C16 − YAW gaps in item 2 are large for T60 (≈ 7× at K = 1, ≈ 19× at K = 8), moderate for EDT and R@1 at K = 8 (≈ 4× and ≈ 3×), and about one SD or less for EDT and R@1 at K = 1. So the T60 gap is a real difference between these two checkpoints, but with a single training run per arm it cannot be attributed to the discretisation itself; training-seed variance is not measured. This is the main limit on item 2.
- **The off-grid and 45° blocks are one evaluation seed.** The registered margin is descriptive, not a test of flatness.
- **The one flagged pair (K = 8, R@10).** C16's own off-grid change is −0.087 points, inside S = 0.237 and below the eval-seed SD of R@10 (0.10–0.19 at K = 8). The flag comes from YAW's R@10 rising by 0.189 points off-grid at this seed, which makes P = −0.276. It is reported as registered; it is most likely seed noise in a seven-metric, two-K family with no multiplicity control, and no other metric at either K shows it.
- **Protocol parity.** C16 differs from YAW in one config key (`training.yaw_aug.offset_step = 32`); same recipe (2 ranks × 32, A6000), same code path (golden test: absent key = byte-identical draws). C16 trained from the local AcousticRooms copy whose content re-hash equals the committed manifest. The end-of-training audit first failed for two audit bugs (scheduled lr; seven never-gradient parameters), fixed at `2b5f4a35` without changing the checkpoint; the re-audit passed.

## Use

- For the rebuttal: C16 supports the statement that augmentation over a modest finite set of headings already gives heading robustness at every angle, at FLAC's inference cost. It does not change the OrbitRIR-vs-Yaw-Aug comparison; it adds a second augmentation baseline with the same pattern.
- Not done: a second training seed for C16 or YAW; C16 on the localization task; the SOP's HTML results page; the `model_comparison.md` regeneration (neuronic only).
