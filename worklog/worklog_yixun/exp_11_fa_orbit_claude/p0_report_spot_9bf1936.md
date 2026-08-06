# exp_11 P0 profiling — measured throughput and peak VRAM

Run `9bf1936-1786033425104073952-d8d84328` · mode `spot` · commit `9bf193697ac2` — 2 expected row(s).

Steady-state rate measured INSIDE one fit (steps/s = 20 / (t30 − t10) from the runner's callback marks), so no startup, rendezvous or teardown time is included. Peak VRAM is the whole-run poller peak; every OK row's poller artifact is hash-verified.

| Cell | rung MBxN | workers | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |
|---|---|---|---|---|---|---|---|---|
| C16_8x8 | 8x8 | 6 | 0.0982 | 10.18 | 16605 | 16605 | OK |  |
| C32_8x8 | 8x8 | 6 | 0.0518 | 19.32 | 30817 | 30817 | OK |  |

## GPU utilisation / power over the step-10 → step-30 window

| Cell | workers | GPU UUID | ticks | mem max MiB | util mean % | power mean W |
|---|---|---|---|---|---|---|
| C16_8x8 | 6 | GPU-241439ca-6a93-19d5-be17-c038969570a3 | 286 | 16605 | 30.1 | 143.4 |
| C16_8x8 | 6 | GPU-2504a508-16eb-e0e6-0745-5bee7be03047 | 286 | 16601 | 31.7 | 143.9 |
| C16_8x8 | 6 | GPU-5d3b4c5d-a85d-c5fa-9041-0447f2ad7793 | 286 | 16587 | 32.2 | 141.3 |
| C16_8x8 | 6 | GPU-6d9f8e33-442d-0eae-968b-19d47848844c | 286 | 16605 | 31.9 | 149.1 |
| C16_8x8 | 6 | GPU-81092ef3-7804-eed7-e11e-d467130d17b6 | 286 | 16601 | 27.9 | 148.3 |
| C16_8x8 | 6 | GPU-d1dc6b04-d070-b672-e156-234247ad05fe | 286 | 16605 | 31.4 | 148.4 |
| C16_8x8 | 6 | GPU-d3013b2f-a166-87d6-3028-07edb77ed4e6 | 286 | 16585 | 30.7 | 141.1 |
| C16_8x8 | 6 | GPU-d9754f01-4a40-1253-b319-1cee2fdc033d | 286 | 16605 | 31.9 | 144.8 |
| C32_8x8 | 6 | GPU-241439ca-6a93-19d5-be17-c038969570a3 | 383 | 30817 | 28.7 | 145.9 |
| C32_8x8 | 6 | GPU-2504a508-16eb-e0e6-0745-5bee7be03047 | 383 | 30813 | 30.0 | 147.6 |
| C32_8x8 | 6 | GPU-5d3b4c5d-a85d-c5fa-9041-0447f2ad7793 | 383 | 30797 | 29.9 | 145.1 |
| C32_8x8 | 6 | GPU-6d9f8e33-442d-0eae-968b-19d47848844c | 383 | 30817 | 28.6 | 152.9 |
| C32_8x8 | 6 | GPU-81092ef3-7804-eed7-e11e-d467130d17b6 | 383 | 30813 | 28.4 | 153.1 |
| C32_8x8 | 6 | GPU-d1dc6b04-d070-b672-e156-234247ad05fe | 383 | 30817 | 30.5 | 151.3 |
| C32_8x8 | 6 | GPU-d3013b2f-a166-87d6-3028-07edb77ed4e6 | 383 | 30797 | 29.6 | 144.7 |
| C32_8x8 | 6 | GPU-d9754f01-4a40-1253-b319-1cee2fdc033d | 383 | 30817 | 29.1 | 149.1 |

## Derived attribution — not applicable to a `spot` run

A `spot` manifest does not carry the FA1+C4L+C8 set; its cells are reported above and enter the matrix analysis only through their own run.

## Rejected rows (not from this run)

- `C4L_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_32x2`: runid smoke-8d53691-1785968197132-99b9c11b is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C4L_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `C8_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `CKPT4_16x4`: runid supp-09d41ca-1785975836736-02cd7f4b is not this run (9bf1936-1786033425104073952-d8d84328)
- `CKPT4_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_32x2`: runid smoke2-8d53691-1785968294781-96a1b43a is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `FA1_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_16x4`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (9bf1936-1786033425104073952-d8d84328)
- `VAN_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (9bf1936-1786033425104073952-d8d84328)

## Source files

- `slurm_p0_p0-C16_8x8-w6_3646146.out`
- `slurm_p0_p0-C32_8x8-w6_3646147.out`
- `slurm_p0_p0-C4L_16x4-w6_3638643.out`
- `slurm_p0_p0-C4L_16x4-w6_3638698.out`
- `slurm_p0_p0-C4L_16x4-w6_3639675.out`
- `slurm_p0_p0-C4L_16x4-w6_3646048.out`
- `slurm_p0_p0-C4L_32x2-w6_3638639.out`
- `slurm_p0_p0-C4L_32x2-w6_3638694.out`
- `slurm_p0_p0-C4L_32x2-w6_3639671.out`
- `slurm_p0_p0-C4L_32x2-w6_3646044.out`
- `slurm_p0_p0-C4L_8x8-w6_3638647.out`
- `slurm_p0_p0-C4L_8x8-w6_3638702.out`
- `slurm_p0_p0-C4L_8x8-w6_3639679.out`
- `slurm_p0_p0-C4L_8x8-w6_3646052.out`
- `slurm_p0_p0-C8_16x4-w6_3638644.out`
- `slurm_p0_p0-C8_16x4-w6_3638699.out`
- `slurm_p0_p0-C8_16x4-w6_3639676.out`
- `slurm_p0_p0-C8_16x4-w6_3646049.out`
- `slurm_p0_p0-C8_32x2-w6_3638640.out`
- `slurm_p0_p0-C8_32x2-w6_3638695.out`
- `slurm_p0_p0-C8_32x2-w6_3639672.out`
- `slurm_p0_p0-C8_32x2-w6_3646045.out`
- `slurm_p0_p0-C8_8x8-w6_3638648.out`
- `slurm_p0_p0-C8_8x8-w6_3638703.out`
- `slurm_p0_p0-C8_8x8-w6_3639680.out`
- `slurm_p0_p0-C8_8x8-w6_3646053.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638649.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638704.out`
- `slurm_p0_p0-FA1_16x4-w6_3638642.out`
- `slurm_p0_p0-FA1_16x4-w6_3638697.out`
- `slurm_p0_p0-FA1_16x4-w6_3639674.out`
- `slurm_p0_p0-FA1_16x4-w6_3646047.out`
- `slurm_p0_p0-FA1_32x2-w6_3638638.out`
- `slurm_p0_p0-FA1_32x2-w6_3638693.out`
- `slurm_p0_p0-FA1_32x2-w6_3639670.out`
- `slurm_p0_p0-FA1_32x2-w6_3646043.out`
- `slurm_p0_p0-FA1_8x8-w6_3638646.out`
- `slurm_p0_p0-FA1_8x8-w6_3638701.out`
- `slurm_p0_p0-FA1_8x8-w6_3639678.out`
- `slurm_p0_p0-FA1_8x8-w6_3646051.out`
- `slurm_p0_p0-VAN_16x4-w6_3638641.out`
- `slurm_p0_p0-VAN_16x4-w6_3638696.out`
- `slurm_p0_p0-VAN_16x4-w6_3639673.out`
- `slurm_p0_p0-VAN_16x4-w6_3646046.out`
- `slurm_p0_p0-VAN_32x2-w6_3638637.out`
- `slurm_p0_p0-VAN_32x2-w6_3638692.out`
- `slurm_p0_p0-VAN_32x2-w6_3639669.out`
- `slurm_p0_p0-VAN_32x2-w6_3646042.out`
- `slurm_p0_p0-VAN_8x8-w6_3638645.out`
- `slurm_p0_p0-VAN_8x8-w6_3638700.out`
- `slurm_p0_p0-VAN_8x8-w6_3639677.out`
- `slurm_p0_p0-VAN_8x8-w6_3646050.out`
- `slurm_p0_p0-smoke-C4L_32x2_3638618.out`
- `slurm_p0_p0-smoke2-FA1_32x2_3638630.out`
- `slurm_p0_p0-supp-CKPT4_16x4_3639145.out`
