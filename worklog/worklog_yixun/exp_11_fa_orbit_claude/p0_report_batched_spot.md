# exp_11 P0 profiling — measured throughput and peak VRAM

Run `bd96575-1786045321895684456-ae4c2f92` · mode `spot` · commit `bd96575d2315` — 2 expected row(s).

Steady-state rate measured INSIDE one fit (steps/s = 20 / (t30 − t10) from the runner's callback marks), so no startup, rendezvous or teardown time is included. Peak VRAM is the whole-run poller peak; every OK row's poller artifact is hash-verified.

| Cell | rung MBxN | workers | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |
|---|---|---|---|---|---|---|---|---|
| C16_8x8 | 8x8 | 6 | 0.2454 | 4.08 | 19629 | 19629 | OK |  |
| C32_8x8 | 8x8 | 6 | 0.1308 | 7.64 | 32063 | 32063 | OK |  |

## GPU utilisation / power over the step-10 → step-30 window

| Cell | workers | GPU UUID | ticks | mem max MiB | util mean % | power mean W |
|---|---|---|---|---|---|---|
| C16_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 118 | 19629 | 84.5 | 259.3 |
| C16_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 118 | 19609 | 83.0 | 262.6 |
| C16_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 118 | 19629 | 82.3 | 259.2 |
| C16_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 118 | 19629 | 84.6 | 257.9 |
| C16_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 118 | 19629 | 84.2 | 259.8 |
| C16_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 118 | 19609 | 83.1 | 256.9 |
| C16_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 118 | 19625 | 83.4 | 260.9 |
| C16_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 118 | 19625 | 85.1 | 261.2 |
| C32_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 215 | 32063 | 90.9 | 274.1 |
| C32_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 215 | 32043 | 89.9 | 278.6 |
| C32_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 215 | 32063 | 89.6 | 272.4 |
| C32_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 215 | 32063 | 89.2 | 273.6 |
| C32_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 215 | 32063 | 90.5 | 272.9 |
| C32_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 215 | 32043 | 87.2 | 270.7 |
| C32_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 215 | 32057 | 90.5 | 273.6 |
| C32_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 215 | 32057 | 90.3 | 275.3 |

## Derived attribution — not applicable to a `spot` run

A `spot` manifest does not carry the FA1+C4L+C8 set; its cells are reported above and enter the matrix analysis only through their own run.

## Rejected rows (not from this run)

- `C16_8x8`: runid 9bf1936-1786033425104073952-d8d84328 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C32_8x8`: runid 9bf1936-1786033425104073952-d8d84328 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_16x4`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_32x2`: runid smoke-8d53691-1785968197132-99b9c11b is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C4L_8x8`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_16x4`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_32x2`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `C8_8x8`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `CKPT4_16x4`: runid supp-09d41ca-1785975836736-02cd7f4b is not this run (bd96575-1786045321895684456-ae4c2f92)
- `CKPT4_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_16x4`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_32x2`: runid smoke2-8d53691-1785968294781-96a1b43a is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `FA1_8x8`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_16x4`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_16x4`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_32x2`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321895684456-ae4c2f92)
- `VAN_8x8`: runid bd96575-1786045321510462046-a3ed28eb is not this run (bd96575-1786045321895684456-ae4c2f92)

## Source files

- `slurm_p0_p0-C16_8x8-w6_3646146.out`
- `slurm_p0_p0-C16_8x8-w6_3646677.out`
- `slurm_p0_p0-C32_8x8-w6_3646147.out`
- `slurm_p0_p0-C32_8x8-w6_3646678.out`
- `slurm_p0_p0-C4L_16x4-w6_3638643.out`
- `slurm_p0_p0-C4L_16x4-w6_3638698.out`
- `slurm_p0_p0-C4L_16x4-w6_3639675.out`
- `slurm_p0_p0-C4L_16x4-w6_3646048.out`
- `slurm_p0_p0-C4L_16x4-w6_3646671.out`
- `slurm_p0_p0-C4L_32x2-w6_3638639.out`
- `slurm_p0_p0-C4L_32x2-w6_3638694.out`
- `slurm_p0_p0-C4L_32x2-w6_3639671.out`
- `slurm_p0_p0-C4L_32x2-w6_3646044.out`
- `slurm_p0_p0-C4L_32x2-w6_3646667.out`
- `slurm_p0_p0-C4L_8x8-w6_3638647.out`
- `slurm_p0_p0-C4L_8x8-w6_3638702.out`
- `slurm_p0_p0-C4L_8x8-w6_3639679.out`
- `slurm_p0_p0-C4L_8x8-w6_3646052.out`
- `slurm_p0_p0-C4L_8x8-w6_3646675.out`
- `slurm_p0_p0-C8_16x4-w6_3638644.out`
- `slurm_p0_p0-C8_16x4-w6_3638699.out`
- `slurm_p0_p0-C8_16x4-w6_3639676.out`
- `slurm_p0_p0-C8_16x4-w6_3646049.out`
- `slurm_p0_p0-C8_16x4-w6_3646672.out`
- `slurm_p0_p0-C8_32x2-w6_3638640.out`
- `slurm_p0_p0-C8_32x2-w6_3638695.out`
- `slurm_p0_p0-C8_32x2-w6_3639672.out`
- `slurm_p0_p0-C8_32x2-w6_3646045.out`
- `slurm_p0_p0-C8_32x2-w6_3646668.out`
- `slurm_p0_p0-C8_8x8-w6_3638648.out`
- `slurm_p0_p0-C8_8x8-w6_3638703.out`
- `slurm_p0_p0-C8_8x8-w6_3639680.out`
- `slurm_p0_p0-C8_8x8-w6_3646053.out`
- `slurm_p0_p0-C8_8x8-w6_3646676.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638649.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638704.out`
- `slurm_p0_p0-FA1_16x4-w6_3638642.out`
- `slurm_p0_p0-FA1_16x4-w6_3638697.out`
- `slurm_p0_p0-FA1_16x4-w6_3639674.out`
- `slurm_p0_p0-FA1_16x4-w6_3646047.out`
- `slurm_p0_p0-FA1_16x4-w6_3646670.out`
- `slurm_p0_p0-FA1_32x2-w6_3638638.out`
- `slurm_p0_p0-FA1_32x2-w6_3638693.out`
- `slurm_p0_p0-FA1_32x2-w6_3639670.out`
- `slurm_p0_p0-FA1_32x2-w6_3646043.out`
- `slurm_p0_p0-FA1_32x2-w6_3646666.out`
- `slurm_p0_p0-FA1_8x8-w6_3638646.out`
- `slurm_p0_p0-FA1_8x8-w6_3638701.out`
- `slurm_p0_p0-FA1_8x8-w6_3639678.out`
- `slurm_p0_p0-FA1_8x8-w6_3646051.out`
- `slurm_p0_p0-FA1_8x8-w6_3646674.out`
- `slurm_p0_p0-VAN_16x4-w6_3638641.out`
- `slurm_p0_p0-VAN_16x4-w6_3638696.out`
- `slurm_p0_p0-VAN_16x4-w6_3639673.out`
- `slurm_p0_p0-VAN_16x4-w6_3646046.out`
- `slurm_p0_p0-VAN_16x4-w6_3646669.out`
- `slurm_p0_p0-VAN_32x2-w6_3638637.out`
- `slurm_p0_p0-VAN_32x2-w6_3638692.out`
- `slurm_p0_p0-VAN_32x2-w6_3639669.out`
- `slurm_p0_p0-VAN_32x2-w6_3646042.out`
- `slurm_p0_p0-VAN_32x2-w6_3646665.out`
- `slurm_p0_p0-VAN_8x8-w6_3638645.out`
- `slurm_p0_p0-VAN_8x8-w6_3638700.out`
- `slurm_p0_p0-VAN_8x8-w6_3639677.out`
- `slurm_p0_p0-VAN_8x8-w6_3646050.out`
- `slurm_p0_p0-VAN_8x8-w6_3646673.out`
- `slurm_p0_p0-smoke-C4L_32x2_3638618.out`
- `slurm_p0_p0-smoke2-FA1_32x2_3638630.out`
- `slurm_p0_p0-supp-CKPT4_16x4_3639145.out`
