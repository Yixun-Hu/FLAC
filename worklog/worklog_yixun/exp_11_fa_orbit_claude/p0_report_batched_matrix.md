# exp_11 P0 profiling — measured throughput and peak VRAM

Run `bd96575-1786045321510462046-a3ed28eb` · mode `matrix` · commit `bd96575d2315` — 12 expected row(s).

Steady-state rate measured INSIDE one fit (steps/s = 20 / (t30 − t10) from the runner's callback marks), so no startup, rendezvous or teardown time is included. Peak VRAM is the whole-run poller peak; every OK row's poller artifact is hash-verified.

| Cell | rung MBxN | workers | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |
|---|---|---|---|---|---|---|---|---|
| VAN_32x2 | 32x2 | 6 | 0.8178 | 1.22 | 6255 | 6255 | OK |  |
| VAN_16x4 | 16x4 | 6 | 0.9216 | 1.09 | 4133 | 4133 | OK |  |
| VAN_8x8 | 8x8 | 6 | 1.0722 | 0.93 | 3251 | 3251 | OK |  |
| FA1_32x2 | 32x2 | 6 | 0.8214 | 1.22 | 6255 | 6255 | OK |  |
| FA1_16x4 | 16x4 | 6 | 1.0011 | 1.00 | 4133 | 4133 | OK |  |
| FA1_8x8 | 8x8 | 6 | 1.0705 | 0.93 | 3251 | 3251 | OK |  |
| C4L_32x2 | 32x2 | 6 | 0.2690 | 3.72 | 15951 | 15951 | OK |  |
| C4L_16x4 | 16x4 | 6 | 0.4529 | 2.21 | 9545 | 9545 | OK |  |
| C4L_8x8 | 8x8 | 6 | 0.6598 | 1.52 | 6003 | 6003 | OK |  |
| C8_32x2 | 32x2 | 6 | 0.1405 | 7.12 | 28779 | 28779 | OK |  |
| C8_16x4 | 16x4 | 6 | 0.2557 | 3.91 | 19053 | 19053 | OK |  |
| C8_8x8 | 8x8 | 6 | 0.4351 | 2.30 | 9503 | 9503 | OK |  |

## GPU utilisation / power over the step-10 → step-30 window

| Cell | workers | GPU UUID | ticks | mem max MiB | util mean % | power mean W |
|---|---|---|---|---|---|---|
| VAN_32x2 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 46 | 6255 | 66.7 | 221.0 |
| VAN_32x2 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 46 | 6255 | 70.5 | 212.3 |
| VAN_16x4 | 6 | GPU-b8652e16-4f8e-5cb0-3f70-f35d1b1efc32 | 39 | 4133 | 46.1 | 161.0 |
| VAN_16x4 | 6 | GPU-b9374a97-1eff-8616-e530-f6ea65a8b2db | 39 | 4117 | 54.7 | 161.8 |
| VAN_16x4 | 6 | GPU-d9d6f311-c639-c652-d739-a6fedf72c35b | 39 | 4133 | 49.4 | 164.9 |
| VAN_16x4 | 6 | GPU-dcc113f2-98e2-bebd-79b5-1b0834dda0a0 | 39 | 4117 | 50.3 | 168.7 |
| VAN_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 29 | 3251 | 35.3 | 126.9 |
| VAN_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 29 | 3231 | 36.6 | 127.0 |
| VAN_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 29 | 3251 | 45.0 | 130.1 |
| VAN_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 29 | 3251 | 39.9 | 122.0 |
| VAN_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 29 | 3251 | 39.5 | 131.0 |
| VAN_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 29 | 3231 | 35.6 | 122.9 |
| VAN_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 29 | 3247 | 42.7 | 131.0 |
| VAN_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 29 | 3247 | 38.3 | 133.1 |
| FA1_32x2 | 6 | GPU-a139a34f-e52a-e5f1-f156-e2f433a0ecf3 | 45 | 6255 | 63.8 | 215.7 |
| FA1_32x2 | 6 | GPU-e9c26fcd-9801-8055-5009-17d220a8607b | 45 | 6255 | 64.4 | 213.9 |
| FA1_16x4 | 6 | GPU-b8652e16-4f8e-5cb0-3f70-f35d1b1efc32 | 35 | 4133 | 40.9 | 164.3 |
| FA1_16x4 | 6 | GPU-b9374a97-1eff-8616-e530-f6ea65a8b2db | 35 | 4117 | 52.3 | 170.8 |
| FA1_16x4 | 6 | GPU-d9d6f311-c639-c652-d739-a6fedf72c35b | 35 | 4133 | 47.1 | 170.2 |
| FA1_16x4 | 6 | GPU-dcc113f2-98e2-bebd-79b5-1b0834dda0a0 | 35 | 4117 | 39.7 | 173.3 |
| FA1_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 29 | 3251 | 30.8 | 130.6 |
| FA1_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 29 | 3231 | 40.6 | 128.5 |
| FA1_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 29 | 3251 | 28.6 | 126.6 |
| FA1_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 29 | 3251 | 32.3 | 124.3 |
| FA1_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 29 | 3251 | 35.0 | 127.9 |
| FA1_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 29 | 3231 | 33.2 | 121.5 |
| FA1_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 29 | 3247 | 35.1 | 132.9 |
| FA1_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 29 | 3247 | 31.6 | 131.5 |
| C4L_32x2 | 6 | GPU-bc15acb5-279d-6dea-00e7-eb3e432008ea | 133 | 15951 | 85.7 | 265.2 |
| C4L_32x2 | 6 | GPU-d01e46e9-813e-11dc-2c93-5a7dfed33e12 | 133 | 15951 | 85.1 | 268.3 |
| C4L_16x4 | 6 | GPU-b8652e16-4f8e-5cb0-3f70-f35d1b1efc32 | 72 | 9473 | 67.8 | 239.5 |
| C4L_16x4 | 6 | GPU-b9374a97-1eff-8616-e530-f6ea65a8b2db | 72 | 9455 | 72.8 | 239.9 |
| C4L_16x4 | 6 | GPU-d9d6f311-c639-c652-d739-a6fedf72c35b | 72 | 9545 | 70.0 | 238.6 |
| C4L_16x4 | 6 | GPU-dcc113f2-98e2-bebd-79b5-1b0834dda0a0 | 72 | 9455 | 71.4 | 243.3 |
| C4L_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 47 | 6003 | 52.6 | 190.9 |
| C4L_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 47 | 5983 | 50.3 | 189.0 |
| C4L_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 47 | 6003 | 51.4 | 189.5 |
| C4L_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 47 | 6003 | 50.3 | 185.6 |
| C4L_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 47 | 6003 | 53.6 | 192.2 |
| C4L_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 47 | 5983 | 50.7 | 185.9 |
| C4L_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 47 | 6001 | 56.5 | 190.8 |
| C4L_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 47 | 5999 | 51.7 | 193.1 |
| C8_32x2 | 6 | GPU-2629c503-45a4-43fb-62fb-8377dbf2a621 | 263 | 28669 | 91.4 | 282.5 |
| C8_32x2 | 6 | GPU-42d82925-ff7b-9d88-1fc1-980d18b3b9e0 | 263 | 28779 | 89.9 | 282.5 |
| C8_16x4 | 6 | GPU-920b08d6-34ff-3e67-aa4d-c8b619d1e06d | 137 | 19051 | 84.9 | 264.8 |
| C8_16x4 | 6 | GPU-b9374a97-1eff-8616-e530-f6ea65a8b2db | 137 | 19037 | 83.9 | 265.5 |
| C8_16x4 | 6 | GPU-d9d6f311-c639-c652-d739-a6fedf72c35b | 137 | 19053 | 82.6 | 266.9 |
| C8_16x4 | 6 | GPU-dcc113f2-98e2-bebd-79b5-1b0834dda0a0 | 137 | 19037 | 83.5 | 267.9 |
| C8_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 71 | 9503 | 69.2 | 234.1 |
| C8_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 71 | 9483 | 69.0 | 237.2 |
| C8_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 71 | 9503 | 75.3 | 234.9 |
| C8_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 71 | 9503 | 73.8 | 233.1 |
| C8_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 71 | 9503 | 72.8 | 238.2 |
| C8_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 71 | 9483 | 74.2 | 231.8 |
| C8_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 71 | 9499 | 73.9 | 237.7 |
| C8_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 71 | 9499 | 71.1 | 236.1 |

## Per-orbit-pass cost (step time vs ViT passes; exact FA1+C4L+C8 set)

| rung | s per orbit pass | unattributed residual (s) | points | R² | verdict |
|---|---|---|---|---|---|
| 32x2 | 0.844 | 0.363 | 3 (C4L+C8+FA1) | 1.0000 | plausible |
| 16x4 | 0.417 | 0.568 | 3 (C4L+C8+FA1) | 0.9998 | plausible |
| 8x8 | 0.195 | 0.738 | 3 (C4L+C8+FA1) | 1.0000 | plausible |

FA1 is `fa_invariant` with a single-angle orbit, so it shares the cylindrical pose path with C4L/C8 and the slope is the cost of one ADDITIONAL ViT pass. The intercept is an **unattributed residual** (the fa base step, including its one pass); naming what it contains needs the utilisation/power trace and the worker contrast above.


## Vanilla vs FA1 (pose-path + fa dispatch overhead, 1 ViT pass each)

| rung | VAN s/step | FA1 s/step | Δ s/step |
|---|---|---|---|
| 32x2 | 1.223 | 1.217 | -0.005 |
| 16x4 | 1.085 | 0.999 | -0.086 |
| 8x8 | 0.933 | 0.934 | 0.002 |

## C8 − C4L marginal contrast (measured difference, not a fit)

| rung | Δ s/step | extra passes | s per extra pass |
|---|---|---|---|
| 32x2 | 3.402 | 4 | 0.850 |
| 16x4 | 1.703 | 4 | 0.426 |
| 8x8 | 0.783 | 4 | 0.196 |

## DDP strong scaling at micro x N = 64

| family | rung | GPUs | steps/s | efficiency vs smallest N |
|---|---|---|---|---|
| VAN | 32x2 | 2 | 0.8178 | 1.000 |
| VAN | 16x4 | 4 | 0.9216 | 0.563 |
| VAN | 8x8 | 8 | 1.0722 | 0.328 |
| FA1 | 32x2 | 2 | 0.8214 | 1.000 |
| FA1 | 16x4 | 4 | 1.0011 | 0.609 |
| FA1 | 8x8 | 8 | 1.0705 | 0.326 |
| C4L | 32x2 | 2 | 0.2690 | 1.000 |
| C4L | 16x4 | 4 | 0.4529 | 0.842 |
| C4L | 8x8 | 8 | 0.6598 | 0.613 |
| C8 | 32x2 | 2 | 0.1405 | 1.000 |
| C8 | 16x4 | 4 | 0.2557 | 0.910 |
| C8 | 8x8 | 8 | 0.4351 | 0.774 |

## Rejected rows (not from this run)

- `C16_8x8`: runid 9bf1936-1786033425104073952-d8d84328 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C16_8x8`: runid bd96575-1786045321895684456-ae4c2f92 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C32_8x8`: runid 9bf1936-1786033425104073952-d8d84328 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C32_8x8`: runid bd96575-1786045321895684456-ae4c2f92 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_32x2`: runid smoke-8d53691-1785968197132-99b9c11b is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C4L_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `C8_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `CKPT4_16x4`: runid supp-09d41ca-1785975836736-02cd7f4b is not this run (bd96575-1786045321510462046-a3ed28eb)
- `CKPT4_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_32x2`: runid smoke2-8d53691-1785968294781-96a1b43a is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `FA1_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_16x4`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_16x4`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_16x4`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_32x2`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_32x2`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_8x8`: runid 1334933-1786032532843128131-8f21c960 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_8x8`: runid 86a752b-1785980874148140138-06d348d6 is not this run (bd96575-1786045321510462046-a3ed28eb)
- `VAN_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (bd96575-1786045321510462046-a3ed28eb)

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
