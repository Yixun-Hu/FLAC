# exp_11 P0 profiling — measured throughput and peak VRAM

Run `86a752b-1785980874148140138-06d348d6` · mode `matrix` · commit `86a752bc118d` — 12 expected row(s).

Steady-state rate measured INSIDE one fit (steps/s = 20 / (t30 − t10) from the runner's callback marks), so no startup, rendezvous or teardown time is included. Peak VRAM is the whole-run poller peak; every OK row's poller artifact is hash-verified.

| Cell | rung MBxN | workers | steps/s | s/step | peak MiB (overall) | peak MiB (max GPU) | status | note |
|---|---|---|---|---|---|---|---|---|
| VAN_32x2 | 32x2 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| VAN_16x4 | 16x4 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| VAN_8x8 | 8x8 | 6 | 1.0443 | 0.96 | 3251 | 3251 | OK |  |
| FA1_32x2 | 32x2 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| FA1_16x4 | 16x4 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| FA1_8x8 | 8x8 | 6 | 1.0434 | 0.96 | 3253 | 3253 | OK |  |
| C4L_32x2 | 32x2 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| C4L_16x4 | 16x4 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| C4L_8x8 | 8x8 | 6 | 0.3639 | 2.75 | 5921 | 5921 | OK |  |
| C8_32x2 | 32x2 | 6 | 0.1523 | 6.57 | 28639 | 28639 | OK |  |
| C8_16x4 | 16x4 | 6 | — | — | 551 | 551 | FAILED | rc=1 |
| C8_8x8 | 8x8 | 6 | 0.1876 | 5.33 | 9437 | 9437 | OK |  |

## GPU utilisation / power over the step-10 → step-30 window

| Cell | workers | GPU UUID | ticks | mem max MiB | util mean % | power mean W |
|---|---|---|---|---|---|---|
| VAN_8x8 | 6 | GPU-5bae6e31-87e7-00ec-b0c2-580318c2e839 | 30 | 3231 | 35.3 | 130.8 |
| VAN_8x8 | 6 | GPU-96b6e334-4e7c-f533-e861-6470c38b4d5d | 30 | 3251 | 32.2 | 124.9 |
| VAN_8x8 | 6 | GPU-9a1afd95-2fa7-f119-6f71-b256e68fd26d | 30 | 3251 | 31.9 | 129.6 |
| VAN_8x8 | 6 | GPU-a3c78d1e-7d32-874b-18e6-adea32889c07 | 30 | 3251 | 42.9 | 124.5 |
| VAN_8x8 | 6 | GPU-b22a19f5-2f1c-7f2a-8b0a-dbb74d9a87b6 | 30 | 3247 | 35.4 | 125.4 |
| VAN_8x8 | 6 | GPU-be380183-eaaf-7dc6-480b-a2d878c7bc02 | 30 | 3231 | 34.8 | 121.0 |
| VAN_8x8 | 6 | GPU-ded19225-ed41-1129-546e-8d0ca53120d6 | 30 | 3247 | 36.9 | 124.6 |
| VAN_8x8 | 6 | GPU-f54cbabb-7cfb-2655-6880-75136c1e95f7 | 30 | 3251 | 34.4 | 126.2 |
| FA1_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 32 | 3251 | 37.2 | 127.0 |
| FA1_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 32 | 3231 | 40.3 | 124.9 |
| FA1_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 32 | 3253 | 46.5 | 126.0 |
| FA1_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 32 | 3251 | 24.8 | 122.0 |
| FA1_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 32 | 3251 | 27.8 | 130.4 |
| FA1_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 32 | 3231 | 30.2 | 123.7 |
| FA1_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 32 | 3245 | 29.1 | 131.0 |
| FA1_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 32 | 3245 | 38.4 | 131.3 |
| C4L_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 82 | 5921 | 33.8 | 143.4 |
| C4L_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 82 | 5899 | 32.7 | 140.1 |
| C4L_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 82 | 5921 | 32.6 | 142.8 |
| C4L_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 82 | 5919 | 34.0 | 136.9 |
| C4L_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 82 | 5919 | 33.0 | 145.3 |
| C4L_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 82 | 5899 | 32.6 | 135.6 |
| C4L_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 82 | 5915 | 32.0 | 147.3 |
| C4L_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 82 | 5917 | 28.1 | 147.7 |
| C8_32x2 | 6 | GPU-c4fe2e80-5c68-8d2f-74b0-20bc97a9c357 | 242 | 28639 | 86.8 | 273.6 |
| C8_32x2 | 6 | GPU-f764ccff-3d57-9013-cdb0-a83f96ba9fb7 | 242 | 28639 | 88.7 | 277.4 |
| C8_8x8 | 6 | GPU-0db9729c-a84e-d217-30dd-583df67bb952 | 167 | 9437 | 36.5 | 145.4 |
| C8_8x8 | 6 | GPU-1a208c15-1ebc-598b-a6ca-9a1063b6d951 | 167 | 9415 | 35.3 | 143.5 |
| C8_8x8 | 6 | GPU-58ee62f8-3ad6-8946-69c3-6b5d09a5c585 | 167 | 9435 | 35.2 | 144.8 |
| C8_8x8 | 6 | GPU-67715b50-d1ce-0b1b-2448-faeb8eb3dfbe | 167 | 9435 | 38.2 | 140.2 |
| C8_8x8 | 6 | GPU-a6faa6d3-ad5b-665e-4af5-56b3c612e837 | 167 | 9435 | 34.6 | 149.0 |
| C8_8x8 | 6 | GPU-c62b55f3-77c5-b082-f9d3-59eb58918ab6 | 167 | 9415 | 38.7 | 139.0 |
| C8_8x8 | 6 | GPU-e476ab82-1493-4ff3-7f37-694224309c7a | 167 | 9431 | 34.6 | 149.8 |
| C8_8x8 | 6 | GPU-e8729bb2-b8a2-9e11-96b7-028e4b6585e2 | 167 | 9433 | 34.9 | 150.2 |

## Derived attribution — **WITHHELD**

The submitted run is not complete and all-OK (see the status column), so the per-orbit-pass fit, scaling efficiency and grad-ckpt cost are NOT computed: a partial matrix cannot support a rung decision.

## Rejected rows (not from this run)

- `C4L_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `C4L_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `C4L_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)
- `C4L_32x2`: runid smoke-8d53691-1785968197132-99b9c11b is not this run (86a752b-1785980874148140138-06d348d6)
- `C4L_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `C8_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `C8_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `C8_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `CKPT4_16x4`: runid supp-09d41ca-1785975836736-02cd7f4b is not this run (86a752b-1785980874148140138-06d348d6)
- `CKPT4_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_32x2`: runid smoke2-8d53691-1785968294781-96a1b43a is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `FA1_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_16x4`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_16x4`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_32x2`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_32x2`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_8x8`: runid 72a8114-1785969226421855487-c8d5b51f is not this run (86a752b-1785980874148140138-06d348d6)
- `VAN_8x8`: runid aa4bc18-1785968431124626318-df9602ea is not this run (86a752b-1785980874148140138-06d348d6)

## Source files

- `slurm_p0_p0-C4L_16x4-w6_3638643.out`
- `slurm_p0_p0-C4L_16x4-w6_3638698.out`
- `slurm_p0_p0-C4L_16x4-w6_3639675.out`
- `slurm_p0_p0-C4L_32x2-w6_3638639.out`
- `slurm_p0_p0-C4L_32x2-w6_3638694.out`
- `slurm_p0_p0-C4L_32x2-w6_3639671.out`
- `slurm_p0_p0-C4L_8x8-w6_3638647.out`
- `slurm_p0_p0-C4L_8x8-w6_3638702.out`
- `slurm_p0_p0-C4L_8x8-w6_3639679.out`
- `slurm_p0_p0-C8_16x4-w6_3638644.out`
- `slurm_p0_p0-C8_16x4-w6_3638699.out`
- `slurm_p0_p0-C8_16x4-w6_3639676.out`
- `slurm_p0_p0-C8_32x2-w6_3638640.out`
- `slurm_p0_p0-C8_32x2-w6_3638695.out`
- `slurm_p0_p0-C8_32x2-w6_3639672.out`
- `slurm_p0_p0-C8_8x8-w6_3638648.out`
- `slurm_p0_p0-C8_8x8-w6_3638703.out`
- `slurm_p0_p0-C8_8x8-w6_3639680.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638649.out`
- `slurm_p0_p0-CKPT4_32x2-w6_3638704.out`
- `slurm_p0_p0-FA1_16x4-w6_3638642.out`
- `slurm_p0_p0-FA1_16x4-w6_3638697.out`
- `slurm_p0_p0-FA1_16x4-w6_3639674.out`
- `slurm_p0_p0-FA1_32x2-w6_3638638.out`
- `slurm_p0_p0-FA1_32x2-w6_3638693.out`
- `slurm_p0_p0-FA1_32x2-w6_3639670.out`
- `slurm_p0_p0-FA1_8x8-w6_3638646.out`
- `slurm_p0_p0-FA1_8x8-w6_3638701.out`
- `slurm_p0_p0-FA1_8x8-w6_3639678.out`
- `slurm_p0_p0-VAN_16x4-w6_3638641.out`
- `slurm_p0_p0-VAN_16x4-w6_3638696.out`
- `slurm_p0_p0-VAN_16x4-w6_3639673.out`
- `slurm_p0_p0-VAN_32x2-w6_3638637.out`
- `slurm_p0_p0-VAN_32x2-w6_3638692.out`
- `slurm_p0_p0-VAN_32x2-w6_3639669.out`
- `slurm_p0_p0-VAN_8x8-w6_3638645.out`
- `slurm_p0_p0-VAN_8x8-w6_3638700.out`
- `slurm_p0_p0-VAN_8x8-w6_3639677.out`
- `slurm_p0_p0-smoke-C4L_32x2_3638618.out`
- `slurm_p0_p0-smoke2-FA1_32x2_3638630.out`
- `slurm_p0_p0-supp-CKPT4_16x4_3639145.out`
