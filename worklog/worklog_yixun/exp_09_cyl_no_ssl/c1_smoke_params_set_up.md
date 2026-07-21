# C1 smoke — params (recorded BEFORE launch)

Exact exp-09 config (`FLAC_AR_exp09.json`, B-F protocol + registered delta), 100 steps,
`--checkpoint-every 100`-or-earlier per the script, seed 42, DDP 2 GPU + SyncBN, bf16
mixed, grad-ckpt ON, cond_method fa_invariant [0.0], gauge-ON. Env flac; co-tenant
with B-F (mutual-slowdown flag stands; the sustained-throughput gate is the fail-closed
arbiter). Aborted/superseded: fit attempts 1–2 (env / missing assets, corrections in
`c1_fit_command.md`).
