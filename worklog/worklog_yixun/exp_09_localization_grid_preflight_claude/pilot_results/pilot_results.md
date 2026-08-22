# Exp_09 64-query localization pilot results

Pilot SHA-256: `6eeeec401c3f63e47ab446b4b43efe2f1db260bad9decf74de423d0872b659a2`. Scope: 64 queries / 16 rooms; N_ctx=8; K_gen=[1, 4, 8].

| Arm | K_gen | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 1 | 1.748 | 1.009 | 0.156 | 0.500 | 0.234 |
| vanilla | 4 | 1.844 | 1.059 | 0.156 | 0.484 | 0.250 |
| vanilla | 8 | 1.849 | 1.059 | 0.125 | 0.484 | 0.219 |
| fa_bf | 1 | 1.816 | 0.928 | 0.281 | 0.531 | 0.406 |
| fa_bf | 4 | 1.718 | 1.085 | 0.250 | 0.484 | 0.359 |
| fa_bf | 8 | 1.886 | 1.085 | 0.203 | 0.484 | 0.328 |
| random candidate | — | 3.209 | 2.320 | 0.078 | 0.188 | 0.141 |

This is a room-stratified diagnostic pilot (four targets per room), not the complete 5,337-query unseen-room evaluation.
