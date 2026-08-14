# Queries — exp_16 are_port

## Q1 (2026-08-14)
**Verbatim:** "First finish a and then run ARE-on-FLAC, and tell me how long it will take to run the whole trianing?" (after asking whether ARE — Anchor-Relative Endpoints, rir2rir exp_15 — was ever tried on FLAC; it was not).
**Summary:** Port ARE to FLAC: analytic direct-sound anchor A(p) (Hann-windowed sinc skeleton from source–receiver geometry → frozen VAE encode → minus silence bias → keep frames 0–2 only, LOS-gated by depth), train FLAC's rectified flow on the residual target z − λ·A(p), add λ·A_query back at inference. λ=1 arm vs λ=0 control.
**Assumption:** the λ=0 control at our recipe IS the existing P1 arm (identical objective bit-for-bit) — so only ONE new training arm is needed for the primary comparison. Anchor is pure geometry (r, t* yaw-invariant) → composable with the FA/cyl_vit equivariance line; direct-sound placement targets EDT, our most persistent gap metric.
**Why:** EDT has been the stubborn cell across every arm; ARE attacks it with an analytic prior instead of more training.
