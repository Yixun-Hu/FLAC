# C1 fit-probe — params (recorded BEFORE launch)

Protocol: plan Rev 4.1 §3 C1 + integrative-r4 §4 checklist. Config
`FLAC_AR_exp09.json` (B-F protocol; delta = 2×implementation + 2×gauge +
frame_avg_angles [0.0]); gauge-ON per the blessed audit. Fit = SHORT train.py
invocation (exact B-F CLI shape) under `gpu_peak_sampler.py` (external nvidia-smi,
both GPUs, fail-on-zero-samples). Bootstrap gate ≥21,900 MiB/GPU (B-F's registered
threshold — upper bound since exp-09 = B-F minus 4× frame-averaging). Env `flac`;
`LOGGER` unset for the fit (no wandb pollution); doc-nit fix (stale CLI help)
included in this records commit as registered at integrative r4.
Aborted/superseded: none yet.
