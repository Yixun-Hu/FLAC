# FEM--AGREE matched-band diagnostic protocol

This diagnostic changes only the FEM candidate selector on the frozen 97-query
Depth-AABB strict-coverage subset. It does not replace the registered
FEM--Room-Helps OMP baseline.

- Geometry: receiver-centred Depth-AABB, 0.05 m padding, `h_max <= 0.22 m`.
- Candidates: byte-identical frozen arrays from the exp_16 matched run.
- Acoustic context: the same ordered eight context RIRs (`K_ctx=8`).
- FEM response: exact 80--300 Hz DFT bins used by the OMP baseline.
- Waveform construction: conjugate-symmetric IFFT with the original beginning
  alignment and 10,240 samples at 22,050 Hz.
- Matched-band control: both the observed RIR and every FEM candidate retain
  exactly the same 80--300 Hz bins; DC and all other bins are zero.
- Gain control: each waveform is independently scaled to absolute peak 0.95.
  This frozen, target-independent operation removes FEM's arbitrary common
  scalar gain without choosing a gain from localization errors.
- Selector: frozen `AGREE_fullAR.pt` audio encoder and cosine similarity.
- AGREE stochasticity: one independently seeded observation encoding and
  nested candidate encoder samples `K_agree={1,4,8}`, aggregated by the same
  log-mean-exp rule with `tau=0.1` used by the FLAC localization protocol.
- Metrics: mean and median localization error, SR@0.5 m, SR@1.0 m, and
  resolution-aware SR@0.5 m. Query-micro and room-macro summaries are retained.

`K_agree` counts stochastic AGREE audio-encoder samples. It is not `K_gen`, and
it does not trigger additional FEM solves.
