# YAWAUG AR 40k — vanilla FLAC localization pilot

The checkpoint loaded through the vanilla `FLAC_AR.json` structure with zero missing and zero stray unexpected model keys. Results cover 64 queries from 16 unseen AcousticRooms, with eight context RIRs and nested `K_gen = 1/4/8` samples.

| K_gen | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.793 | 0.892 | 0.219 | 0.547 | 0.344 |
| 4 | 1.754 | 0.994 | 0.203 | 0.500 | 0.312 |
| 8 | 1.784 | 0.871 | 0.203 | 0.547 | 0.328 |
| Random candidate | 3.209 | 2.320 | 0.078 | 0.188 | 0.141 |

The summed per-query inference time was 3,688.9 seconds and peak allocated GPU memory was 7,185,682,944 bytes. The run manifest SHA-256 is `5ab30cb35c24221eee14a73232305856d8e5c0d22b858beff7120b830fc4bbf7`.

This is the room-stratified 64-query diagnostic pilot (192 model readouts), corresponding to one complete arm of the existing 384-readout two-model protocol.
