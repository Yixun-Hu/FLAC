# exp_12 mem_probe — analysis (Planner)

## Reliability

High. The measurement is fail-closed end-to-end: the job ran the reviewed commit exactly (`EXPECT_SHA` gate), the config was byte-pinned to the reviewed canonical SHA, the semantic gate proved the paper path (vanilla conditioning, EMA on, no ViT grad-ckpt), and the peak number comes from 228 validated samples bound to the single allocated L40's UUID with `measurement_valid=1`. The OOM classification used the flushed synchronous train log. One caveat inherent to the method: a 0.5 s poller can undersample sub-second allocator spikes — irrelevant here since the run OOMed (the binding fact) and the sampled peak (45,437 MiB, 98.6%) is within ~630 MiB of the card's capacity anyway.

## Interpretation

The paper configuration needs more than an L40 provides. At failure the process held 44.36 GiB on a 44.39 GiB-capacity device while asking for another 98 MiB, with 36.83 GiB PyTorch-allocated + 7.03 GiB reserved-but-unallocated. Even a perfectly defragmented allocator (`expandable_segments`) would buy back at most ~7 GiB, and the OOM struck at step 0 of the first forward/backward — activation demand was still climbing. This is consistent with the exp_07 audit's inference that the release trained micro-64 BN-64 on H100-class (80 GB) hardware, and it retro-justifies exp_07/10's SyncBN-64 DDP workaround (32×2 with ViT grad-ckpt, measured 15.7 GiB/rank): on 46 GB cards the paper's single-GPU micro-64 recipe is not attainable without changing memory behavior (checkpointing / smaller micro / more cards), each of which this probe was explicitly forbidden to do — correctly, since any of them would have changed what was being measured.

## Consequence for exp_11 (fast-recipe rung selection)

Rung 64×1 (single-card, no ckpt) is eliminated by direct measurement. For micro×N = 64 rungs WITHOUT grad-ckpt: 32×2 is borderline-plausible (well under 64's activation load, but no-ckpt roughly doubles ViT activation residency vs exp_07's checkpointed 15.7 GiB/rank — P0 must measure before trusting it), while 16×4 and 8×8 have comfortable priors. P0's fit matrix decides; per-arm orbit size adds retained per-pass ViT outputs on top.

## Recommended next step

None within exp_12 — the question is answered at the requested scope (single probe, decision-grade). The number feeds exp_11 P0 as a rung prior. Optional (only if Yixun wants the paper recipe on this cluster for its own sake): a follow-up probe series on rungs 32×2/16×4/8×8 is already subsumed by exp_11's P0.
