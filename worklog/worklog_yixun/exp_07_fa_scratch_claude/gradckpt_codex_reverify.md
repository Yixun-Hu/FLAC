1. **Round-1 P0: WRONG.** Installed Transformers 4.57.0 has `DINOv3ViTLayer(GradientCheckpointingLayer)`; its inherited `__call__` invokes `_gradient_checkpointing_func` during training. Dynamic forward/backward confirmed 12 checkpoint segments and gradients in all 12 layers.

2. **SHIP — [conditioners.py:180](/home/yixunhu/codespace/FLAC/src/models/conditioners.py:180).** Lines 180–266 are correct:

   - Loop closures bind each current `layer` and bound `orig_forward`.
   - `_flac_ckpt_wrapped` prevents double wrapping; `functools.wraps` metadata does not interfere.
   - Bound-forward wrapping preserves hooks and state dict. Deep-copying an already-wrapped model would retain closures bound to the original—real caveat, unused by this DDP path.
   - Non-reentrant checkpoint correctly carries nested kwargs and their gradients; DINOv3 RoPE has no trainable embedding parameter.
   - Compatible with `find_unused_parameters_true`; repeated C4 calls create independent checkpoint frames. SyncBN/audio BatchNorm remains outside the wrapped ViT.

3. **SHIP — [test_vit_gradient_checkpointing.py:206](/home/yixunhu/codespace/FLAC/src/tests/test_vit_gradient_checkpointing.py:206).** Lines 206–513 collect 14 cases and passed:

   - 12/12 checkpoint invocations, all `use_reentrant=False`.
   - ON/OFF gradients identical across 210 tensors; max difference `0`.
   - Eval and `no_grad` produce zero checkpoint calls.
   - Idempotency, state-dict identity, and structural fail-closed behavior are pinned.

No correctness findings or requested changes.