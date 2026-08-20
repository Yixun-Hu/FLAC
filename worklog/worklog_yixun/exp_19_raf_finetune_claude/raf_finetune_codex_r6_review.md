APPROVE

No open in-boundary findings. The exp_19 code round can close.

Verified closures:

- F1: canonical pointer required, output root bound with `os.path.samefile`—including symlinked NAS paths—and combined verification forced canonical at [RAF_md.py:193](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:193).
- F2: `RAFPublicationError` is RAF-only and narrowly re-raised before the unchanged substitution handler at [dataset.py:20](/home/yixunhu/codespace/exp-19-raf-finetune/src/data/dataset.py:20) and [dataset.py:373](/home/yixunhu/codespace/exp-19-raf-finetune/src/data/dataset.py:373). The `+19/−0` diff has no behavioral effect on non-RAF datasets.
- F3: complete marker parameter identities, extra/missing keys, and pinned digests are independently checked at [publish.py:391](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/publish.py:391).
- F4: exact canonical miss cap and receiver-count identity/wiring are enforced at [render_depth.py:340](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:340) and [render_depth.py:955](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:955).
- F5: absent or unverified raw-mask evidence cannot pass; authoritative count/rate/hash values are mask-derived at [render_depth.py:244](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:244) and [render_depth.py:403](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:403).

Residuals:

1. MEDIUM — [readback_audit.py:373](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:373)  
   Problem: The 43-GB raw audio corpus remains outside full content binding.  
   Fix: Optional future full or sampled audio-hash manifest.

2. LOW — [readback_audit.py:357](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/readback_audit.py:357), [RAF_md.py:265](/home/yixunhu/codespace/exp-19-raf-finetune/src/configs/dataset_configs/custom_metadata/RAF_md.py:265)  
   Problem: No signing, opened-file `fstat` hardening, or per-item rehashing protects against a malicious local actor.  
   Fix: Optional hardening if the threat model expands.

3. MEDIUM — [render_depth.py:93](/home/yixunhu/codespace/exp-19-raf-finetune/data/RAF/render_depth.py:93)  
   Problem: A globally consistent horizontal permutation/chirality remains render-undetectable.  
   Fix: Obtain an independently surveyed landmark or compass bearing.

The archived “511 passed” count was not relied upon or rerun because of strict read-only constraints; commits and committed oracles were independently inspected with non-writing probes. The worktree remains unchanged.