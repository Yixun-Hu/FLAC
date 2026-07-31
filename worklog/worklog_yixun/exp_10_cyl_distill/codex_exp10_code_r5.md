## Findings

1. **Blocking — the SHA gate does not bind the bytes actually loaded.** [init_backbone.py:35](/home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/init_backbone.py:35) hashes the path, closes it, then [reopens it for `torch.load`](</home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/init_backbone.py:50>). A replacement or concurrent rewrite between those operations can pass the hash for artifact A but load artifact B. Hash and load the same immutable byte buffer, or otherwise guarantee the same open file contents.

2. **Blocking — conflicting initialization sources are silently accepted.** The distilled backbone loads at [train.py:139](/home/yixunhu/codespace/exp-10-cyl-distill/train.py:139), then `pretrained_ckpt_path` strictly reloads the entire model at [train.py:151](/home/yixunhu/codespace/exp-10-cyl-distill/train.py:151), overwriting it while leaving a misleading successful init-backbone log. Likewise, `ckpt_path` restores the wrapper later at [train.py:242](/home/yixunhu/codespace/exp-10-cyl-distill/train.py:242). For exp_10, `init_backbone` should explicitly refuse combination with either full-model source. The stated S3 command avoids the defect, but the code does not fail closed against misconfiguration.

3. **Class-name adjudication — non-blocking for this narrow pipeline.** A deliberately same-named unrelated class can spoof [the `__name__` check](</home/yixunhu/codespace/exp-10-cyl-distill/worklog/worklog_yixun/exp_10_cyl_distill/init_backbone.py:47>); the fixture itself demonstrates that. However, the object comes from reviewed factory code rather than artifact deserialization, and the real-model probe confirmed the exact package type. An ordinary differently named subclass is refused. Exact `type(obj) is CylindricalDINOv3ViTModel` would be stronger, but the present check is acceptable within this trust boundary.

Verified:

- Hook ordering is otherwise correct: model creation → backbone initialization → full-model initialization → wrapper/EMA creation.
- Missing `init_backbone_sha` refuses. The enabled-only `sys.path` insertion is permanent, but its directory currently exposes no likely later-import collision; non-blocking.
- Installed `prefigure==0.0.9` produced `''`/`str` for both defaults and correctly parsed nonempty overrides.
- The exp-09 pin surface includes `train.py` and `defaults.ini`; old pins therefore fail closed on this changed exp_10 HEAD. The dedicated exp-09 worktree is unaffected.
- CPU-only real factory probe: shared object `True`, exact type `cylindrical_dinov3.modeling_cylindrical_dinov3.CylindricalDINOv3ViTModel`, 21,596,544 parameters, CUDA unavailable with zero visible devices.
- Suite: **19 passed**. The SHA-comparison, shared-object-check, and `strict=True → False` mutants were all killed by their intended tests.

Reviewed SHA: `1d72167b25a1162751536bf18d0aaeaa0b245f6a`

NOT CLEARED