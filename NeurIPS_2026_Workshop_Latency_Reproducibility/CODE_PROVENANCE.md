# 代码来源

计时代码快照取自：

- repository：`NeuriPs_Workshop/localization-exp`
- 基础 commit：`872e62cfb094131e742a7c205db576e6e01aee2c`
- snapshot date：2026-08-31（America/New_York）

Latency 相关代码在快照时包含尚未提交的工作树修改，因此不能只用基础 commit 恢复。本目录中的 `code/` 是实际生成这些结果时的文件级快照，`SHA256SUMS` 是其权威内容校验。

`code/calculation/` 中的两个汇总程序负责从逐-query数据计算单次 repeat 和最终表格；其中 aggregate 的数值逻辑保持原样，只把单-repeat输出文字和 `aggregation_protocol` 改成明确的 “single repeat”，避免误写成跨三轮中位数。`recompute_all.py` 与 `verify_numeric_equivalence.py` 是便携入口和数值验证程序。`code/measurement/` 保存 orchestrator、AGREE/OMP、FEM fallback 与 FEM core 的原始计时程序。`code/reference/` 保存这些程序调用的推理入口、`src/` 源码和模型配置快照。

模型 checkpoint 与 AcousticRooms 原始数据集不属于 latency 结果数据，体积也远大于本包，因此不重复分发。它们的内容哈希保存在各方法的 `run_manifest.json` 和 `data/selector/selector_latency_128.json` 中；下载者无需这些大文件即可重算并验证论文 latency 表。如需重新执行神经模型或重新测量 selector，则需另行取得相同哈希的 checkpoint 与原始数据集。
