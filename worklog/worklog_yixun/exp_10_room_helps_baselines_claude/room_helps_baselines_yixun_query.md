# exp_10 — Room Helps baselines: Yixun queries and decisions

## Driving query 1

> 对于Few-shotRiR
>
> 1. Geometry 输入与FLAC相同
> 2. Acoustic context和坐标表示都与FLAC相同，适配AR
> 3. 输出格式保留 Few-ShotRIR 风格
> 4. 不加入定位损失
> 5. Train from scratch
>
> 对于FEM - Sabine
>
> 1. T60估计和FLAC看齐
> 2. Sabine 边界假设，体积和表面积定义我同意你的设计
> 3. FEM 频率范围80–300 Hz
> 4. 定位评分和Flac看齐
>
> Scorer都统一和现有FLAC看齐，K=1，8都做，还有问题吗

## Driving query 2 — Few-ShotRIR output amendment

> 那用方法2吧，AR 的所有 RIR都是做过裁剪的吧

The accepted “method 2” is direct time-domain RIR prediction. It supersedes the earlier request to retain the original magnitude-only output, because the frozen FLAC/AGREE scorer consumes a waveform.

## Driving query 3 — confirmation and requested artifact

> 两种方法都确定下来了吧

> 写成一份文档供我做审核

## Summary of the frozen hypothesis

Two material-blind baselines will test whether room information helps localization under the same AcousticRooms observations, candidate grid, and metrics, using their frozen method-specific selectors:

1. **Few-ShotRIR-Waveform** learns a deterministic context-conditioned waveform predictor from scratch, without RGB, material labels, or localization loss.
2. **FEM-Sabine** replaces explicit surface material labels by a single Sabine-equivalent boundary inferred from context-RIR T60, solves the low-frequency Helmholtz problem from 80–300 Hz, and applies Room Helps pulse-source stacked complex OMP directly to the observed/FEM frequency responses.

Both are evaluated with nested acoustic context counts `K_ctx ∈ {1, 8}`. `K_ctx` is not the number of stochastic FLAC generations.

## Why the experiment is needed

The central comparison is not merely which forward model best reconstructs an RIR. It tests whether localization improves when a baseline explicitly or implicitly uses room structure, and whether FLAC remains stronger despite not receiving explicit surface-material assignments. The comparison keeps observations, candidate locations, and localization metrics fixed; after Driving query 4, the selection rule is intentionally method-specific and must be reported as such.

## Driving query 4 — FEM selection amendment

> 调整一个地方FEM的选点标准还是采用Room Helps 的稀疏恢复算法

This supersedes the earlier common-AGREE selection decision for FEM only. Few-ShotRIR continues to use frozen AGREE cosine similarity. FEM now uses the Room Helps pulse-source special case because AcousticRooms supplies one-microphone unit impulse responses: stack the exact 80–300 Hz complex observation/dictionary equations across frequency, run complex OMP with `source_count=1`, and select the stable maximum first-step projection score. The candidate grid and localization metrics remain shared, but the numerical selection score intentionally differs by method.
