# 计时环境记录

生成模型、AGREE 与 OMP 计时所在主机：

- OS：Ubuntu 22.04 系列，Linux 6.8.0-136-generic，x86_64
- CPU：Intel Xeon Gold 5220R @ 2.20 GHz，24 核 / 48 线程
- GPU：2 × NVIDIA RTX A6000 48 GB
- GPU UUID：物理卡 0 `GPU-d4f1b10f-c28a-4b76-8b49-1c9c432719b6`；物理卡 1 `GPU-295d7083-e6e4-1854-703e-44bac67d2690`
- NVIDIA driver：595.84
- Python：3.10.12
- NumPy：1.23.5
- PyTorch：2.7.0+cu126
- TorchAudio：2.7.0+cu126
- SciPy：1.15.3
- CUDA runtime reported by PyTorch：12.6

学习模型的有效 repeat 1 结果均来自 RTX A6000：Vanilla/FA-BF 来自物理卡 1，Yaw-Augmented/Few-ShotRIR 来自物理卡 0；两张卡型号相同。曾在物理卡 1 与其他负载竞争时产生的 18 条 Yaw-Augmented 临时结果已删除，未进入本复现包、repeat-1 summary 或最终统计。后续 repeat 即使存在于原工作目录，也不属于本发布包。

FEM 的 112 条成功求解时间来自本地与 external server 的混合 CPU 环境，没有做硬件归一化。因此 FEM 与 GPU 模型之间是实际观测运行时间比较，不应解释成同一硬件上的纯算法 microbenchmark。
