# NeurIPS 2026 Workshop Sound Localization：Latency 可复现数据包

本目录与论文 `_Neurips_2026_workshop__Sound_Localization.pdf` 并列，保存本次统一 latency 评估实际使用的逐-query数据、计时程序、汇总程序、执行日志和预期结果。下载本目录后，无需模型 checkpoint、GPU 或 AcousticRooms 原始数据集，即可从原始计时记录重新计算每次 repeat 和论文表格，并逐项验证数值。

本包只讨论 **FEM–OMP**；不再使用 FEM–AGREE。

## 最快复现方式

计算环境只需要 Python 3.10+ 与 NumPy：

```bash
python3 -m pip install -r requirements-calculation.txt
python3 code/calculation/recompute_all.py
```

程序将：

1. 校验 selection、所有逐-query JSON、FEM源数据和 selector 数据内嵌的 SHA-256；
2. 重算唯一纳入发布包的 `repeat_001`；
3. 生成单-repeat最终表；
4. 与 `expected/summary_final.json` 做逐-query数值等价验证。

输出位于 `recomputed/`。移动整个目录后，summary 中记录的绝对 provenance path 会改变，因此新旧 summary 的顶层 SHA-256 不要求相同；验证程序严格比较全部 latency 数值、query身份、scope和protocol字段。

发布包完整性可用下列命令检查：

```bash
sha256sum -c SHA256SUMS
```

`SHA256SUMS` 不覆盖可重新生成的 `recomputed/` 目录。

## 统一评估协议

- query scope：16 rooms × 8 queries/room = 128 queries；固定选择见 `data/selection/frozen_16room_128.json`。
- 总 candidate evaluations：92,608；各 query candidate 数量不同，范围 27–5,296。
- context：`K_ctx = 8`。
- generation：每个 candidate 只生成一个 RIR，即 `K_gen = 1`。
- learned model candidate batch size：64。
- learned model timing：每个方法、每次完整 repeat 前有 1 个不计入结果的 warm-up query；CUDA阶段前后显式同步。
- repeat：发布包只采用四个学习模型的 `repeat_001`，每个方法各有一次完整 128-query计时。AGREE/OMP selector 每个 query 连续计时 3 次并先取该 query 的中位数。FEM acoustic core 不重跑，保留 112 条原始实测；16 条失败路径实测 3 次后取中位数。
- random seed：生成/评分和 FEM fallback 均使用 seed 42；fallback 的选择是确定性的 random candidate policy。

16 个房间均各占 8 条 query：

`Apartments_idx_42`, `Apartments_idx_50`, `Auditorium_idx_1`, `Bathrooms_idx_14`, `Bathrooms_idx_18`, `Bedrooms_idx_18`, `Bedrooms_idx_33`, `Cafe_idx_1`, `LivingRoomsWithHallway_idx_25`, `LivingRoomsWithHallway_idx_30`, `MeetingRoom_idx_20`, `MeetingRoom_idx_32`, `Office_idx_10`, `Office_idx_11`, `Restaurants_idx_22`, `Restaurants_idx_24`。

这套 `K_ctx=8, K_gen=1, 16-room/128-query, FEM–OMP` 协议是本次修订后的统一口径，取代相邻 PDF 当前 appendix 草稿中仍可见的旧口径/TODO（例如八次生成、FEM–AGREE、排除极端房间等文字）。因此应以本目录的 frozen selection、程序和 expected output 为最终 latency 复现依据。

## 每个方法的 latency 如何计算

### 四个生成方法

对方法 `m`、query `q`、完整 repeat `r`：

```text
L[m,q,r] = T_context_conditioning[m,q,r]
           + T_candidate_conditioning_and_one_generation[m,q,r]
           + T_AGREE_selector[q]
```

Vanilla FLAC、FA-BF FLAC/OrbitRIR 和 Yaw-Augmented FLAC 的前两项分别记录在逐-query JSON 的：

```text
latency_seconds.context_conditioning
latency_seconds.candidate_conditioning_and_generation
```

Few-ShotRIR 不单独拆 context 项；它的两项之和直接记录为：

```text
latency_seconds.core_forward_total
```

所有生成方法最终都使用：

```text
L[m,q,r] = core_forward_total[m,q,r]
           + selector_latency_128[q].agree.median.scoring_total_seconds
```

其中 AGREE selector 包含：

1. observed RIR 的一次 AGREE 编码；
2. 所有 generated RIR 的 AGREE 编码（batch size 64）；
3. generated/observed embedding cosine similarity；
4. 拼接全部 score 并执行 `argmax`。

AGREE 在这里是单独计时的公共评分阶段。generated 输入使用与实际输出完全相同的 GPU resident tensor shape `[batch, 1, 10240]`，以零 tensor 代替 waveform 值，避免重跑生成器；AGREE计算量由 shape 决定。这解释了为什么四个生成方法的 **AGREE mean 完全一样**：不是四次碰巧测得一样，而是有意给四个方法加入同一条逐-query公共 selector latency 向量，以统一评分边界。

128-query 上 AGREE 各分量的 query mean 为：

| 分量 | Mean [s/query] |
|---|---:|
| observed RIR encoding | 0.007005106 |
| generated RIR encoding | 0.669236953 |
| cosine similarity | 0.001924929 |
| argmax | 0.000039887 |
| scoring total | 0.678263996 |

### FEM–OMP：112 条成功 query

成功 query 的公式是：

```text
L[FEM-OMP,q] = T_mesh_construction[q]
               + T_operator_construction[q]
               + T_fullband_solve[q]
               + T_OMP_selector[q]
```

前三项来自 FEM result JSON 的 `runtime_seconds`，且：

```text
runtime_seconds.total
  = mesh_construction + operator_construction + fullband_solve
```

112 条成功 query 的来源不是 donor、补值或混合统计量，而是逐-query observed runtime：

| 来源 | Query 数 | 本包位置 | 含义 |
|---|---:|---|---|
| local primary | 97 | `data/fem/primary_97/` | `exp_16_depth_aabb_matched_97` 原始 result JSON |
| local oversized | 9 | `data/fem/oversized_9/` | 后续在大房间完成的原始 result JSON |
| external server | 6 | `data/fem/external_server_6query_runtime_recovery.json` | Cafe 4 条、Auditorium 2 条逐-query内部 FEM total |

external server 六条使用 `fem_internal_total_seconds`（即 FEM 内部三阶段之和），不用包含调度/进程启动/文件写出的 `wall_clock_elapsed_seconds`。两个字段都原样保留，便于审计。

OMP 的来源为：

- 97 条：使用 `data/selector/fem_response_cache_97/` 的真实 FEM complex response；
- 15 条（9 local oversized + 6 external）：原计算服务器没有保存 response cache，因此使用相同 `[candidate_count, 102 frequency bins]` shape 的复数 response 测量 OMP计算时间；
- 16 条 fallback：没有 FEM solve，也不运行 OMP，OMP latency 为 0。

128-query 的 OMP 中位计时总和为 0.197519663 s，mean 为 0.001543122 s/query。OMP 相比 FEM求解很小，但仍被计入。

### FEM–OMP：16 条严格覆盖失败 query

这 16 条不从同房间成功 query 随机选择 runtime donor，也不假装执行 FEM。计入的是实际部署策略所走的失败路径：

```text
L[FEM-OMP,q] = T_frozen_candidate_preparation[q]
               + T_Depth-AABB_strict_coverage_detection[q]
               + T_deterministic_random_candidate_selection[q]
```

每条失败路径实测 3 次，使用逐-query中位数；数据在 `data/fem/fem_random_fallback_latency_16.json`。因此 FEM统计覆盖完整的 16 rooms / 128 queries：112 条成功 acoustic solve + OMP，16 条失败检测 + random choice。

## 计入与未计入的环节

| 环节 | 生成方法 | FEM成功 | FEM失败 |
|---|:---:|:---:|:---:|
| 8 个 context 的模型 conditioning | ✓ | 由 FEM求解问题体现 | 不适用 |
| candidate conditioning | ✓ | 由 FEM operator/solve体现 | candidate preparation ✓ |
| 每 candidate 生成/求解 1 个 response | ✓ | ✓ | 不执行 |
| observed AGREE encoding | ✓ | 不使用 | 不使用 |
| generated AGREE encoding + similarity + argmax | ✓ | 不使用 | 不使用 |
| OMP scoring/selection | 不使用 | ✓ | 不执行，记 0 |
| strict coverage detection + random choice | 不使用 | 不计入成功路径 | ✓ |
| checkpoint/model loading | ✗ | ✗ | ✗ |
| manifest/geometry audit加载 | ✗ | ✗ | ✗ |
| 普通 query input/depth/RIR 文件加载 | ✗ | ✗ | depth 文件加载属于 strict gate 实际路径 |
| 成功 query 的 candidate filtering | ✗ | ✗ | candidate preparation 属于 fallback 实际路径 |
| localization error metric | ✗ | ✗ | ✗ |
| JSON/NPZ serialization | ✗ | ✗ | ✗ |

因此本表不是从进程启动到结果文件落盘的“完整程序 end-to-end elapsed”。它是统一的 **inference + localization selector latency**：保留真正用于 acoustic inference 和候选选择的阶段，排除一次性初始化、通用 I/O、评估指标和写盘。FEM失败行是 policy-aware exception，计量其实际失败检测/随机选择路径。

## 聚合方法

先对每个 query 建立统一 latency：

- 学习方法：直接采用 `repeat_001` 的逐-query值；不在发布包内跨 repeat 聚合；
- FEM–OMP：每个 query 采用一份 acoustic/fallback计时及其 OMP计时。

然后在 128 条 frozen query 上做 query-micro 统计：

```text
Mean   = arithmetic mean of 128 per-query values
Median = NumPy median of 128 per-query values
P90    = NumPy quantile(values, 0.9), default linear interpolation
```

候选归一化表包含两种容易混淆的量：

```text
dataset-wide mean ms/candidate
  = 1000 * sum_q L[q] / sum_q candidate_count[q]

query median / P90 ms/candidate
  = median / P90 of {1000 * L[q] / candidate_count[q]}
```

它们不能通过把 per-query Mean/Median/P90 除以平均 candidate 数得到。

## 数据目录

```text
data/
  selection/                 冻结的 128-query全集与 112-query FEM成功子集
  protocol/                  context manifest、geometry audit、原始 benchmark config
  repeats/repeat_001/        四个生成方法的逐-query JSON/NPZ、run manifest、日志和单次汇总
  selector/
    selector_latency_128.json    128 条 AGREE/OMP逐-query计时
    fem_response_cache_97/       97 条真实 FEM response 与审计 JSON
  fem/
    primary_97/              97 条原始 FEM成功 result JSON
    oversized_9/             9 条原始大房间 FEM成功 result JSON
    external_server_...json  6 条 external server逐-query runtime
    fem_random_...16.json    16 条真实 fallback逐-query计时
  execution_logs/            驱动程序的有效执行/恢复日志
expected/
  summary_final.json/.md     发布采用的 repeat-1 最终结果
  latency_summary.csv        五方法最终汇总
  latency_per_query.csv      五方法逐-query repeat-1值
  selector_latency_...csv    AGREE/OMP逐-query分量
  fem_latency_...csv         FEM阶段、OMP与最终逐-query值
code/
  calculation/               可直接运行的重算与验证程序
  measurement/               原始 latency计时程序
  reference/                 推理入口及所调用源码快照
```

学习方法每个 query 的 `.json` 是汇总程序实际读取的计时与身份记录；同名 `.npz` 是该次 localization 输出的数组 artifact，也一并保留。全包每个文件的下载完整性由 `SHA256SUMS` 覆盖。

## 原始测量能否在下载后直接重跑

有两层复现能力：

1. **表格计算复现：完全自包含。** 下载者可直接运行 `recompute_all.py`，不需要 GPU、checkpoint 或 AcousticRooms。
2. **重新测量 wall-clock：代码和所有 latency配置/身份/hash均已保存，但需要外部大文件。** 重新生成 1×4×128 条神经模型 timing，需要取得与 run manifest 哈希一致的四个模型 checkpoint、AGREE checkpoint 和 AcousticRooms 数据集；重新跑 FEM还需要原始 depth 与 FEM依赖。它们不是本次 latency结果文件，也不适合重复放入论文附件。

checkpoint 内容哈希：

| 模型 | SHA-256 |
|---|---|
| Vanilla FLAC | `da12748586912c5fe9683a6d27b2507ff13c0a89c458abcbdc63aecd4f35c643` |
| FA-BF FLAC / OrbitRIR | `0f61277f45367fb0e75d7ee70c0627b8948a23eb62be58f13fce91662551557a` |
| Yaw-Augmented FLAC | `ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7` |
| Few-ShotRIR | `f1c83309b7821f6998ed8f754fb6e35fb2a871af836e47240fc101fdb9ae2119` |
| AGREE | `3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787` |

硬件与软件版本见 `ENVIRONMENT.md`，代码工作树来源见 `CODE_PROVENANCE.md`，相邻论文信息见 `PAPER_REFERENCE.md`。
