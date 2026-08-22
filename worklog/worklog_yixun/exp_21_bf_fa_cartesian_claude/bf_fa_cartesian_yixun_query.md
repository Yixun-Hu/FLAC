# exp_21 bf_target_frame — Yixun's driving queries

## Query 1 — 2026-08-21 ~13:00 EDT (verbatim)

> Currently I think the exp_07 B-F FA C4 method is not correct. Please try these two way of doing the B-F. 1. This revision replaces B-F's reused Cartesian position embedder with a cylindrical-aware Fourier encoder: \(r\) and \(z\) use separate distance scales, while \(\Delta\phi\) uses periodic sine/cosine harmonics. It preserves yaw invariance without mixing meters and radians or creating a discontinuity at \(\pm\pi\). 我同意，而且这不是简单把 `max_val=5` 改成别的数字，而是应该把 embedding 按柱坐标的拓扑重新设计。
>
> 当前做法：
>
> \[(r,z,\Delta\phi) \overset{\text{旧 XYZ PE}}{\longrightarrow} \text{embedding}\]
>
> 最大的问题有两个：
>
> - \(r,z\) 是长度，\(\Delta\phi\) 是角度，却使用同一个 `/5`；
> - 角度是周期变量，但旧 PE 把它当普通实数，在 \(-\pi/\pi\) 处不连续。
>
> ### 建议的 Cylindrical Fourier Embedding
>
> 距离和角度应分别编码：
>
> \[E(r,z,\Delta\phi) = \left[E_r(r),\, E_z(z),\, E_\phi(\Delta\phi)\right]\]
>
> 距离部分可以继续使用 multi-scale Fourier features：
>
> \[E_r(r)= \left[r/L_r,\, \sin(\omega_j r/L_r),\, \cos(\omega_j r/L_r)\right]_{j=1}^{N_r}\]
>
> \[E_z(z)= \left[z/L_z,\, \sin(\omega_j z/L_z),\, \cos(\omega_j z/L_z)\right]_{j=1}^{N_z}\]
>
> 角度部分使用整数 circular harmonics：
>
> \[E_\phi(\Delta\phi)= \left[\sin(k\Delta\phi),\, \cos(k\Delta\phi)\right]_{k=1}^{K_\phi}\]
>
> 关键点是：
>
> - 不直接输入 raw \(\Delta\phi\)；
> - 不对 \(\Delta\phi\) 除以 5；
> - \(k\) 必须是整数，使 embedding 满足严格的 \(2\pi\) 周期性；
> - \(-\pi\) 和 \(+\pi\) 会得到完全相同的 embedding。
>
> 例如：\[E_\phi(-\pi)=E_\phi(+\pi)\] 因此没有当前的 wrap discontinuity。
>
> ### Target 和 context 如何编码
>
> 现有柱坐标转换可以保留：
>
> \[\text{target}\rightarrow(r_t,z_t,0)\qquad \text{context}_i\rightarrow(r_i,z_i,\Delta\phi_i)\]
>
> target 的 angular feature 不再是全零，因为：\[\sin(k\cdot0)=0,\qquad \cos(k\cdot0)=1\]
>
> 所以它会明确表示“这是角度参考位置”。
>
> ### 尺度应该分开
>
> 初始候选可以是：
>
> ```text
> Lr = 5 m
> Lz = 1 m
> Kφ = 8 或 16
> Nr = Nz = 20
> ```
>
> 但更严谨的做法是只根据 AR training split 统计：
>
> - \(L_r\)：例如 horizontal radius 的 p90 或 robust scale；
> - \(L_z\)：例如 \(|z|\) 的 p90；
> - 然后把这两个值固定到 checkpoint/config；
> - HAA fine-tune 时不能重新估计，否则又改变了预训练输入语义。
>
> `Lr=5m` 仍可能合理，因为它大致是 AR 常见水平距离尺度；但 `Lz` 明显不应该也等于 5m。
>
> ### 建议的新模块
>
> ```python
> class CylindricalPoseEmbedderConditioner(Conditioner):
>     def forward(self, x_input, device="cuda"):
>         r, z, dphi = ...
>         r = r / self.r_scale
>         z = z / self.z_scale
>         r_feat = [r, sin(r * radial_freqs), cos(r * radial_freqs)]
>         z_feat = [z, sin(z * vertical_freqs), cos(z * vertical_freqs)]
>         angle_feat = [sin(dphi * harmonics), cos(dphi * harmonics)]
>         feat = cat(r_feat, z_feat, angle_feat)
>         return self.proj(feat)
> ```
>
> 然后只替换：
>
> ```text
> source
> context_poses
> ```
>
> 对应的 `DistEmbedderConditioner`。
>
> `source_vit/context_poses_vit` 仍保持 Cartesian xyz，因为 Geometry Module 需要计算：\[q_{\mathrm{xyz}}-P_{\mathrm{depth,xyz}}\]
>
> ### 不能直接沿用旧 projection 权重
>
> 新 embedding 的输入含义和维度已经改变，所以不应该假装旧的 `dist_embedder_proj` 可以无损 warm-start。
>
> 最可靠的实验是从相同初始化和训练预算重新训练 AR：
>
> | Arm | Pose embedding | Geometry branch |
> |---|---|---|
> | Current BF | 旧 XYZ Fourier 用于 `(r,z,Δφ)` | C4 DINO mean |
> | Cyl-PE BF | 专门的 cylindrical Fourier | C4 DINO mean |
> | Canonical-XYZ BF | target-aligned Cartesian | single/C4 geometry |
>
> 之后再用完全一致的 HAA fine-tune 比较。
>
> 只在 48 条 HAA 数据上随机初始化新的 Cylindrical projection，大概率学不充分，不能公平判断这个设计。
>
> ### 必须通过的单元测试
>
> 新模块至少应验证：
>
> 1. 任意全局 yaw 后 embedding 完全不变；
> 2. `Δφ=-π` 与 `Δφ=+π` embedding 相同；
> 3. target 的 `Δφ=0` 编码稳定；
> 4. 米和角度使用独立尺度；
> 5. context 顺序和 batch shape 不变；
> 6. source/context 共享 projection 时梯度正常。
>
> 我认为这是当前 B-F 最重要的表示修正之一。不过它只修复 pose branch；如果修正后仍明显输 Vanilla，下一个主要嫌疑仍是 ViT 分支过早进行 C4 feature averaging。 2. 可以，而且我认为这是更值得优先测试的方案：不要使用原始全局 \(x,y,z\)，而是以 target 的方位角 \(\phi_t\) 将所有坐标旋转到 target-aligned frame：
>
> \[p'_t=(r_t,0,z_t),\qquad p'_i=(r_i\cos\Delta\phi_i,\ r_i\sin\Delta\phi_i,\ z_i)\]
>
> 这样仍然保持 yaw-invariant，同时可以继续使用原本为 Cartesian \(x,y,z\) 设计的 Fourier embedding，也避免把“米”和“弧度”混进同一个 embedder。对应的 panorama/depth 也应旋转 \(-\phi_t\)，使 geometry 与 poses 位于同一坐标系；如果直接使用未旋转的全局 \(x,y,z\)，就会退回 Vanilla，并失去 yaw invariance。 B-F的时候yaw-invariant,不用cylindrical coordinates，就是直接对应旋转就可以了. Please train the second choice first. You should follow @worklog/experiment_SOP.md as your experiment rule.

## Summary

Yixun judges the registered exp_07 B-F fa_invariant conditioning to be a flawed *representation*, independent of the frame-averaging idea itself: the cylindrical pose triplet `(r, z, Δφ)` is pushed through the reused Cartesian `DistEmbedderConditioner` (`x / max_val=5` + Fourier features), which (a) divides an angle in radians by a 5-meter length scale and (b) treats the periodic Δφ as an unbounded real, creating a discontinuity at ±π. He proposes two corrected B-F variants and orders them:

1. **(second priority) Cyl-PE B-F** — a purpose-built `CylindricalPoseEmbedderConditioner`: separate `L_r`/`L_z` distance scales, integer circular harmonics `sin(kΔφ), cos(kΔφ)` for the angle; only replaces the `source`/`context_poses` dist-embedders; ViT branch keeps C4 DINO mean.
2. **(train FIRST — this experiment) Canonical/Target-aligned-XYZ B-F** — no cylindrical coordinates at all. Rotate the whole conditioning scene by −φ_t (the target source's azimuth) into a target-aligned Cartesian frame: `p'_t=(r_t,0,z_t)`, `p'_i=(r_i cosΔφ_i, r_i sinΔφ_i, z_i)`, and rotate the depth panorama by the same −φ_t so geometry and poses share one frame. The existing Cartesian Fourier embedder is reused *unchanged and in-domain* (meters only). Yaw invariance holds by construction ("就是直接对应旋转就可以了").

Both variants must be retrained on AR from the same initialization and training budget as the current B-F (warm-starting the old projection is explicitly disallowed for variant 1; a 48-sample HAA-only fit of new projections would be an unfair test). Comparison arms per his table: Current BF / Cyl-PE BF / Canonical-XYZ BF, followed later by an identical HAA finetune comparison.

## Assumption / hypothesis

The B-F underperformance vs Vanilla (exp_11's seed-paired reversal: K8 T60 +0.366, EDT +4.18 fa-worse; exp_19's sim2real ordering) is at least partly caused by the pose-representation defect, not by yaw-symmetrization per se. Fixing the representation while keeping exact yaw invariance should recover (part of) the gap. If the corrected arms still clearly lose to Vanilla, the next suspect is the ViT branch's premature C4 feature averaging — which the canonical-frame arm *also* addresses if its geometry branch runs single-forward on the canonicalized panorama.

## Why the experiment needs to run

- FA's value proposition (exact yaw invariance) is currently entangled with a representation artifact; no existing arm separates them.
- The canonical-frame variant is the cheapest strong test: zero new learned parameters, embedder stays in-domain, invariance is exact on the panorama column group (and at 45° = exactly 64 columns of W=512), and it removes the 4× ViT orbit cost.
- Standing goal (2026-08-19): FA must still beat FLAC on HAA — a corrected B-F is the natural candidate to carry that.

## Query 2 — 2026-08-21 ~13:50 EDT (verbatim) — REDIRECT: the arm is full C4 FA with Cartesian poses, NOT canonicalization

> This is not canonical Cartesian. Keep C4 frame averaging and remove cylindrical_pose_features. For every C4 frame, jointly rotate depth and all four pose keys; feed the rotated source/context_poses into the unchanged DistEmbedder as Cartesian xyz, and frame-average all four pose/geometry conditioner outputs. Keep context_audio single-pass. Do not use target-aligned −φ_t canonicalization or a single ViT pass.

### Summary

Delivered as the change list for his "Approve with changes" on plan Rev 1. The planner's reading of "choice 2" as per-sample −φ_t canonicalization + single ViT pass is REJECTED. The arm Yixun wants (experiment renamed `bf_fa_cartesian`):

- **Keep the C4 orbit.** For each frame angle g ∈ {0°, 90°, 180°, 270°}: rotate depth AND all four pose keys (`source`, `source_vit`, `context_poses`, `context_poses_vit`) together — one rigid frame per angle.
- **Cartesian poses into the unchanged embedder.** The rotated `source`/`context_poses` go into the existing `DistEmbedderConditioner` as plain Cartesian xyz (meters only — the representation fix). `cylindrical_pose_features` is removed from the method entirely.
- **Frame-average ALL FOUR pose/geometry conditioner outputs** (the two dist-embedders now join the two ViT conditioners in the orbit mean). `context_audio` stays single-pass.
- Explicitly forbidden: target-aligned −φ_t canonicalization; a single ViT pass.

### Assumption / hypothesis (updated)

vs the registered B-F, this changes ONLY the pose branch's symmetrization scheme (cylindrical exact-C∞ features through an out-of-domain embedder → C4 frame-averaged in-domain Cartesian embeddings); the ViT branch (C4 mean) is bit-identical in structure. It is therefore a clean single-mechanism test of the representation-defect hypothesis. Invariance class: exact (numerical) on the C4 subgroup for all conditioning branches; 45° is again a genuine negative control (expected to break).
