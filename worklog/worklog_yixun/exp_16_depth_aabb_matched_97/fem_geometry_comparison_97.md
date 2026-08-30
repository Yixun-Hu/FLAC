# FEM geometry comparison on the matched 97-query subset

Scope: the same 97 strict-coverage queries, byte-identical frozen candidates,
`K_ctx=8`, 80--300 Hz, and `h_max <= 0.22 m` for both methods.

| Model | Median Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ |
|---|---:|---:|---:|---:|
| FEM-Sabine (Depth-AABB) | 0.718 | 33.0% | 59.8% | 46.4% |
| FEM-Sabine (Full Geometry Oracle) | **0.354** | **64.9%** | **76.3%** | **69.1%** |

Diagnostic mean localization errors are 1.185 m for Depth-AABB and 0.760 m
for Full Geometry. Full Geometry has a lower error on 59 queries, the two
methods tie on 17, and Depth-AABB has a lower error on 21.

The Full Geometry row consumes the official unseen-room mesh and is therefore
a privileged-geometry oracle, not a matched-input main-table method.

## Copyable LaTeX

```latex
\begin{table}[t]
\centering
\caption{FEM geometry comparison on the same 97-query strict-coverage subset.}
\label{tab:fem_geometry_comparison}
\begin{tabular}{lcccc}
\toprule
Model & Median Error [m] $\downarrow$ & SR@0.5m $\uparrow$ & SR@1.0m $\uparrow$ & Resolution-Aware SR@0.5m $\uparrow$ \\
\midrule
FEM-Sabine (Depth-AABB) & 0.718 & 33.0\% & 59.8\% & 46.4\% \\
FEM-Sabine (Full Geometry Oracle) & \textbf{0.354} & \textbf{64.9\%} & \textbf{76.3\%} & \textbf{69.1\%} \\
\bottomrule
\end{tabular}
\end{table}
```

Full Geometry uses the official room mesh and must be labeled as an oracle.
