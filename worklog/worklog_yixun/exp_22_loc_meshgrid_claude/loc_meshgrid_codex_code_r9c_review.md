# exp_22 Codex re-review — round r9c (consolidated r9b+r9c)
Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, read-only static.
Scope: r9b `07a7242 dc46a70 9ba13cf b3f08e0` + r9c `0a06416 f92cb26`. Date: 2026-08-25.

# Verdict: REJECT

Do not authorize downstream tools to consume the P1 merged run as canonical yet. Result-corrupting blockers remain in merge provenance, truth authentication, registered checkpoint enforcement, and the new retrieval control.

## r9 checklist

| Finding | Status | Static re-review |
|---|---|---|
| B1 — provenance/merge | **PARTIALLY** | Manifest swaps are now hash-joined at [meshgrid_report.py:319](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:319), but [meshgrid_report.py:392](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:392) trusts a copyable `merge_report.json` and does not recheck row batching or derive `source_rows`; a complete hand-assembled mixed-batching directory can still pass. |
| B2 — 5,336/duplicate G1 | **RESOLVED** | Duplicate-free G1 enumeration and exact D1≡G1≡rows identity equality are enforced before metrics at [meshgrid_report.py:466](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:466). |
| B3 — mirrored truth | **NOT** | The metadata digest is optional/TOFU at [meshgrid_report.py:773](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:773). The probe comparison at [meshgrid_offgrid_probe.py:641](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:641) is circular: loader `md["source"]` comes from the same pair JSON at [AR_md.py:31](/home/yixunhu/codespace/FLAC/src/configs/dataset_configs/custom_metadata/AR_md.py:31). A trusted digest supplied before evaluation would close this; recording then feeding it back proves stability, not origin. |
| B4 — probe run/row join | **PARTIALLY** | Foreign-row acceptance is fixed by [meshgrid_offgrid_probe.py:561](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:561), and the full census/join is reused at line 127; however, the copied-merge-receipt path from B1 still admits a never-merged run. |
| B5 — device before gate | **PARTIALLY** | Model/scorer transfer is correctly after the gate at [meshgrid_offgrid_probe.py:829](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:829), but checkpoint validation imports `eval_FLAC`, whose function default evaluates `torch.cuda.is_available()` at [eval_FLAC.py:1192](/home/yixunhu/codespace/FLAC/eval_FLAC.py:1192) before `gate_run` returns. No allocation occurs, but strict “no device touch” is not met. |
| M6 — registered protocol | **PARTIALLY** | Most settings are enforced at [meshgrid_report.py:423](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:423), but any `ckpt_sha256` is considered registered unless an optional expectation is supplied at line 445. Decision 2d identifies the admitted wrapper `c4c67882…`, so it should be pinned. The claim that it “was hash-checked” against the EMA extract contradicts [loc_meshgrid_worklog.md:42](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude/loc_meshgrid_worklog.md:42), which says that cross-check remained optional. Dataset-config identity is also omitted. |
| M7 — float16 bound | **PARTIALLY** | Dtype enforcement, Lipschitz reasoning, unconditional aggregate checks, and the non-causal rename are good. But [meshgrid_report.py:886](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:886) uses only `np.spacing(y)`, which follows the smaller toward-zero gap for negative float16 binade boundaries. For stored `-0.5`, honest roundoff can be roughly twice the computed half-ULP bound, causing false refusal. Both adjacent representables must be considered. |
| M8 — latency | **PARTIALLY** | Room-first/bootstrap aggregation is implemented at [meshgrid_report.py:1266](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1266), but incomplete rows are excluded while a canonical-looking endpoint is still emitted. Selective missingness can bias the result or remove a room. |
| M9 — staged NPZs | **PARTIALLY** | Mid-run files are quarantined and labels are embedded, but [meshgrid_offgrid_probe.py:366](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:366) moves finals sequentially before JSON/Markdown are written at line 436. A later failure still leaves partial, unmanifested final NPZs. |
| Minor — rank-one ties | **RESOLVED** | Strict wins and best-score ties are separated at [meshgrid_offgrid_probe.py:460](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:460). |
| Minor — visualization tie-break | **RESOLVED** | All three selections choose the smallest global position at [meshgrid_report.py:1582](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:1582). |
| Minor — Markdown columns | **RESOLVED** | Mean `e_excess` and both 1.0 m baseline columns are present at [meshgrid_report.py:2052](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_report.py:2052) and line 2114. |
| Minor — disclosure propagation | **PARTIALLY** | Cases JSON and off-grid JSON are fixed, but off-grid Markdown [meshgrid_offgrid_probe.py:492](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_offgrid_probe.py:492) renders neither latency scope nor controls elsewhere; standalone NPZ labels at line 350 omit them and sensitivity status. |

## r9b retrieval-control review

The bank intersection itself is sound: metadata plus an existing WAV, same receiver, numeric self-pair exclusion, and collocated-target refusal at [meshgrid_retrieval_control.py:341](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:341). `numeric_identity` is clearly the registered rule and `released_eligible_pool` is visibly labeled as sensitivity. Sparse-oracle arithmetic and labeling are also sound at [meshgrid_retrieval_control.py:793](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:793).

New result-corrupting findings:

- **BLOCKER — sparse-bank inputs are unbound.** [meshgrid_retrieval_control.py:690](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:690) reads caller-selected pair metadata and RIR bytes, but neither bank is hashed or gated. Alternate bytes can change membership, similarities, coordinates, sparse oracle, and every reported error.

- **BLOCKER — retrieval truth remains scalar-spoofable.** [meshgrid_retrieval_control.py:602](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:602) uses only the non-injective grid-oracle scalar and nevertheless reports that it “pins_the_truth” at line 942. It uses neither the metadata-bank gate nor a genuinely independent vector witness.

- **BLOCKER — `model_config_sha256` is wrongly classified as irrelevant.** It is omitted at [meshgrid_retrieval_control.py:1179](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:1179), yet that model config builds the observed-RIR loader at line 1297. Sample-rate/size changes can alter `obs_wav` and every cosine while the binding passes.

- **MAJOR — K=1 LME equals cosine only algebraically.** [meshgrid_retrieval_control.py:496](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:496) performs float32 divide/logsumexp/multiply. This is not bit-identical to the cosine and can turn adjacent scores into a tie. Arbitrary positive `tau` is accepted at line 1251, yet actual tau is neither gated nor faithfully recorded.

- **MAJOR — binding partition is syntactically complete but semantically unsound.** `RETRIEVAL_BINDING_FIELDS ∪ NOT_CHECKED == RUN_BINDING_FIELDS` does hold at [meshgrid_retrieval_control.py:155](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_retrieval_control.py:155). It nevertheless omits inputs outside `RUN_BINDING_FIELDS`, misclassifies model config and tau, and only matches the run rather than enforcing the registered AGREE/protocol.

Static inspection only. No code, imports, tests, installs, file changes, or GPU operations were performed.
