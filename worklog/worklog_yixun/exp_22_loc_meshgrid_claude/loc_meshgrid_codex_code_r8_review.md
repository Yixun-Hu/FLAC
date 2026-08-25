**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.147.0, read-only) · **Date:** 2026-08-25

*Round exp22-r8. 5 RESOLVED / 3 PARTIAL; three narrow residuals (2ε argmax boundary; merge digest recompute + restart-safe source census; dump digest in strict binding). Post-fix launch commands included verbatim. Body verbatim.*

---

## Verdict: REJECT — launch blocked

At `localization-exp@5f63cc7`, scoped to `0c10d42..6a69808`: **5 RESOLVED, 3 PARTIALLY RESOLVED, 0 NOT RESOLVED**. Do not launch the probe or P1 from current HEAD.

| # | Gate | Status | Finding |
|---|---|---|---|
| 1 | CRN | **RESOLVED** | Shared-across-candidates is the default; the driver unconditionally refuses `per_candidate`, while the engine alternative requires explicit test-only opt-in ([engine](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:120), [driver](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:161)). |
| 2 | GT | **RESOLVED** | `GuardedMetadata` blocks both target fields, including `dict()` copies; the mid-pass conditioner sentinel and real-stack cache-parity run demonstrate that target access aborts while the released conditioner succeeds ([guard](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:486), [pass](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1775)). |
| 3 | Resume | **RESOLVED** | Resume requires a pre-existing matching binding; rows carry a full-row digest and binding digest, named sidecars are checked, and skipped rows are matched against their exact G1 identity/candidate list ([driver](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:221), [verification](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1014)). |
| 4 | Binding | **RESOLVED** | `cond_autocast` and dataset-config SHA are strict fields; effective batching is row-stamped and advisory changes are persisted ([binding](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:881)). |
| 5 | Determinism | **PARTIALLY** | The tooling exists, but a per-score bound `ε` only guarantees argmax stability when margin `>2ε`; the code uses `>ε` and counts risk at `≤ε`, falsely declaring margins in `(ε,2ε]` stable. The real-stack replay also remained skipped ([classification](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:283), [comparison](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1237)). |
| 6 | Probe | **RESOLVED** | Raw records carry complete binding/advisory provenance, receiver-group identity, union size and separate waveform/source/context denominators, sufficient to rederive the registered projection independently ([projection](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1416)). |
| 7 | `--rooms` / merge | **PARTIALLY** | Filtering satisfies the room contract, but merge trusts stored `binding_sha256` strings instead of recomputing every shard’s digest; additionally, source-row census uses only the final invocation’s summary, so a normal resume after completed receiver groups cannot merge ([merge binding](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:2173), [source census](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:2257)). |
| 8 | Dump digest | **PARTIALLY** | The supplied digest and named waveform sidecar are verified, but the case-list digest is absent from the strict run binding; a resume may change dump authority and skip an already-complete query without producing its newly requested dump ([dump check](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:1304), [binding fields](/home/yixunhu/codespace/FLAC/src/localization/meshgrid_engine.py:881)). |

Required fixes: use the `2*SCORE_TOLERANCE` stability boundary; recompute/compare every shard binding and derive or accumulate restart-safe source census; put the registered dump-case digest into the strict binding.

Nit: [localize_meshgrid.py](/home/yixunhu/codespace/FLAC/localize_meshgrid.py:13) still describes the now-rejected per-candidate key as a deviation.

The ledger reports 3,036 passed / 11 skipped; tests were not rerun because of the strict read-only constraint. Review followed the repository SOP and the [OpenAI Docs local-diff review workflow](https://learn.chatgpt.com/use-cases/scan-code-changes-for-security).

## Post-fix launch call 1: registered probe sweep

```bash
for batch_rows in 64 128 256; do
  python localize_meshgrid.py \
    --ckpt-path weights/exp20/P1_40k.ckpt \
    --agree-ckpt weights/AGREE/AGREE_fullAR.pt \
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --context-manifest outputs_loc/exp22/d1_context_manifest.json \
    --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json \
    --branch z_band \
    --device cuda:0 \
    --probe 1 \
    --probe-room Bathrooms/Bathrooms_idx_14 \
    --probe-stem "P1_CRN_br${batch_rows}" \
    --out-dir "outputs_loc/exp22/i1_probe_P1_br${batch_rows}" \
    --noise-policy shared_across_candidates \
    --seed 42 --tau 0.1 --num-samples 8 --k-prefixes 1 4 8 \
    --steps 1 --cfg-scale 1.0 --cond-autocast default \
    --source-chunk 16 --batch-rows "${batch_rows}"
done
```

Select the smallest `batch_rows` attaining the minimum independently rederived projected total. Proceed only if that total is `≤175` GPU-hours.

## Post-fix launch call 2: conditional P1 sharded pass and mandatory merge

Set `BATCH_ROWS` to the registered probe winner first.

```bash
BATCH_ROWS=<64|128|256>
RUN_TAG="P1_CRN_br${BATCH_ROWS}_$(date +%Y%m%d_%H%M%S)"
CAFE_OUT="outputs_loc/exp22/i1_${RUN_TAG}_cafe"
REST_OUT="outputs_loc/exp22/i1_${RUN_TAG}_rest15"
MERGED_OUT="outputs_loc/exp22/i1_${RUN_TAG}_merged"

COMMON=(
  --ckpt-path weights/exp20/P1_40k.ckpt
  --agree-ckpt weights/AGREE/AGREE_fullAR.pt
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
  --context-manifest outputs_loc/exp22/d1_context_manifest.json
  --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json
  --branch z_band
  --device cuda:0
  --noise-policy shared_across_candidates
  --seed 42 --tau 0.1 --num-samples 8 --k-prefixes 1 4 8
  --steps 1 --cfg-scale 1.0 --cond-autocast default
  --source-chunk 16 --batch-rows "${BATCH_ROWS}"
)

CUDA_VISIBLE_DEVICES=0 python localize_meshgrid.py "${COMMON[@]}" \
  --out-dir "${CAFE_OUT}" \
  --rooms Cafe/Cafe_idx_1 &
p0=$!

CUDA_VISIBLE_DEVICES=1 python localize_meshgrid.py "${COMMON[@]}" \
  --out-dir "${REST_OUT}" \
  --rooms \
    Apartments/Apartments_idx_42 \
    Apartments/Apartments_idx_50 \
    Auditorium/Auditorium_idx_1 \
    Bathrooms/Bathrooms_idx_14 \
    Bathrooms/Bathrooms_idx_18 \
    Bedrooms/Bedrooms_idx_18 \
    Bedrooms/Bedrooms_idx_33 \
    LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25 \
    LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30 \
    MeetingRoom/MeetingRoom_idx_20 \
    MeetingRoom/MeetingRoom_idx_32 \
    Office/Office_idx_10 \
    Office/Office_idx_11 \
    Restaurants/Restaurants_idx_22 \
    Restaurants/Restaurants_idx_24 &
p1=$!

wait "${p0}" && wait "${p1}"

python localize_meshgrid.py \
  --merge-shards "${CAFE_OUT}" "${REST_OUT}" \
  --merge-out "${MERGED_OUT}" \
  --context-manifest outputs_loc/exp22/d1_context_manifest.json \
  --audit-report outputs_loc/exp22/g1_audit/geometry_audit_report.json \
  --branch z_band
```

This preserves the registered split: Cafe `4,559,398` pairs; remaining rooms `4,337,142`; merge must certify all `5,337 / 8,896,540 / 966,147 / 71,172,320` totals.