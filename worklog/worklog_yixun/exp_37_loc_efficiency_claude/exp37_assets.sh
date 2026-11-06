#!/usr/bin/env bash
# exp_37 loc_efficiency, Coder round B-a item 1: the worktree's UNTRACKED asset links -- create them, or verify them.
#
#   bash exp37_assets.sh link   <worktree>   create the links (refuses if any exists), then verify
#   bash exp37_assets.sh verify <worktree>   read-only: every link, its target, its pinned sha256; the 16 audit meshes
#
# Layout: AcousticRooms -> the NAS dataset; weights/{exp20,AGREE}/ are REAL directories of the worktree holding one
# symlink per pinned file of the main checkout (read only; nothing of the main checkout is ever written). The
# geometry audit binds the other host's absolute mesh paths: the 16 meshes are resolved here under
# AcousticRooms/room_mesh_obj_format/<Scene>/<Room>.obj and must hash to the audit's mesh_sha256 (the runtime
# remap `--mesh-root` re-checks every mesh before use). Every other input of localize_FLAC.py is tracked in the
# branch (model config, manifests, audit, AGREE/AGREE/model_configs/dinov3_vits16_local/config.json).
# Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a item 1.
set -uo pipefail
MAIN=/home/yixunhu/codespace/FLAC
ROOMS=/media/diskstation/yixunhu/FLAC/AcousticRooms
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
BRANCH=exp-37-loc-efficiency
AUDIT_REL=worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json
ORDER=(weights/exp20/P1_40k.ckpt weights/exp20/BF_40k.ckpt weights/exp20/YAW_40k.ckpt weights/AGREE/AGREE_fullAR.pt)
declare -A SHA=(  # P1 / BF / YAW: localization-exp src/localization/meshgrid_report.py REGISTERED_CKPT_SHA256;
                  # AGREE: the latency bundle's run manifests (agree_checkpoint_sha256)
  [weights/exp20/P1_40k.ckpt]=c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c
  [weights/exp20/BF_40k.ckpt]=5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328
  [weights/exp20/YAW_40k.ckpt]=ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7
  [weights/AGREE/AGREE_fullAR.pt]=3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787
)
REQ_DIRS=(AcousticRooms/single_channel_ir_1 AcousticRooms/depth_map AcousticRooms/room_mesh_obj_format)

if ! { [ "$#" -eq 2 ] && [[ "$1" =~ ^(link|verify)$ ]]; }; then
  echo "usage: bash $0 link|verify <worktree>"; exit 2
fi
MODE=$1; WT=$2
die() { echo "REFUSED: $*"; exit 1; }
[ "$(realpath -m -- "$WT")" = "$WT" ] || die "the worktree path must be canonical (no link, no trailing slash)"
case "$WT/" in "$MAIN"/*) die "the worktree must not lie inside the main checkout ${MAIN}" ;; esac
[ "$(git -C "$WT" rev-parse --show-toplevel 2>/dev/null)" = "$WT" ] || die "${WT} is not the top of a git worktree"
[ "$(git -C "$WT" symbolic-ref -q --short HEAD)" = "$BRANCH" ] || die "${WT} is not on branch ${BRANCH}"
echo "=== exp37_assets ${MODE} -- $(date -Is) -- ${WT} @ $(git -C "$WT" rev-parse HEAD) -- this script $(sha256sum < "$0" | cut -c1-64)"

if [ "$MODE" = link ]; then
  for rel in AcousticRooms weights "${ORDER[@]}"; do
    [ ! -e "$WT/$rel" ] && [ ! -L "$WT/$rel" ] || die "${WT}/${rel} exists"
  done
  ln -s "$ROOMS" "$WT/AcousticRooms" || die "the AcousticRooms link"
  for rel in "${ORDER[@]}"; do
    mkdir -p "$(dirname "$WT/$rel")" || die "mkdir $(dirname "$rel")"
    ln -s "$MAIN/$rel" "$WT/$rel" || die "the link ${rel}"
  done
  echo "linked; verifying"
fi

FAIL=0
ok() { echo "OK: $*"; }
bad() { echo "FAILED: $*"; FAIL=1; }
echo "--- ls -la (links, then their targets)"
ls -la "$WT/AcousticRooms" "$WT/weights" "$WT/weights/exp20" "$WT/weights/AGREE" 2>&1
ls -laL "$WT/AcousticRooms/" "${ORDER[@]/#/$WT/}" 2>&1
echo "---"
if [ -L "$WT/AcousticRooms" ] && [ "$(readlink "$WT/AcousticRooms")" = "$ROOMS" ]; then ok "AcousticRooms -> ${ROOMS}"; else bad "AcousticRooms -> ${ROOMS}"; fi
for rel in "${REQ_DIRS[@]}"; do if [ -d "$WT/$rel" ]; then ok "required ${rel}/"; else bad "required ${rel}/"; fi; done
for d in weights weights/exp20 weights/AGREE; do
  if [ -d "$WT/$d" ] && [ ! -L "$WT/$d" ]; then ok "${d}/ is a real directory"; else bad "${d}/ is a real directory"; fi
done
got="$(cd "$WT" && find weights -mindepth 1 | sort)"
want="$(printf '%s\n' weights/AGREE weights/exp20 "${ORDER[@]}" | sort)"
if [ "$got" = "$want" ]; then ok "weights/ holds exactly the pinned links"; else bad "weights/ holds: ${got//$'\n'/ }"; fi
for rel in "${ORDER[@]}"; do
  if [ -L "$WT/$rel" ] && [ "$(readlink "$WT/$rel")" = "$MAIN/$rel" ] && [ "$WT/$rel" -ef "$MAIN/$rel" ]; then
    ok "${rel} -> ${MAIN}/${rel} (same device + inode)"
  else
    bad "${rel} -> ${MAIN}/${rel}"
  fi
  digest="$(sha256sum -- "$WT/$rel" 2>/dev/null | cut -c1-64)"
  if [ "$digest" = "${SHA[$rel]}" ]; then ok "sha256 ${rel} = ${digest}"; else bad "sha256 ${rel} = ${digest:-unreadable}, pinned ${SHA[$rel]}"; fi
  if git -C "$WT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then bad "${rel} is TRACKED"; fi
done
if git -C "$WT" ls-files --error-unmatch -- AcousticRooms >/dev/null 2>&1; then bad "AcousticRooms is TRACKED"; fi
echo "--- the 16 audit meshes under AcousticRooms/room_mesh_obj_format (vs the audit's mesh_sha256)"
if (cd "$WT" && "$PY" - "$AUDIT_REL" <<'EOF'
import hashlib, json, os, sys
audit = json.load(open(sys.argv[1]))
root = audit["mesh_root"].rstrip("/")
bad = 0
for room, row in sorted(audit["rooms"].items()):
    want = f"{root}/{row['scene']}/{room}.obj"
    local = os.path.join("AcousticRooms", "room_mesh_obj_format", row["scene"], f"{room}.obj")
    digest = hashlib.sha256(open(local, "rb").read()).hexdigest() if os.path.isfile(local) else None
    good = row["mesh_path"] == want and digest == row["mesh_sha256"]
    bad += not good
    print(("OK" if good else "FAILED") + f": {room}: {local} sha256 {digest} audit {row['mesh_sha256']}")
print(f"{len(audit['rooms']) - bad}/{len(audit['rooms'])} meshes match")
sys.exit(1 if bad or len(audit["rooms"]) != 16 else 0)
EOF
); then ok "16/16 audit meshes"; else bad "the audit meshes"; fi
if [ "$FAIL" = 0 ]; then echo "VERIFY PASS"; else echo "VERIFY FAIL"; fi
exit "$FAIL"
