#!/usr/bin/env bash
# exp_22 A7 r6 evidence (Planner, Claude Opus 5.5): why a worktree at PIN2 b722d5f0 cannot pass the merged launcher's
# two checks -- `readlink -f weights` = the main checkout's weights AND a clean tracked tree -- when weights/ is TRACKED.
# A throwaway repo under the scratchpad mirrors the layout (a tracked weights/FLAC/m.json); nothing else is touched.
#   bash exp22_weights_link_repro.sh <empty scratch dir>
set -u
T=$1; [ ! -e "$T" ] || { echo "REFUSED: $T exists"; exit 1; }
mkdir -p "$T/repo/weights/FLAC" "$T/repo/src" "$T/main_weights/FLAC" || exit 1
cd "$T/repo" && git init -q && echo a > weights/FLAC/m.json && echo code > src/x.py && echo ckpt > "$T/main_weights/FLAC/model.ckpt" \
  && cp weights/FLAC/m.json "$T/main_weights/FLAC/" && git add . && git -c user.name=t -c user.email=t@t commit -q -m t || exit 1
echo "git $(git --version | cut -d' ' -f3); the repo tracks: $(git ls-files | tr '\n' ' ')"
echo "## (a) a worktree whose weights IS a link to the main weights (what the merged launcher demands)"
git worktree add -q --detach "$T/wt_a" HEAD && rm -rf "$T/wt_a/weights" && ln -s "$T/main_weights" "$T/wt_a/weights"
echo "readlink -f weights = $(readlink -f "$T/wt_a/weights")"; echo "status --porcelain -uno: [$(git -C "$T/wt_a" status --porcelain --untracked-files=no)]"
echo "## (b) a sparse checkout excluding /weights/, then the link"
git worktree add -q --detach "$T/wt_b" HEAD && git -C "$T/wt_b" sparse-checkout set --no-cone '/*' '!/weights/'
echo "after sparse-checkout: ls-files -v = [$(git -C "$T/wt_b" ls-files -v | tr '\n' ' ')]"
ln -s "$T/main_weights" "$T/wt_b/weights"; echo "status --porcelain -uno: [$(git -C "$T/wt_b" status --porcelain --untracked-files=no)]"
echo "## (c) the A7 r6 layout: the tracked weights/ kept, one link per untracked entry"
git worktree add -q --detach "$T/wt_c" HEAD && ln -s "$T/main_weights/FLAC/model.ckpt" "$T/wt_c/weights/FLAC/model.ckpt"
echo "status --porcelain -uno: [$(git -C "$T/wt_c" status --porcelain --untracked-files=no)]"
[ "$T/wt_c/weights/FLAC/model.ckpt" -ef "$T/main_weights/FLAC/model.ckpt" ] && echo "-ef identity of the linked asset: yes"
cp "$T/main_weights/FLAC/model.ckpt" "$T/copy.ckpt"; [ "$T/copy.ckpt" -ef "$T/main_weights/FLAC/model.ckpt" ] || echo "-ef of a byte-identical copy: no (as required)"
