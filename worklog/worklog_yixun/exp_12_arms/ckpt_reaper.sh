#!/bin/bash
# RETIRED 2026-08-20 by Yixun's standing rule: "Please do not delete the checkpoints
# without my permission, never." This script previously deleted intermediate checkpoints
# of a live run; under the rule NO automation may delete checkpoints. It now refuses.
# Disk pressure is handled by disk_guard.sh, which STOPS a training gracefully at a
# complete checkpoint and leaves every file in place for a human decision.
echo "REFUSE: ckpt_reaper.sh is retired -- checkpoints are never deleted automatically" >&2
exit 2
