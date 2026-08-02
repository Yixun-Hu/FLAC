#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
ASSETS="${ASSETS:-none}"

"$PYTHON" worklog/worklog_zhixuan/cyl_vit_test/verify_config.py \
  --instantiate --assets "$ASSETS"
