#!/usr/bin/env bash
# Copyright 2026 Google LLC.
#
# Push the files the decode engine needs onto REMOTE_HOST at the SAME absolute
# paths, then verify with md5. Run this after build_raw_transfer.sh so both
# hosts execute byte-identical code. Mirrors scripts/sync_to_decode.sh.
#
# Usage:  ./sync_oss_to_remote.sh
# Env:    REMOTE_HOST (default 10.128.1.8), OSS_DIR

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/disagg_env.sh"

# Files that must match on the decode host:
#   - the compiled nanobind extension (the actual transfer engine)
#   - the python wrappers
#   - this harness (so the remote runs the same runner)
FILES=(
  "${OSS_DIR}/api/jax/_kv_cache_manager.so"
  "${OSS_DIR}/api/jax/disagg_kv_cache_manager.py"
  "${OSS_DIR}/kv_cache/disagg_proxy.py"
  "${DISAGG_SCRIPTS_DIR}/disagg_env.sh"
  "${DISAGG_SCRIPTS_DIR}/disagg_node_runner.py"
)

echo "Syncing $(printf '%s\n' "${#FILES[@]}") files to ${REMOTE_HOST} ..."
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "FAIL  missing locally: $f" >&2
    exit 1
  fi
  abs="$(readlink -f "$f")"
  # Remote artifacts (e.g. the .so) may be mode 0555 from a prior build, which
  # blocks scp overwrite — clear the destination first (dir is writable).
  ssh "${REMOTE_HOST}" "mkdir -p '$(dirname "$abs")'; chmod u+w '$abs' 2>/dev/null; rm -f '$abs'"
  scp -q "$abs" "${REMOTE_HOST}:$abs"
  local_md5=$(md5sum "$abs" | awk '{print $1}')
  remote_md5=$(ssh "${REMOTE_HOST}" "md5sum '$abs'" | awk '{print $1}')
  if [[ "$local_md5" == "$remote_md5" ]]; then
    echo "OK    $abs"
  else
    echo "FAIL  $abs (local=$local_md5 remote=$remote_md5)" >&2
    exit 1
  fi
done
echo "Sync complete."
