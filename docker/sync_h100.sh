#!/bin/bash
# Push the working tree to the H100 host through `smc toc` (no scp/rsync there).
# Usage: docker/sync_h100.sh [paths...]   (default: python tests examples tools docker)
set -euo pipefail
REMOTE="${REMOTE:-10.249.32.139}"
REMOTE_DIR="${REMOTE_DIR:-/data1/min.yang/mbsa-h100/mbsa}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$ROOT"
PATHS=("$@")
[ ${#PATHS[@]} -eq 0 ] && PATHS=(python tests examples tools docs docker pyproject.toml README.md PROVENANCE.md MANIFEST.in .dockerignore)
COPYFILE_DISABLE=1 tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' -czf "$TMP/code.tgz" "${PATHS[@]}"
base64 -i "$TMP/code.tgz" | tr -d '\n' | fold -w 60000 > "$TMP/code.b64"
MD5=$(md5 -q "$TMP/code.tgz")
smc toc "$REMOTE" "rm -f $REMOTE_DIR.b64 && mkdir -p $REMOTE_DIR && echo ok" >/dev/null 2>&1
while IFS= read -r line || [ -n "$line" ]; do
  smc toc "$REMOTE" "printf '%s' '$line' >> $REMOTE_DIR.b64" </dev/null >/dev/null 2>&1
done < "$TMP/code.b64"
smc toc "$REMOTE" "cd $REMOTE_DIR && base64 -d $REMOTE_DIR.b64 > /tmp/mbsa_code.tgz && [ \$(md5sum /tmp/mbsa_code.tgz | cut -d' ' -f1) = $MD5 ] && tar xzf /tmp/mbsa_code.tgz && rm -f $REMOTE_DIR.b64 && echo SYNC_OK || echo SYNC_FAIL" 2>&1 | grep -E 'SYNC_(OK|FAIL)'
