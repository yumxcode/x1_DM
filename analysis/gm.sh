#!/bin/bash
# gm wrapper: sandbox-safe gradmotion CLI (HOME redirect + pooled API key)
# usage: ./analysis/gm.sh <gm args...>
# NOTE: active training account = pool id 27 (fayadag545, r16 TASK_20261005_033)
#       19-26 exhausted (1002056); 27/28 funded (user-added)
set -euo pipefail
REPO=/Users/yumx/code/x1_DM
KEY=$(awk -F' ' '$1=="27"{print $2}' "$REPO/.repos/keys.txt")
[ -z "$KEY" ] && { echo "no key for account 27 in .repos/keys.txt" >&2; exit 1; }
export HOME="$REPO/.gmhome"
export GM_API_KEY="$KEY"
exec gm "$@"
