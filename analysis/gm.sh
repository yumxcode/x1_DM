#!/bin/bash
# gm wrapper: sandbox-safe gradmotion CLI (HOME redirect + pooled API key)
# usage: ./analysis/gm.sh <gm args...>
# NOTE: active training account = pool id 26 (mecoxos612, r14 TASK_20261004_089)
#       19-25 exhausted (1002056); 26 is the last funded account
set -euo pipefail
REPO=/Users/yumx/code/x1_DM
KEY=$(awk -F' ' '$1=="26"{print $2}' "$REPO/.repos/keys.txt")
[ -z "$KEY" ] && { echo "no key for account 26 in .repos/keys.txt" >&2; exit 1; }
export HOME="$REPO/.gmhome"
export GM_API_KEY="$KEY"
exec gm "$@"
