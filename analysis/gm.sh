#!/bin/bash
# gm wrapper: sandbox-safe gradmotion CLI (HOME redirect + pooled API key)
# usage: ./analysis/gm.sh <gm args...>
# NOTE: active training account = pool id 28 (jefavi2847, r17 TASK_20261005_039)
#       19-27 exhausted (1002056); 28 is the LAST funded account
set -euo pipefail
REPO=/Users/yumx/code/x1_DM
KEY=$(awk -F' ' '$1=="28"{print $2}' "$REPO/.repos/keys.txt")
[ -z "$KEY" ] && { echo "no key for account 28 in .repos/keys.txt" >&2; exit 1; }
export HOME="$REPO/.gmhome"
export GM_API_KEY="$KEY"
exec gm "$@"
