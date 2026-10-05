#!/bin/bash
# gm wrapper: sandbox-safe gradmotion CLI (HOME redirect + pooled API key)
# usage: ./analysis/gm.sh <gm args...>
# NOTE: active training account = pool id 30 (solokot916, r22 TASK_20261006_018)
#       19-29 exhausted (1002056); 30 is the LAST funded account
set -euo pipefail
REPO=/Users/yumx/code/x1_DM
KEY=$(awk -F' ' '$1=="30"{print $2}' "$REPO/.repos/keys.txt")
[ -z "$KEY" ] && { echo "no key for account 30 in .repos/keys.txt" >&2; exit 1; }
export HOME="$REPO/.gmhome"
export GM_API_KEY="$KEY"
exec gm "$@"
