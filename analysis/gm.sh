#!/bin/bash
# gm wrapper: sandbox-safe gradmotion CLI (HOME redirect + pooled API key)
# usage: ./analysis/gm.sh <gm args...>
# NOTE: active training account = pool id 31 (lavah77607, r23 TASK_20261006_054)
#       19-34 exhausted (1002056); 35 funded (batch 6, I87: log batch size)
set -euo pipefail
REPO=/Users/yumx/code/x1_DM
KEY=$(awk -F' ' '$1=="43"{print $2}' "$REPO/.repos/keys.txt")
[ -z "$KEY" ] && { echo "no key for account 43 in .repos/keys.txt" >&2; exit 1; }
export HOME="$REPO/.gmhome"
export GM_API_KEY="$KEY"
exec gm "$@"
