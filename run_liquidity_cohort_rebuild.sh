#!/bin/bash
# Rebuild each flat%>5 market in its own process (crash isolation).
set -u
cd "$(dirname "$0")"
PY=/Users/Work/NoteBooks/orklys_env/bin/python
LOG=liquidity_roll_rebuild_needs.log
CLIENT=110

echo "=== Cohort rebuild started $(date) ===" | tee "$LOG"

# Refresh audit list
$PY -u liquidity_roll_pilot.py --rank-only | tee -a "$LOG"

# Read needs_rebuild markets from audit JSON
MARKETS=$($PY - <<'PY'
import json
for m in json.load(open("liquidity_roll_audit.json"))["needs_rebuild"]:
    print(m["market"])
PY
)

i=0
ok=0
fail=0
while IFS= read -r mkt; do
  [ -z "$mkt" ] && continue
  i=$((i+1))
  cid=$((CLIENT + i))
  echo "" | tee -a "$LOG"
  echo "######## [$i] $mkt (clientId=$cid) ########" | tee -a "$LOG"
  # Per-market timeout: IB hangs shouldn't kill the whole cohort.
  if $PY -u liquidity_roll_pilot.py --market "$mkt" --client-id "$cid" >> "$LOG" 2>&1; then
    ok=$((ok+1))
    echo "######## [$i] OK ########" | tee -a "$LOG"
  else
    rc=$?
    fail=$((fail+1))
    echo "######## [$i] FAILED (exit $rc) ########" | tee -a "$LOG"
  fi
  sleep 2
done <<< "$MARKETS"

echo "" | tee -a "$LOG"
echo "=== Rebuild finished $(date): ok=$ok fail=$fail ===" | tee -a "$LOG"

# Promote all validation_passed liquidity markets and refresh cot_data
echo "=== Promoting passed markets ===" | tee -a "$LOG"
$PY -u liquidity_roll_pilot.py --promote-passed --refresh-cot >> "$LOG" 2>&1
echo "=== Promote/refresh exit $? ===" | tee -a "$LOG"

# Final audit
$PY -u liquidity_roll_pilot.py --rank-only | tee -a "$LOG"
echo "=== DONE $(date) ===" | tee -a "$LOG"
