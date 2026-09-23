#!/usr/bin/env bash
# Resilient driver for the EPR fauna fine-tune.
#
# The Windows host has rebooted twice mid-run.  Ultralytics writes
# weights/last.pt after every epoch, so a restart only ever costs the time
# since the last epoch boundary -- provided something actually restarts it.
# This script is that something: it re-invokes the trainer with resume=True
# until the run writes its completion marker (train_meta_<tax>.json).
#
#   nohup setsid ./fauna_train_supervise.sh v2 > supervise_v2.log 2>&1 &
#
# After a host reboot, re-run the exact same line: it picks up from last.pt.
# Idempotent -- if the run already finished it exits immediately.

set -uo pipefail

TAX="${1:-v2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OUT="/home/troyboland/models/finetune"
RUN_DIR="$MODEL_OUT/epr_fauna_${TAX}"
LAST="$RUN_DIR/weights/last.pt"
MARKER="$MODEL_OUT/train_meta_${TAX}.json"
WEIGHTS_OUT="epr_fauna_yolo_${TAX}.pt"
LOG="$MODEL_OUT/supervise_${TAX}.log"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
BACKOFF="${BACKOFF:-20}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [[ -f "$MARKER" ]]; then
  log "marker $MARKER already present - training complete, nothing to do"
  exit 0
fi

# Refuse to run two supervisors (or a supervisor beside a manual trainer).
if pgrep -f "[f]auna_finetune.py train --taxonomy ${TAX}" >/dev/null; then
  log "a trainer for ${TAX} is already running - refusing to start a second"
  exit 0
fi

attempt=0
while [[ ! -f "$MARKER" ]]; do
  attempt=$((attempt + 1))
  if (( attempt > MAX_ATTEMPTS )); then
    log "giving up after ${MAX_ATTEMPTS} attempts - investigate $LOG"
    exit 1
  fi

  if [[ -f "$LAST" ]]; then
    ep=$(tail -1 "$RUN_DIR/results.csv" 2>/dev/null | cut -d, -f1)
    log "attempt ${attempt}: resuming from $LAST (last completed epoch: ${ep:-?})"
    python3 "$REPO/fauna_finetune.py" train --taxonomy "$TAX" \
            --resume "$LAST" --weights-out "$WEIGHTS_OUT"
  else
    log "attempt ${attempt}: no checkpoint, starting fresh"
    python3 "$REPO/fauna_finetune.py" train --taxonomy "$TAX" \
            --epochs 60 --batch 8 --device 0 --workers 4 \
            --name "epr_fauna_${TAX}" --weights-out "$WEIGHTS_OUT"
  fi
  rc=$?

  if [[ -f "$MARKER" ]]; then
    log "training finished cleanly (exit ${rc})"
    break
  fi
  log "trainer exited ${rc} without a marker - retrying in ${BACKOFF}s"
  sleep "$BACKOFF"
done

log "done: $(cat "$MARKER" 2>/dev/null | tr -d '\n')"
