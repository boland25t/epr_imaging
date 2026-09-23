#!/bin/bash
# Launch the simple UI with a workspace (default J1756); avoids nested-quote loss.
cd /home/troyboland/epr_imaging/epr_imaging
exec python3 simple_main.py "${1:-/mnt/f/EPR_2026_PROCESSED/J1756_down.eprproj}"
