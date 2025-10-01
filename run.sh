#!/usr/bin/env bash
set -euo pipefail

export TORCH_COMPILE_OFF=0

# Defaults
NPROC=8
CHECKPOINT=""

# Require positional CONFIG first
if [ $# -lt 1 ]; then
  echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH]" >&2
  exit 1
fi

CONFIG="$1"
shift

# Parse options appearing AFTER CONFIG
OPTIND=1
while getopts ":n:p:" opt; do
  case "$opt" in
    n)
      NPROC="$OPTARG"
      ;;
    p)
      CHECKPOINT="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH]" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH]" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

export RUN_ID=1
export OMP_NUM_THREADS=8

CMD=(torchrun --standalone --nproc_per_node="$NPROC" train.py "$CONFIG")
if [ -n "$CHECKPOINT" ]; then
  CMD+=("--checkpoint=$CHECKPOINT")
endif

"${CMD[@]}"