#!/usr/bin/env bash
set -euo pipefail

export TORCH_COMPILE_OFF=0

# Default values
NPROC=8

# Parse options: -n NUM_PROCS
while getopts ":n:" opt; do
  case "$opt" in
    n)
      NPROC="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 [-n NUM_PROCS] CONFIG_FILE" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 [-n NUM_PROCS] CONFIG_FILE" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

# Positional argument: config file (required)
if [ $# -lt 1 ]; then
  echo "Usage: $0 [-n NUM_PROCS] CONFIG_FILE" >&2
  exit 1
fi
CONFIG="$1"

export RUN_ID=1
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node="$NPROC" \
    train.py "$CONFIG"
