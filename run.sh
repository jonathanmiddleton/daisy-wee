#!/usr/bin/env bash
set -euo pipefail

export TORCH_COMPILE_OFF=0

# Defaults
NPROC=8
CHECKPOINT=""
BEGIN_SHARD=""

# Require positional CONFIG first
if [ $# -lt 1 ]; then
  echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [--ignore-prior-steps] [key=value ...]" >&2
  exit 1
fi

CONFIG="$1"
shift

# Parse options appearing AFTER CONFIG
OPTIND=1
while getopts ":n:p:s:" opt; do
  case "$opt" in
    n)
      NPROC="$OPTARG"
      ;;
    p)
      CHECKPOINT="$OPTARG"
      ;;
    s)
      BEGIN_SHARD="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [--ignore-prior-steps] [key=value ...]" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [--ignore-prior-steps] [key=value ...]" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

export RUN_ID=1
export OMP_NUM_THREADS=8

# Optionally set beginning shard index for training data
if [ -n "$BEGIN_SHARD" ]; then
  export BEGIN_SHARD
fi

CMD=(torchrun --standalone --nproc_per_node="$NPROC" train.py "$CONFIG")
if [ -n "$CHECKPOINT" ]; then
  CMD+=("--init_checkpoint=$CHECKPOINT")
fi

# Forward any remaining args as overrides. Accept key=value, --key=value, and supported bare flags.
for arg in "$@"; do
  if [[ "$arg" == --ignore-prior-steps || "$arg" == --ignore_prior_steps ]]; then
    # Normalize bare flag to key=value for train.py parser
    CMD+=("--ignore-prior-steps=true")
  elif [[ "$arg" == --* ]]; then
    CMD+=("$arg")
  else
    # if it's key=value without leading dashes, prepend --
    if [[ "$arg" == *"="* ]]; then
      CMD+=("--$arg")
    else
      echo "Ignoring unexpected argument: $arg (expected key=value, --key=value, or a supported bare flag)" >&2
    fi
  fi
done

"${CMD[@]}"