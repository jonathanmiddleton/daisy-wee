#!/usr/bin/env bash
set -euo pipefail

# Setup logging to logs/<timestamp>.log while still printing to console
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
# Redirect all stdout/stderr to tee (append) so it prints and saves
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

export TORCH_COMPILE_OFF=0

# Defaults
NPROC=8
CHECKPOINT=""
BEGIN_SHARD=""
RUN_ID="1"

# Require positional CONFIG first
if [ $# -lt 1 ]; then
  echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [-r RUN_ID] [key=value ...]" >&2
  exit 1
fi

CONFIG="$1"
shift

# Parse options appearing AFTER CONFIG (accept short and long options)
OPTIND=1
EXTRA_ARGS=()
while getopts ":-:n:p:s:r:" opt; do
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
    r)
      RUN_ID="$OPTARG"
      ;;
    -)
      case "$OPTARG" in
        full_windows|full-windows)
          EXTRA_ARGS+=("--full_windows=true")
          ;;
        run_id)
          RUN_ID="${!OPTIND}"; OPTIND=$((OPTIND + 1))
          ;;
        run-id)
          RUN_ID="${!OPTIND}"; OPTIND=$((OPTIND + 1))
          ;;
        run_id=*)
          RUN_ID="${OPTARG#run_id=}"
          ;;
        run-id=*)
          RUN_ID="${OPTARG#run-id=}"
          ;;
        *)
          # Preserve other long options (with or without =)
          EXTRA_ARGS+=("--$OPTARG")
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [-r RUN_ID] [key=value ...]" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 CONFIG_FILE [-n NUM_PROCS] [-p CHECKPOINT_PATH] [-s BEGIN_SHARD] [-r RUN_ID] [key=value ...]" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

export RUN_ID
export OMP_NUM_THREADS=8

# Optionally set beginning shard index for training data
if [ -n "$BEGIN_SHARD" ]; then
  export BEGIN_SHARD
fi

CMD=(torchrun --standalone --nproc_per_node="$NPROC" train.py "$CONFIG")
if [ -n "$CHECKPOINT" ]; then
  CMD+=("--init_checkpoint=$CHECKPOINT")
fi
# Include any parsed long options handled by getopts
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

# Forward any remaining args as overrides. Accept key=value, --key=value, and supported bare flags.
for arg in "$@"; do
  if [[ "$arg" == --full_windows || "$arg" == --full-windows ]]; then
    CMD+=("--full_windows=true")
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