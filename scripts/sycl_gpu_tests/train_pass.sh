#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.2.0
# @adai-reviewed: 2026-11-10

# One small GPU training pass for run_gpu_tests.sh: a private config (no MNS, registry or metrics),
# the packaged pairs queued in a private session directory, <epochs> epochs (default 3) with dropout
# and SEED=<seed>.
#
#   train_pass.sh <work_dir> <seed> [epochs]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$1"
SEED="$2"
EPOCHS="${3:-3}"
mkdir -p "$DIR/sessions"
cat >"$DIR/config.trainer.conf" <<CONF
LOG_FILE_PATH=$DIR/trainer.log
SESSION_DIR=$DIR/sessions
VOCAB_PATH=$DIR/sessions/vocab
MODEL_PATH=$DIR/sessions/model
NAME_SERVICE_URL=
REGISTRY_SERVER_URL=
METRICS_SERVER_URL=
MODEL_NAME=gpu-test
GPU_ENABLED=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.3
D_MODEL=64
NUM_HEADS=2
D_FF=128
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
MAX_SEQ_LENGTH=16
NUM_EPOCHS=$EPOCHS
DROPOUT=0.1
SEED=$SEED
CONF
cd "$DIR"
"$HERE/bin/dataset_manager" --config "$DIR/config.trainer.conf" --kind chatbot add "$HERE/data/pairs.jsonl"
"$HERE/bin/incremental_trainer" --config "$DIR/config.trainer.conf" --foreground train "$EPOCHS"
