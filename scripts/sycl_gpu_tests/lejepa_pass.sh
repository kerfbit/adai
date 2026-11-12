#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-12

# LeJEPA world-model passes (incremental_trainer --objective=lejepa) for run_gpu_tests.sh (TD-572):
# a private config (no MNS, registry or metrics), a vocabulary built from the packaged texts, and
# the texts queued in the world_model pool of a private session directory.
#
#   lejepa_pass.sh setup <work_dir> <seed> [epochs]   config, vocabulary, texts queued
#   lejepa_pass.sh train <work_dir> <epochs>          one `train` pass (foreground)
#   lejepa_pass.sh resume <work_dir>                  one `resume` pass (NUM_EPOCHS epochs)
#   lejepa_pass.sh requeue <work_dir>                 trained texts back to pending
#   lejepa_pass.sh status <work_dir>                  the world_model pool's status
#
# `resume` and `train` exit as incremental_trainer does: 0 trained, 3 nothing pending, 1 failed.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COMMAND="${1:-}"
DIR="${2:-}"
[[ -n "$COMMAND" && -n "$DIR" ]] || {
    sed -n '/^#   lejepa_pass.sh setup/,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
CONF="$DIR/config.trainer.conf"

case "$COMMAND" in
    setup)
        SEED="${3:?setup needs a seed}"
        EPOCHS="${4:-3}"
        mkdir -p "$DIR/sessions"
        cat >"$CONF" <<CONF
LOG_FILE_PATH=$DIR/trainer.log
SESSION_DIR=$DIR/sessions
VOCAB_PATH=$DIR/vocab
MODEL_PATH=$DIR/sessions/model
NAME_SERVICE_URL=
REGISTRY_SERVER_URL=
METRICS_SERVER_URL=
MODEL_NAME=gpu-test-world-model
GPU_ENABLED=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.3
MAX_SEQ_LENGTH=32
NUM_EPOCHS=$EPOCHS
SEED=$SEED
AUTO_SAVE_ENABLED=false
WORLD_MODEL_ENABLED=true
WORLD_MODEL_D_MODEL=128
WORLD_MODEL_NUM_HEADS=4
WORLD_MODEL_D_FF=256
WORLD_MODEL_NUM_LAYERS=2
WORLD_MODEL_SIGREG_NUM_SKETCHES=16
WORLD_MODEL_SIGREG_BATCH_SIZE=8
CONF
        "$HERE/bin/vocab_builder" --input "$HERE/data/world_model_texts.jsonl" --output "$DIR/vocab" \
            --vocab-size 400
        "$HERE/bin/dataset_manager" --config "$CONF" --kind world_model add "$HERE/data/world_model_texts.jsonl"
        ;;
    train)
        cd "$DIR"
        exec "$HERE/bin/incremental_trainer" --config "$CONF" --objective=lejepa --foreground train "${3:?train needs epochs}"
        ;;
    resume)
        cd "$DIR"
        exec "$HERE/bin/incremental_trainer" --config "$CONF" --objective=lejepa --foreground resume
        ;;
    requeue)
        "$HERE/bin/dataset_manager" --config "$CONF" --kind world_model requeue --all
        ;;
    status)
        "$HERE/bin/dataset_manager" --config "$CONF" --kind world_model status
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        exit 2
        ;;
esac
