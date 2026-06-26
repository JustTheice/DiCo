#!/bin/bash

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$PROJECT_ROOT/common_env.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dllmfine}"
MODEL_PATH="$LLADA_PATH"
MASK_ID=126336

# MODEL_PATH="$DREAM_PATH"
# MASK_ID=151666

MODEL_NAME=$([ "$MODEL_PATH" = "$DREAM_PATH" ] && echo "Dream" || echo "LLaDA")

GPU_IDS=(0 1 2)
MASTER_PORT=8090

# Select the evaluated tasks

# TASKS="gsm8k"
# NUM_FEWSHOT=4

TASKS="humaneval"
NUM_FEWSHOT=0

# TASKS="mbpp"
# NUM_FEWSHOT=3

# math-500 is a dataset on huggingface
# TASKS="math-500"
# INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"
# NUM_FEWSHOT=4

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

BATCH_SIZE=1
MC_NUM=128
CFG_SCALE=0.0
TEMPERATURE=0.0
POSITIONAL_WEIGHTS_TYPE="ratio"
WEIGHT_FUNCTION_TYPE="exponential"
MAX_WEIGHT=1.0
TG_ALPHA=0.5
TG_BETA=0.05
CACHE_BACKEND="none"

MAX_EXPLORATION_STEPS=10
EXPLORATION_N=4
EXPLORATION_THRESHOLD=0.3
# EXPLORATION_SEED_METHOD="soft_nms"
ACCELERATION_PARALLEL_METHOD="factor"
ACCELERATION_THRESHOLD=0.95
ACCELERATION_LOW_THRESHOLD=0.6
ACCELERATION_FACTOR=1
R_GATE=0.8
MOPUP_MARGIN_THRESHOLD=3.0
MOPUP_SPEED=1
TOLERANCE_M=2
RECORD_TIMING_BREAKDOWN=0

SL_VALUES=(256)
BLOCK_LENGTHS=(128)
HYPERPARAM_DIR="explore${MAX_EXPLORATION_STEPS}_n${EXPLORATION_N}_m${TOLERANCE_M}_th${EXPLORATION_THRESHOLD}"
if [ -n "$EXPLORATION_SEED_METHOD" ]; then
  HYPERPARAM_DIR="${HYPERPARAM_DIR}_seed-${EXPLORATION_SEED_METHOD}"
fi
HYPERPARAM_DIR="${HYPERPARAM_DIR}_acc-${ACCELERATION_PARALLEL_METHOD}_hi${ACCELERATION_THRESHOLD}_lo${ACCELERATION_LOW_THRESHOLD}_fa${ACCELERATION_FACTOR}_Rgate${R_GATE}_m${MOPUP_MARGIN_THRESHOLD}_v${MOPUP_SPEED}"
if [ "$RECORD_TIMING_BREAKDOWN" -eq 1 ]; then
  HYPERPARAM_DIR="${HYPERPARAM_DIR}_timing"
fi

for SL in "${SL_VALUES[@]}"
do
  GEN_LENGTH=$SL
  STEPS=$SL

  for BL in "${BLOCK_LENGTHS[@]}"
  do
    OUTPUT_DIR="eval/outputs/DiCo_pw-${POSITIONAL_WEIGHTS_TYPE}-${WEIGHT_FUNCTION_TYPE}_a${TG_ALPHA}_b${TG_BETA}_cache-${CACHE_BACKEND}/${HYPERPARAM_DIR}/SL${SL}_BL${BL}/${TASKS}_${NUM_FEWSHOT}shot_${N_LIMIT:+limit_${N_LIMIT}}/${MODEL_NAME}"
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    MODEL_ARGS="model_path=$MODEL_PATH"
    MODEL_ARGS+=",output_dir=$OUTPUT_DIR"
    MODEL_ARGS+=",mc_num=$MC_NUM"
    MODEL_ARGS+=",gen_length=$GEN_LENGTH"
    MODEL_ARGS+=",steps=$STEPS"
    MODEL_ARGS+=",block_length=$BL"

    MODEL_ARGS+=",cfg_scale=$CFG_SCALE"
    MODEL_ARGS+=",temperature=$TEMPERATURE"
    MODEL_ARGS+=",positional_weights_type=$POSITIONAL_WEIGHTS_TYPE"
    MODEL_ARGS+=",weight_function_type=$WEIGHT_FUNCTION_TYPE"
    MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
    MODEL_ARGS+=",TG_alpha=$TG_ALPHA"
    MODEL_ARGS+=",TG_beta=$TG_BETA"
    MODEL_ARGS+=",mask_id=$MASK_ID"
    MODEL_ARGS+=",cache_backend=$CACHE_BACKEND"

    MODEL_ARGS+=",max_exploration_steps=$MAX_EXPLORATION_STEPS"
    MODEL_ARGS+=",exploration_N=$EXPLORATION_N"
    MODEL_ARGS+=",tolerance_M=$TOLERANCE_M"
    MODEL_ARGS+=",exploration_threshold=$EXPLORATION_THRESHOLD"
    if [ -n "$EXPLORATION_SEED_METHOD" ]; then
      MODEL_ARGS+=",exploration_seed_method=$EXPLORATION_SEED_METHOD"
    fi
    MODEL_ARGS+=",acceleration_parallel_method=$ACCELERATION_PARALLEL_METHOD"
    MODEL_ARGS+=",acceleration_threshold=$ACCELERATION_THRESHOLD"
    MODEL_ARGS+=",acceleration_low_threshold=$ACCELERATION_LOW_THRESHOLD"
    MODEL_ARGS+=",acceleration_factor=$ACCELERATION_FACTOR"
    MODEL_ARGS+=",R_gate=$R_GATE"
    MODEL_ARGS+=",mopup_margin_threshold=$MOPUP_MARGIN_THRESHOLD"
    MODEL_ARGS+=",mopup_speed=$MOPUP_SPEED"
    MODEL_ARGS+=",record_timing_breakdown=$RECORD_TIMING_BREAKDOWN"

    echo "================================================="
    echo "Project Root: $PROJECT_ROOT"
    echo "Runtime Env: $RUNTIME_ENV"
    echo "Using GPUs: $GPU_LIST (Total: $NUM_GPUS)"
    echo "Model: $MODEL_PATH"
    echo "Tasks: $TASKS"
    echo "Model Args: $MODEL_ARGS"
    echo "Output Dir: $OUTPUT_DIR"
    echo "================================================="

    cd "$PROJECT_ROOT" || exit

    set +e
    CUDA_VISIBLE_DEVICES=$GPU_LIST stdbuf -o0 "$CONDA_EXE" run -n "$CONDA_ENV_NAME" --no-capture-output \
      accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $MASTER_PORT \
        -m eval.eval_model.eval_dico \
          --model eval_sampler \
          --confirm_run_unsafe_code \
          --tasks $TASKS \
          ${INCLUDE_PATH:+--include_path $INCLUDE_PATH} \
          ${NUM_FEWSHOT:+--num_fewshot $NUM_FEWSHOT} \
          --batch_size $BATCH_SIZE \
          --model_args $MODEL_ARGS \
          --log_samples \
          --output_path $OUTPUT_DIR \
          ${N_LIMIT:+--limit $N_LIMIT} \
          > "${OUTPUT_DIR}/log.txt" 2>&1
    set -e
  done
done

shutdown_if_autodl
