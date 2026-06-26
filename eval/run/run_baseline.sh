#!/bin/bash

set -e
# export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_ENDPOINT=https://huggingface.co
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
MASTER_PORT=8083


# Select the evaluated tasks

TASKS="gsm8k"
NUM_FEWSHOT=4

# TASKS="humaneval"
# NUM_FEWSHOT=0

#TASKS="humaneval_instruct"

# TASKS="mbpp"
# NUM_FEWSHOT=3

# TASKS="gsm8k"
# unset INCLUDE_PATH

# math-500 is a dataset on huggingface
#TASKS="math-500"
#INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"
#NUM_FEWSHOT=4

#TASKS=sudoku
#INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/sudoku/"
#NUM_FEWSHOT=4

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}


# HYPERPARAMETERS for different methods
# Vanilla: DECODING_METHOD="topk", K = 1 
# Fast-dllm: DECODING_METHOD="fixed", CONFIDENCE_THRESHOLD = 0.95
# EB-Sampler: DECODING_METHOD="entropy_bound", ENTROPY_BOUND_GAMMA = 0.1

BATCH_SIZE=1
MC_NUM=128
CFG_SCALE=0.0
TEMPERATURE=0.0
POSITIONAL_WEIGHTS_TYPE='none'
# WEIGHT_FUNCTION_TYPE="exponential"
MAX_WEIGHT=1.0
INITIAL_MIN_WEIGHT=0.0
REMASKING="low_confidence"
DECODING_METHOD="topk"
FACTOR=0.7
CONFIDENCE_THRESHOLD=0.95
K=1
ENTROPY_BOUND_GAMMA=0.1
CACHE_BACKEND="none"

SL_VALUES=(256)
BLOCK_LENGTHS=(128)

for SL in "${SL_VALUES[@]}"
do
  GEN_LENGTH=$SL
  STEPS=$SL

  if [ "$DECODING_METHOD" = "fixed" ]; then
    METHOD_SUFFIX="conf_tr${CONFIDENCE_THRESHOLD}"
  elif [ "$DECODING_METHOD" = "factor" ]; then
    METHOD_SUFFIX="factor${FACTOR}"
  elif [ "$DECODING_METHOD" = "topk" ]; then
    METHOD_SUFFIX="k${K}"
  elif [ "$DECODING_METHOD" = "entropy_bound" ]; then
    METHOD_SUFFIX="eb${ENTROPY_BOUND_GAMMA}"
  else
    METHOD_SUFFIX=""
  fi

  for BL in "${BLOCK_LENGTHS[@]}"
  do
    OUTPUT_DIR="eval/outputs/Baseline_${METHOD_SUFFIX}_cache-${CACHE_BACKEND}/SL${SL}_BL${BL}/${TASKS}_${NUM_FEWSHOT}shot_${N_LIMIT:+limit_${N_LIMIT}}/${MODEL_NAME}"
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
    # MODEL_ARGS+=",weight_function_type=$WEIGHT_FUNCTION_TYPE"
    MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
    MODEL_ARGS+=",initial_min_weight=$INITIAL_MIN_WEIGHT"
    MODEL_ARGS+=",remasking=$REMASKING"
    MODEL_ARGS+=",decoding_method=$DECODING_METHOD"
    MODEL_ARGS+=",factor=$FACTOR"
    MODEL_ARGS+=",confidence_threshold=$CONFIDENCE_THRESHOLD"
    MODEL_ARGS+=",k=$K"
    MODEL_ARGS+=",entropy_bound_gamma=$ENTROPY_BOUND_GAMMA"
    MODEL_ARGS+=",mask_id=$MASK_ID"
    MODEL_ARGS+=",cache_backend=$CACHE_BACKEND"

    echo "================================================="
    echo "Project Root: $PROJECT_ROOT"
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
        -m eval.eval_model.eval_baseline \
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
