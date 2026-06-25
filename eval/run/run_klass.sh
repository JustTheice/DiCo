#!/bin/bash

set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_ALLOW_CODE_EVAL=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$PROJECT_ROOT/common_env.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-dllmfine}"
INCLUDE_PATH=""

GPU_IDS=(0 1 2 3 7)
MASTER_PORT=8080


# LLaDA + MATH
# MODEL_PATH="$LLADA_PATH"
# MASK_ID=126336
# TASKS="math-500"
# INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"
# NUM_FEWSHOT=4
# CONFIDENCE_THRESHOLD=0.6
# KL_THRESHOLD=0.010

# LLaDA + GSM8K
# MODEL_PATH="$LLADA_PATH"
# MASK_ID=126336
# TASKS="gsm8k"
# NUM_FEWSHOT=4
# CONFIDENCE_THRESHOLD=0.6
# KL_THRESHOLD=0.015

# LLaDA + HumanEval
# MODEL_PATH="$LLADA_PATH"
# MASK_ID=126336
# TASKS="humaneval"
# NUM_FEWSHOT=0
# CONFIDENCE_THRESHOLD=0.9
# KL_THRESHOLD=0.010

# LLaDA + MBPP
# MODEL_PATH="$LLADA_PATH"
# MASK_ID=126336
# TASKS="mbpp"
# NUM_FEWSHOT=0
# CONFIDENCE_THRESHOLD=0.7
# KL_THRESHOLD=0.010

# Dream + MATH
# MODEL_PATH="$DREAM_PATH"
# MASK_ID=151666
# TASKS="math-500"
# INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"
# NUM_FEWSHOT=4
# CONFIDENCE_THRESHOLD=0.9
# KL_THRESHOLD=0.005

# Dream + GSM8K
MODEL_PATH="$DREAM_PATH"
MASK_ID=151666
TASKS="gsm8k"
NUM_FEWSHOT=4
CONFIDENCE_THRESHOLD=0.9
KL_THRESHOLD=0.001

# Dream + HumanEval
# MODEL_PATH="$DREAM_PATH"
# MASK_ID=151666
# TASKS="humaneval"
# NUM_FEWSHOT=0
# CONFIDENCE_THRESHOLD=0.8
# KL_THRESHOLD=0.001

# Dream + MBPP
# MODEL_PATH="$DREAM_PATH"
# MASK_ID=151666
# TASKS="mbpp"
# NUM_FEWSHOT=3
# CONFIDENCE_THRESHOLD=0.9
# KL_THRESHOLD=0.001

MODEL_NAME=$([ "$MODEL_PATH" = "$DREAM_PATH" ] && echo "Dream" || echo "LLaDA")

# N_LIMIT=100

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

BATCH_SIZE=1
MC_NUM=128
CFG_SCALE=0.0
TEMPERATURE=0.0
POSITIONAL_WEIGHTS_TYPE='none'
MAX_WEIGHT=1.0
INITIAL_MIN_WEIGHT=0.0
REMASKING="low_confidence"
DECODING_METHOD="fixed"
FACTOR=0.7
K=1
ENTROPY_BOUND_GAMMA=0.1
KL_HISTORY_LENGTH=2
CACHE_BACKEND="none"

SL_VALUES=(256)
BLOCK_LENGTHS=(256)

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
    OUTPUT_DIR="eval/outputs/KLASS_${METHOD_SUFFIX}_KLtr${KL_THRESHOLD}_KLhis${KL_HISTORY_LENGTH}_cache-${CACHE_BACKEND}/SL${SL}_BL${BL}/${TASKS}_${NUM_FEWSHOT}shot_${N_LIMIT:+limit_${N_LIMIT}}/${MODEL_NAME}"
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
    MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
    MODEL_ARGS+=",initial_min_weight=$INITIAL_MIN_WEIGHT"
    MODEL_ARGS+=",remasking=$REMASKING"
    MODEL_ARGS+=",decoding_method=$DECODING_METHOD"
    MODEL_ARGS+=",factor=$FACTOR"
    MODEL_ARGS+=",confidence_threshold=$CONFIDENCE_THRESHOLD"
    MODEL_ARGS+=",k=$K"
    MODEL_ARGS+=",entropy_bound_gamma=$ENTROPY_BOUND_GAMMA"
    MODEL_ARGS+=",kl_threshold=$KL_THRESHOLD"
    MODEL_ARGS+=",kl_history_length=$KL_HISTORY_LENGTH"
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
        -m eval.eval_model.eval_klass \
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
