#!/usr/bin/env bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT

export RUNTIME_ENV="${RUNTIME_ENV:-local}"
export CONDA_EXE="${CONDA_EXE:-conda}"
if [ -z "${LLADA_PATH:-}" ]; then
  if [ -f "$PROJECT_ROOT/models/LLaDA-8B-Instruct/config.json" ]; then
    export LLADA_PATH="$PROJECT_ROOT/models/LLaDA-8B-Instruct"
  else
    export LLADA_PATH="GSAI-ML/LLaDA-8B-Instruct"
  fi
fi

if [ -z "${DREAM_PATH:-}" ]; then
  if [ -f "$PROJECT_ROOT/models/Dream-7B-Instruct/config.json" ]; then
    export DREAM_PATH="$PROJECT_ROOT/models/Dream-7B-Instruct"
  else
    export DREAM_PATH="Dream-org/Dream-v0-Instruct-7B"
  fi
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RUNTIME_ENV=$RUNTIME_ENV"

shutdown_if_autodl() {
  :
}
