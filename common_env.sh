#!/usr/bin/env bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT

export RUNTIME_ENV="${RUNTIME_ENV:-local}"
export CONDA_EXE="${CONDA_EXE:-conda}"
export LLADA_PATH="${LLADA_PATH:-$PROJECT_ROOT/models/LLaDA-8B-Instruct}"
export DREAM_PATH="${DREAM_PATH:-$PROJECT_ROOT/models/Dream-7B-Instruct}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "RUNTIME_ENV=$RUNTIME_ENV"

shutdown_if_autodl() {
  :
}
