#!/bin/bash
# Run only inside the separate GPU experiment container, never the live service.
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E1_COMPONENT="${1:-tilelang}"
case "$E1_COMPONENT" in tilelang|deepgemm) ;; *) exit 2 ;; esac
export OMP_NUM_THREADS=2
export MAX_JOBS=2
export DEEPINFER_ACCELERATOR=nvidia
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$($E1_ROOT/venv-release/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
timeout 600 "$E1_ROOT/venv-release/bin/python" "$E1_ROOT/gpu-smoke.py" \
  --source "$E1_ROOT/DeepInfer" \
  --report "$E1_ROOT/artifacts/$E1_COMPONENT-gpu-smoke.json" \
  --component "$E1_COMPONENT"
