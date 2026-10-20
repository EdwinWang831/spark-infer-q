#!/bin/bash
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_EXECUTOR_ROOT="${E2_EXECUTOR_ROOT:-$E1_ROOT/e2-executor}"
export OMP_NUM_THREADS=2
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
timeout 1800 "$E1_ROOT/venv-release/bin/python" "$E2_EXECUTOR_ROOT/e2_executor_smoke.py" \
  --root "$E2_EXECUTOR_ROOT" "${@}"
