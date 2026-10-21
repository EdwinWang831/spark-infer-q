#!/bin/bash
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_LOAD_ROOT="${E2_LOAD_ROOT:-$E1_ROOT/e2-load}"
export OMP_NUM_THREADS=4
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
timeout 3600 "$E1_ROOT/venv-release/bin/python" "$E2_LOAD_ROOT/e2_load_smoke.py" \
  --root "$E2_LOAD_ROOT" --checkpoint "${E2_CHECKPOINT:?set checkpoint directory}" "$@"
