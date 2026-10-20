#!/bin/bash
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_BLOCK_ROOT="${E2_BLOCK_ROOT:-$E1_ROOT/e2-block}"
export OMP_NUM_THREADS=2 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=12.1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
case "${1:-smoke}" in
  build) timeout 1200 "$E1_ROOT/venv-release/bin/python" "$E2_BLOCK_ROOT/e2_build_blocks.py" --root "$E2_BLOCK_ROOT" ;;
  smoke) timeout 1800 "$E1_ROOT/venv-release/bin/python" "$E2_BLOCK_ROOT/e2_block_smoke.py" --root "$E2_BLOCK_ROOT" ;;
  *) exit 2 ;;
esac
