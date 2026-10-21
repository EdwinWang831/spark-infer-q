#!/bin/bash
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_MOE_ROOT="${E2_MOE_ROOT:-$E1_ROOT/e2-moe}"
export OMP_NUM_THREADS=2 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=12.1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
case "${1:-smoke}" in
  build) timeout 1200 "$E1_ROOT/venv-release/bin/python" "$E2_MOE_ROOT/e2_build_moe.py" --root "$E2_MOE_ROOT" ;;
  smoke) timeout 1200 "$E1_ROOT/venv-release/bin/python" "$E2_MOE_ROOT/e2_moe_smoke.py" --root "$E2_MOE_ROOT" ;;
  router) timeout 120 "$E1_ROOT/venv-release/bin/python" "$E2_MOE_ROOT/e2_moe_router_probe.py" --root "$E2_MOE_ROOT" ;;
  *) exit 2 ;;
esac
