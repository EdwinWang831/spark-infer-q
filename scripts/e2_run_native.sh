#!/bin/bash
# Run inside a separate experiment container; never install into the live service.
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_NATIVE_ROOT="${E2_NATIVE_ROOT:-$E1_ROOT/e2-native}"
export OMP_NUM_THREADS=2
export MAX_JOBS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=12.1
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
case "${1:-smoke}" in
  build)
    timeout 600 "$E1_ROOT/venv-release/bin/python" "$E2_NATIVE_ROOT/e2_build_native.py" --root "$E2_NATIVE_ROOT"
    ;;
  smoke)
    timeout 300 "$E1_ROOT/venv-release/bin/python" "$E2_NATIVE_ROOT/e2_native_expert_smoke.py" \
      --root "$E2_NATIVE_ROOT" --expert-root "$E1_ROOT/e2-expert"
    ;;
  *) exit 2 ;;
esac
