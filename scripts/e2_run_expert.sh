#!/bin/bash
# Inside the separate GPU experiment container, after preparing external references.
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E2_ROOT="${E2_ROOT:-$E1_ROOT/e2-expert}"
case "${1:-expert}" in
  expert) E2_RUNNER=e2_expert_smoke.py ;;
  bf16-control) E2_RUNNER=e2_expert_bf16_control.py ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ld_library_path"])' \
  "$E1_ROOT/artifacts/release-linker-env.json")"
timeout 300 "$E1_ROOT/venv-release/bin/python" "$E2_ROOT/$E2_RUNNER" --root "$E2_ROOT"
