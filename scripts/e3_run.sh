#!/bin/bash
set -euo pipefail
E1_ROOT="${E1_ROOT:-/workspace/e1}"
E3_ROOT="${E3_ROOT:-$E1_ROOT/e3-dflash}"
export OMP_NUM_THREADS=4
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export MAX_JOBS=1
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$("$E1_ROOT/venv-release/bin/python" -c 'import json,sys; print(":".join(p for p in json.load(open(sys.argv[1]))["ld_library_path"].split(":") if not p.endswith("/driver-libs")))' "$E1_ROOT/artifacts/release-linker-env.json")"
exec "$E1_ROOT/venv-release/bin/python" "$E3_ROOT/$@"
