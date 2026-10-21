#!/bin/bash
# Build only the upstream C++ extension in an isolated, GPU-less E1 builder.
set -euo pipefail

E1_ROOT="${E1_ROOT:-/workspace/e1}"
E1_VENV="${E1_VENV:-$E1_ROOT/venv}"
E1_EXPECTED_SHA="${E1_EXPECTED_SHA:-774ce244b87cc0eb8d4b94ab9f5de2ab22049f01}"

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=void
export DEEPINFER_ACCELERATOR=nvidia
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS=2
export TORCH_CUDA_ARCH_LIST=12.1

python3 - <<'PY'
from pathlib import Path
if any(Path('/dev').glob('nvidia[0-9]*')):
    raise SystemExit('Run this build in the isolated container without GPU devices')
PY

cd "$E1_ROOT/DeepInfer"
test "$(git rev-parse HEAD)" = "$E1_EXPECTED_SHA"
git diff --quiet
if [[ ! -x "$E1_VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$E1_VENV"
fi
source "$E1_VENV/bin/activate"
python - <<'PY'
import json, sys, torch
print(json.dumps({
    'python': sys.version.split()[0], 'torch': torch.__version__,
    'torch_cuda_build': torch.version.cuda,
    'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'cuda_initialized': torch.cuda.is_initialized(),
}))
if torch.cuda.is_initialized():
    raise SystemExit('CUDA unexpectedly initialized before compilation')
PY

python setup.py build_ext --inplace
