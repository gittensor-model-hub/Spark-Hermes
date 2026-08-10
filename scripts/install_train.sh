#!/usr/bin/env bash
# Install Axolotl training stack for SparkDistill recipes.
#
#   scripts/install_train.sh
#
# Qwen3.5 processors require torchvision. On Blackwell (SM120+), skip FlashAttention
# 2 source builds and let train.sh fall back to SDPA. On Hopper and earlier, prefer
# the FlashAttention 3 wheel, then attempt a FlashAttention 2 build.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found — run SparkProof scripts/install.sh first" >&2
  exit 1
fi

echo ">>> syncing SparkDistill base deps"
uv sync --extra dev

# Qwen3.5 binds flash-linear-attention, whose Triton CUDA extensions compile against
# Python.h. Without python3-dev the build fails and FLA rolls back to a CPU path that
# crashes mid-train (issue #283). Install the headers when we can (root or passwordless
# sudo); otherwise warn — the miner image often has no sudo.
train_has_headers() {
  uv run --no-sync python -c \
    "import os,sysconfig,sys;inc=sysconfig.get_paths().get('include');sys.exit(0 if inc and os.path.exists(os.path.join(inc,'Python.h')) else 1)" \
    2>/dev/null
}

if ! train_has_headers; then
  echo ">>> Python.h (python3-dev) missing — needed for Qwen3.5 flash-linear-attention Triton kernels (issue #283)"
  if command -v apt-get >/dev/null 2>&1 && { [ "$(id -u)" = 0 ] || { command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }; }; then
    if [ "$(id -u)" = 0 ]; then SUDO=""; else SUDO="sudo"; fi
    echo "  installing python3-dev..."
    $SUDO apt-get update -qq && $SUDO apt-get install -y python3-dev || true
  fi
  if train_has_headers; then
    echo "  python3-dev installed"
  else
    echo "  warning: Python.h still missing — training will crash on a CPU rollback."
    echo "           Fix: sudo apt-get install -y python3-dev   (see issue #283)"
  fi
fi

echo ">>> installing Axolotl + torchvision"
uv pip install -q axolotl torchvision numba ninja packaging einops "numpy<2.5"

echo ">>> installing CutCrossEntropy (Axolotl fork)"
if uv run --no-sync python -c "import cut_cross_entropy" 2>/dev/null; then
  echo "  cut-cross-entropy: already installed"
else
  uv pip install -q "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@fec1a88"
  uv run --no-sync python -c "import cut_cross_entropy; print('  cut-cross-entropy: installed')"
fi

install_fa2() {
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${gpu_major}0}"
  if uv run --no-sync python -c "import flash_attn" 2>/dev/null; then
    uv run --no-sync python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
    return 0
  fi

  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "  flash-attn: CUDA compiler not found at $CUDA_HOME/bin/nvcc (training will use SDPA)" >&2
    return 1
  fi

  echo ">>> building FlashAttention 2 (first install takes several minutes)"
  build_jobs="${MAX_JOBS:-$(nproc)}"
  echo "  CUDA arch: ${FLASH_ATTN_CUDA_ARCHS}; parallel jobs: $build_jobs ($(nproc) vCPUs available)"
  if MAX_JOBS="$build_jobs" uv pip install "flash-attn==2.8.3.post1" --no-build-isolation; then
    uv run --no-sync python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__}')"
    return 0
  fi
  echo "  flash-attn: build failed (training will use SDPA via scripts/train.sh)" >&2
  return 1
}

gpu_major="$(uv run --no-sync python -c "import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)")"

if [ "${SPARKDISTILL_SKIP_FLASH_ATTN:-0}" = "1" ]; then
  echo "  flash-attn: skipped (SPARKDISTILL_SKIP_FLASH_ATTN=1; training will use SDPA)"
elif [ "$gpu_major" -ge 10 ]; then
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${gpu_major}0}"
  echo ">>> Blackwell SM${FLASH_ATTN_CUDA_ARCHS}: skipping FlashAttention 2 source build (compile fails; train.sh uses SDPA)"
  if uv run --no-sync python -c "import flash_attn" 2>/dev/null; then
    uv run --no-sync python -c "import flash_attn; print(f'  flash-attn: {flash_attn.__version__} (preinstalled)')"
  fi
elif uv run --no-sync python -c "from transformers.utils import is_flash_attn_3_available; import sys; sys.exit(0 if is_flash_attn_3_available() else 1)" 2>/dev/null; then
  echo "  flash-attn-3: installed"
else
  cuda_tag="$(uv run --no-sync python -c "import torch; v=torch.version.cuda or ''; print('cu' + v.replace('.', '')[:3] if v else 'cu130')")"
  fa3_index="https://download.pytorch.org/whl/${cuda_tag}"
  echo ">>> installing FlashAttention 3 wheel (${cuda_tag}, ~30s)"
  if uv pip install -q "flash-attn-3" --index-url "$fa3_index" \
    && uv run --no-sync python -c "from transformers.utils import is_flash_attn_3_available; import sys; sys.exit(0 if is_flash_attn_3_available() else 1)" 2>/dev/null; then
    echo "  flash-attn-3: installed from ${fa3_index}"
  else
    install_fa2 || true
  fi
fi

if uv run --no-sync axolotl --help >/dev/null 2>&1; then
  echo ">>> Axolotl ready"
else
  echo "error: axolotl CLI not available after install" >&2
  exit 1
fi

echo ""
echo "Next:"
echo "  scripts/prepare_mining_sft.sh"
echo "  scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml"
