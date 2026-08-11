#!/usr/bin/env bash
# Serve the agent benchmark's model on SGLang.
#
#   scripts/serve_agent.sh /path/to/model [served-name] [port]
#   python -m hermesbench.runner --suite all --model <served-name> --base-url http://127.0.0.1:8001/v1
#
# SGLang rather than vLLM, and that is a measurement rather than a preference. Against
# Muse-Glimmer-30B on 2026-08-11, vLLM 0.27.0 had no native support for the architecture and its
# `--model-impl transformers` fallback served the model while returning ten tokens of multilingual
# noise; SGLang from the muse-glimmer branch returned correct tool calls. Plain transformers on the
# same weights, revision and card agreed with SGLang, so the fallback was what was broken.
# docs/serving-muse-glimmer.md has the evidence and the eight startup failures that preceded it.
#
# For a Hermes-dialect model this script works unchanged -- drop the two `muse` parser flags, which
# are what teaches SGLang the ATEM wire format.
#
# NOT the TritonBench stack. `scripts/install_serve.sh` pins vLLM 0.25.0+cu129 deliberately: the
# Triton domain score is only comparable across miners if every checkpoint is served by the same
# engine, and every published Triton number was measured on that one. Switching it would invalidate
# them. The two paths serve different benchmarks and are meant to stay apart.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?usage: scripts/serve_agent.sh <model-path-or-repo> [served-name] [port]}"
SERVED="${2:-$(basename "$MODEL")}"
PORT="${3:-8001}"
VENV="${SGLANG_VENV:-$ROOT/.venv-sglang}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "error: no SGLang venv at $VENV" >&2
  echo "  python3 -m venv $VENV" >&2
  echo "  SGLANG_BUILD_RUST_EXTS=none $VENV/bin/pip install sglang" >&2
  echo "  $VENV/bin/pip install ninja        # SGLang JIT-compiles kernels through it" >&2
  echo "  apt-get install -y ffmpeg          # torchcodec dlopens libavutil" >&2
  echo "See docs/serving-muse-glimmer.md; for Muse-Glimmer the install is a branch build." >&2
  exit 1
fi

# The venv's own CUDA toolchain, and only that one. Pointing CUDA_HOME at a different venv's copy
# made flashinfer's bundled cccl headers disagree with nvcc and every JIT compile failed with
# "CUDA compiler and CUDA toolkit headers are incompatible". CPATH is deliberately not set for the
# same reason: adding the toolkit's includes is what let the bundled headers be found at all.
CUDA_DIR="$(cd "$VENV" && "$VENV/bin/python" -c "
import pathlib, sysconfig
site = pathlib.Path(sysconfig.get_paths()['purelib'])
found = sorted((site / 'nvidia').glob('cu*'))
print(found[-1] if found else '')" 2>/dev/null || true)"

if [ -n "$CUDA_DIR" ] && [ -d "$CUDA_DIR" ]; then
  export CUDA_HOME="$CUDA_DIR"
  export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  # The wheel ships libcudart.so.13 and no bare .so, which `ld -lcudart` cannot find, so every JIT
  # link failed. A versioned library alone is not linkable.
  for lib in "$CUDA_HOME"/lib/libcudart.so.*; do
    [ -e "$CUDA_HOME/lib/libcudart.so" ] || ln -sf "$(basename "$lib")" "$CUDA_HOME/lib/libcudart.so"
    break
  done
fi

# The venv's bin too. Invoking $VENV/bin/python by absolute path leaves the venv's bin off PATH, so
# a pip-installed ninja is present and invisible -- which killed the scheduler three times with
# FileNotFoundError.
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH"

ARGS=(
  --model-path "$MODEL"
  --served-model-name "$SERVED"
  --context-length "${SERVE_CONTEXT_LENGTH:-32768}"
  --mem-fraction-static "${SERVE_MEM_FRACTION:-0.88}"
  --host 127.0.0.1 --port "$PORT"
  # Neither backend needs a CUDA toolchain. The default flashinfer paths JIT-compile at startup and
  # again on the first request: attention alone was not enough, because the server came up, served
  # its warmup, and died on the first real completion inside the sampling kernel.
  --attention-backend "${SERVE_ATTENTION_BACKEND:-triton}"
  --sampling-backend "${SERVE_SAMPLING_BACKEND:-pytorch}"
  --trust-remote-code
)

# ATEM, when the served model speaks it. Without a tool-call parser SGLang does not render the tool
# definitions into the prompt AT ALL: the first working run came back with prompt_tokens=77 and the
# model wondering aloud which command to use, because it had never been told it had any tools.
if [ "${SERVE_DIALECT:-atem}" = "atem" ]; then
  ARGS+=(--tool-call-parser muse --reasoning-parser muse)
fi

echo "serving $MODEL as $SERVED on 127.0.0.1:$PORT (sglang)" >&2
exec "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}"
