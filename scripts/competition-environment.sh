#!/usr/bin/env bash
# Run after trusted checkout, before any credential-bearing or evaluation step.
set -euo pipefail
: "${SPARK_PYTHON:?Configure the external trusted Python interpreter}"
: "${SPARK_STATE_ROOT:?Configure durable private state}"
[[ "$SPARK_PYTHON" = /* && -x "$SPARK_PYTHON" ]]
"$SPARK_PYTHON" - <<'PY'
import hashlib
import os
import sys
from pathlib import Path

checkout = Path.cwd().resolve()
runtime = Path(sys.prefix).resolve()
state_input = Path(os.environ["SPARK_STATE_ROOT"])
state = state_input.resolve()
for name, path in (("runtime", runtime), ("state", state)):
    if path == Path("/") or path.is_relative_to(checkout) or path.is_relative_to(Path("/tmp").resolve()):
        raise SystemExit(f"{name} must be outside checkout and /tmp on a private persistent filesystem")
if not state_input.is_absolute() or not state.is_dir():
    raise SystemExit("SPARK_STATE_ROOT must be an existing absolute directory")
if runtime == Path(sys.base_prefix).resolve() or sys.version_info < (3, 12):
    raise SystemExit("SPARK_PYTHON must select a provisioned Python 3.12+ virtual environment")
expected = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
if (runtime / ".spark-uv-lock.sha256").read_text().strip() != expected:
    raise SystemExit("trusted runtime dependencies need provisioning for the current uv.lock")
# Import the actual entry points and evaluation dependencies, with no credentials.
import openai
import yaml
from validator import crown, judge, pr_admission, settlement

for module in (crown, judge, pr_admission, settlement):
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise SystemExit("competition code must come from the trusted current checkout")
print(f"Trusted competition interpreter ready: {sys.executable}")
PY
