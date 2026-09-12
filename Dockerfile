# The benchmark's execution environment, so `container_image_digest` has something to pin.
#
# Two things this fixes, both measured rather than theoretical. `LocalToolExecutor` refuses
# to run model-authored shell without `allow_unsandboxed=True`, described as an assertion by
# the operator that the process is already inside a container -- and no container shipped,
# so every reproducer had to make that assertion about their own laptop. And the agent's
# `python` tool resolved through PATH: on one developer machine that found an unrelated
# vendored virtualenv, and on a stock host it finds nothing at all, so the score was a
# function of whose shell was running it.
#
# Pinned by digest rather than tag on purpose. `python:3.12-slim` moves; a benchmark that
# claims reproducibility while floating on a mutable tag is claiming something it does not
# have, which is the same failure `PriceBook` and `harness_digest` refuse elsewhere.
# Resolved from the official registry on 2026-09-11 (Python 3.12.14, slim-trixie).
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# Coreutils the task verifiers actually invoke. Left implicit, a slim base silently changes
# which tasks are solvable, and the diff is invisible in every task file.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates make gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/spark

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv==0.9.7 && uv sync --frozen --no-install-project

COPY admin/ admin/
COPY hermes/ hermes/
COPY hermesbench/ hermesbench/
COPY eval/ eval/
COPY proof/ proof/
COPY teacher/ teacher/
COPY validator/ validator/
COPY miner/ miner/
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HERMESBENCH_IN_CONTAINER=1

# Not a root shell by default: the executor runs model-authored commands, and the container
# is the isolation the flag on LocalToolExecutor only asserts.
RUN useradd --create-home --uid 1000 agent && chown -R agent /opt/spark
USER agent

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "hermesbench.runner"]
