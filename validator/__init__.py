"""The validator's outward-facing surface.

Kept apart from `hermes/` and `hermesbench/` deliberately. Those are libraries: importing
them costs nothing and pulls in nothing that listens on a socket. This package is the one
place that serves requests, so the dependency stays optional (`uv sync --extra validator`)
and a benchmark run on a machine with no web stack is unaffected.
"""
