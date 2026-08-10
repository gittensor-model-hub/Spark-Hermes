"""Shared pytest fixtures.

CI runs the suite without a SparkProof checkout (a plain ``pull_request`` job must
never expose the private-repo checkout token to untrusted PR code), so tests that
genuinely need SparkProof's publish/novelty code must skip cleanly rather than error.
"""

import os
from pathlib import Path

import pytest

_SPARKPROOF_ROOT = Path(
    os.environ.get("SPARKPROOF_ROOT", Path(__file__).resolve().parents[1] / ".." / "SparkProof")
).resolve()


@pytest.fixture
def sparkproof_root() -> Path:
    """Path to a SparkProof checkout, or skip the test when none is available."""
    marker = _SPARKPROOF_ROOT / "sparkproof" / "publish" / "hf_dataset.py"
    if not marker.is_file():
        pytest.skip("SparkProof checkout required beside SparkDistill (or set SPARKPROOF_ROOT)")
    return _SPARKPROOF_ROOT
