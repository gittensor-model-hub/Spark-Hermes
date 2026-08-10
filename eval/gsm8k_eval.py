"""GSM8K prompt + grading aligned with lm-evaluation-harness gsm8k.yaml (v3).

Matches the harness basket: 5-shot `Question: …\\nAnswer:` prompts, greedy generation,
and `exact_match,strict-match` extraction with flexible-extract fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FEWSHOT_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_fewshot_5.jsonl"
_STRICT_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_NORMALIZE_PATTERNS = (
    re.compile(r","),
    re.compile(r"\$"),
    re.compile(r"(?s).*#### "),
    re.compile(r"\.$"),
)


def load_fewshot_examples(path: Path = FEWSHOT_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "answer" not in row:
                raise ValueError(f"{path}:{line_no}: each fewshot row needs question and answer")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: fewshot set is empty")
    return rows


def format_gsm8k_prompt(question: str, *, fewshot_path: Path = FEWSHOT_PATH) -> str:
    """Build the lm-eval gsm8k 5-shot prompt for a test question."""
    parts: list[str] = []
    for row in load_fewshot_examples(fewshot_path):
        parts.append(f"Question: {row['question']}\nAnswer: {row['answer']}\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)


def extract_gsm8k_prediction(text: str) -> str | None:
    """Extract a numeric answer using strict-match, then flexible-extract."""
    strict = _STRICT_PATTERN.search(text)
    if strict:
        return strict.group(1)
    matches = _FLEXIBLE_PATTERN.findall(text)
    if not matches:
        return None
    # findall returns tuples for alternation groups; take last non-empty group.
    last = matches[-1]
    if isinstance(last, tuple):
        for group in reversed(last):
            if group:
                return group
        return None
    return last


def normalize_gsm8k_answer(text: str) -> str:
    """Normalize like lm-eval gsm8k `regexes_to_ignore` before exact match."""
    value = str(text)
    for pattern in _NORMALIZE_PATTERNS:
        value = pattern.sub("", value)
    return value.strip()


def grade_gsm8k_response(gold: str, model_response: str) -> bool:
    prediction = extract_gsm8k_prediction(model_response)
    if prediction is None:
        return False
    return normalize_gsm8k_answer(gold) == normalize_gsm8k_answer(prediction)


def gold_answer(row: dict[str, Any]) -> str:
    """Full GSM8K gold answer (reasoning + ####), matching lm-eval doc_to_target."""
    if "answer" in row and "####" in str(row["answer"]):
        return str(row["answer"])
    if "answer_short" in row:
        return f"#### {row['answer_short']}"
    return str(row["answer"])
