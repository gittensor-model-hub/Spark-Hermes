"""CLI: teacher trajectories -> SFT-ready records with reasoning wrapped in <think> tags.

    python -m teacher.format \
        --in data/processed/phase1_trajectories.jsonl \
        --out data/processed/phase1_sft.jsonl \
        --format messages

Qwen3's chat template natively supports `<think>...</think>` blocks preceding
the final answer, so we target that format rather than inventing a new one --
the trained student's output shape then matches what the base model already
expects at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from teacher.cot_recovery import normalize_training_reasoning


def _assistant_content(trajectory: dict[str, Any]) -> str:
    # A null/absent response (e.g. a completion that returned content: null, or a
    # partial/legacy row) must not crash SFT conversion / mix aggregation (issue #212).
    response = str(trajectory.get("response") or "").strip()
    # Drop encrypted/JSON reasoning dumps: only plaintext CoT belongs in <think>.
    # CoT recovery (teacher.cot_recovery) backfills a Fable rationale upstream when a
    # teacher returns no usable trace, but this guard also protects older/raw data.
    reasoning = normalize_training_reasoning(trajectory.get("reasoning"))
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{response}"
    return response


def to_sft_record(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Alpaca-style record for legacy Axolotl `type: alpaca` recipes."""
    return {
        "prompt": trajectory["prompt"],
        "response": _assistant_content(trajectory),
        "system": trajectory.get("system"),
    }


def to_messages_record(trajectory: dict[str, Any]) -> dict[str, Any]:
    """OpenAI messages record for Axolotl `type: chat_template` + `chat_template: qwen3_5`."""
    messages: list[dict[str, str]] = []
    system = trajectory.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": trajectory["prompt"]})
    messages.append({"role": "assistant", "content": _assistant_content(trajectory)})
    return {"messages": messages}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="jsonl of trajectory records")
    parser.add_argument("--out", type=Path, required=True, help="output jsonl of SFT-ready records")
    parser.add_argument(
        "--format",
        choices=["alpaca", "messages"],
        default="messages",
        help="messages: {messages:[...]} for chat_template: qwen3_5; alpaca: legacy {prompt,response,system}",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    convert = to_messages_record if args.format == "messages" else to_sft_record

    count = 0
    with args.in_path.open(encoding="utf-8") as in_f, args.out.open("w", encoding="utf-8", newline="\n") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            out_f.write(json.dumps(convert(json.loads(line))) + "\n")
            count += 1

    print(f"wrote {count} SFT records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
