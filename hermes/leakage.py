"""How much of a corpus's assistant text was caused by the system prompt.

A trajectory is produced with some system prompt S in context. Train on the assistant
turns and serve without S, and the model has to reproduce behaviour whose only cause is
gone. That is the whole bet of this project -- miner guidance is scaffolding, the model
internalises it, users run stock Hermes -- and nothing measured whether it holds.

Measured on the canonical mining export (133 rows, one system prompt across all of them):
133 assistant turns emit the validator sentinel `SPARKPROOF_TRITON_PASS`; 51 user turns
ask for it. The other **82 rows, 61.7%, emit it because S told them to**. 111 rows name a
GPU architecture that appears nowhere but S. Strip S and train on that and the model
learns to print a CI sentinel unconditionally.

**The obvious detector does not work.** Explicit back-references -- "as instructed", "per
the ledger" -- are the rare case: 7 of those 133 rows, roughly one genuinely dangling. The
universal case leaves no textual trace of its cause at all, so a blacklist of giveaway
phrases catches almost nothing. What is detectable is narrower and sound: text the
assistant produced that appears in S and does *not* appear anywhere the assistant could
have got it from in this conversation. That is not a guess about influence, it is an
observation about provenance.

This measures. It does not repair. There is no automatic fix -- deciding whether a
behaviour should survive without its prompt is a judgement about what the model ought to
do, and the honest options are to keep S in the row, replace it with a canonical prompt
that ships with the model, or accept the shift knowingly. `hermes.format` exposes those
three as an explicit choice and records which was taken; this tells you what it costs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# Phrases below this length are noise: "the", "in the file", "you should" appear in every
# system prompt and every answer, and reporting them would bury the real findings.
MIN_PHRASE_WORDS = 3
# Above this, a match is almost always the assistant quoting a whole instruction back,
# which the shorter windows already caught as their prefix.
MAX_PHRASE_WORDS = 8

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
# A single token worth reporting on its own: a sentinel, an identifier, a version, a
# hardware target. Anything with an underscore, an internal digit, or an interior capital.
#
# Word *phrases* alone miss the leak that matters. Measured on the canonical export, the
# three-word floor found 5.3% of rows; the actual figure is 61.7%, because the leaked text
# is `SPARKPROOF_TRITON_PASS` -- one token, emitted inside `print(...)`, never part of a
# three-word run shared with the prompt. Prose repetition is the rare case; a bare
# identifier copied out of the system prompt is the common one.
_MARKER = re.compile(
    r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[A-Za-z]+\d[A-Za-z0-9.]*|[A-Z]{2,}[A-Za-z0-9_]*|\w*[a-z][A-Z]\w*)\b"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _markers(text: str) -> set[str]:
    """Distinctive single tokens, case-folded for comparison but detected with case intact."""
    return {m.lower() for m in _MARKER.findall(text) if len(m) >= 4}


def _phrases(text: str) -> set[str]:
    """Distinctive tokens plus every word window, normalised for comparison.

    Windows rather than sentences: an assistant turn rarely repeats a whole sentence from
    the prompt, but it routinely repeats the distinctive three or four words in it.
    """
    words = _words(text)
    out: set[str] = _markers(text)
    for size in range(MIN_PHRASE_WORDS, MAX_PHRASE_WORDS + 1):
        for start in range(len(words) - size + 1):
            out.add(" ".join(words[start : start + size]))
    return out


@dataclass(frozen=True)
class RowLeakage:
    """One row's assistant text that traces to the system prompt and nowhere else."""

    index: int
    phrases: tuple[str, ...]

    @property
    def leaked(self) -> bool:
        return bool(self.phrases)


@dataclass(frozen=True)
class CorpusLeakage:
    """What a corpus would lose if its system prompt were removed before training."""

    rows: int
    system_prompts: int
    affected: tuple[RowLeakage, ...]

    @property
    def rate(self) -> float:
        return len(self.affected) / self.rows if self.rows else 0.0

    def top_phrases(self, limit: int = 12) -> list[tuple[str, int]]:
        """The phrases that leak most often, which is where a fix would start."""
        counts: dict[str, int] = {}
        for row in self.affected:
            for phrase in row.phrases:
                counts[phrase] = counts.get(phrase, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    @property
    def measurable(self) -> bool:
        """Whether a rate over this corpus means anything.

        A corpus with no system turns cannot be measured: there is nothing to trace text
        back to, so every row comes out clean and the rate reads 0.0. That is exactly what
        an already-stripped corpus looks like -- the leakage did not go away, it went
        invisible, and it is now baked into rows that no longer say what caused them.

        Measure on the `keep` corpus, before applying a system policy. The ordering is not
        a convention; it is the only order in which the question can be asked.
        """
        return self.rows == 0 or self.system_prompts > 0

    def to_record(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "system_prompts": self.system_prompts,
            "rows_affected": len(self.affected),
            "rate": round(self.rate, 4),
            "measurable": self.measurable,
            "top_phrases": [{"phrase": p, "rows": n} for p, n in self.top_phrases()],
            # Stated rather than implied. This measures provenance, not influence: a row
            # with no leaked phrase can still have been shaped by the prompt in ways no
            # string comparison reaches, and a clean report is not a clean corpus.
            "measures": "assistant text traceable to the system prompt and to nothing else in the row",
            "does_not_measure": "behaviour caused by the system prompt that left no textual trace",
        }


def _role_text(messages: Iterable[dict[str, Any]], role: str) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(str(function.get("name", "")))
            parts.append(str(function.get("arguments", "")))
    return "\n".join(parts)


def leakage(records: Iterable[dict[str, Any]]) -> CorpusLeakage:
    """Measure a corpus of `{"messages": [...]}` rows.

    A phrase counts as leaked when it appears in the system turn AND in an assistant turn
    AND in no user or tool turn. The last condition is what makes this provenance rather
    than coincidence: if the user asked for the sentinel, the assistant emitting it is
    explained without S, and removing S changes nothing about that row.
    """
    rows = list(records)
    prompts: set[str] = set()
    affected: list[RowLeakage] = []

    for index, record in enumerate(rows):
        messages = record.get("messages") or []
        system = _role_text(messages, "system")
        if not system.strip():
            continue
        prompts.add(system)
        assistant = _role_text(messages, "assistant")
        if not assistant.strip():
            continue
        # Everything the assistant could have taken the text from other than S. Tool
        # results count: an agent echoing a filename it read is not reciting the prompt.
        elsewhere = _phrases(_role_text(messages, "user") + "\n" + _role_text(messages, "tool"))
        leaked = (_phrases(system) & _phrases(assistant)) - elsewhere
        if leaked:
            # Longest first: "sparkproof_triton_pass after tests pass" is the finding;
            # its three-word prefixes are the same finding reported four more times.
            ordered = sorted(leaked, key=lambda p: (-len(p.split()), p))
            kept: list[str] = []
            for phrase in ordered:
                if not any(phrase in longer for longer in kept):
                    kept.append(phrase)
            affected.append(RowLeakage(index=index, phrases=tuple(kept)))

    return CorpusLeakage(rows=len(rows), system_prompts=len(prompts), affected=tuple(affected))


def main(argv: list[str] | None = None) -> int:
    """Report what a corpus would lose if its system prompt were stripped before training.

    Exits non-zero above `--max-rate`, so this can gate an export rather than only
    describe one. The default of 1.0 reports without failing: a number nobody has looked
    at yet should not start breaking pipelines on the day it lands.
    """
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="source", type=Path, required=True, help="messages jsonl to measure")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument("--max-rate", type=float, default=1.0, help="fail above this leaked-row fraction")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = leakage(records)
    record = report.to_record()
    print(json.dumps(record, indent=2))
    if not report.measurable:
        import sys

        print(
            f"none of these {report.rows} rows carries a system turn, so this rate is 0.0 because "
            "there is nothing to trace text back to -- not because the corpus is clean. An "
            "already-stripped corpus looks exactly like this. Measure before applying a system policy.",
            file=sys.stderr,
        )
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if report.rate > args.max_rate:
        import sys

        print(
            f"{len(report.affected)}/{report.rows} rows ({report.rate:.1%}) carry assistant text whose only "
            f"source is the system prompt, above the {args.max_rate:.1%} limit. Training on these and serving "
            "without that prompt asks the model to reproduce behaviour whose cause is gone.",
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = ["CorpusLeakage", "RowLeakage", "leakage", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
