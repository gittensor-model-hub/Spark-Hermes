"""Reading public trace datasets into `TaskDNA`, one dataset shape at a time.

`dna.extract` is deliberately generic -- it takes a dict and reads whatever keys it recognises. This
module holds the part that is not generic: what a particular published dataset actually looks like on
disk, which of its rows this harness can learn anything from, and what its licence permits.

## Licence decides where a seed may go, and it is recorded per row

`lambda/hermes-agent-reasoning-traces` is apache-2.0 and is the clean seed for anything commercial.
`kai-os/carnice-glm5-hermes-traces` is labelled `other` upstream, so it is available for research and
kept out of the canonical corpus until its provenance is resolved. Both are read the same way; the
licence travels on every DNA so a downstream corpus can be filtered by it rather than by memory.

None of that permits copying content. Seeds are read for structure only -- see `dna.assert_abstract`,
which fails a DNA that quotes its seed -- so the licence governs where the *shape* may travel, and
the shape is all that travels.

## Rows this harness cannot learn from are dropped, loudly

Roughly a seventh of the Lambda traces are browser automation. This runtime has a terminal, two file
tools and a Python interpreter, and no browser; a DNA that asks for one produces a task that cannot be
executed. Those rows are dropped and counted rather than being mapped onto a terminal, because a
"browser task done with curl" is a different task and pretending otherwise puts a category into the
corpus that the seed never demonstrated.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from hermes.taskgen.dna import DNAError, TaskDNA, extract

# Published datasets this module knows how to read. `path` is the file inside the repo; `licence` is
# what Hugging Face reports and is what governs where the DNA may travel.
SOURCES: dict[str, dict[str, str]] = {
    "lambda": {
        "repo": "lambda/hermes-agent-reasoning-traces",
        "path": "data/kimi/train.parquet",
        "licence": "apache-2.0",
    },
    "lambda-glm": {
        "repo": "lambda/hermes-agent-reasoning-traces",
        "path": "data/glm-5.1/train.parquet",
        "licence": "apache-2.0",
    },
}

# Categories this harness cannot reproduce. Dropped rather than remapped: a browser task performed
# with curl is a different task, and calling it the same one puts a capability in the corpus that no
# seed ever demonstrated.
UNSUPPORTED_CATEGORIES = ("Browser Automation",)


@dataclass
class SeedStats:
    """What a pass over a dataset actually yielded, including everything it threw away.

    Reported rather than summarised into a single count, for the reason this repository keeps
    rediscovering: "300 seeds" and "300 seeds out of 8,000, 5,000 of them dropped for reasons nobody
    recorded" look identical in a number.
    """

    read: int = 0
    unsupported_category: int = 0
    no_mappable_tool: int = 0
    unusable: int = 0
    yielded: int = 0

    def to_record(self) -> dict[str, int]:
        return {
            "read": self.read,
            "unsupported_category": self.unsupported_category,
            "no_mappable_tool": self.no_mappable_tool,
            "unusable": self.unusable,
            "yielded": self.yielded,
        }


def _normalise(row: dict[str, Any]) -> dict[str, Any]:
    """One published row into the generic shape `dna.extract` reads.

    The Lambda traces are ShareGPT-style: `conversations` of `{from, value}` rather than
    `{role, content}`, and `tools` as a JSON *string* rather than a list. Translating here keeps the
    dataset's quirks out of `dna`, which has to stay readable against the next dataset too.
    """
    turns = []
    for turn in row.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") if "value" in turn else turn.get("content")
        turns.append({"role": {"human": "user", "gpt": "assistant"}.get(role, role), "content": content})

    tools: list[Any] = []
    raw = row.get("tools")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                tools = parsed
        except json.JSONDecodeError:
            # A row whose tool block does not parse tells us nothing about which tools it used, and
            # guessing from the prose would invent a capability. Left empty so `extract` drops it.
            tools = []
    elif isinstance(raw, list):
        tools = raw

    return {
        "task": row.get("task"),
        "messages": turns,
        "tools": tools,
        # Passed through so `extract` can prefer the dataset's own label over inference. Its authors
        # categorised these deliberately; a keyword vote over the prompt is a worse signal.
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
    }


def _observed_calls(row: dict[str, Any]) -> int:
    """Tool calls in a ShareGPT trace, which records them as text rather than as structured fields.

    Counted by the `<tool_call>` marker: these traces are Hermes-dialect, so the calls are in the
    assistant's text. Only the COUNT is taken -- it sets the horizon band a generated task aims at --
    and none of the content, which is what keeps this dialect-agnostic. A generated task is rendered
    in whatever dialect the pinned model speaks, not in the seed's.
    """
    count = 0
    for turn in row.get("conversations") or []:
        if isinstance(turn, dict) and turn.get("from") in ("gpt", "assistant"):
            count += str(turn.get("value") or "").count("<tool_call>")
    return count


def read(source: str, *, limit: int | None = None, stats: SeedStats | None = None) -> Iterator[TaskDNA]:
    """Stream one named source as DNA, skipping what this harness cannot use.

    Downloads through `huggingface_hub`, which caches, so a second pass over the same source costs
    nothing. Streaming rather than materialising: these files are tens of thousands of rows and only
    a few hundred are needed per generation run.
    """
    if source not in SOURCES:
        raise DNAError(f"unknown seed source {source!r}; known: {sorted(SOURCES)}")
    spec = SOURCES[source]
    stats = stats if stats is not None else SeedStats()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(spec["repo"], spec["path"], repo_type="dataset")
    table = pq.read_table(local)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}

    for index in range(table.num_rows):
        if limit is not None and stats.yielded >= limit:
            return
        row = {name: values[index] for name, values in columns.items()}
        stats.read += 1

        if str(row.get("category") or "") in UNSUPPORTED_CATEGORIES:
            stats.unsupported_category += 1
            continue

        record = _normalise(row)
        calls = _observed_calls(row)
        if calls:
            # `extract` counts structured `tool_calls`, which a ShareGPT row does not have. Handing
            # it the observed count keeps the horizon band real rather than falling back to a guess.
            record["steps"] = [None] * (calls * 3)

        try:
            dna = extract(record, dataset=spec["repo"], licence=spec["licence"], index=index)
        except DNAError as exc:
            if "no tool" in str(exc):
                stats.no_mappable_tool += 1
            else:
                stats.unusable += 1
            continue

        dna.source["category"] = row.get("category")
        dna.source["subcategory"] = row.get("subcategory")
        stats.yielded += 1
        yield dna


__all__ = ["SOURCES", "UNSUPPORTED_CATEGORIES", "SeedStats", "read"]
