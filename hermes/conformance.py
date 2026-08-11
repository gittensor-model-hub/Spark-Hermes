"""The Hermes wire format, pinned as bytes rather than as code.

Hermes is upstream and fixed. This project's product is a better model *for* the official
Hermes Agent, not a variant of it -- so the format is not ours to move, and a change to it
is a defect regardless of how good the change is.

But the format lives in `hermes/protocol.py` as code that assembles strings. Code can be
edited, and an edit to a prompt template does not look like a breaking change in review:
no signature moves, no test asserts the sentence that was reworded, and the suite stays
green because everything downstream was rebuilt from the same edited source. The model
trained afterwards is the one that pays, and it pays silently -- it will emit whatever it
was trained on, to a runtime expecting what it was not.

So the rendered prompt is committed as a file. `hermes/templates/system-<dialect>.txt` is
the exact text a model is conditioned on for a fixed reference tool set, and a test
compares it byte for byte. The point is not that the file is more correct than the code.
The point is that changing the code now requires changing a checked-in artifact in the
same commit, which puts the diff in front of a reviewer instead of leaving it implicit.

Regenerate deliberately, never reflexively:

    python -m hermes.conformance --update

**What this does not do.** It pins the format against *our own* drift, not against
upstream's. Nothing here fetches Nous's template, because a test that reaches the network
fails for reasons unrelated to the code and gets deleted. When a real base model is chosen,
its `tokenizer_config.json` carries the template it was actually trained with, and
comparing against that -- at a pinned revision, via `eval.hf_pin` -- is the check that
catches upstream moving. That check belongs with the model decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hermes.protocol import DIALECTS, render_system, tool_schema

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# A fixed tool set, chosen for coverage rather than realism: a required string, an optional
# enum, and a nested object. The rendered prompt has to exercise the parts of the envelope
# that could plausibly change shape -- a single no-argument tool would pin almost nothing.
# These are reference values for the pin and are not the suite's tools.
REFERENCE_TOOLS = [
    tool_schema(
        "terminal",
        "Run a shell command in the task workspace.",
        {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    ),
    tool_schema(
        "file_write",
        "Write text to a file, creating parent directories as needed.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            "required": ["path", "content"],
        },
    ),
]


# Markers `hermes.atem` parses. Pinning these against the model's own committed template is what
# a rendered-prompt pin does for Hermes: it makes a format change require editing a checked-in
# artifact in the same commit. If a later revision of the model renames a tag, this fails and the
# parser is revisited -- rather than the parser quietly returning zero calls forever.
ATEM_MARKERS = (
    "<atem:function_calls>",
    "</atem:function_calls>",
    '<atem:invoke name="',
    '<atem:parameter name="',
    "</atem:parameter>",
    '<tool_output name="',
    "</tool_output>",
    "to=self",
    "# Valid recipients:",
    "Reasoning strength:",
)


def template_path(dialect_name: str) -> Path:
    """Where a dialect's pinned artifact lives.

    For Hermes that is the system prompt this repo renders, because the repo is what writes it.
    For ATEM it is the model's own `chat_template.jinja` at the pinned revision, because the repo
    writes no tool block at all -- the template does, and it is the thing the model is actually
    conditioned on. This module's own docstring named that as the check belonging with the model
    decision; this is that decision.
    """
    if DIALECTS.get(dialect_name) is not None and not DIALECTS[dialect_name].tools_in_prompt:
        return TEMPLATE_DIR / f"chat-template-{dialect_name}.jinja"
    return TEMPLATE_DIR / f"system-{dialect_name}.txt"


def render(dialect_name: str) -> str:
    """The system turn a model in this dialect is conditioned on, for the reference tools."""
    dialect = DIALECTS[dialect_name]
    return render_system(REFERENCE_TOOLS, dialect=dialect, scratch_pad=dialect.supports_scratch_pad)


def pinned(dialect_name: str) -> str | None:
    path = template_path(dialect_name)
    return path.read_text(encoding="utf-8") if path.is_file() else None


def drift(dialect_name: str) -> str:
    """Empty when the rendered prompt matches its pin; otherwise why it does not.

    Reported as text rather than a boolean because the useful output is the first line that
    differs. "The Hermes 4 system prompt changed" sends a reviewer to diff two 650-character
    strings by eye; naming the line does not.
    """
    expected = pinned(dialect_name)
    if expected is None:
        return f"no pinned template for {dialect_name}; run `python -m hermes.conformance --update`"

    dialect = DIALECTS.get(dialect_name)
    if dialect is not None and not dialect.tools_in_prompt:
        # Nothing of ours to render, so the check is the other way round: every marker the parser
        # depends on must still be in the model's template. A comparison of the committed file
        # against itself would be a check that cannot fail.
        absent = [m for m in ATEM_MARKERS if m not in expected]
        if absent:
            return (
                f"{dialect_name}: the pinned chat template no longer contains {absent!r}, which "
                f"hermes.{dialect_name} parses. A renamed tag makes the parser return zero calls on "
                "every turn, which reads as a model that never calls tools."
            )
        return ""

    actual = render(dialect_name)
    if actual == expected:
        return ""
    exp_lines, act_lines = expected.splitlines(), actual.splitlines()
    for index, (want, got) in enumerate(zip(exp_lines, act_lines, strict=False), start=1):
        if want != got:
            return f"{dialect_name} system prompt drifted at line {index}:\n  pinned:   {want!r}\n  rendered: {got!r}"
    return (
        f"{dialect_name} system prompt drifted in length: pinned has {len(exp_lines)} line(s), "
        f"rendered has {len(act_lines)}"
    )


def update() -> list[str]:
    """Rewrite every pin from the current code. Returns the dialects that changed."""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    changed = []
    for name in sorted(DIALECTS):
        if not DIALECTS[name].tools_in_prompt:
            # Not generated from this repo's code. The artifact is the model's own template, fetched
            # once at a pinned revision and committed; regenerating it here would overwrite an
            # upstream file with something this repo made up.
            continue
        text = render(name)
        if pinned(name) != text:
            template_path(name).write_text(text, encoding="utf-8")
            changed.append(name)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="rewrite the pins from the current code")
    args = parser.parse_args(argv)

    if args.update:
        changed = update()
        print("no change" if not changed else f"updated: {', '.join(changed)}")
        return 0

    problems = [d for name in sorted(DIALECTS) if (d := drift(name))]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
