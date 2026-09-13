"""Deterministic pull-request labels, and the table that decides whether one is worth anything.

`.gittensor/weights.json` declares two label families and they do different jobs:

    eval:* / dataset:*   carry a multiplier -- these are what a reward reads
    area:*               "Categorization only ... NOT emission weights"

Both are declared there and, until this module, nothing computed either. A merged pull request
left the repository with whatever a human remembered to click, which is the wrong place for a
number that feeds a payout to come from.

## Why the multiplier table is read rather than copied

The weights file is the single declaration of what a label is worth. Restating those values here
would create a second copy that drifts silently: the labels would keep being applied, the
multipliers would keep being read from the file, and the disagreement would only surface as
someone being paid the wrong amount. So `emission_multiplier` reads the file, and a label that is
not in it returns None rather than a guess.

## The check this exists to make loud

`unknown_emission_labels` exists because of a specific, live gap. `eval/strategy_track.py` emits
`strategy:ACCEPT` and `strategy:REJECT`, and `strategy:*` appears NOWHERE in `label_multipliers`.
A miner whose surface wins a round, merged with `strategy:ACCEPT` and nothing else, scores against
no multiplier at all.

That is not something this module can fix by choosing a tier -- what a won round is worth is a
policy decision with payout consequences, and inventing one here would bury it in a helper. What
it can do is refuse to let the gap be silent: anything shaped like an emission label that the
table does not price is reported.

## Area labels are a pure function of which paths changed

Deterministic so two runs over one diff agree, and derived from the areas the weights file already
names rather than a second list invented here. A path matching no area contributes no label; that
is correct rather than a fallback, because `non_quality_prs` in the same file says tooling, docs
and refactors earn no model-quality credit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

WEIGHTS_PATH = Path(".gittensor/weights.json")

# Which paths belong to which declared area. The AREA NAMES are not invented here -- they are the
# keys of `areas` in the weights file, and `check_areas_match_weights` fails if this drifts from
# it. Only the path mapping lives here, because the weights file describes areas in prose and
# prose cannot be matched against a diff.
AREA_PATHS: dict[str, tuple[str, ...]] = {
    "teacher": ("teacher/",),
    "recipes": ("recipes/", "hermes/recipes/"),
    "eval": ("eval/", "hermesbench/"),
    "proof": ("proof/",),
}

EMISSION_PREFIXES = ("eval:", "dataset:")

__all__ = [
    "AREA_PATHS",
    "EMISSION_PREFIXES",
    "LabelError",
    "area_labels",
    "check_areas_match_weights",
    "emission_multiplier",
    "load_weights",
    "unknown_emission_labels",
]


class LabelError(ValueError):
    """The weights file cannot be read, or declares something this module cannot honour."""


def load_weights(path: Path | None = None) -> dict[str, Any]:
    source = path or WEIGHTS_PATH
    if not source.is_file():
        raise LabelError(f"no weights file at {source}; label values are declared there and nowhere else")
    return json.loads(source.read_text(encoding="utf-8"))


def check_areas_match_weights(path: Path | None = None) -> list[str]:
    """Empty when `AREA_PATHS` names exactly the areas the weights file declares.

    Checked rather than assumed because the two halves live in different files by necessity --
    the weights file describes an area in prose, and a diff can only be matched against paths. An
    area declared there and unmapped here silently never gets applied; one mapped here and absent
    there applies a label nothing recognises.
    """
    declared = {k for k in (load_weights(path).get("areas") or {}) if not k.startswith("_")}
    mapped = set(AREA_PATHS)
    problems = []
    if missing := sorted(declared - mapped):
        problems.append(f"areas declared in weights.json but not mapped to paths here: {missing}")
    if extra := sorted(mapped - declared):
        problems.append(f"areas mapped here but not declared in weights.json: {extra}")
    return problems


def area_labels(changed_paths: Iterable[str]) -> list[str]:
    """The `area:*` labels a diff earns, sorted and deduplicated.

    A path under no declared area contributes nothing. That is the intended outcome, not a gap:
    the weights file says tooling, benchmarks, docs and refactors carry no model-quality credit,
    so labelling them would assert a category the reward model does not have.
    """
    found = set()
    for raw in changed_paths:
        path = raw.strip().lstrip("./")
        if not path:
            continue
        for area, prefixes in AREA_PATHS.items():
            if any(path.startswith(prefix) for prefix in prefixes):
                found.add(f"area:{area}")
    return sorted(found)


def emission_multiplier(label: str, path: Path | None = None) -> float | None:
    """What the weights file says this label is worth, or None if it does not price it.

    None is not zero. `eval:none` is priced AT zero deliberately -- a judged submission that
    earned nothing -- while an unpriced label means nobody has decided, and the two must not
    collapse into the same answer.
    """
    multipliers = load_weights(path).get("label_multipliers") or {}
    value = multipliers.get(label)
    return None if value is None else float(value)


def unknown_emission_labels(labels: Iterable[str], path: Path | None = None) -> list[str]:
    """Labels that look like they should carry a multiplier but that the table does not price.

    The live example is `strategy:*`. It is not caught by the prefix test below, which is exactly
    why the prefix test is not the whole check: `report` takes any label family a track actually
    emits and asks the table about it.
    """
    multipliers = load_weights(path).get("label_multipliers") or {}
    return sorted(label for label in labels if label.startswith(EMISSION_PREFIXES) and label not in multipliers)


def report(labels: list[str], path: Path | None = None) -> dict[str, Any]:
    """What to apply, what it is worth, and what nobody has priced."""
    priced, unpriced = {}, []
    for label in labels:
        value = emission_multiplier(label, path)
        if value is None:
            if not label.startswith("area:"):
                unpriced.append(label)
        else:
            priced[label] = value
    return {
        "labels": sorted(labels),
        "priced": priced,
        "unpriced": sorted(unpriced),
        "area_only": sorted(label for label in labels if label.startswith("area:")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--changed-files", type=Path, help="file holding one changed path per line")
    parser.add_argument("--label", action="append", default=[], help="an additional label to price (repeatable)")
    parser.add_argument("--check-areas", action="store_true", help="verify AREA_PATHS matches weights.json")
    args = parser.parse_args(argv)

    if problems := check_areas_match_weights():
        for problem in problems:
            print(problem, file=sys.stderr)
        if args.check_areas:
            return 1
    if args.check_areas:
        print("areas match weights.json")
        return 0

    paths = args.changed_files.read_text(encoding="utf-8").splitlines() if args.changed_files else []
    labels = area_labels(paths) + list(args.label)
    result = report(labels)
    print(json.dumps(result, indent=2, sort_keys=True))
    for label in result["unpriced"]:
        print(f"warning: {label} is emission-shaped but .gittensor/weights.json does not price it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
