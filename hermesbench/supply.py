"""Turn source projects into a task suite: mutate, dedup, and split by lineage.

`hermesbench.mutation` can emit a task per mutation site and has had no caller. This is the
caller. It reads a manifest of working projects, mutates each declared target file, keeps
the mutants a verifier actually catches, discards the near-copies, and writes a suite.

The whole point is volume the arena can consume: a hand-written task needs its verifier
checked in both directions by a person, so fifteen is what a person produces. A mutant's
correct answer is known by construction and its verifier already exists, so the supply is
bounded by CPU rather than by attention.

Three properties this adds on top of `mutation.generate`:

**The split is assigned by lineage, not by task.** Every mutant of one source file shares a
`lineage_digest`, and the split is drawn once per lineage. This is the property that makes a
generated suite safe to benchmark on. Splitting per task would put `flip_comparison-0003`
in training and `flip_comparison-0007` in sealed evaluation -- the same file, the same
function, one operator apart -- and the model would arrive at the sealed set having already
been trained on its answer. The score would be real, reproducible, and meaningless.

**Lineage is content-addressed, not path-addressed.** It digests the original source, so
vendoring the same file into two projects yields one lineage rather than two. A path-based
id would call them different and let the split straddle them anyway, which is the same leak
wearing a different name.

**Dedup runs across the whole supply, not per project.** `DuplicateIndex` catches the same
verifier against the same environment outright, and the coverage matrix caps how many tasks
one (category, topic) bucket may hold -- because twelve genuinely distinct tasks that all
exercise one narrow skill is the failure that neither exact nor lexical dedup can see.

What this does not do is decide that a suite is good. `report()` states the yield, the
rejection reasons and the split sizes; a supply that emitted four hundred tasks from one
lineage is visible there rather than discovered when the benchmark stops discriminating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hermesbench.identity import (
    DEFAULT_SIMILARITY_THRESHOLD,
    CoverageMatrix,
    DuplicateIndex,
    Rejection,
)
from hermesbench.mutation import DEFAULT_OPERATORS, GenerationReport, MutationError, generate

# Where a task may be used. Assigned once per lineage and never per task.
TRAIN = "train"
DEV = "dev"
SEALED_EVAL = "sealed_eval"

SPLITS = (TRAIN, DEV, SEALED_EVAL)

# Default shares. Sealed is small because it is spent slowly: every public report against it
# leaks a little, and a large sealed set does not buy more confidence than a small one, only
# more surface to leak.
DEFAULT_SHARES: dict[str, float] = {TRAIN: 0.70, DEV: 0.15, SEALED_EVAL: 0.15}


class SupplyError(ValueError):
    """The supply cannot be generated as configured."""


@dataclass(frozen=True)
class SourceProject:
    """One working project to mutate, and the command that proves it works."""

    name: str
    project_dir: Path
    target_files: tuple[str, ...]
    verify: str
    category: str = "swe"
    topic: str = "debug"
    protected_paths: tuple[str, ...] = ()
    timeout_s: int = 120
    limit_per_file: int | None = None

    @staticmethod
    def from_record(record: dict[str, Any], *, root: Path) -> SourceProject:
        missing = [k for k in ("name", "project_dir", "target_files", "verify") if not record.get(k)]
        if missing:
            raise SupplyError(f"source project record is missing {missing}")
        directory = Path(record["project_dir"])
        return SourceProject(
            name=str(record["name"]),
            # Relative paths resolve against the manifest, so a manifest can be moved with
            # its projects and keep working.
            project_dir=directory if directory.is_absolute() else (root / directory).resolve(),
            target_files=tuple(record["target_files"]),
            verify=str(record["verify"]),
            category=str(record.get("category", "swe")),
            topic=str(record.get("topic", "debug")),
            protected_paths=tuple(record.get("protected_paths", ())),
            timeout_s=int(record.get("timeout_s", 120)),
            limit_per_file=record.get("limit_per_file"),
        )


def lineage_digest(source: str) -> str:
    """Content address of the file every mutant of it descends from.

    Over the source rather than its path: the same file vendored into two projects is one
    lineage, and calling them two would let the split put a mutant of one in training and a
    mutant of the other in sealed evaluation -- the leak this exists to prevent, wearing a
    different filename.
    """
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def split_for(lineage: str, *, shares: dict[str, float] | None = None, salt: str = "") -> str:
    """Which split a whole lineage belongs to.

    Deterministic in the lineage, so re-running the supply reproduces the same assignment
    and a task cannot drift between splits when a project is regenerated. `salt` lets a
    fresh sealed set be drawn deliberately; changing it reassigns everything, which is the
    honest cost of a new sealed set rather than something to work around.
    """
    weights = shares or DEFAULT_SHARES
    total = sum(weights.values())
    if total <= 0:
        raise SupplyError("split shares sum to zero; every task would be unassigned")
    # A digest rather than hash(): Python's string hash is salted per process, so the same
    # lineage would land in a different split on every run.
    draw = int(hashlib.sha256(f"{salt}:{lineage}".encode()).hexdigest()[:16], 16) / float(1 << 64)
    cursor = 0.0
    for name in SPLITS:
        cursor += weights.get(name, 0.0) / total
        if draw < cursor:
            return name
    return SPLITS[-1]


@dataclass
class SupplyReport:
    """What the supply produced, what it refused, and where it landed."""

    emitted: int = 0
    accepted: int = 0
    rejected: list[tuple[str, str, str]] = field(default_factory=list)
    per_project: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_split: dict[str, int] = field(default_factory=dict)
    lineages: dict[str, str] = field(default_factory=dict)

    @property
    def rejection_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _task_id, reason, _detail in self.rejected:
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def to_record(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "accepted": self.accepted,
            "rejected": len(self.rejected),
            "rejection_reasons": self.rejection_reasons,
            "per_project": self.per_project,
            "per_split": dict(sorted(self.per_split.items())),
            "lineages": len(set(self.lineages.values())),
        }


def problems(report: SupplyReport) -> list[str]:
    """What is wrong with a finished supply, worst first. Empty means it is fit to use.

    Reported rather than raised: a supply can be lopsided and still worth keeping, and the
    caller is the one who knows whether this run was meant to be broad.
    """
    found: list[str] = []
    if not report.accepted:
        found.append("no task survived generation and dedup; the suite would be empty")
        return found
    lineage_count = len(set(report.lineages.values()))
    if lineage_count < len(SPLITS):
        found.append(
            f"only {lineage_count} lineage(s) across {report.accepted} tasks, and the split is drawn "
            f"per lineage -- fewer lineages than splits means at least one split is empty, however "
            "many tasks were generated"
        )
    empty = [name for name in SPLITS if not report.per_split.get(name)]
    if empty and lineage_count >= len(SPLITS):
        found.append(f"splits {empty} are empty; nothing can be evaluated on them")
    if report.emitted and report.accepted / report.emitted < 0.25:
        found.append(
            f"only {report.accepted} of {report.emitted} generated tasks survived dedup; the "
            "operators are mostly producing variations of one another on this source"
        )
    return found


def build(
    projects: list[SourceProject],
    *,
    workspace: Path,
    operators: tuple[str, ...] = DEFAULT_OPERATORS,
    shares: dict[str, float] | None = None,
    salt: str = "",
    cap_per_bucket: int | None = None,
    similarity_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], SupplyReport]:
    """Mutate every project, dedup across all of them, and stamp lineage and split."""
    if not projects:
        raise SupplyError("no source projects; there is nothing to mutate")

    coverage = CoverageMatrix(cap_per_bucket=cap_per_bucket) if cap_per_bucket else CoverageMatrix()
    index = DuplicateIndex(
        coverage=coverage,
        similarity_threshold=(DEFAULT_SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold),
    )
    report = SupplyReport()
    records: list[dict[str, Any]] = []

    for project in projects:
        per_file: dict[str, Any] = {}
        for target in project.target_files:
            source_path = project.project_dir / target
            if not source_path.is_file():
                raise SupplyError(f"{project.name}: no such target file {source_path}")
            lineage = lineage_digest(source_path.read_text(encoding="utf-8"))
            split = split_for(lineage, shares=shares, salt=salt)

            tasks, generation = generate(
                project_dir=project.project_dir,
                target_file=target,
                verify=project.verify,
                workspace=workspace / project.name / target.replace("/", "_"),
                task_prefix=f"{project.name}-{Path(target).stem}",
                protected_paths=project.protected_paths,
                operators=operators,
                timeout_s=project.timeout_s,
                limit=project.limit_per_file,
            )
            per_file[target] = {**generation.to_record(), "lineage": lineage, "split": split}
            report.emitted += generation.emitted

            for task in tasks:
                record = dict(task.to_record())
                # Stamped before the dedup check so a rejected task still carries the
                # lineage that explains which family it duplicated.
                record["metadata"] = {
                    **record.get("metadata", {}),
                    "lineage_digest": lineage,
                    "split": split,
                    "source_project": project.name,
                    "generator": "hermesbench.supply",
                }
                record["tags"] = sorted({*record.get("tags", []), split, project.category})
                report.lineages[record["task_id"]] = lineage

                rejection = index.add(record, category=project.category, topic=project.topic)
                if rejection is not None:
                    report.rejected.append((record["task_id"], rejection.reason, rejection.detail))
                    continue
                records.append(record)
                report.accepted += 1
                report.per_split[split] = report.per_split.get(split, 0) + 1

        report.per_project[project.name] = per_file

    return records, report


def write_suite(records: list[dict[str, Any]], out_dir: Path) -> int:
    """Write one YAML task per record, named by task id.

    One file per task rather than one suite file, because that is what `iter_tasks` reads
    and because a per-task file makes a bad generated task deletable without a merge
    conflict against every other task generated in the same run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = out_dir / f"{record['task_id']}.yaml"
        path.write_text(yaml.safe_dump(record, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return len(records)


def load_manifest(path: Path) -> list[SourceProject]:
    """Source projects from a JSON or YAML manifest."""
    body = path.read_text(encoding="utf-8")
    parsed = json.loads(body) if path.suffix == ".json" else yaml.safe_load(body)
    if not isinstance(parsed, dict) or "projects" not in parsed:
        raise SupplyError(f"{path}: manifest needs a top-level 'projects' list")
    return [SourceProject.from_record(record, root=path.parent) for record in parsed["projects"]]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="source-project manifest (JSON or YAML)")
    parser.add_argument("--out", type=Path, required=True, help="suite directory to write tasks into")
    parser.add_argument("--workspace", type=Path, required=True, help="scratch directory for mutation trials")
    parser.add_argument("--salt", default="", help="redraw the splits; changing it reassigns every lineage")
    parser.add_argument("--cap-per-bucket", type=int, default=None, help="max tasks per (category, topic)")
    parser.add_argument("--similarity", type=float, default=None, help="objective-duplicate threshold")
    parser.add_argument("--report", type=Path, default=None, help="write the generation report here")
    parser.add_argument("--dry-run", action="store_true", help="generate and report without writing the suite")
    args = parser.parse_args(argv)

    try:
        projects = load_manifest(args.manifest)
        records, report = build(
            projects,
            workspace=args.workspace,
            salt=args.salt,
            cap_per_bucket=args.cap_per_bucket,
            similarity_threshold=args.similarity,
        )
    except (SupplyError, MutationError) as exc:
        print(f"hermesbench.supply: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.to_record(), indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    found = problems(report)
    if not args.dry_run:
        written = write_suite(records, args.out)
        print(f"wrote {written} tasks to {args.out}", file=sys.stderr)

    if found:
        print("\nthis supply is not fit to benchmark on:", file=sys.stderr)
        for problem in found:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "DEFAULT_SHARES",
    "DEV",
    "SEALED_EVAL",
    "SPLITS",
    "TRAIN",
    "GenerationReport",
    "Rejection",
    "SourceProject",
    "SupplyError",
    "SupplyReport",
    "build",
    "lineage_digest",
    "load_manifest",
    "main",
    "problems",
    "split_for",
    "write_suite",
]


if __name__ == "__main__":
    raise SystemExit(main())
