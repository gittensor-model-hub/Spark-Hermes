"""The stages, what each one needs, and what it refuses to do without.

The pipeline is deliberately a state machine over a directory rather than one long script. Every
stage reads what the previous one wrote and writes a manifest of its own. Task generation resumes;
rollout logs are preserved on failure and require a fresh run directory before retrying.

    generate -> rollout -> corpus -> prepare -> train -> merge -> evaluate

## Stages refuse rather than degrade

Each stage states what it needs and stops if it is absent. The alternative -- proceeding with less --
is how a corpus ends up built from three tasks, or an evaluation ends up run against a baseline
nobody measured. Both look like results.

## What is measured where

`rollout` runs the TRAIN tasks. `evaluate` runs the EVAL tasks, which are never trained on, and
compares before against after through `hermes.promotion`. Keeping those two commands separate is not
tidiness: a single `run everything` that happened to include the eval tasks in the rollout set would
produce a number nobody could distinguish from a real one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, file_digest, read_record, write_record
from admin.split import SplitError, refuse_eval_tasks
from hermes.base_model import load as load_base

STAGES = ("generate", "rollout", "corpus", "prepare", "train", "merge", "evaluate")

MANIFEST = "stage.json"


@dataclass
class Workspace:
    """A pipeline run on disk. Every stage writes into its own subdirectory.

    Gitignored by living under `var/`: a corpus is regenerable and a rollout log is large, and the
    repository is not a place to keep either.
    """

    root: Path
    tasks: Path = field(init=False)
    rollouts: Path = field(init=False)
    corpus: Path = field(init=False)
    models: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.tasks = self.root / "tasks"
        self.rollouts = self.root / "rollouts"
        self.corpus = self.root / "corpus"
        self.models = self.root / "models"
        self.reports = self.root / "reports"

    @property
    def identity(self) -> dict[str, str]:
        from validator.persistence import state_identity

        return state_identity(self.root)

    def stage_dir(self, stage: str) -> Path:
        return {
            "generate": self.tasks,
            "rollout": self.rollouts,
            "corpus": self.corpus,
            "prepare": self.models / "preparation",
            "train": self.models,
            "merge": self.models / "merging",
            "evaluate": self.reports,
        }[stage]

    def manifest_of(self, stage: str) -> dict[str, Any] | None:
        path = self.stage_dir(stage) / MANIFEST
        if not path.is_file():
            return None
        try:
            payload = read_record(path)
            return payload if isinstance(payload, dict) else None
        except StageError:
            # A half-written manifest is worse than none: it makes a stage look complete. Reported as
            # absent so the stage re-runs rather than being skipped on corrupt evidence.
            return None

    def record(self, stage: str, payload: dict[str, Any]) -> Path:
        directory = self.stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST
        write_record(path, payload)
        for dependent in STAGES[STAGES.index(stage) + 1 :]:
            (self.stage_dir(dependent) / MANIFEST).unlink(missing_ok=True)
        return path


def status(workspace: Workspace) -> list[dict[str, Any]]:
    """What each stage has produced, in order, with the first unmet dependency named.

    Showing every stage matters because a pipeline whose third stage
    was re-run has a fourth stage holding stale output, and a status that reported only "next: train"
    would hide it.
    """
    rows = []
    for stage in STAGES:
        manifest = workspace.manifest_of(stage)
        rows.append(
            {
                "stage": stage,
                "done": manifest is not None,
                "summary": (manifest or {}).get("summary", ""),
                "at": (manifest or {}).get("completed_at", ""),
            }
        )
    return rows


def _require(workspace: Workspace, stage: str) -> dict[str, Any]:
    manifest = workspace.manifest_of(stage)
    if manifest is None:
        raise StageError(f"stage {stage!r} has not run; {workspace.stage_dir(stage) / MANIFEST} does not exist")
    return manifest


def generated_task_ids(workspace: Workspace) -> list[str]:
    """Task ids the generator produced and the gate accepted."""
    manifest = _require(workspace, "generate")
    accepted = manifest.get("accepted")
    if not isinstance(accepted, list) or not all(isinstance(t, str) and t for t in accepted):
        raise StageError("generate manifest must contain a list of accepted task IDs")
    if len(set(accepted)) != len(accepted):
        raise StageError("generate manifest contains duplicate task IDs")
    return accepted


def run_rollouts(
    workspace: Workspace,
    *,
    repeats: int,
    base_url: str,
    model: str,
    concurrency: int = 12,
    dialect: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    allow_unsandboxed: bool = False,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """N attempts on every accepted TRAIN task, through the real harness.

    `refuse_eval_tasks` runs first and raises rather than filtering. A rollout set that quietly
    included an eval task would produce a corpus that is correct and a count that is wrong, and the
    operator would go on believing they trained on what they asked for.
    """
    accepted = generated_task_ids(workspace)
    if not accepted:
        raise StageError("the generate stage accepted no tasks; there is nothing to roll out")
    refuse_eval_tasks(accepted)
    if repeats < 1 or concurrency < 1:
        raise StageError("repeats and concurrency must be positive")
    from hermesbench.tasks import load_suite

    tasks = load_suite("generated", root=workspace.tasks)
    if set(accepted) != {t.task_id for t in tasks}:
        raise StageError("generated task files disagree with the accepted task manifest")
    from admin.split import contamination

    overlaps = contamination(
        {t.task_id: t.prompt for t in tasks},
        {t.task_id: t.prompt for t in load_suite("v0,v1")},
    )
    if overlaps:
        raise SplitError("training/evaluation prompt overlap: " + "; ".join(map(str, overlaps)))
    environment = withheld_environment(tasks, workspace.tasks / "withheld")

    workspace.rollouts.mkdir(parents=True, exist_ok=True)
    episodes = workspace.rollouts / "episodes.jsonl"
    command = [
        sys.executable,
        "-m",
        "hermesbench.runner",
        "--suite",
        "generated",
        "--task-root",
        str(workspace.tasks),
        "--task-ids",
        ",".join(accepted),
        "--workspace-root",
        str(workspace.rollouts / "ws"),
        "--model",
        model,
        "--base-url",
        base_url,
        "--dialect",
        dialect or load_base().hermes_dialect,
        "--api-key-env",
        api_key_env,
        "--repeats",
        str(repeats),
        "--concurrency",
        str(concurrency),
        "--keep-trajectories",
        "--episodes-out",
        str(episodes),
        "--out",
        str(workspace.rollouts / "manifest.json"),
    ]
    if allow_unsandboxed:
        command.append("--allow-unsandboxed")
    if episodes.exists() and episodes.stat().st_size:
        raise StageError("rollout episodes already exist; use a new --root to avoid mixing runs")
    done = runner(command, env=environment)
    if getattr(done, "returncode", 1) != 0:
        raise StageError(f"the runner exited {done.returncode}; see its output above")
    return {
        "tasks": len(accepted),
        "repeats": repeats,
        "episodes": str(episodes),
        "origin": workspace.identity,
        "model": model,
    }


def build_corpus(
    workspace: Workspace, *, max_pairs_per_task: int = 8, data_policy: Path | None = None
) -> dict[str, Any]:
    """Rollouts to SFT rows and preference pairs, with the split enforced at the boundary.

    The cap defaults lower than `validator.aggregate`'s 32. Measured on a real 152-episode run, 33 of
    36 pairs came from two tasks; at that concentration a preference set teaches those two tasks and
    calls it a policy.
    """
    import hashlib

    from admin.artifacts import AuthorityStore, content_digest
    from admin.data_policy import DataPolicy
    from hermesbench.sink import decode_episodes
    from hermesbench.tasks import load_suite
    from validator.aggregate import Episode, admit_episode, aggregate
    from validator.score import normalize_episode

    rollout = _require(workspace, "rollout")
    if rollout.get("origin") != workspace.identity or not isinstance(rollout.get("model"), str) or not rollout["model"]:
        raise StageError("operator rollout requires recorded origin and model identity")
    accepted = set(generated_task_ids(workspace))
    episodes_path = workspace.rollouts / "episodes.jsonl"
    if not episodes_path.is_file():
        raise StageError(f"{episodes_path} does not exist; the rollout stage recorded no episodes")

    tasks = {t.task_id: t for t in load_suite("generated", root=workspace.tasks)}
    if set(tasks) != accepted:
        raise StageError("generated task files disagree with accepted corpus tasks")
    policy = DataPolicy(data_policy or workspace.root / "data-policy.json", identity=workspace.identity)
    episode_bytes = episodes_path.read_bytes()
    episode_sha256 = hashlib.sha256(episode_bytes).hexdigest()
    contribution = "sha256:" + episode_sha256
    episodes: list[Episode] = []
    for number, row in enumerate(decode_episodes(episode_bytes, source=str(episodes_path)), 1):
        metrics = normalize_episode(row)
        public = metrics.get("public_passed")
        if public is not True and public is not False:
            raise StageError("episode has no boolean public-check result")
        hidden = metrics.get("hidden_passed")
        if hidden is not True and hidden is not False:
            raise StageError("episode has no withheld-check result; refusing public-only training data")
        if metrics.get("task_id") not in accepted:
            raise StageError(f"episode task {metrics.get('task_id')!r} is not in the accepted training set")
        if metrics.get("dialect") != load_base().hermes_dialect:
            raise StageError("episode dialect does not match the pinned training model")
        task = tasks[metrics["task_id"]]
        if not task.declares_hidden_tests:
            raise StageError("operator training task must declare withheld checks")
        if metrics.get("verify_digest") != "sha256:" + hashlib.sha256(task.verify.encode()).hexdigest():
            raise StageError("operator episode public verifier differs from generated task")
        if row.get("trajectory", {}).get("task") != task.prompt:
            raise StageError("executed task prompt differs from generated task")
        version = content_digest(asdict(task))
        members = [m for m in policy.members if m["task_id"] == task.task_id and m["version"] == version]
        if len(members) != 1:
            raise StageError("operator task requires exact family/version membership")
        provenance = {
            "origin": workspace.identity,
            "miner_id": "operator",
            "round_id": "admin",
            "task_id": task.task_id,
            "task_version": version,
            "contribution_id": contribution,
            "model_revision": rollout["model"],
            "attempt_id": str(number),
            "episode_hash": content_digest(row),
            "log": {"path": str(episodes_path), "sha256": contribution},
            **policy.provenance(
                task_id=task.task_id, repository=members[0]["repository"], version=version, contribution=contribution
            ),
        }
        episodes.append(
            admit_episode(row, round_id="admin", miner_id="operator", private_required=True, provenance=provenance)
        )
    if not episodes:
        raise StageError(f"{episodes_path} holds no episodes; refusing to write an empty corpus")

    # The last line of defence. Everything upstream should have kept eval tasks out; this is where
    # a mistake anywhere upstream becomes visible instead of becoming training data.
    refuse_eval_tasks(sorted({e.task_id for e in episodes}))

    if file_digest(episodes_path) != episode_sha256 or file_digest(policy.path) != policy.sha256:
        raise StageError("operator episode/policy source changed during admission")

    summary = aggregate(episodes, workspace.corpus, max_per_task=max_pairs_per_task)
    if not summary.sft_rows:
        raise StageError("no usable verified SFT rows; collect successful executed trajectories first")
    payload = {
        **summary.to_record(),
        "schema": "spark-operator-corpus-v1",
        "origin": workspace.identity,
        "accepted": sorted(accepted),
        "episodes_sha256": episode_sha256,
        "sha256": {name: file_digest(workspace.corpus / name) for name in ("sft.jsonl", "preference.jsonl")},
        "inputs": {
            str(p.resolve()): file_digest(p)
            for p in [
                episodes_path,
                workspace.tasks / MANIFEST,
                workspace.rollouts / MANIFEST,
                policy.path,
                *sorted((workspace.tasks / "generated").glob("*.yaml")),
                *sorted((workspace.tasks / "generated").glob("*.yml")),
            ]
        },
        "authorizes_model_promotion": False,
    }
    authority = AuthorityStore(
        workspace.root / "learning-authority",
        role="operator-corpus",
        mode=workspace.identity["mode"],
        namespace=workspace.identity["namespace"],
    )
    record = authority.put("mixture", payload)
    return {**payload, "authority": {"root": str(authority.root), "identity": authority.identity, "id": record["id"]}}


def evaluate_command(
    workspace: Workspace,
    *,
    base_url: str,
    model: str,
    dialect: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    allow_unsandboxed: bool = False,
    sampling: dict[str, float] | None = None,
) -> list[str]:
    """The command that measures a model on the EVAL suite.

    Returned rather than run, because this is the number everything else is judged by and it should
    be launched deliberately -- with the withheld environment set, on a quiet machine, by someone who
    means to. A convenience wrapper that ran it as a side effect of another stage is how a baseline
    gets measured under load and then compared against one that was not.
    """
    return (
        [
            sys.executable,
            "-m",
            "hermesbench.runner",
            "--suite",
            "v0,v1",
            "--workspace-root",
            str(workspace.reports / "ws"),
            "--model",
            model,
            "--base-url",
            base_url,
            "--dialect",
            dialect or load_base().hermes_dialect,
            "--api-key-env",
            api_key_env,
            "--keep-trajectories",
            "--repeats",
            "10",
            "--episodes-out",
            str(workspace.reports / "episodes.jsonl"),
            "--out",
            str(workspace.reports / "manifest.json"),
        ]
        + (["--allow-unsandboxed"] if allow_unsandboxed else [])
        + [value for key, number in (sampling or {}).items() for value in ("--" + key.replace("_", "-"), str(number))]
    )


def withheld_environment(tasks: Any, root: Path | None = None) -> dict[str, str]:
    """Validate commitments before paying for any model requests."""
    from hermesbench.withheld import overlay, unscorable

    environment = os.environ.copy()
    salt = environment.get("HERMESBENCH_WITHHELD_SALT", "")
    if len(salt) < 16:
        raise StageError("HERMESBENCH_WITHHELD_SALT must contain at least 16 characters")
    resolved = overlay(tasks, root=root, salt=salt)
    missing = unscorable(resolved)
    if missing:
        raise StageError("missing withheld checks: " + ", ".join(missing))
    if root is not None:
        environment["SPARKDISTILL_WITHHELD_ROOT"] = str(root.resolve())
    return environment


__all__ = [
    "MANIFEST",
    "STAGES",
    "SplitError",
    "StageError",
    "Workspace",
    "build_corpus",
    "evaluate_command",
    "generated_task_ids",
    "run_rollouts",
    "status",
]
