"""Spark Hermes operator commands, with explicit fixture cycles and production prerequisites."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from admin.pipeline import (
    StageError,
    Workspace,
    build_corpus,
    evaluate_command,
    run_rollouts,
    status,
)
from admin.split import SplitError
from admin.training import doctor, merge, prepare_training, train
from hermes.base_model import load as load_base
from hermes.protocol import DIALECTS
from hermesbench.tasks import TaskError
from hermesbench.withheld import WithheldError
from validator.aggregate import AggregateError


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "runtime":
        from admin.runtime_transition import main as runtime_main

        return runtime_main(argv[1:])
    if argv and argv[0] == "cycle":
        from admin.cycles import main as cycle_main

        return cycle_main(argv[1:])
    if argv and argv[0] in {"replay", "curriculum", "parents", "candidates", "cotraining", "release"}:
        import importlib

        return importlib.import_module("admin." + argv[0]).main(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Additional command groups: runtime, cycle, replay, curriculum, parents, candidates, cotraining, release. "
            "Use spark-hermes GROUP --help for their options. CPU demonstration: "
            "spark-hermes cycle demo --root PATH --mode fixture. "
            "No command starts implicitly; production namespaces are the default. "
            "doctor/status do not launch training, contact providers, or deliver external actions."
        ),
    )
    parser.add_argument(
        "command",
        choices=(
            "status",
            "doctor",
            "selfcheck",
            "generate",
            "rollout",
            "corpus",
            "prepare",
            "train",
            "merge",
            "evaluate",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("var/admin/run-1"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", help="served model name; defaults to the selected profile")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dialect", default=load_base().hermes_dialect, choices=sorted(DIALECTS))
    parser.add_argument("--salt-file", type=Path)
    parser.add_argument("--max-pairs-per-task", type=int, default=8)
    parser.add_argument("--training-stage", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--sequence-len", type=int, help="default: 8192 for bf16; 2048 for rtx5090-poc")
    parser.add_argument("--profile", choices=("bf16", "rtx5090-poc"), default="bf16")
    parser.add_argument("--max-steps", type=int, help="prepare: cap training optimizer steps for a smoke test")
    parser.add_argument("--print-only", action="store_true", help="evaluate: print the command")
    parser.add_argument("--dry-run", action="store_true", help="train: validate and print without launching")
    parser.add_argument(
        "--software-only", action="store_true", help="doctor: exit based on CPU software, still list real prerequisites"
    )
    parser.add_argument(
        "--offline", action="store_true", help="prepare: only use tokenizer files already cached locally"
    )
    parser.add_argument("--serving-config", type=Path, help="evaluate: engine/device/precision and sampling JSON")
    parser.add_argument("--data-policy", type=Path, help="corpus: reviewed rights/family/exposure declarations")
    parser.add_argument("--parent-approval", help="prepare SFT: committed release decision ID")
    parser.add_argument("--release-root", type=Path, help="configured private release authority")
    parser.add_argument(
        "--fixture-tokenizer", action="store_true", help="prepare: CPU template fixture, fixture namespace only"
    )
    parser.add_argument("--allow-unsandboxed", action="store_true", help="assert this is a disposable execution host")
    args = parser.parse_args(argv)
    for flag, enabled, command in (
        ("--dry-run", args.dry_run, "train"),
        ("--print-only", args.print_only, "evaluate"),
        ("--software-only", args.software_only, "doctor"),
        ("--offline", args.offline, "prepare"),
        ("--fixture-tokenizer", args.fixture_tokenizer, "prepare"),
    ):
        if enabled and args.command != command:
            parser.error(f"{flag} is only valid for {command}")
    args.model = args.model or ("qwen3.5-4b" if args.profile == "rtx5090-poc" else "qwen3.8-27b")
    workspace = Workspace(args.root.resolve())
    try:
        if args.salt_file:
            salt = args.salt_file.read_text(encoding="utf-8").strip()
            if len(salt) < 16:
                raise StageError("salt file must contain at least 16 characters")
            os.environ["HERMESBENCH_WITHHELD_SALT"] = salt
        if args.command == "status":
            from admin.readiness import prerequisites

            print(f"pipeline at {workspace.root}")
            for row in status(workspace):
                print(f"  [{'done' if row['done'] else '  --'}] {row['stage']:<9} {row['summary']}")
            report = prerequisites(workspace, profile=args.profile)
            print(f"\nMode: {report['mode']}; profile: {report['base_model']}@{report['revision']}")
            for name, check in report["prerequisites"].items():
                print(f"  [{'ready' if check['ready'] else 'pending'}] {name}: {check['summary']}")
            print("\nStage records are workflow progress. Use doctor for JSON prerequisites; --help lists commands.")
            return 0
        if args.command == "doctor":
            report = doctor(workspace, profile=args.profile, software_only=args.software_only)
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 1
        if args.command == "selfcheck":
            from admin.selfcheck import run

            report = run()
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 1
        if args.command == "generate":
            if not args.allow_unsandboxed:
                raise StageError(
                    "generation executes model-written scripts; use --allow-unsandboxed on a disposable host"
                )
            if not args.salt_file:
                raise StageError("generate requires --salt-file")
            if args.count < 1 or args.concurrency < 1:
                raise StageError("count and concurrency must be positive")
            command = [
                sys.executable,
                "-m",
                "hermes.taskgen.cli",
                "--count",
                str(args.count),
                "--out",
                str(workspace.tasks / "generated"),
                "--withheld-out",
                str(workspace.tasks / "withheld"),
                "--salt-file",
                str(args.salt_file.resolve()),
                "--concurrency",
                str(args.concurrency),
                "--base-url",
                args.base_url,
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
            ]
            done = subprocess.run(command)
            if done.returncode:
                raise StageError(f"generation exited {done.returncode}")
            payload = json.loads((workspace.tasks / "generated/stage.json").read_text())
            payload["summary"] = f"{len(payload['accepted'])} accepted tasks"
        elif args.command == "rollout":
            payload = run_rollouts(
                workspace,
                repeats=args.repeats,
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                dialect=args.dialect,
                api_key_env=args.api_key_env,
                allow_unsandboxed=args.allow_unsandboxed,
            )
            payload["summary"] = f"{payload['tasks']} tasks x {payload['repeats']} attempts"
        elif args.command == "corpus":
            if args.max_pairs_per_task < 1:
                raise StageError("max-pairs-per-task must be positive")
            payload = build_corpus(workspace, max_pairs_per_task=args.max_pairs_per_task, data_policy=args.data_policy)
            payload["summary"] = f"{payload['sft_rows']} SFT rows, {payload['preference_pairs']} pairs"
        elif args.command == "prepare":
            tokenizer = None
            if args.fixture_tokenizer:
                if workspace.identity["mode"] != "fixture":
                    raise StageError("fixture tokenizer requires an explicitly created fixture workspace")
                from admin.selfcheck import FixtureTokenizer

                tokenizer = FixtureTokenizer()
            payload = prepare_training(
                workspace,
                stage=args.training_stage,
                sequence_len=args.sequence_len,
                profile=args.profile,
                max_steps=args.max_steps,
                local_files_only=args.offline,
                tokenizer=tokenizer,
                parent_approval=args.parent_approval,
                release_root=args.release_root,
            )
        elif args.command == "train":
            payload = train(workspace, stage=args.training_stage, dry_run=args.dry_run)
        elif args.command == "merge":
            payload = merge(workspace, stage=args.training_stage)
        else:
            from admin.evaluation import evaluate, read_serving

            serving = read_serving(args.serving_config) if args.serving_config else None
            command = evaluate_command(
                workspace,
                base_url=args.base_url,
                model=args.model,
                dialect=args.dialect,
                api_key_env=args.api_key_env,
                allow_unsandboxed=args.allow_unsandboxed,
                sampling={"temperature": serving.temperature, "top_p": serving.top_p} if serving else None,
            )
            if args.print_only:
                print(shlex.join(command))
                return 0
            if serving is None:
                raise StageError("evaluate requires --serving-config so the promotion gate can compare the run")
            payload = evaluate(
                workspace,
                serving=serving,
                base_url=args.base_url,
                model=args.model,
                dialect=args.dialect,
                api_key_env=args.api_key_env,
                allow_unsandboxed=args.allow_unsandboxed,
            )
        payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if args.command in {"generate", "rollout", "corpus", "train", "evaluate"} and not args.dry_run:
            workspace.record(args.command, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (StageError, SplitError, TaskError, WithheldError, AggregateError, OSError, ValueError, RuntimeError) as exc:
        print(f"admin: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
