"""Durable local training supervisor used by the cycle controller.

The supervisor, not the requesting CLI, owns completion. A lost CLI response can be
reconciled without relaunching training. A killed supervisor with no completion is
uncertain and requires a new reviewed run; checkpoint existence is never success.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, canonical, file_digest, write_record
from admin.candidates import file_identity
from admin.pipeline import Workspace
from admin.release import ReleaseAuthority
from admin.training import merge, train, training_recipe
from hermes.evidence_json import evidence_object
from validator.persistence import locked


class FixtureTraining:
    """Explicit external Axolotl substitute; all recipe/adapter/merge checks still run."""

    def __init__(self, workspace: Workspace, job_id: str):
        if workspace.identity["mode"] != "fixture":
            raise StageError("fixture training is forbidden in production")
        self.workspace, self.job_id = workspace, job_id

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        import yaml

        config = yaml.safe_load(Path(argv[2]).read_text())
        output = Path(config["output_dir"])
        if argv[1] == "train":
            output.mkdir(parents=True, exist_ok=True)
            write_record(
                output / "adapter_config.json",
                {
                    "base_model_name_or_path": config["base_model"],
                    "peft_type": "LORA",
                    "r": config["lora_r"],
                    "lora_alpha": config["lora_alpha"],
                    "target_modules": config["lora_target_modules"],
                    "fixture_only": True,
                },
            )
            (output / "adapter_model.safetensors").write_text("CPU_FIXTURE_NOT_WEIGHTS:" + self.job_id)
        elif argv[1] == "merge-lora":
            output = output / "merged"
            output.mkdir(parents=True)
            write_record(output / "config.json", {"model_type": "qwen3_5", "fixture_only": True})
            (output / "model.safetensors").write_text("CPU_FIXTURE_NOT_MODEL_WEIGHTS:" + self.job_id)
            write_record(output / "tokenizer.json", {"fixture_only": True})
            write_record(
                output / "tokenizer_config.json",
                {
                    "fixture_only": True,
                    "chat_template": config["chat_template_jinja"],
                },
            )
        else:
            raise StageError("unknown fixture training command")
        import json

        with (self.workspace.root / "fixture-training.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "fixture_only": True,
                        "trained": False,
                        "job_id": self.job_id,
                        "argv": argv,
                        "recipe_sha256": file_digest(Path(argv[2])),
                        "outputs": [str(p) for p in sorted(output.iterdir()) if p.is_file()],
                        "exit_code": 0,
                    }
                )
                + "\n"
            )
        return subprocess.CompletedProcess(argv, 0)


def run_job(root: Path, job_id: str) -> None:
    authority = ReleaseAuthority(root)
    with locked(root / "cycle-jobs" / (job_id.removeprefix("sha256:") + ".lock")):
        with authority.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request,status FROM cycle_external_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise StageError("external job was not submitted by the controller")
            request, state = evidence_object(row[0]), row[1]
            if state == "complete":
                return
            if state != "submitted":
                raise StageError("external job has uncertain/failed execution; never relaunch it")
            db.execute("UPDATE cycle_external_jobs SET status='running',pid=? WHERE id=?", (os.getpid(), job_id))
        try:
            ws = Workspace(Path(request["workspace"]))
            if ws.identity != request["workspace_identity"]:
                raise StageError("external job workspace issuer changed")
            recipe = training_recipe(ws, "sft")
            if file_identity(recipe) != request["recipe"]:
                raise StageError("external job recipe changed before execution")
            if file_identity(ws.models / "sft/prepared.json") != request["preparation"]:
                raise StageError("external job preparation changed before execution")
            fixture = request["execution"] == "fixture"
            if request["execution"] not in {"fixture", "local"}:
                raise StageError("unknown external training execution mode")
            executor = FixtureTraining(ws, job_id) if fixture else None
            if request["stage"] == "train":
                result = train(ws, stage="sft", fixture_executor=executor)
                output = Path(result["adapter"])
                files = [file_identity(p) for p in sorted(output.iterdir()) if p.is_file()]
            elif request["stage"] == "merge":
                result = merge(ws, stage="sft", fixture_executor=executor)
                files = [file_identity(ws.models / "sft/merged.json")]
                files += [file_identity(Path(result["merged"]) / name) for name in result["files"]]
            else:
                raise StageError("external job stage is not train or merge")
            if file_digest(recipe) != request["recipe"]["sha256"]:
                raise StageError("external job inputs changed during execution")
            training_recipe(ws, "sft")
            completion = {
                "schema": "spark-cycle-execution-v1",
                "job_id": job_id,
                "request": request,
                "origin": authority.identity,
                "returncode": 0,
                "result": result,
                "files": files,
                "fixture_only": fixture,
                "trained": not fixture,
                "execution_status": "fixture-complete"
                if fixture
                else "trained"
                if request["stage"] == "train"
                else "merged",
            }
            with authority.store.connect() as db:
                db.execute(
                    "UPDATE cycle_external_jobs SET status='complete',completion=? WHERE id=?",
                    (canonical(completion), job_id),
                )
        except BaseException:
            with authority.store.connect() as db:
                db.execute("UPDATE cycle_external_jobs SET status='failed' WHERE id=?", (job_id,))
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    try:
        run_job(args.root, args.id)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        # Exceptions from training can contain commands but never credentials supplied here.
        print(f"cycle training job: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
