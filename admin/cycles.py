"""Resumable operator cycles over the real admission, learning and release producers.

The release database is the single authority for cycles, jobs, incumbent and epochs.
Per-cycle locks serialize long producers; SQLite commits the durable job before execution.
Inputs and completed outputs are immutable, and missing evidence fails closed on resume.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from admin.artifacts import (
    AuthorityStore,
    StageError,
    canonical,
    content_digest,
    read_record,
    same_domain,
    write_record,
)
from admin.candidates import bound_record, checked_file, file_identity
from admin.curriculum import build_curriculum
from admin.pipeline import Workspace, _require
from admin.release import ReleaseAuthority
from admin.replay import ReplayStore, source_stores, verify_corpus
from admin.runtime_protocol import campaign_writer
from admin.training import prepare_training, train, training_recipe
from hermes.evidence_json import evidence_object
from validator.persistence import locked, state_identity
from validator.pr_admission import GitHubSource, admission_for, admit, round_identity

STAGES = (
    "admission",
    "settlement",
    "experience",
    "replay",
    "curriculum",
    "prepare",
    "train",
    "merge",
    "candidate",
    "plan",
    "evaluate",
    "decide",
    "activate",
)


class Pending(StageError):
    """A named external prerequisite is still pending; no stage completion is issued."""


def _result(value: dict[str, Any], *paths: Path) -> dict[str, Any]:
    return {"value": value, "files": [file_identity(p) for p in paths]}


class CycleController:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.release = ReleaseAuthority(root, mode=mode, namespace=namespace)
        self.store, self.identity = self.release.store, self.release.identity
        self.root = self.store.root
        with self.store.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS cycle_configuration (singleton INTEGER PRIMARY KEY CHECK(singleton=1), value BLOB NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS cycles (id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, request BLOB NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS cycle_jobs (cycle TEXT NOT NULL, stage TEXT NOT NULL, id TEXT UNIQUE NOT NULL, inputs TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL, output BLOB, error TEXT, PRIMARY KEY(cycle,stage))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS cycle_external_jobs (id TEXT PRIMARY KEY, request BLOB NOT NULL, status TEXT NOT NULL, pid INTEGER, completion BLOB)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS cycle_workspaces (cycle TEXT PRIMARY KEY, identity BLOB NOT NULL)")

    @campaign_writer
    def configure(
        self,
        *,
        replay: Path,
        bootstrap_parent: dict[str, Any] | None = None,
        github: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.release.configuration()
        source = ReplayStore(replay)
        same_domain(self.identity, source.identity)
        config = {
            "origin": self.identity,
            "replay": {"root": str(source.root), "identity": source.identity, "configuration": source.configuration()},
            "bootstrap_parent": None,
            "github": github,
        }
        if github is not None:
            if set(github) != {"repository", "credential_env"} or any(
                not isinstance(v, str) or not v for v in github.values()
            ):
                raise StageError("GitHub configuration requires repository and credential_env")
        if bootstrap_parent:
            parent = AuthorityStore(Path(bootstrap_parent["root"]), role="release")
            same_domain(self.identity, parent.identity)
            approval = parent.get(bootstrap_parent["id"], kind="release-decision")
            decision = approval["payload"]
            current = self.release.active_pair()
            if (
                decision.get("result") != "accepted"
                or decision.get("candidate") != current["model"]
                or decision.get("agent") != current["agent_id"]
            ):
                raise StageError("bootstrap parent must approve the exact configured initial pair")
            if self.identity["mode"] == "production":
                ReleaseAuthority(parent.root).resolve_decision(approval["id"])
            config["bootstrap_parent"] = {"root": str(parent.root), "identity": parent.identity, "id": approval["id"]}
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO cycle_configuration VALUES (1,?)", (canonical(config),))
            if db.execute("SELECT value FROM cycle_configuration WHERE singleton=1").fetchone()[0] != canonical(config):
                raise StageError("cycle controller configuration is immutable")
        return config

    def configuration(self) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute("SELECT value FROM cycle_configuration WHERE singleton=1").fetchone()
        if row is None:
            raise StageError("cycle init must configure the release and replay authorities first")
        config = evidence_object(row[0])
        if config["origin"] != self.identity:
            raise StageError("cycle controller issuer changed")
        source = ReplayStore(Path(config["replay"]["root"]))
        same_domain(self.identity, source.identity)
        if (
            source.identity != config["replay"]["identity"]
            or source.configuration() != config["replay"]["configuration"]
        ):
            raise StageError("configured replay source changed")
        return config

    def replay(self, identifier: str | None = None) -> ReplayStore:
        binding = self.configuration()["replay"]
        if identifier is not None:
            with self.store.connect() as db:
                row = db.execute("SELECT request FROM cycles WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise StageError("unknown cycle for replay lookup")
            request = evidence_object(row[0])
            if content_digest(request) != identifier:
                raise StageError("cycle replay binding changed")
            binding = request["replay"]
        replay = ReplayStore(Path(binding["root"]))
        same_domain(self.identity, replay.identity)
        if replay.identity != binding["identity"] or replay.configuration() != binding["configuration"]:
            raise StageError("cycle's frozen replay version changed")
        return replay

    @campaign_writer
    def start(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Freeze one operator request; repeating its name may only repeat identical inputs."""
        if not isinstance(name, str) or not name.strip():
            raise StageError("cycle requires a stable nonempty name")
        spec = evidence_object(canonical(spec))
        required = {"rounds", "agent", "workload", "training", "evaluation", "curriculum"}
        if not required <= set(spec) or set(spec) - required - {"replay", "history_ids", "confirmation_policy"}:
            raise StageError(
                "cycle specification requires: " + ", ".join(sorted(required)) + "; optional replay root/history_ids"
            )
        # Persist absolute paths once. A resumed installed CLI may have a different
        # working directory; its cwd must never select different stage inputs.
        for key in ("agent", "workload", "replay", "confirmation_policy"):
            if key in spec:
                spec[key] = str(Path(spec[key]).resolve())
        if "fixture" in spec["evaluation"]:
            spec["evaluation"]["fixture"] = str(Path(spec["evaluation"]["fixture"]).resolve())
        for item in spec["rounds"]:
            for key in ("scorecards", "episodes"):
                item[key] = str(Path(item[key]).resolve())
        with locked(self.root / ".cycle-start.lock"):
            with self.store.connect() as db:
                prior = db.execute("SELECT id,request FROM cycles WHERE name=?", (name,)).fetchone()
            if prior:
                if canonical(evidence_object(prior[1])["spec"]) != canonical(spec):
                    raise StageError("cycle name already binds different immutable inputs")
                return self.status(prior[0])
            config = self.configuration()
            replay = ReplayStore(Path(spec["replay"])) if "replay" in spec else self.replay()
            same_domain(self.identity, replay.identity)
            replay_binding = {
                "root": str(replay.root),
                "identity": replay.identity,
                "configuration": replay.configuration(),
            }
            baseline = self.release.active_pair()
            training = spec["training"]
            if set(training) != {"profile", "sequence_len", "max_steps", "execution"} or training["execution"] not in {
                "fixture",
                "local",
            }:
                raise StageError("training requires profile, sequence_len, max_steps and execution=local/fixture")
            if training["profile"] not in {"bf16", "rtx5090-poc"} or any(
                type(training[k]) is not int or training[k] < 1 for k in ("sequence_len", "max_steps")
            ):
                raise StageError("training profile/sequence_len/max_steps are invalid")
            if training["execution"] == "fixture" and self.identity["mode"] != "fixture":
                raise StageError("production cycles cannot consume fixture training")
            evaluation = spec["evaluation"]
            if set(evaluation) not in (
                {"schedule", "budget", "sampling", "serving"},
                {"schedule", "budget", "sampling", "fixture"},
            ):
                raise StageError("evaluation requires frozen schedule, budget, sampling and serving or fixture")
            if "fixture" in evaluation and self.identity["mode"] != "fixture":
                raise StageError("production cycles cannot consume fixture serving")
            files = {k: file_identity(Path(spec[k])) for k in ("agent", "workload")}
            if "confirmation_policy" in spec:
                files["confirmation_policy"] = file_identity(Path(spec["confirmation_policy"]))
            if "fixture" in evaluation:
                files["serving_fixture"] = file_identity(Path(evaluation["fixture"]))
                fixture = bound_record(files["serving_fixture"])
                if fixture.get("schema") != "spark-cycle-serving-fixture-v1" or fixture.get("origin") != self.identity:
                    raise StageError("cycle serving fixture schema/issuer mismatch")
            elif set(evaluation["serving"]) != {"old", "new"}:
                raise StageError("serving configuration must name old/new deployments")
            if not isinstance(spec["rounds"], list) or not spec["rounds"]:
                raise StageError("cycle requires explicit contribution rounds")
            rounds = []
            for item in spec["rounds"]:
                if set(item) != {"source", "round_id", "prs", "scorecards", "episodes"}:
                    raise StageError("round requires source, round_id, prs, scorecards and episodes")
                source = self._source(item["source"], replay=replay)
                store, _, _ = source_stores(source)
                win = store.load(item["round_id"])
                # Next-round producers must retain the activated pair binding. Initial
                # pre-controller rounds remain usable as the configured initial epoch.
                if (baseline["generation"] or baseline.get("authority_kind")) and canonical(
                    win.challenge.epoch.get("incumbent")
                ) != canonical(self.epoch_binding(baseline)):
                    raise StageError("round does not bind the active released model/agent/epoch")
                if not item["prs"] or any(
                    set(p) != {"number", "head", "author"} or type(p["number"]) is not int or p["number"] <= 0
                    for p in item["prs"]
                ):
                    raise StageError("round requires exact PR number/head/author commitments")
                if len({p["author"] for p in item["prs"]}) != len(item["prs"]):
                    raise StageError("duplicate admitted author in cycle round")
                rounds.append({"source": item["source"], "round_id": item["round_id"], "identity": round_identity(win)})
            if len({(r["source"], r["round_id"]) for r in rounds}) != len(rounds):
                raise StageError("duplicate cycle round")
            parent = (
                {"root": str(self.root), "identity": self.identity, "id": baseline["approval"]}
                if baseline["approval"]
                else config["bootstrap_parent"]
            )
            if parent is None:
                raise StageError("initial pair requires configured approved-parent authority before repeated SFT")
            request = {
                "schema": "spark-cycle-v1",
                "origin": self.identity,
                "name": name,
                "spec": spec,
                "configuration_hash": content_digest(config),
                "replay": replay_binding,
                "baseline": baseline,
                "parent": parent,
                "files": files,
                "rounds": rounds,
            }
            identifier = content_digest(request)
            ws = self.workspace(identifier)
            state_identity(ws.root, mode=self.identity["mode"], namespace=self.identity["namespace"])
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT candidate,generation FROM incumbent WHERE singleton=1").fetchone()
                if current != (baseline["candidate"], baseline["generation"]):
                    raise StageError("incumbent changed while creating cycle; retry with the current pair")
                db.execute("INSERT INTO cycles VALUES (?,?,?)", (identifier, name, canonical(request)))
                db.execute("INSERT INTO cycle_workspaces VALUES (?,?)", (identifier, canonical(ws.identity)))
        return self.status(identifier)

    @staticmethod
    def epoch_binding(pair: dict[str, Any]) -> dict[str, Any]:
        return {
            k: pair[k] for k in ("origin", "candidate", "approval", "generation", "epoch_id", "model_id", "agent_id")
        }

    def active_pair(self) -> dict[str, Any]:
        return self.release.active_pair()

    def workspace(self, identifier: str) -> Workspace:
        if (
            not isinstance(identifier, str)
            or not identifier.startswith("sha256:")
            or len(identifier) != 71
            or any(c not in "0123456789abcdef" for c in identifier[7:])
        ):
            raise StageError("invalid cycle ID")
        return Workspace(self.root / "cycles" / identifier[7:])

    def request(self, identifier: str) -> dict[str, Any]:
        self.workspace(identifier)
        with self.store.connect() as db:
            row = db.execute("SELECT request FROM cycles WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise StageError("unknown cycle ID")
        result = evidence_object(row[0])
        if content_digest(result) != identifier or result["origin"] != self.identity:
            raise StageError("cycle request identity changed")
        if content_digest(self.configuration()) != result["configuration_hash"]:
            raise StageError("cycle configuration changed")
        for binding in result["files"].values():
            checked_file(binding)
        for binding in result["rounds"]:
            store, _, _ = source_stores(self._source(binding["source"], replay=self.replay(identifier)))
            if round_identity(store.load(binding["round_id"])) != binding["identity"]:
                raise StageError("cycle round inputs changed")
        source = self.release.candidates()
        baseline = source.resolve(result["baseline"]["candidate"])["payload"]
        if (
            baseline["model_id"] != result["baseline"]["model_id"]
            or baseline["agent_id"] != result["baseline"]["agent_id"]
        ):
            raise StageError("cycle baseline pair changed")
        same_domain(self.identity, self.workspace(identifier).identity)
        with self.store.connect() as db:
            workspace = db.execute("SELECT identity FROM cycle_workspaces WHERE cycle=?", (identifier,)).fetchone()
        if workspace is None or workspace[0] != canonical(self.workspace(identifier).identity):
            raise StageError("cycle workspace issuer changed")
        return result

    def _source(self, name: str, *, replay: ReplayStore | None = None) -> dict[str, Any]:
        sources = [s for s in (replay or self.replay()).configuration()["sources"] if s["name"] == name]
        if len(sources) != 1:
            raise StageError("unknown configured contribution source")
        source = sources[0]
        stores = source_stores(source)
        if {k: s.identity for k, s in zip(("rounds", "settlement", "intake"), stores)} != source["identities"]:
            raise StageError("cycle source issuer changed")
        return source

    def _jobs(self, identifier: str) -> dict[str, Any]:
        with self.store.connect() as db:
            return {
                r[0]: {
                    "stage": r[0],
                    "id": r[1],
                    "inputs": r[2],
                    "status": r[3],
                    "attempts": r[4],
                    "output": evidence_object(r[5]) if r[5] else None,
                    "error": r[6],
                }
                for r in db.execute(
                    "SELECT stage,id,inputs,status,attempts,output,error FROM cycle_jobs WHERE cycle=?", (identifier,)
                )
            }

    def _validate(self, identifier: str, stage: str, output: dict[str, Any]) -> None:
        if set(output) != {"value", "files", "digest"} or output["digest"] != content_digest(
            {k: output[k] for k in ("value", "files")}
        ):
            raise StageError("cycle output journal changed")
        for item in output["files"]:
            checked_file(item)
        value = output["value"]
        ws = self.workspace(identifier)
        replay = self.replay(identifier)
        if stage == "admission":
            for item in value["admissions"]:
                store, _, intake = source_stores(self._source(item["source"], replay=replay))
                record = admission_for(store.load(item["round_id"]), item["record"]["author"])
                if record != item["record"]:
                    raise StageError("admitted contribution changed")
                from validator.intake import receipt_for_digest

                receipt = receipt_for_digest(
                    intake.read_receipts(),
                    round_id=item["round_id"],
                    digest=record["bundle_sha256"],
                    miner_id=record["author"],
                )
                if receipt is None:
                    raise StageError("admission receipt is missing")
                intake.verify(receipt)
        elif stage == "settlement":
            for item in value["settlements"]:
                _, settlement, _ = source_stores(self._source(item["source"], replay=replay))
                if content_digest(settlement.record(item["round_id"])) != item["hash"]:
                    raise StageError("committed cycle settlement changed")
                replay.read_round(item["source"], item["round_id"])
        elif stage == "experience":
            replay.experiences(value["ids"])
        elif stage == "replay":
            verify_corpus(ws, value)
            if _require(ws, "corpus") != value:
                raise StageError("cycle replay manifest changed")
        elif stage == "curriculum":
            issued = replay.authority.get(value["authority_id"], kind="curriculum")["payload"]
            if {k: v for k, v in value.items() if k != "authority_id"} != issued:
                raise StageError("cycle curriculum changed")
        elif stage == "prepare":
            training_recipe(ws, "sft")
            if read_record(ws.models / "sft/prepared.json") != value:
                raise StageError("cycle preparation changed")
        elif stage in {"train", "merge"}:
            if self._external_completion(value["job_id"]) != value:
                raise StageError("external training completion changed")
            training_recipe(ws, "sft")
        elif stage == "candidate":
            candidate = self.release.candidates().resolve(value["id"])["payload"]
            with self.store.connect() as db:
                request = evidence_object(
                    db.execute("SELECT request FROM cycles WHERE id=?", (identifier,)).fetchone()[0]
                )
            if any(candidate[key] != request["files"][key] for key in ("agent", "workload")):
                raise StageError("candidate differs from cycle's original agent/workload expectation")
            if (
                candidate["workspace"] != str(ws.root)
                or candidate["parent"]["path"] != request["baseline"]["model"]["merged"]
            ):
                raise StageError("candidate workspace/parent differs from cycle inputs")
        elif stage == "plan":
            self.release.plan(value["id"])
        elif stage == "evaluate":
            self.release.evaluation(value["id"])
        elif stage == "decide":
            record = self.store.get(value["id"], kind="release-decision")
            if record["payload"] != value["payload"]:
                raise StageError("cycle decision changed")
            if value["payload"]["result"] == "accepted":
                self.release.resolve_decision(value["id"])
        elif stage == "activate":
            with self.store.connect() as db:
                row = db.execute(
                    "SELECT generation FROM activation_operations WHERE id=?", (value["operation_id"],)
                ).fetchone()
            if row is None or row[0] != value["generation"]:
                raise StageError("cycle activation has no matching committed history")

    def status(self, identifier: str | None = None) -> dict[str, Any]:
        if identifier is None:
            with self.store.connect() as db:
                rows = [{"id": r[0], "name": r[1]} for r in db.execute("SELECT id,name FROM cycles ORDER BY rowid")]
            return {"origin": self.identity, "incumbent": self.release.status(), "cycles": rows}
        request = self.request(identifier)
        jobs = self._jobs(identifier)
        for stage in STAGES:
            job = jobs.get(stage)
            if job and job["status"] == "complete":
                self._validate(identifier, stage, job["output"])
        next_stage = next((s for s in STAGES if s not in jobs or jobs[s]["status"] != "complete"), None)
        refused = (
            "decide" in jobs
            and jobs["decide"]["status"] == "complete"
            and jobs["decide"]["output"]["value"]["payload"]["result"] == "refused"
        )
        state = (
            "refused"
            if refused
            else "complete"
            if next_stage is None
            else jobs.get(next_stage, {}).get("status", "ready")
        )
        return {
            "id": identifier,
            "name": request["name"],
            "origin": self.identity,
            "status": state,
            "next_stage": None if refused else next_stage,
            "workspace": str(self.workspace(identifier).root),
            "baseline": self.epoch_binding(request["baseline"]),
            "jobs": [jobs[s] for s in STAGES if s in jobs],
            "incumbent": self.release.status(),
            "fixture_only": self.identity["mode"] == "fixture",
            "trained": jobs.get("train", {}).get("output", {}).get("value", {}).get("trained", False)
            if jobs.get("train", {}).get("output")
            else False,
        }

    @campaign_writer
    def resume(
        self,
        identifier: str,
        *,
        through: str = "activate",
        execute_training: bool = False,
        allow_unsandboxed: bool = False,
        hook: Callable[[str], None] | None = None,
        github_transport: Any = None,
    ) -> dict[str, Any]:
        if through not in STAGES:
            raise StageError("unknown cycle target stage")
        if github_transport is not None and self.identity["mode"] != "fixture":
            raise StageError("injected GitHub transport requires fixture cycle authority")
        with locked(self.workspace(identifier).root / ".cycle.lock"):
            request = self.request(identifier)
            jobs = self._jobs(identifier)
            outputs: dict[str, Any] = {}
            for stage in STAGES[: STAGES.index(through) + 1]:
                expected = content_digest(
                    {"cycle": identifier, "stage": stage, "previous": {k: v["digest"] for k, v in outputs.items()}}
                )
                job_id = content_digest({"cycle": identifier, "stage": stage})
                job = jobs.get(stage)
                if job and (job["id"] != job_id or job["inputs"] != expected):
                    raise StageError("cycle job prerequisite/input identity changed")
                if job and job["status"] == "complete":
                    self._validate(identifier, stage, job["output"])
                    outputs[stage] = job["output"]
                else:
                    with self.store.connect() as db:
                        db.execute(
                            "INSERT OR IGNORE INTO cycle_jobs VALUES (?,?,?,?, 'ready',0,NULL,NULL)",
                            (identifier, stage, job_id, expected),
                        )
                        db.execute(
                            "UPDATE cycle_jobs SET status='running',attempts=attempts+1,error=NULL WHERE id=?",
                            (job_id,),
                        )
                    if hook:
                        hook(stage + ":before_producer")
                    try:
                        value = self._produce(
                            identifier,
                            request,
                            stage,
                            job_id,
                            outputs,
                            execute_training=execute_training,
                            allow_unsandboxed=allow_unsandboxed,
                            hook=hook,
                            github_transport=github_transport,
                        )
                        if hook:
                            hook(stage + ":after_producer")
                        output = {**value, "digest": content_digest(value)}
                        self._validate(identifier, stage, output)
                        with self.store.connect() as db:
                            db.execute(
                                "UPDATE cycle_jobs SET status='complete',output=?,error=NULL WHERE id=?",
                                (canonical(output), job_id),
                            )
                        outputs[stage] = output
                        if hook:
                            hook(stage + ":after_commit")
                    except Pending as exc:
                        with self.store.connect() as db:
                            db.execute("UPDATE cycle_jobs SET status='pending',error=? WHERE id=?", (str(exc), job_id))
                        break
                    except BaseException as exc:
                        with self.store.connect() as db:
                            db.execute(
                                "UPDATE cycle_jobs SET status='failed',error=? WHERE id=?", (type(exc).__name__, job_id)
                            )
                        raise
                if stage == "decide" and outputs[stage]["value"]["payload"]["result"] == "refused":
                    break
            return self.status(identifier)

    def _external_completion(self, job_id: str) -> dict[str, Any]:
        with self.store.connect() as db:
            row = db.execute(
                "SELECT request,status,completion FROM cycle_external_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None or row[1] != "complete" or row[2] is None:
            raise Pending("training/merge job has no supervisor completion; inspect its stable job ID")
        request, completion = evidence_object(row[0]), evidence_object(row[2])
        if (
            completion["request"] != request
            or completion["job_id"] != job_id
            or completion["origin"] != self.identity
            or type(completion["returncode"]) is not int
            or completion["returncode"] != 0
        ):
            raise StageError("external job completion binding changed")
        fixture = request["execution"] == "fixture"
        if completion["fixture_only"] is not fixture or completion["trained"] is not (not fixture):
            raise StageError("external job completion execution mode changed")
        if fixture and self.identity["mode"] != "fixture":
            raise StageError("production cannot consume fixture completion")
        for record in completion["files"]:
            checked_file(record)
        return completion

    def _external(
        self, identifier: str, request: dict[str, Any], stage: str, job_id: str, execute: bool, hook: Any
    ) -> dict[str, Any]:
        ws = self.workspace(identifier)
        recipe = training_recipe(ws, "sft")
        spec = {
            "cycle": identifier,
            "workspace": str(ws.root),
            "workspace_identity": ws.identity,
            "stage": stage,
            "recipe": file_identity(recipe),
            "preparation": file_identity(ws.models / "sft/prepared.json"),
            "execution": request["spec"]["training"]["execution"],
        }
        with self.store.connect() as db:
            row = db.execute("SELECT request,status FROM cycle_external_jobs WHERE id=?", (job_id,)).fetchone()
        if row:
            if canonical(spec) != row[0]:
                raise StageError("training job immutable inputs changed")
            if row[1] == "complete":
                return self._external_completion(job_id)
            if row[1] in {"running", "failed"}:
                raise Pending("original supervisor job is running or uncertain/failed; no duplicate launch")
        if not execute:
            raise Pending("prepared only; resume --execute-training to launch the recorded external job")
        with self.store.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO cycle_external_jobs VALUES (?,?,'submitted',NULL,NULL)",
                (job_id, canonical(spec)),
            )
        if hook:
            hook(stage + ":after_submit")
        log = self.root / "cycle-jobs" / (job_id[7:] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as stream:
            process = subprocess.Popen(
                [sys.executable, "-m", "admin.cycle_jobs", "--root", str(self.root), "--id", job_id],
                stdout=stream,
                stderr=stream,
                start_new_session=True,
            )
            if hook:
                hook(stage + ":after_launch")
            code = process.wait()
        if code:
            raise Pending("training supervisor did not complete; inspect the retained job log")
        return self._external_completion(job_id)

    def _produce(
        self,
        identifier: str,
        request: dict[str, Any],
        stage: str,
        job_id: str,
        outputs: dict[str, Any],
        *,
        execute_training: bool,
        allow_unsandboxed: bool,
        hook: Any,
        github_transport: Any,
    ) -> dict[str, Any]:
        spec, ws = request["spec"], self.workspace(identifier)
        replay = self.replay(identifier)
        values = {k: v["value"] for k, v in outputs.items()}
        if stage == "admission":
            records = []
            for item in spec["rounds"]:
                store, _, intake = source_stores(self._source(item["source"], replay=replay))
                for pr in item["prs"]:
                    win = store.load(item["round_id"])
                    if pr["author"] in win.admissions:
                        record = admission_for(win, pr["author"])
                    else:
                        github = self.configuration()["github"]
                        if github is None:
                            raise Pending(
                                "configure authenticated GitHub metadata or admit the pinned PR through validator.pr_admission"
                            )
                        source = GitHubSource(**github, transport=github_transport)
                        metadata = source.collect(pr["number"], round_id=item["round_id"], expected_head=pr["head"])
                        if metadata.author != pr["author"]:
                            raise StageError("PR author differs from frozen cycle request")
                        record = admit(metadata=metadata, round_id=item["round_id"], store=store, intake=intake)
                    if (
                        record["pr_number"] != pr["number"]
                        or record["head_sha"] != pr["head"]
                        or record["author"] != pr["author"]
                    ):
                        raise StageError("existing admission differs from frozen cycle commitment")
                    records.append({"source": item["source"], "round_id": item["round_id"], "record": record})
            return _result({"admissions": records})
        if stage == "settlement":
            records = []
            for item in spec["rounds"]:
                store, settlement, _ = source_stores(self._source(item["source"], replay=replay))
                from hermes.round import GRADED, SETTLED

                win = store.load(item["round_id"])
                if win.state not in {GRADED, SETTLED}:
                    raise Pending(
                        "round must be frozen and graded by validator.judge before settlement: " + item["round_id"]
                    )
                if set(win.admissions) != {p["author"] for p in item["prs"]}:
                    raise StageError("round admission set expanded after cycle input freeze")
                # A committed settlement remains recoverable even after another round is
                # activated in this source. Never rewind the source's active round on retry.
                try:
                    record = settlement.record(item["round_id"])
                except ValueError:
                    repository = next(
                        r["record"]["repository"]
                        for r in values["admission"]["admissions"]
                        if r["source"] == item["source"] and r["round_id"] == item["round_id"]
                    )
                    settlement.activate(store, item["round_id"], repository)
                    record = settlement.settle_round(
                        store,
                        scorecards=Path(item["scorecards"]),
                        episodes=Path(item["episodes"]),
                        round_id=item["round_id"],
                    )
                replay.read_round(item["source"], item["round_id"])
                records.append(
                    {
                        "source": item["source"],
                        "round_id": item["round_id"],
                        "id": record["settlement_id"],
                        "hash": content_digest(record),
                    }
                )
            return _result({"settlements": records})
        if stage == "experience":
            history = spec.get("history_ids", [])
            if not isinstance(history, list) or len(set(history)) != len(history):
                raise StageError("replay history requires distinct retained experience IDs")
            if history:
                replay.experiences(history)
            return _result(
                {
                    "ids": sorted(
                        set(history) | {replay.import_round(r["source"], r["round_id"])["id"] for r in spec["rounds"]}
                    )
                }
            )
        if stage == "replay":
            manifest = replay.freeze(ws, identifiers=values["experience"]["ids"])
            return _result(manifest, ws.corpus / "stage.json", *(ws.corpus / name for name in manifest["sha256"]))
        if stage == "curriculum":
            curriculum = build_curriculum(replay, config=spec["curriculum"], identifiers=values["experience"]["ids"])
            path = ws.root / "curriculum.json"
            write_record(path, curriculum)
            return _result(curriculum, path)
        if stage == "prepare":
            from admin.selfcheck import FixtureTokenizer

            parent = request["parent"]
            source = AuthorityStore(Path(parent["root"]), role="release")
            if source.identity != parent["identity"]:
                raise StageError("configured parent issuer changed")
            training = spec["training"]
            prepared = prepare_training(
                ws,
                stage="sft",
                profile=training["profile"],
                sequence_len=training["sequence_len"],
                max_steps=training["max_steps"],
                local_files_only=True,
                tokenizer=FixtureTokenizer() if training["execution"] == "fixture" else None,
                parent_approval=parent["id"],
                release_root=source.root,
            )
            return _result(
                prepared, ws.models / "sft/prepared.json", Path(prepared["data"]), Path(prepared["prepared_recipe"])
            )
        if stage in {"train", "merge"}:
            completion = self._external(identifier, request, stage, job_id, execute_training, hook)
            return _result(completion)
        if stage == "candidate":
            candidate = self.release.candidates().register(
                workspace=ws,
                merged_record=ws.models / "sft/merged.json",
                agent=checked_file(request["files"]["agent"]),
                workload=checked_file(request["files"]["workload"]),
                parent=Path(request["baseline"]["model"]["merged"]),
            )
            return _result({"id": candidate["id"]})
        if stage == "plan":
            current = self.release.status()
            if (
                current["candidate"] != request["baseline"]["candidate"]
                or current["generation"] != request["baseline"]["generation"]
            ):
                raise StageError("cycle baseline is stale before evaluation; no families spent")
            evaluation = spec["evaluation"]
            candidates = [
                self.release.candidates().resolve(i)["payload"]
                for i in (request["baseline"]["candidate"], values["candidate"]["id"])
            ]
            if "fixture" in evaluation:
                script = bound_record(request["files"]["serving_fixture"])
                serving = {}
                for m, candidate in enumerate(candidates):
                    artifact = {
                        "schema": "spark-serving-fixture-v1",
                        "origin": self.identity,
                        "model_id": candidate["model_id"],
                        "agents": {a["agent_id"]: script["cells"][f"Q{i}{m}"] for i, a in enumerate(candidates)},
                    }
                    path = ws.root / f"serving-{m}.json"
                    write_record(path, artifact)
                    serving[candidate["model_id"]] = {"fixture": file_identity(path)}
            else:
                serving = {c["model_id"]: evaluation["serving"][role] for role, c in zip(("old", "new"), candidates)}
            plan = self.release.freeze(
                old=request["baseline"]["candidate"],
                new=values["candidate"]["id"],
                schedule=evaluation["schedule"],
                budget=evaluation["budget"],
                sampling=evaluation["sampling"],
                serving=serving,
                **(
                    {"data_policy": checked_file(request["files"]["confirmation_policy"])}
                    if "confirmation_policy" in request["files"]
                    else {}
                ),
            )
            return _result({"id": plan["id"]})
        if stage == "evaluate":
            plan_id = values["plan"]["id"]
            prior = [
                r
                for r in self.release.experiments.records(kind="crossed-evaluation")
                if r["payload"]["plan_id"] == plan_id
            ]
            if len(prior) > 1:
                raise StageError("multiple evaluation completions for one cycle plan")
            if prior:
                self.release.evaluation(prior[0]["id"])
                return _result({"id": prior[0]["id"]})
            from admin.evaluation import execute_crossed

            evaluated = execute_crossed(self.release, plan_id, allow_unsandboxed=allow_unsandboxed)
            return _result({"id": evaluated["id"]})
        if stage == "decide":
            evaluation_id = values["evaluate"]["id"]
            prior = [
                r
                for r in self.store.records(kind="release-decision")
                if r["payload"].get("strict") is True and r["payload"]["evaluation"]["id"] == evaluation_id
            ]
            if len(prior) > 1:
                raise StageError("conflicting release decisions require operator review")
            decision = prior[0] if prior else self.release.decide(evaluation_id)
            return _result({"id": decision["id"], "payload": decision["payload"]})
        if stage == "activate":
            self.release.activate(
                values["decide"]["id"],
                operation_id=job_id,
                expected_candidate=request["baseline"]["candidate"],
                expected_generation=request["baseline"]["generation"],
                hook=(lambda name: hook("activate:" + name)) if hook else None,
            )
            with self.store.connect() as db:
                generation = db.execute(
                    "SELECT generation FROM activation_operations WHERE id=?", (job_id,)
                ).fetchone()[0]
            return _result(
                {
                    "operation_id": job_id,
                    "generation": generation,
                    "epoch_id": self.release.epoch_id(generation),
                    "approval": values["decide"]["id"],
                    "candidate": values["candidate"]["id"],
                }
            )
        raise StageError("unknown cycle stage")

    def dry_run(self, identifier: str) -> dict[str, Any]:
        self.status(identifier)
        return {**train(self.workspace(identifier), stage="sft", dry_run=True), "trained": False, "cycle": identifier}

    @campaign_writer
    def rollback(
        self, approval: str, *, operation_id: str, expected_generation: int, expected_candidate: str, hook: Any = None
    ) -> dict[str, Any]:
        return self.release.activate(
            approval,
            rollback=True,
            operation_id=operation_id,
            expected_generation=expected_generation,
            expected_candidate=expected_candidate,
            hook=hook,
        )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "demo":
        from admin.cycle_demo import main as demo_main

        return demo_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "start", "status", "resume", "dry-run", "active", "rollback"))
    parser.add_argument("--root", required=True, type=Path, help="same authoritative root as release")
    parser.add_argument("--mode", choices=("fixture", "production"))
    parser.add_argument("--namespace")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--id")
    parser.add_argument("--through", choices=STAGES, default="activate")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--allow-unsandboxed", action="store_true")
    parser.add_argument("--operation-id")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--expected-candidate")
    args = parser.parse_args(argv)
    try:
        controller = CycleController(args.root, mode=args.mode, namespace=args.namespace)
        if args.command == "init":
            config = read_record(args.config)
            if "release" in config:
                release = config["release"]
                controller.release.configure(
                    candidates=Path(release["candidates"]),
                    incumbent=release["incumbent"],
                    data_policy=Path(release["data_policy"]),
                    policy=release["policy"],
                )
            result = controller.configure(
                replay=Path(config["replay"]),
                bootstrap_parent=config.get("bootstrap_parent"),
                github=config.get("github"),
            )
        elif args.command == "start":
            result = controller.start(args.name, read_record(args.spec))
        elif args.command == "status":
            result = controller.status(args.id)
        elif args.command == "active":
            result = controller.active_pair()
        elif args.command == "resume":
            result = controller.resume(
                args.id,
                through=args.through,
                execute_training=args.execute_training,
                allow_unsandboxed=args.allow_unsandboxed,
            )
        elif args.command == "dry-run":
            result = controller.dry_run(args.id)
        else:
            if not args.operation_id or args.expected_generation is None or not args.expected_candidate:
                raise StageError("rollback requires operation-id and exact expected-generation/expected-candidate")
            result = controller.rollback(
                args.id,
                operation_id=args.operation_id,
                expected_generation=args.expected_generation,
                expected_candidate=args.expected_candidate,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 3 if result.get("status") == "refused" else 4 if result.get("status") == "pending" else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, sqlite3.Error) as exc:
        print(f"cycle: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
