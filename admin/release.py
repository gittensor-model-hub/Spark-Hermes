"""Durable strict release authority: frozen experiments, confirmation use and exact pairs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

from admin.artifacts import AuthorityStore, StageError, canonical, content_digest, read_record, same_domain
from admin.candidates import CandidateStore, bound_record, checked_file, file_identity
from admin.data_policy import DataPolicy
from admin.runtime_protocol import campaign_writer, historical_checked, historical_reading, verification_call
from hermes.cotraining import DEFAULT_POLICY, crossed_report, validate_policy


class ReleaseAuthority:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.store = AuthorityStore(root, role="release", mode=mode, namespace=namespace)
        self.identity = self.store.identity
        if not historical_reading():
            with self.store.connect() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS runtime_cutover (singleton INTEGER PRIMARY KEY CHECK(singleton=1), successor TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS release_config (singleton INTEGER PRIMARY KEY CHECK(singleton=1), payload BLOB NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS confirmation_use (family TEXT PRIMARY KEY, experiment TEXT NOT NULL)"
                )
                db.execute("CREATE TABLE IF NOT EXISTS experiment_runs (id TEXT PRIMARY KEY, status TEXT NOT NULL)")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS incumbent (singleton INTEGER PRIMARY KEY CHECK(singleton=1), candidate TEXT NOT NULL, approval TEXT, generation INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS activation_history (generation INTEGER PRIMARY KEY, candidate TEXT NOT NULL, approval TEXT, intent TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS activation_operations (id TEXT PRIMARY KEY, request BLOB NOT NULL, generation INTEGER NOT NULL)"
                )
        self.experiments = AuthorityStore(
            self.store.root / "experiments",
            role="crossed-evaluation",
            mode=self.identity["mode"],
            namespace=self.identity["namespace"],
        )

    @campaign_writer
    def configure(
        self, *, candidates: Path, incumbent: str, data_policy: Path, policy: dict[str, Any]
    ) -> dict[str, Any]:
        validate_policy(policy)
        source = CandidateStore(candidates)
        same_domain(self.identity, source.identity)
        from admin.pipeline import Workspace
        from admin.runtime_protocol import campaign_owners, check_owner, check_workspace

        for binding in campaign_owners(source.store):
            check_owner(binding)
        for record in source.store.records(kind="candidate"):
            check_workspace(Workspace(Path(record["payload"]["workspace"])))
        source.resolve(incumbent)
        DataPolicy(data_policy, identity=self.identity)
        config = {
            "candidates": {"root": str(source.store.root), "identity": source.identity},
            "initial_incumbent": incumbent,
            "campaign_protocol": "spark-runtime-transition-v1",
            "data_policy": file_identity(data_policy),
            "policy": policy,
            "policy_hash": content_digest(policy),
            "experiments": self.experiments.identity,
        }
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO release_config VALUES (1,?)", (canonical(config),))
            if db.execute("SELECT payload FROM release_config WHERE singleton=1").fetchone()[0] != canonical(config):
                raise StageError("release configuration is immutable; policy cannot be retuned")
            db.execute("INSERT OR IGNORE INTO incumbent VALUES (1,?,NULL,0)", (incumbent,))
            db.execute(
                "INSERT OR IGNORE INTO activation_history VALUES (0,?,NULL,'configured initial pair')", (incumbent,)
            )
        from admin.runtime_protocol import bind_campaign, preparation_store

        bind_campaign(source.store, self)
        for record in source.store.records(kind="candidate"):
            bind_campaign(preparation_store(Workspace(Path(record["payload"]["workspace"]))), self)
        return config

    def configuration(self) -> dict[str, Any]:
        from hermes.evidence_json import evidence_object

        with self.store.connect() as db:
            row = db.execute("SELECT payload FROM release_config WHERE singleton=1").fetchone()
        if row is None:
            raise StageError("release authority must be configured before use")
        config = evidence_object(row[0])
        checked_file(config["data_policy"])
        if (
            validate_policy(config["policy"]) != config["policy_hash"]
            or config["experiments"] != self.experiments.identity
        ):
            raise StageError("release configuration policy/issuer changed")
        source = CandidateStore(Path(config["candidates"]["root"]))
        if source.identity != config["candidates"]["identity"]:
            raise StageError("configured candidate issuer changed")
        same_domain(self.identity, source.identity)
        if config.get("campaign_protocol"):
            from admin.runtime_protocol import campaign_owners

            if {"root": str(self.store.root), "identity": self.identity} not in campaign_owners(source.store):
                raise StageError("configured candidate campaign binding changed")
        return config

    def candidates(self) -> CandidateStore:
        return CandidateStore(Path(self.configuration()["candidates"]["root"]))

    def status(self) -> dict[str, Any]:
        self.configuration()
        with self.store.connect() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT candidate,approval,generation FROM incumbent WHERE singleton=1").fetchone()
            history = [
                dict(zip(("generation", "candidate", "approval", "intent"), r))
                for r in db.execute("SELECT * FROM activation_history ORDER BY generation")
            ]
        return {
            "origin": self.identity,
            "candidate": row[0],
            "approval": row[1],
            "generation": row[2],
            "epoch_id": self.epoch_id(row[2]),
            "history": history,
        }

    def epoch_id(self, generation: int) -> str:
        return content_digest({"origin": self.identity, "generation": generation})

    @verification_call
    def active_pair(self) -> dict[str, Any]:
        """Resolve one coherent incumbent snapshot for next-epoch producers.

        The upstream revision is ancestry; model_id names the actual merged representation.
        Callers retain this returned expectation throughout execution, then compare the epoch
        again before admitting/crediting work. No global prompt or upstream pin is changed.
        """
        state = self.status()
        candidate = self.candidates().resolve(state["candidate"])["payload"]
        if state["approval"]:
            if self.store.kind(state["approval"]) == "runtime-parent":
                from admin.runtime_transition import baseline_parent

                baseline_parent(self, state["approval"])
            else:
                self.resolve_decision(state["approval"])
        return {
            **{k: state[k] for k in ("origin", "candidate", "approval", "generation", "epoch_id")},
            "model_id": candidate["model_id"],
            "agent_id": candidate["agent_id"],
            "model": candidate["model"],
            "agent": candidate["agent"],
            "agent_record": candidate["agent_record"],
            "representation": candidate["representation"],
            "runtime": candidate["runtime"],
            **({"authority_kind": "verified-historical-baseline"} if candidate.get("baseline_only") else {}),
        }

    @campaign_writer
    def freeze(
        self,
        *,
        old: str,
        new: str,
        schedule: list[dict[str, Any]],
        budget: dict[str, Any],
        sampling: dict[str, Any],
        serving: dict[str, Any],
        data_policy: Path | None = None,
    ) -> dict[str, Any]:
        from dataclasses import asdict

        from hermesbench.tasks import Task

        config = self.configuration()
        source = self.candidates()
        a, b = source.resolve(old)["payload"], source.resolve(new)["payload"]
        if old == new or a["agent_id"] == b["agent_id"] or a["model_id"] == b["model_id"]:
            raise StageError("crossed evaluation requires distinct old/new agent and model artifacts")
        if a["runtime"] != b["runtime"]:
            raise StageError("candidate evaluator/environment differ")
        approval = b["parent"]["approval"]
        if (
            not approval
            or approval["decision"]["candidate"] != a["model"]
            or approval["decision"]["agent"] != a["agent_id"]
        ):
            raise StageError("new candidate preparation does not bind the approved incumbent pair")
        if b["parent"]["files"] != a["model"]["files"]:
            raise StageError("new candidate does not descend from the incumbent model artifact")
        if set(budget) != {"max_steps", "max_tokens", "tool_timeout_s"} or any(
            type(v) is not int or v <= 0 for v in budget.values()
        ):
            raise StageError("missing/invalid fixed execution budget")
        import math

        if (
            set(sampling) != {"temperature", "top_p"}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in sampling.values())
            or sampling["temperature"] < 0
            or not 0 < sampling["top_p"] <= 1
        ):
            raise StageError("invalid fixed sampling")
        if set(serving) != {a["model_id"], b["model_id"]}:
            raise StageError("serving must bind both exact model representations")
        if self.identity["mode"] == "production" and any("fixture" in s for s in serving.values()):
            raise StageError("fixture serving cannot enter production")
        remote = [s for s in serving.values() if "fixture" not in s]
        if remote and len(remote) != 2:
            raise StageError("mixed fixture and remote serving is not comparable")
        if len(remote) == 2 and any(
            remote[0].get(k) != remote[1].get(k) for k in ("engine", "precision", "device", "environment")
        ):
            raise StageError("remote serving environments are not comparable")
        workload = bound_record(b["workload"])
        if (
            set(workload) != {"schema", "tasks"}
            or workload["schema"] != "spark-crossed-workload-v1"
            or not workload["tasks"]
        ):
            raise StageError("missing exact crossed workload")
        catalog = file_identity(data_policy) if data_policy is not None else config["data_policy"]
        data = self._catalog(catalog)
        from admin.pipeline import Workspace, _require
        from hermesbench.sink import read_episodes

        for candidate in (a, b):
            ws = Workspace(Path(candidate["workspace"]))
            corpus = _require(ws, "corpus")
            policy_paths = set()
            if "configuration" in corpus:
                policy_paths.add(corpus["configuration"]["policy"])
            for filename in ("sft.jsonl", "preference.jsonl"):
                for row in read_episodes(ws.corpus / filename):
                    for provenance in row.get("provenance", []):
                        policy_paths.add(provenance["policy"]["path"])
            for path in policy_paths:
                prior = DataPolicy(Path(path), identity=self.identity)
                self._preserve_catalog(data, prior)
        full_schedule, tasks, families = [], [], set()
        for item in workload["tasks"]:
            if not isinstance(item.get("task", {}).get("task_id"), str) or not item["task"]["task_id"]:
                raise StageError("workload task IDs must be explicit nonempty strings")
            task = Task.from_record(item["task"])
            if not task.hidden_verify or not task.declares_hidden_tests:
                raise StageError("strict confirmation requires actual withheld verification")
            version = content_digest(asdict(task))
            member = data.membership(
                task_id=task.task_id, repository=item["repository"], version=version, purpose="release"
            )
            family = member["canonical_family"]
            families.add(family)
            tasks.append(
                {"task": asdict(task), "repository": item["repository"], "version": version, "family_id": family}
            )
            for attempt in schedule:
                if set(attempt) != {"attempt_id", "seed"}:
                    raise StageError("missing attempt/seed schedule")
                full_schedule.append({"task_id": task.task_id, "family_id": family, **attempt})
        runtime = a["runtime"]
        plan = {
            "schema": "spark-crossed-plan-v1",
            "origin": self.experiments.identity,
            "release": {"root": str(self.store.root), "identity": self.identity},
            "configuration_hash": content_digest(config),
            "data_policy": catalog,
            "old": old,
            "new": new,
            "baseline_generation": self.status()["generation"],
            "policy": config["policy"],
            "policy_hash": config["policy_hash"],
            "factors": {"agents": [a["agent_id"], b["agent_id"]], "models": [a["model_id"], b["model_id"]]},
            "workload": b["workload"],
            "evaluator": runtime["evaluator"],
            "environment": runtime["environment"],
            "tasks": tasks,
            "schedule": full_schedule,
            "budget": budget,
            "sampling": sampling,
            "serving": serving,
        }
        # Validate schedule before reserving anything, using the same strict structural consumer.
        from hermes.cotraining import CELLS

        dummy = {**plan, "cells": {}}
        for cell in CELLS:
            dummy["cells"][cell] = {
                **{k: plan[k] for k in ("evaluator", "environment", "workload", "budget", "sampling", "policy_hash")},
                "agent": plan["factors"]["agents"][int(cell[1])],
                "model": plan["factors"]["models"][int(cell[2])],
                "rows": [{**r, "success": False, "tokens": 1, "latency": 1} for r in full_schedule],
            }
        crossed_report(dummy)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current, generation = db.execute("SELECT candidate,generation FROM incumbent WHERE singleton=1").fetchone()
            if current != old or generation != plan["baseline_generation"]:
                raise StageError("experiment baseline is not the current incumbent")
            # Serialize catalog extension with confirmation reservations. Even refused or
            # abandoned evaluations retain all knowledge from their successfully frozen plan.
            prior_catalogs = {content_digest(config["data_policy"]): config["data_policy"]}
            inherited = self.inherited_history()
            if inherited:
                for binding in inherited["catalogs"]:
                    prior_catalogs[content_digest(binding)] = binding
            inherited_ids = {r[1] for r in inherited["reservations"]} if inherited else set()
            for (identifier,) in db.execute("SELECT DISTINCT experiment FROM confirmation_use ORDER BY experiment"):
                if identifier in inherited_ids:
                    continue
                prior_plan = self.experiments.get(identifier, kind="crossed-plan")["payload"]
                known = [prior_plan.get("data_policy", config["data_policy"])]
                known.extend(prior_plan.get("prior_data_policies", []))
                for binding in known:
                    prior_catalogs[content_digest(binding)] = binding
            for binding in prior_catalogs.values():
                self._preserve_catalog(data, self._catalog(binding))
            # Retain exact dependencies for subsequent reads; a path cannot silently acquire
            # new memberships after this experiment has declared its confirmation workload.
            plan["prior_data_policies"] = [
                prior_catalogs[key] for key in sorted(prior_catalogs) if prior_catalogs[key] != catalog
            ]
            self._catalog(catalog)
            issued = self.experiments.put("crossed-plan", plan)
            for family in sorted(families):
                existing = db.execute("SELECT experiment FROM confirmation_use WHERE family=?", (family,)).fetchone()
                if existing and existing[0] != issued["id"]:
                    raise StageError("confirmation family was already used; a renamed workload is not fresh")
                db.execute("INSERT OR IGNORE INTO confirmation_use VALUES (?,?)", (family, issued["id"]))
            # This original commit record distinguishes a successfully frozen but
            # interrupted experiment from an orphan plan after a refused freeze.
            reservation = self.reservation_record(issued["id"], plan)
            db.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?)",
                (reservation["id"], canonical({k: v for k, v in reservation.items() if k != "id"})),
            )

        return issued

    def reservation_record(self, identifier, plan):
        return self.store.envelope(
            "confirmation-reservation",
            {
                "schema": "spark-confirmation-reservation-v1",
                "origin": self.identity,
                "plan": identifier,
                "families": sorted({r["family_id"] for r in plan["schedule"]}),
            },
        )

    def inherited_history(self):
        config = self.configuration()
        if "runtime_transition" not in config:
            return None
        from admin.runtime_transition import resolve_bridge

        return resolve_bridge(self, config["runtime_transition"])["history"]

    def _catalog(self, binding: dict[str, Any]) -> DataPolicy:
        data = DataPolicy(checked_file(binding), identity=self.identity)
        if data.sha256 != binding["sha256"]:
            raise StageError("confirmation catalog changed at final byte capture")
        return data

    @staticmethod
    def _preserve_catalog(data: DataPolicy, prior: DataPolicy) -> None:
        for alias, target in prior.aliases.items():
            if data.aliases.get(alias) != target or data.family(alias) != prior.family(alias):
                raise StageError("confirmation catalog dropped/changed a known training-family relationship")
        for member in prior.members:
            matching = [
                m
                for m in data.members
                if all(m[k] == member[k] for k in ("task_id", "repository", "version", "family_id"))
            ]
            if (
                len(matching) != 1
                or not set(member["exposure"]) <= set(matching[0]["exposure"])
                or matching[0]["partition"] != member["partition"]
            ):
                raise StageError("confirmation catalog dropped prior membership/exposure knowledge")

    @historical_checked
    def plan(self, identifier: str) -> dict[str, Any]:
        plan = self.experiments.get(identifier, kind="crossed-plan")["payload"]
        config = self.configuration()
        if (
            plan["origin"] != self.experiments.identity
            or plan["release"] != {"root": str(self.store.root), "identity": self.identity}
            or plan["configuration_hash"] != content_digest(config)
        ):
            raise StageError("experiment origin/configuration changed")
        data = self._catalog(plan.get("data_policy", config["data_policy"]))
        for binding in plan.get("prior_data_policies", [config["data_policy"]]):
            self._preserve_catalog(data, self._catalog(binding))
        source = self.candidates()
        for key in ("old", "new"):
            source.resolve(plan[key])
        if config.get("campaign_protocol"):
            reservation = self.reservation_record(identifier, plan)
            self.store.get(reservation["id"], kind="confirmation-reservation")
        with self.store.connect() as db:
            for family in {r["family_id"] for r in plan["schedule"]}:
                row = db.execute("SELECT experiment FROM confirmation_use WHERE family=?", (family,)).fetchone()
                if row is None or row[0] != identifier:
                    raise StageError("experiment has no original confirmation reservation")
        return plan

    @historical_checked
    def evaluation(self, identifier: str) -> dict[str, Any]:
        from admin.evaluation import verify_crossed_evaluation

        return verify_crossed_evaluation(self, identifier)

    @campaign_writer
    def decide(self, evaluation: str) -> dict[str, Any]:
        evidence = self.evaluation(evaluation)
        plan = self.plan(evidence["plan_id"])
        report = crossed_report(evidence["matrix"], require_execution=True)
        candidate = self.candidates().resolve(plan["new"])["payload"]
        reasons = list(report["reasons"])
        current = self.status()
        if current["candidate"] != plan["old"] or current["generation"] != plan["baseline_generation"]:
            reasons.append("stale evaluation baseline; incumbent changed")
        payload = {
            "schema": "spark-release-decision-v1",
            "strict": True,
            "origin": self.identity,
            "result": "refused" if reasons else "accepted",
            "reasons": reasons,
            "candidate": candidate["model"],
            "candidate_id": plan["new"],
            "old_id": plan["old"],
            "baseline_generation": plan["baseline_generation"],
            "agent": candidate["agent_id"],
            "policy": plan["policy"],
            "policy_hash": plan["policy_hash"],
            "evaluation": {
                "id": evaluation,
                "path": evidence["artifact"]["path"],
                "sha256": evidence["artifact"]["sha256"],
                "origin": self.experiments.identity,
                "cells": list(evidence["matrix"]["cells"]),
            },
            "corpus": candidate["corpus"],
            "fixture_only": self.identity["mode"] == "fixture",
            "authorizes_production_promotion": self.identity["mode"] == "production" and not reasons,
            "report": report,
        }
        return self.store.put("release-decision", payload)

    @historical_checked
    def resolve_decision(self, identifier: str) -> dict[str, Any]:
        decision = self.store.get(identifier, kind="release-decision")["payload"]
        if (
            decision.get("strict") is not True
            or decision.get("result") != "accepted"
            or decision.get("origin") != self.identity
        ):
            raise StageError("activation requires an accepted strict release decision")
        if decision["fixture_only"] != (self.identity["mode"] == "fixture"):
            raise StageError("release fixture trust domain mismatch")
        evidence = self.evaluation(decision["evaluation"]["id"])
        plan = self.plan(evidence["plan_id"])
        candidate = self.candidates().resolve(plan["new"])["payload"]
        report = crossed_report(evidence["matrix"], require_execution=True)
        if (
            not report["eligible"]
            or report != decision["report"]
            or decision["candidate_id"] != plan["new"]
            or decision["old_id"] != plan["old"]
            or decision["baseline_generation"] != plan["baseline_generation"]
            or decision["candidate"] != candidate["model"]
            or decision["agent"] != candidate["agent_id"]
            or decision["corpus"] != candidate["corpus"]
            or decision["policy_hash"] != plan["policy_hash"]
        ):
            raise StageError("release decision provenance/measurement changed")
        return decision

    @campaign_writer
    def activate(
        self,
        identifier: str,
        *,
        rollback: bool = False,
        operation_id: str | None = None,
        expected_generation: int | None = None,
        expected_candidate: str | None = None,
        hook: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        decision = self.resolve_decision(identifier)
        if expected_generation is not None and (type(expected_generation) is not int or expected_generation < 0):
            raise StageError("expected generation must be a nonnegative integer")
        if (expected_generation is None) != (expected_candidate is None):
            raise StageError("activation requires both expected generation and candidate")
        if operation_id is not None and (not isinstance(operation_id, str) or not operation_id.strip()):
            raise StageError("activation operation ID must be nonempty")
        # Legacy rollback callers get one durable operation per target, so even a delayed
        # replay after another activation cannot overwrite a newer pair. Controllers supply
        # explicit IDs and expected snapshots for distinct operator rollback intents.
        explicit_operation = operation_id is not None
        operation_id = operation_id or ("rollback:" if rollback else "activate:") + identifier
        request = canonical(
            {
                "approval": identifier,
                "rollback": rollback,
                "expected_generation": expected_generation,
                "expected_candidate": expected_candidate,
            }
        )
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT request,generation FROM activation_operations WHERE id=?", (operation_id,)
            ).fetchone()
            if prior:
                if prior[0] != request:
                    raise StageError("activation operation ID already binds a different request")
                if not rollback and not explicit_operation:
                    current = db.execute("SELECT candidate,approval FROM incumbent WHERE singleton=1").fetchone()
                    if current != (decision["candidate_id"], identifier):
                        raise StageError("stale approval replay; incumbent unchanged")
                return self.status()
            current, approval, generation = db.execute(
                "SELECT candidate,approval,generation FROM incumbent WHERE singleton=1"
            ).fetchone()
            if expected_generation is not None and (generation != expected_generation or current != expected_candidate):
                raise StageError("stale expected incumbent/epoch; incumbent unchanged")
            if not rollback and current == decision["candidate_id"] and approval == identifier:
                db.execute("INSERT INTO activation_operations VALUES (?,?,?)", (operation_id, request, generation))
                return self.status()
            if rollback:
                if (
                    db.execute(
                        "SELECT 1 FROM activation_history WHERE candidate=? AND approval=?",
                        (decision["candidate_id"], identifier),
                    ).fetchone()
                    is None
                ):
                    raise StageError("rollback requires an already activated accepted pair")
            elif current != decision["old_id"] or generation != decision["baseline_generation"]:
                raise StageError("stale approval baseline; incumbent unchanged")
            generation += 1
            intent = "rollback to previously accepted pair" if rollback else "activate measured accepted pair"
            db.execute(
                "UPDATE incumbent SET candidate=?,approval=?,generation=? WHERE singleton=1",
                (decision["candidate_id"], identifier, generation),
            )
            if hook:
                hook("after_pointer")
            db.execute(
                "INSERT INTO activation_history VALUES (?,?,?,?)",
                (generation, decision["candidate_id"], identifier, intent),
            )
            db.execute("INSERT INTO activation_operations VALUES (?,?,?)", (operation_id, request, generation))
            if hook:
                hook("before_commit")
        if hook:
            hook("after_commit")
        return self.status()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "decide", "activate", "rollback", "status", "policy"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mode", choices=("fixture", "production"))
    parser.add_argument("--namespace")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--id")
    args = parser.parse_args(argv)
    try:
        if args.command == "policy":
            result = DEFAULT_POLICY
        else:
            if args.root is None:
                raise StageError("release requires --root")
            authority = ReleaseAuthority(args.root, mode=args.mode, namespace=args.namespace)
            if args.command == "init":
                config = read_record(args.config)
                result = authority.configure(
                    candidates=Path(config["candidates"]),
                    incumbent=config["incumbent"],
                    data_policy=Path(config["data_policy"]),
                    policy=config["policy"],
                )
            elif args.command == "status":
                result = authority.status()
            elif args.command == "decide":
                result = authority.decide(args.id)
            else:
                result = authority.activate(args.id, rollback=args.command == "rollback")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 3 if isinstance(result, dict) and result.get("payload", {}).get("result") == "refused" else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
