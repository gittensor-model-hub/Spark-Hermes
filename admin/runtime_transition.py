"""Deliberate, retained-runtime single-successor campaign cutover."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from admin.artifacts import StageError, canonical, content_digest, file_digest, read_record, same_domain
from admin.candidates import CandidateStore
from admin.release import ReleaseAuthority
from admin.runtime_protocol import (
    PROTOCOL,
    campaign_lock,
    campaign_owners,
    historical_checked,
    historical_scope,
    resolution_runtime,
    writable,
)
from hermes.harness import crossed_runtime_identity


def _binding(store):
    return {"root": str(store.root), "identity": store.identity}


def _source(binding):
    root = Path(binding["root"])
    if not (root / "authority.sqlite3").is_file():
        raise StageError("missing original authority database; retain original roots")
    source = ReleaseAuthority(root)
    if _binding(source.store) != binding:
        raise StageError("original authority issuer/path changed")
    return source


def runtime_description():
    import importlib.util

    roots = {}
    for name in ("admin", "eval", "hermes", "hermesbench", "miner", "proof", "teacher", "validator"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise StageError("missing supported runtime package: " + name)
        roots[name] = str(Path(spec.origin).parent.resolve())
    return {
        "protocol": PROTOCOL,
        "runtime": crossed_runtime_identity(),
        "python": sys.executable,
        "python_sha256": file_digest(Path(sys.executable)),
        "prefix": sys.prefix,
        "packages": roots,
    }


def capture(authority):
    with campaign_lock(authority.store.root):
        writable(authority)
        return authority.store.put("runtime-retention", {"source": _binding(authority.store), **runtime_description()})


def _retention(source, identifier):
    payload = source.store.get(identifier, kind="runtime-retention")["payload"]
    if payload["protocol"] != PROTOCOL or payload["source"] != _binding(source.store):
        raise StageError("unsupported original runtime cutover protocol; original installation must support v1")
    if file_digest(Path(payload["python"])) != payload["python_sha256"]:
        raise StageError("retained original interpreter changed")
    files = {}
    for package, root in payload["packages"].items():
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".jinja", ".txt"}:
                files[package + "/" + str(path.relative_to(root))] = file_digest(path)
    if files != payload["runtime"]["files"]:
        raise StageError("retained original runtime files missing or changed; restore original installation")
    return payload


def _invoke(retained, *args):
    # Explicit interpreter, isolated import path and no caller PYTHONPATH. The original
    # installation is executable authority, not a JSON report supplied by the caller.
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}}
    try:
        result = subprocess.run(
            [retained["python"], "-I", "-B", "-m", "admin.runtime_transition", *args],
            cwd=retained["prefix"],
            env=env,
            capture_output=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageError("retained runtime could not execute supported verification") from exc
    if result.returncode:
        raise StageError(
            "retained runtime verification refused: " + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    from hermes.evidence_json import evidence_object

    response = evidence_object(result.stdout)
    if response["installation"] != {k: retained[k] for k in runtime_description()}:
        raise StageError("retained runtime execution identity changed")
    return response


def _table_snapshot(store, *, release=False):
    from hermes.evidence_json import evidence_object

    tables = {
        "metadata",
        "records",
        "release_config",
        "confirmation_use",
        "experiment_runs",
        "incumbent",
        "activation_history",
        "activation_operations",
        "cycle_configuration",
        "cycles",
        "cycle_jobs",
        "cycle_external_jobs",
        "cycle_workspaces",
        "runtime_cutover",
    }
    result = {}
    with store.connect() as db:
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            if name not in tables:
                raise StageError("unsupported original authority table: " + name)
            if name == "runtime_cutover":
                continue
            rows = []
            for row in db.execute('SELECT * FROM "' + name + '"'):
                if name == "records" and evidence_object(row[1])["kind"].startswith("runtime-"):
                    continue
                rows.append([{"bytes": v.hex()} if isinstance(v, bytes) else v for v in row])
            result[name] = sorted(rows, key=canonical)
    return result


def current_history(source, runtime):
    """Read-only CURRENT eligibility over original authority, including strict ancestry.

    No runtime is claimed to have executed here. Exact declared historical runtime
    identities are checked inside this bounded verifier; normal callers stay strict.
    """
    with historical_scope(runtime, source.identity) as graph:
        config = source.configuration()
        state = source.status()
        if not state["approval"]:
            raise StageError("runtime transition requires an originally activated strict approval")
        source.resolve_decision(state["approval"])
        candidates = source.candidates()
        with source.store.connect() as db:
            if (
                db.execute("SELECT 1 FROM sqlite_master WHERE name='cycle_external_jobs'").fetchone()
                and db.execute("SELECT 1 FROM cycle_external_jobs WHERE status IN ('submitted','running')").fetchone()
            ):
                raise StageError("source has an outstanding supervised job; reconcile it before cutover")
        from admin.pipeline import Workspace
        from admin.runtime_protocol import preparation_store

        preparations = {}
        initial = candidates.resolve(config["initial_incumbent"])["payload"]
        for record in candidates.store.records(kind="candidate"):
            value = candidates.resolve(record["id"])["payload"]
            workspace = Workspace(Path(value["workspace"]))
            preparation = preparation_store(workspace)
            if _binding(source.store) not in campaign_owners(preparation):
                raise StageError("source lost original workspace/campaign ownership")
            preparations[str(preparation.root)] = _table_snapshot(preparation)
            approval = value["parent"]["approval"]
            if (
                approval
                and approval["decision"].get("strict") is not True
                and approval["decision"].get("schema") != "spark-runtime-parent-v1"
            ):
                decision = approval["decision"]
                if (
                    source.identity["mode"] != "fixture"
                    or decision.get("candidate") != initial["model"]
                    or decision.get("agent") != initial["agent_id"]
                    or decision.get("policy", {}).get("version") != "cpu-parent-handoff-v1"
                ):
                    raise StageError("non-strict historical approval is not the exact initial fixture bootstrap")
        catalogs = {content_digest(config["data_policy"]): config["data_policy"]}
        with source.store.connect() as db:
            reservations = [
                list(r) for r in db.execute("SELECT family,experiment FROM confirmation_use ORDER BY family")
            ]
        plans = set()
        inherited = source.inherited_history()
        inherited_reservations = {tuple(r) for r in inherited["reservations"]} if inherited else set()
        if inherited:
            if not inherited_reservations <= {tuple(r) for r in reservations}:
                raise StageError("source dropped inherited confirmation reservations")
            for binding in inherited["catalogs"]:
                source._catalog(binding)
                catalogs[content_digest(binding)] = binding
        for family, identifier in reservations:
            if (family, identifier) in inherited_reservations:
                continue
            plan = source.plan(identifier)
            if family not in {r["family_id"] for r in plan["schedule"]}:
                raise StageError("reservation does not belong to its original plan")
            plans.add(identifier)
            for binding in [plan.get("data_policy", config["data_policy"]), *plan.get("prior_data_policies", [])]:
                source._catalog(binding)
                catalogs[content_digest(binding)] = binding
        for record in source.store.records(kind="confirmation-reservation"):
            plan = source.plan(record["payload"]["plan"])
            if source.reservation_record(record["payload"]["plan"], plan) != record:
                raise StageError("original confirmation reservation commit changed")
        for record in source.experiments.records(kind="crossed-evaluation"):
            source.evaluation(record["id"])
        for record in source.store.records(kind="release-decision"):
            if record["payload"].get("strict") is True and record["payload"].get("result") == "accepted":
                source.resolve_decision(record["id"])
        pair = source.active_pair()
        if graph.get("strict_roots", set()) - {str(source.store.root)}:
            raise StageError(
                "unsupported unbridged cross-authority strict ancestry; retain its full campaign and use a supported bridge"
            )
        return {
            "source": _binding(source.store),
            "state": state,
            "pair": pair,
            "configuration": config,
            "reservations": reservations,
            "catalogs": [catalogs[k] for k in sorted(catalogs)],
            "snapshot": {
                "release": _table_snapshot(source.store, release=True),
                "preparations": preparations,
                "candidates": _table_snapshot(candidates.store),
                "experiments": _table_snapshot(source.experiments),
            },
        }


def inspect(source, retention_id):
    retained = _retention(source, retention_id)
    if {k: retained[k] for k in runtime_description()} != runtime_description():
        raise StageError("inspect must run in the retained matching installation")
    return {"installation": runtime_description(), "history": current_history(source, retained["runtime"])}


def verified_history(source, retention_id):
    retained = _retention(source, retention_id)
    original = _invoke(retained, "inspect", "--root", str(source.store.root), "--id", retention_id)["history"]
    current = current_history(source, retained["runtime"])
    if canonical(original) != canonical(current):
        raise StageError("current historical eligibility disagrees with original issuance or source advanced")
    return current


def propose(source, retention_id, target):
    with campaign_lock(source.store.root), campaign_lock(target.store.root):
        writable(source)
        writable(target)
        same_domain(source.identity, target.identity)
        if source.store.root == target.store.root:
            raise StageError("transition requires a separate target authority")
        with target.store.connect() as db:
            if db.execute("SELECT 1 FROM release_config").fetchone():
                raise StageError("target must be an unconfigured authority")
        retained_source = _retention(source, retention_id)
        if set(retained_source["packages"]) != {
            "admin",
            "eval",
            "hermes",
            "hermesbench",
            "miner",
            "proof",
            "teacher",
            "validator",
        }:
            raise StageError(
                "retained runtime lacks the complete supported owned-package inventory; read-only eligibility remains available"
            )
        if source.configuration().get("campaign_protocol") != PROTOCOL:
            raise StageError(
                "original campaign lacks the supported cutover protocol; retain it for read-only eligibility"
            )
        if campaign_owners(source.candidates().store) != [_binding(source.store)]:
            raise StageError("supported cutover requires an exclusively owned candidate store")
        history = verified_history(source, retention_id)
        from admin.competition_pair import base_profile

        model = history["pair"]["model"]
        profile = base_profile(model["profile"])
        if profile["repository"] != model["base_model"] or profile["revision"] != model["revision"]:
            raise StageError("target runtime does not support the original model profile/base/revision")
        if history["pair"]["runtime"] == crossed_runtime_identity():
            raise StageError("runtime is unchanged; continue ordinary cycles without transition")
        retained_target = capture(target)
        candidates = CandidateStore(
            target.store.root / "candidates", mode=target.identity["mode"], namespace=target.identity["namespace"]
        )
        if candidates.store.records(kind="candidate"):
            raise StageError("target candidate authority must be empty")
        return target.store.put(
            "runtime-proposal",
            {
                "protocol": PROTOCOL,
                "source": _binding(source.store),
                "retention": retention_id,
                "target": _binding(target.store),
                "target_retention": retained_target["id"],
                "target_candidates": _binding(candidates.store),
                "target_snapshot": {
                    "release": _table_snapshot(target.store),
                    "candidates": _table_snapshot(candidates.store),
                    "experiments": _table_snapshot(target.experiments),
                },
                "history": history,
            },
        )


def _proposal(target, identifier, *, historical=False):
    from validator.persistence import state_identity

    if state_identity(target.store.root) != target.identity:
        raise StageError("target authority issuer changed at commit/verification")
    proposal = target.store.get(identifier, kind="runtime-proposal")["payload"]
    if proposal["protocol"] != PROTOCOL or proposal["target"] != _binding(target.store):
        raise StageError("target proposal issuer/path changed")
    retained = _retention(target, proposal["target_retention"])
    if (
        retained["runtime"] != resolution_runtime()
        if historical
        else {k: retained[k] for k in runtime_description()} != runtime_description()
    ):
        raise StageError("target runtime changed since verification; proposal cannot be committed")
    candidates = CandidateStore(Path(proposal["target_candidates"]["root"]))
    if _binding(candidates.store) != proposal["target_candidates"]:
        raise StageError("target candidate issuer substituted")
    for store in (target.store, candidates.store):
        with store.connect() as db:
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise StageError("supported target publication requires SQLite DELETE journals before cutover")

    return proposal


def commit(target, identifier, *, hook=None):
    proposal = _proposal(target, identifier)
    source = _source(proposal["source"])
    with campaign_lock(source.store.root), campaign_lock(target.store.root):
        proposal = _proposal(target, identifier)
        with source.store.connect() as db:
            row = db.execute("SELECT successor FROM runtime_cutover WHERE singleton=1").fetchone()
        expected = content_digest({"target": proposal["target"], "proposal": identifier})
        if row and row[0] != expected:
            raise StageError("source already committed a different single successor")
        history = verified_history(source, proposal["retention"])
        if canonical(history) != canonical(proposal["history"]):
            raise StageError("source incumbent/epoch/reservation snapshot advanced; prepare a fresh proposal")
        if hook:
            hook("verified")
        # Locks serialize all supported source writers. Repeat verification after the
        # hook as an explicit testable final boundary against nonparticipating changes.
        if canonical(current_history(source, _retention(source, proposal["retention"])["runtime"])) != canonical(
            history
        ):
            raise StageError("source changed at cutover commit")
        _proposal(target, identifier)
        with target.store.connect() as db:
            config = db.execute("SELECT payload FROM release_config WHERE singleton=1").fetchone()
        if not config:
            actual = {
                "release": _table_snapshot(target.store),
                "candidates": _table_snapshot(CandidateStore(Path(proposal["target_candidates"]["root"])).store),
                "experiments": _table_snapshot(target.experiments),
            }
            if actual != proposal["target_snapshot"]:
                raise StageError("target authority changed since proposal")
        if config:
            from hermes.evidence_json import evidence_object

            if evidence_object(config[0]).get("runtime_transition") != identifier:
                raise StageError("target acquired another configuration")
        with source.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            successor_id, successor_bytes = _issued(
                source.store,
                "runtime-successor",
                {
                    "protocol": PROTOCOL,
                    "target": proposal["target"],
                    "proposal": identifier,
                    "retention": proposal["retention"],
                    "history_hash": content_digest(history),
                },
            )
            db.execute("INSERT OR IGNORE INTO records VALUES (?,?)", (successor_id, successor_bytes))
            db.execute("INSERT OR IGNORE INTO runtime_cutover VALUES (1,?)", (expected,))
            if db.execute("SELECT successor FROM runtime_cutover").fetchone()[0] != expected:
                raise StageError("competing source transition")
        if hook:
            hook("source_committed")
        _proposal(target, identifier)
        _publish(target, identifier, proposal, hook=hook)
        return target.status()


def _issued(store, kind, payload):
    record = {
        "schema": "spark-authority-v1",
        "kind": kind,
        "issuer": store.identity,
        "role": store.role,
        "payload": payload,
    }
    return content_digest(record), canonical(record)


def _publish(target, identifier, proposal, *, hook=None):
    candidates = CandidateStore(Path(proposal["target_candidates"]["root"]))
    bridge = {"target": _binding(target.store), "proposal": identifier}
    baseline_id, baseline_bytes = _issued(candidates.store, "runtime-baseline", bridge)
    parent_id, parent_bytes = _issued(target.store, "runtime-parent", bridge)
    history = proposal["history"]
    config = {
        **history["configuration"],
        "candidates": _binding(candidates.store),
        "initial_incumbent": baseline_id,
        "experiments": target.experiments.identity,
        "runtime_transition": identifier,
    }
    with target.store.connect() as db:
        db.execute("ATTACH DATABASE ? AS candidate_publication", (str(candidates.store.path),))
        if any(
            db.execute(f"PRAGMA {name}.journal_mode").fetchone()[0] != "delete"
            for name in ("main", "candidate_publication")
        ):
            raise StageError("supported atomic publication requires SQLite DELETE journals for both authorities")
        db.execute("PRAGMA candidate_publication.synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT payload FROM release_config WHERE singleton=1").fetchone()
        if existing:
            if existing[0] != canonical(config):
                raise StageError("target publication conflicts with existing configuration")
            return
        binding = _binding(target.store)
        db.execute("INSERT INTO candidate_publication.records VALUES (?,?)", (baseline_id, baseline_bytes))
        db.execute(
            "INSERT INTO candidate_publication.metadata VALUES (?,?)",
            ("campaign:" + content_digest(binding), canonical(binding)),
        )
        db.execute("INSERT INTO records VALUES (?,?)", (parent_id, parent_bytes))
        db.execute("INSERT INTO release_config VALUES (1,?)", (canonical(config),))
        db.execute("INSERT INTO incumbent VALUES (1,?,?,0)", (baseline_id, parent_id))
        db.execute(
            "INSERT INTO activation_history VALUES (0,?,?,?)",
            (baseline_id, parent_id, "verified historical baseline import; no new measurement or promotion"),
        )
        # Exact family IDs and original plan IDs are retained. Their source issuer is
        # supplied by the immutable transition, never interpreted as target experiments.
        db.executemany("INSERT INTO confirmation_use VALUES (?,?)", history["reservations"])
        if hook:
            hook("before_target_commit")
    if hook:
        hook("published")


@historical_checked
def resolve_bridge(target, identifier):
    proposal = _proposal(target, identifier, historical=True)
    source = _source(proposal["source"])
    expected = content_digest({"target": proposal["target"], "proposal": identifier})
    with source.store.connect() as db:
        row = db.execute("SELECT successor FROM runtime_cutover WHERE singleton=1").fetchone()
    if row != (expected,):
        raise StageError("baseline bridge lacks original committed source cutover")
    successor_id, _ = _issued(
        source.store,
        "runtime-successor",
        {
            "protocol": PROTOCOL,
            "target": proposal["target"],
            "proposal": identifier,
            "retention": proposal["retention"],
            "history_hash": content_digest(proposal["history"]),
        },
    )
    source.store.get(successor_id, kind="runtime-successor")
    with target.store.connect() as db:
        row = db.execute("SELECT payload FROM release_config WHERE singleton=1").fetchone()
    from hermes.evidence_json import evidence_object

    if row is None or evidence_object(row[0]).get("runtime_transition") != identifier:
        raise StageError("target baseline publication is incomplete; resume runtime commit")
    history = verified_history(source, proposal["retention"])
    if canonical(history) != canonical(proposal["history"]):
        raise StageError("original cutover authority/artifact/reservation history changed")
    return proposal


def baseline_candidate(store, identifier):
    bridge = store.get(identifier, kind="runtime-baseline")["payload"]
    target = _source(bridge["target"])
    proposal = resolve_bridge(target, bridge["proposal"])
    if _binding(store) != proposal["target_candidates"]:
        raise StageError("baseline belongs to another candidate authority")
    original = (
        _source(proposal["source"]).candidates().store.get(proposal["history"]["state"]["candidate"], kind="candidate")
    )
    return {
        "id": identifier,
        "kind": "runtime-baseline",
        "issuer": store.identity,
        "payload": {
            **original["payload"],
            "origin": store.identity,
            "runtime": resolution_runtime(),
            "historical_candidate": {"id": original["id"], "issuer": original["issuer"]},
            "baseline_only": True,
            "transition": bridge,
        },
    }


def baseline_parent(target, identifier):
    bridge = target.store.get(identifier, kind="runtime-parent")["payload"]
    if bridge["target"] != _binding(target.store):
        raise StageError("baseline parent belongs to another authority")
    proposal = resolve_bridge(target, bridge["proposal"])
    history = proposal["history"]
    return {
        "schema": "spark-runtime-parent-v1",
        "origin": target.identity,
        "result": "baseline-import",
        "strict": False,
        "candidate": history["pair"]["model"],
        "agent": history["pair"]["agent_id"],
        "policy": history["configuration"]["policy"],
        "policy_hash": history["configuration"]["policy_hash"],
        "historical_approval": {
            "root": proposal["source"]["root"],
            "identity": proposal["source"]["identity"],
            "id": history["state"]["approval"],
        },
        "transition": bridge,
        "authorizes_production_promotion": False,
        "measured_gain": False,
    }


def readonly_eligibility(root, identifier=None):
    """Inspect authentic legacy records without adding protocol tables to originals."""
    from admin.evaluation import crossed_rows
    from admin.parents import merged_identity

    identity = read_record(root / ".identity")
    with historical_scope(None, identity):
        source = ReleaseAuthority(root)
        state = source.status()
        identifier = identifier or state["approval"]
        if not identifier:
            raise StageError("eligibility needs an original strict approval ID when none was activated")
        decision = source.store.get(identifier, kind="release-decision")["payload"]
        if decision.get("strict") is not True or decision.get("result") != "accepted":
            raise StageError("historical eligibility requires an original accepted strict approval")
        candidate = source.candidates().store.get(decision["candidate_id"], kind="candidate")["payload"]
        with historical_scope(candidate["runtime"], identity):
            # These checks run on the actual original record bytes before the legacy
            # report comparison, yielding the corrected semantic refusal directly.
            for record in source.candidates().store.records(kind="candidate"):
                value = record["payload"]
                if merged_identity(Path(value["model"]["record"]), identity=identity) != value["model"]:
                    raise StageError("original checkpoint identity changed")
            evidence = source.experiments.get(decision["evaluation"]["id"], kind="crossed-evaluation")["payload"]
            plan = source.experiments.get(evidence["plan_id"], kind="crossed-plan")["payload"]
            reasons = set()
            for cell, binding in evidence["logs"].items():
                for row in crossed_rows(binding, plan=plan, cell=cell):
                    reasons.update(row["execution"]["reasons"])
            if reasons:
                raise StageError("original execution is ineligible: " + "; ".join(sorted(reasons)))
            source.resolve_decision(identifier)
            return {
                "eligible_originals": True,
                "baseline_authority": False,
                "source": _binding(source.store),
                "approval": identifier,
                "runtime": candidate["runtime"],
            }


def readonly_command(root, command, identifier=None):
    """Do not initialize or upgrade even an unsupported authority during inspection."""
    with historical_scope(None, read_record(root / ".identity")):
        target = ReleaseAuthority(root)
        if command == "inspect":
            return inspect(target, identifier)
        with target.store.connect() as db:
            supported = bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_cutover'").fetchone())
            row = (
                db.execute("SELECT successor FROM runtime_cutover WHERE singleton=1").fetchone() if supported else None
            )
        return {
            "origin": target.identity,
            "protocol": PROTOCOL if supported else None,
            "retired": row is not None,
            "successor": row[0] if row else None,
            "successor_records": target.store.records(kind="runtime-successor"),
            "recovery": "use successor_records target/proposal to resume runtime commit"
            if row
            else "source is writable; proposals do not retire it"
            if supported
            else "source lacks supported cutover protocol; retain its original runtime for historical inspection",
            "proposals": [
                {"id": r["id"], "source": r["payload"]["source"], "target": r["payload"]["target"]}
                for r in target.store.records(kind="runtime-proposal")
            ],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("capture", "inspect", "eligibility", "propose", "commit", "status", "identity")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--retention")
    parser.add_argument("--mode", choices=("production", "fixture"))
    parser.add_argument("--namespace")
    args = parser.parse_args(argv)
    try:
        if args.command == "identity":
            result = runtime_description()
        elif args.command == "eligibility":
            if args.root is None:
                raise StageError("eligibility requires --root")
            result = readonly_eligibility(args.root, args.id)
        elif args.command in {"inspect", "status"}:
            if args.root is None:
                raise StageError("runtime command requires --root")
            result = readonly_command(args.root, args.command, args.id)
        elif args.command == "propose":
            if args.root is None or args.source is None or not args.retention:
                raise StageError("propose requires --root, --source and original --retention ID")
            # Inspect unsupported originals before any constructor may initialize
            # their schema. Only the separate, authorized target may be created.
            with historical_scope(None, read_record(args.source / ".identity")):
                source = ReleaseAuthority(args.source)
                if source.configuration().get("campaign_protocol") != PROTOCOL:
                    raise StageError("original campaign lacks supported cutover protocol; use read-only eligibility")
            target = ReleaseAuthority(args.root, mode=args.mode, namespace=args.namespace)
            result = propose(source, args.retention, target)
        else:
            if args.root is None:
                raise StageError("runtime command requires --root")
            target = ReleaseAuthority(args.root, mode=args.mode, namespace=args.namespace)
            if args.command == "capture":
                result = capture(target)
            elif args.command == "commit":
                result = commit(target, args.id)
            else:
                raise StageError("unsupported runtime command")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error) as exc:
        print("runtime: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
