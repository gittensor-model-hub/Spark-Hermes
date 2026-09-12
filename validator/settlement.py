"""Durable crown outcomes and an ordered, reconciled GitHub action outbox.

SQLite transactions persist winner, standing and all intended actions together.
Delivery is at-least-once with reconciliation, never a claim of exactly-once HTTP.
Commands default to printing pending intent without network calls or acknowledgments.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote

from hermes.round import GRADED, SETTLED
from validator.crown import CROWN_LABEL, Standing, close_actions, contenders_from, select, settle
from validator.persistence import atomic_write, locked, state_identity, sync_directory
from validator.pr_admission import GitHubSource, admission_for, digest, round_identity
from validator.score import metrics_from_bytes, policy_hash, policy_record, score
from validator.store import RoundStore

SCHEMA = "spark-settlement-v1"
Hook = Callable[[str], None]


class SettlementError(ValueError):
    """State or remote evidence cannot authorize this transition."""


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def checkpoint(hook: Hook | None, name: str) -> None:
    if hook is not None:
        hook(name)


class SettlementStore:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.root = root.resolve()
        self.identity = state_identity(self.root, mode=mode, namespace=namespace)
        self.path = self.root / "settlement.sqlite3"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS rounds (round_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
                round_id TEXT NOT NULL REFERENCES rounds(round_id), action TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','acknowledged')),
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT, receipt TEXT)""")
            prior = self._get(db, "identity")
            if prior is None:
                self._put(db, "identity", self.identity)
                self._put(db, "schema", SCHEMA)
            elif prior != self.identity or self._get(db, "schema") != SCHEMA:
                raise SettlementError("settlement database identity/schema mismatch")
        sync_directory(self.root)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db: sqlite3.Connection, key: str) -> Any:
        row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _put(db: sqlite3.Connection, key: str, value: Any) -> None:
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, encoded(value)))

    def _scope(self, window: Any, repository: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise SettlementError("configure owner/repository")
        origin = window.store_identity
        if any(origin[k] != self.identity[k] for k in ("mode", "namespace")):
            raise SettlementError("round and settlement trust domains differ")
        return {
            "round_id": window.round_id,
            "round_identity": round_identity(window),
            "task_id": window.task_id,
            "epoch": window.challenge.epoch,
            "origin": origin,
            "repository": repository,
        }

    def activate(self, store: RoundStore, round_id: str, repository: str) -> dict[str, Any]:
        with store.lock(round_id), self.transaction() as db:
            window = store.load(round_id)
            scope = self._scope(window, repository)
            binding = self._get(db, "source")
            source = {"origin": store.identity, "repository": repository}
            if binding is not None and binding != source:
                raise SettlementError("settlement root is bound to another producer/repository")
            previous = self._get(db, "active")
            if previous == scope:
                return scope
            if db.execute("SELECT 1 FROM rounds WHERE round_id=?", (round_id,)).fetchone():
                raise SettlementError("cannot reactivate a historical settled round")
            if previous and not db.execute("SELECT 1 FROM rounds WHERE round_id=?", (previous["round_id"],)).fetchone():
                raise SettlementError("previous active round has not settled")
            self._put(db, "source", source)
            self._put(db, "active", scope)
            return scope

    def active(self) -> dict[str, Any]:
        with self.transaction() as db:
            active = self._get(db, "active")
            if active is None:
                raise SettlementError("activate an explicit round first")
            return active

    def record(self, round_id: str) -> dict[str, Any]:
        with self.transaction() as db:
            row = db.execute("SELECT record FROM rounds WHERE round_id=?", (round_id,)).fetchone()
            if row is None:
                raise SettlementError("round has no committed settlement")
            return json.loads(row[0])

    def actions(self, round_id: str | None = None) -> list[dict[str, Any]]:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT * FROM outbox" + (" WHERE round_id=?" if round_id else "") + " ORDER BY sequence",
                (round_id,) if round_id else (),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "action": json.loads(row["action"]),
                    "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
                }
                for row in rows
            ]

    def settle_round(
        self,
        store: RoundStore,
        *,
        scorecards: Path,
        episodes: Path,
        round_id: str | None = None,
        available_tasks: list[str] | None = None,
        hook: Hook | None = None,
    ) -> dict[str, Any]:
        round_id = round_id or str(self.active()["round_id"])
        # Same lock ordering as activation. Round snapshot settlement is a recoverable
        # projection: the database commit remains authoritative if snapshot saving crashes.
        with store.lock(round_id):
            window = store.load(round_id)
            with self.transaction() as db:
                active = self._get(db, "active")
                if active is None or self._scope(window, active["repository"]) != active:
                    raise SettlementError("candidate round/task/epoch is not active")
                existing = db.execute("SELECT record FROM rounds WHERE round_id=?", (round_id,)).fetchone()
                if existing:
                    result = json.loads(existing[0])
                else:
                    if window.state not in (GRADED, SETTLED):
                        raise SettlementError("active round must be graded")
                    field = contenders_from(scorecards, store=store, round_id=round_id)
                    entries = []
                    identities = {}
                    for contender in field:
                        admission = admission_for(window, contender.miner_id)
                        if admission["repository"] != active["repository"]:
                            raise SettlementError("admission repository differs from active repository")
                        pr = admission["pr_number"]
                        if pr in identities:
                            raise SettlementError("one PR cannot represent two candidates in a round")
                        identities[pr] = {
                            "pr": pr,
                            "head_sha": admission["head_sha"],
                            "author": admission["author"],
                        }
                        log = (episodes / round_id / f"{contender.miner_id}.jsonl").resolve()
                        before = log.read_bytes()
                        regenerated = score(
                            window=window,
                            miner_id=contender.miner_id,
                            rows=metrics_from_bytes(before, source=str(log)),
                            model_revision=window.challenge.epoch["model_revision"],
                            harness_digest=window.challenge.epoch["harness_digest"],
                        ).to_record()
                        if before != log.read_bytes() or regenerated != contender.evidence:
                            raise SettlementError("episode log changed or disagrees with scorecard")
                        entries.append(
                            {
                                "miner_id": contender.miner_id,
                                "admission": admission,
                                "scorecard": regenerated,
                                "scorecard_hash": digest(regenerated),
                                "episodes": {
                                    "path": str(log),
                                    "sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
                                },
                            }
                        )
                    outcome = select(field)
                    previous: dict[str, Any] = self._get(db, "standing") or {"task_id": window.task_id}
                    standing, labels = settle(previous, outcome, available_tasks=available_tasks)
                    old = previous.get("winner")
                    if old:
                        identities.setdefault(old["pr"], previous["winner_pr_identity"])
                    intentions = []
                    for operation, pr in labels:
                        intentions.append({**identities[pr], "kind": "label_" + operation, "label": CROWN_LABEL})
                    # Reviews bind the result to the evaluated commit. Close only this round's
                    # challengers; no historical/refused identity is guessed from registry JSON.
                    for entry in entries:
                        pr = entry["admission"]["pr_number"]
                        accepted = entry["scorecard"]["decision"]["accepted"]
                        winner = outcome.winner is not None and outcome.winner.contender.pr == pr
                        intentions.append(
                            {
                                **identities[pr],
                                "kind": "review",
                                "body": f"Round {round_id}: {'crowned' if winner else 'accepted, not crowned' if accepted else 'refused'}. "
                                "This records strategy evaluation only; it does not authorize model promotion or payment.",
                            }
                        )
                    for pr, reason in close_actions(outcome, Standing.from_record(previous)):
                        intentions.append({**identities[pr], "kind": "close", "reason": reason})
                    settlement_id = digest({"schema": SCHEMA, "scope": active, "producer": self.identity})
                    actions = []
                    for index, intent in enumerate(intentions):
                        body = {
                            **intent,
                            "repository": active["repository"],
                            "origin": store.identity,
                            "settlement_id": settlement_id,
                            "round_id": round_id,
                        }
                        key = digest({"settlement_id": settlement_id, "index": index, "intent": body})
                        actions.append({"key": key, **body})
                    standing_record = dataclasses.asdict(standing)
                    if standing.winner:
                        standing_record["winner_pr_identity"] = identities[standing.pr]
                    result = {
                        "schema": SCHEMA,
                        "settlement_id": settlement_id,
                        "scope": active,
                        "origin": store.identity,
                        "producer": self.identity,
                        "mode": self.identity["mode"],
                        "namespace": self.identity["namespace"],
                        "policy": policy_record(),
                        "policy_hash": policy_hash(policy_record()),
                        "outcome": outcome.to_record(),
                        "standing": standing_record,
                        "entries": entries,
                        "actions": actions,
                        "committed_at": time.time(),
                        "external_delivery": "consult outbox receipts",
                        "authorizes_model_promotion": False,
                        "authorizes_payment": False,
                    }
                    checkpoint(hook, "before_persist")
                    db.execute("INSERT INTO rounds VALUES (?,?)", (round_id, encoded(result)))
                    for action in actions:
                        db.execute(
                            "INSERT INTO outbox(key,round_id,action) VALUES (?,?,?)",
                            (action["key"], round_id, encoded(action)),
                        )
                    self._put(db, "standing", standing_record)
                    checkpoint(hook, "before_commit")
            checkpoint(hook, "after_persist")
            if window.state == GRADED:
                window.settle(now=time.time())
                store.save(window)
            return result

    def deliver(self, adapter: GitHubActions | None = None, *, hook: Hook | None = None) -> list[dict[str, Any]]:
        if adapter is None:
            return [
                {"key": row["key"], "status": "dry-run", "intent": row["action"], "delivered": False}
                for row in self.actions()
                if row["status"] == "pending"
            ]
        if self.identity["mode"] != "production" and not adapter.controlled:
            raise SettlementError("fixture state cannot emit live GitHub actions")
        receipts = []
        # Held through reconcile/write/ack across processes. OS releases it on death;
        # global sequence prevents an old label removal racing a later addition.
        with locked(self.root / ".delivery.lock"):
            for row in self.actions():
                if row["status"] == "acknowledged":
                    continue
                with self.transaction() as db:
                    db.execute("UPDATE outbox SET attempts=attempts+1 WHERE key=?", (row["key"],))
                try:
                    checkpoint(hook, "before_deliver")
                    receipt = adapter.reconcile(row["action"])
                    if receipt is None:
                        adapter.apply(row["action"])
                        checkpoint(hook, "after_deliver")
                        receipt = adapter.reconcile(row["action"])
                    if receipt is None:
                        raise SettlementError("remote effect not yet observable; left pending")
                    checkpoint(hook, "before_ack")
                    with self.transaction() as db:
                        db.execute(
                            "UPDATE outbox SET status='acknowledged',receipt=?,error=NULL WHERE key=?",
                            (encoded(receipt), row["key"]),
                        )
                    checkpoint(hook, "after_ack")
                    receipts.append(receipt)
                except Exception:
                    with self.transaction() as db:
                        db.execute(
                            "UPDATE outbox SET error=? WHERE key=?", ("delivery failed; reconcile on retry", row["key"])
                        )
                    raise
        return receipts


class GitHubActions(GitHubSource):
    """Real gh API action adapter; injected subprocess transport exercises the same code.

    Reviews use a stable body marker and authenticated author for reconciliation.
    Label/close operations converge on desired state. A lost response remains pending
    until a read proves the effect. GitHub supplies no atomic idempotency guarantee.
    """

    def __init__(
        self, repository: str, *, credential_env: str = "GH_TOKEN", transport: Callable[..., Any] | None = None
    ):
        super().__init__(repository, credential_env=credential_env, transport=transport)
        self.controlled = transport is not None
        who = self.get("user")
        if not isinstance(who, dict) or not isinstance(who.get("login"), str) or not who["login"]:
            raise SettlementError("unknown authenticated action identity")
        self.actor = who["login"]

    def _pr(self, action: dict[str, Any]) -> dict[str, Any]:
        if action["repository"] != self.repository or type(action["pr"]) is not int or action["pr"] <= 0:
            raise SettlementError("action repository/PR mismatch")
        pr = self.get(f"repos/{self.repository}/pulls/{action['pr']}")
        try:
            if (
                type(pr["number"]) is not int
                or pr["number"] != action["pr"]
                or pr["base"]["repo"]["full_name"] != self.repository
                or pr["head"]["sha"] != action["head_sha"]
                or pr["user"]["login"] != action["author"]
                or pr["state"] not in ("open", "closed")
                or type(pr["merged"]) is not bool
                or (pr["merged"] and (action["kind"] != "label_remove" or pr["state"] != "closed"))
                or pr["draft"] is not False
            ):
                raise ValueError("changed PR")
        except (KeyError, TypeError, ValueError) as exc:
            raise SettlementError("remote PR identity/state differs from admitted commit") from exc
        return pr

    def _pages(self, endpoint: str) -> list[dict[str, Any]]:
        results = []
        for page in range(1, 1001):
            batch = self.get(f"{endpoint}?per_page=100&page={page}")
            if not isinstance(batch, list) or any(not isinstance(x, dict) for x in batch):
                raise SettlementError("malformed GitHub collection")
            results.extend(batch)
            if len(batch) < 100:
                return results
        raise SettlementError("GitHub collection pagination limit; cannot reconcile")

    @staticmethod
    def marker(action: dict[str, Any]) -> str:
        return f"<!-- spark-settlement:{action['key']} -->"

    def reconcile(self, action: dict[str, Any]) -> dict[str, Any] | None:
        pr = self._pr(action)
        kind = action["kind"]
        endpoint = f"repos/{self.repository}"
        evidence: Any = None
        if kind in ("label_add", "label_remove"):
            labels = self._pages(f"{endpoint}/issues/{action['pr']}/labels")
            if any(not isinstance(x.get("name"), str) for x in labels):
                raise SettlementError("malformed label evidence")
            present = any(x["name"] == action["label"] for x in labels)
            if present == (kind == "label_add"):
                evidence = {"label": action["label"], "present": present}
        elif kind == "review":
            reviews = self._pages(f"{endpoint}/pulls/{action['pr']}/reviews")
            for review in reviews:
                if (
                    review.get("body") == action["body"] + "\n\n" + self.marker(action)
                    and review.get("user", {}).get("login") == self.actor
                    and review.get("commit_id") == action["head_sha"]
                    and review.get("state") == "COMMENTED"
                    and type(review.get("id")) is int
                ):
                    evidence = {"review_id": review["id"]}
                    break
        elif kind == "close":
            if pr["state"] == "closed":
                evidence = {"state": "closed"}
        else:
            raise SettlementError("unknown action kind")
        return (
            {
                "key": action["key"],
                "transport": "controlled-github" if self.controlled else "github",
                "repository": self.repository,
                "pr": action["pr"],
                "evidence": evidence,
            }
            if evidence is not None
            else None
        )

    def apply(self, action: dict[str, Any]) -> None:
        self._pr(action)
        endpoint = f"repos/{self.repository}"
        kind = action["kind"]
        payload: dict[str, Any] = {}
        if kind == "label_add":
            method, endpoint = "POST", f"{endpoint}/issues/{action['pr']}/labels"
            payload = {"labels": [action["label"]]}
        elif kind == "label_remove":
            method, endpoint = "DELETE", f"{endpoint}/issues/{action['pr']}/labels/{quote(action['label'], safe='')}"
        elif kind == "review":
            method, endpoint = "POST", f"{endpoint}/pulls/{action['pr']}/reviews"
            payload = {
                "event": "COMMENT",
                "commit_id": action["head_sha"],
                "body": action["body"] + "\n\n" + self.marker(action),
            }
        elif kind == "close":
            method, endpoint = "PATCH", f"{endpoint}/pulls/{action['pr']}"
            payload = {"state": "closed"}
        else:
            raise SettlementError("unknown action kind")
        try:
            response = self._transport(
                ["gh", "api", "--hostname", "github.com", "--method", method, endpoint, "--input", "-"],
                input=encoded(payload),
                env=self._env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if response.returncode:
                raise SettlementError("GitHub action response failed; retry must reconcile")
        except (OSError, subprocess.SubprocessError) as exc:
            raise SettlementError("GitHub action response unknown; retry must reconcile") from exc


def crown_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Commit crown selection or read its persisted action intent; no network"
    )
    parser.add_argument("action", choices=("select", "actions"))
    parser.add_argument("--settlement-root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--round")
    parser.add_argument("--scorecards", type=Path)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--out", type=Path, help="optional export of the immutable committed record")
    parser.add_argument("--challenges", type=Path)
    args = parser.parse_args(argv)
    try:
        state = SettlementStore(args.settlement_root)
        round_id = args.round or state.active()["round_id"]
        if args.action == "actions":
            state.record(round_id)
            result: Any = state.actions(round_id)
        else:
            if args.scorecards is None or args.episodes is None:
                raise SettlementError("select requires explicit --scorecards and --episodes roots")
            result = state.settle_round(
                RoundStore(args.store),
                round_id=round_id,
                scorecards=args.scorecards,
                episodes=args.episodes,
                available_tasks=sorted(p.stem for p in args.challenges.glob("*.json")) if args.challenges else None,
            )
        if args.out:
            atomic_write(args.out, (encoded(result) + "\n").encode())
        print(encoded(result))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"crown refused: {exc}", file=sys.stderr)
        return 2


def main(
    argv: list[str] | None = None, *, transport: Callable[..., Any] | None = None, hook: Hook | None = None
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("activate", "status", "export", "deliver"))
    parser.add_argument("--settlement-root", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--round")
    parser.add_argument("--repository")
    parser.add_argument("--mode", choices=("production", "fixture"), help="immutable at creation; default production")
    parser.add_argument("--namespace")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-external", action="store_true", help="explicitly enable GitHub delivery (production only)"
    )
    parser.add_argument("--credential-env", default="GH_TOKEN")
    args = parser.parse_args(argv)
    try:
        state = SettlementStore(args.settlement_root, mode=args.mode, namespace=args.namespace)
        result: Any
        if args.action == "activate":
            if args.store is None or not args.round or not args.repository:
                raise SettlementError("activate requires --store, --round and --repository")
            result = state.activate(RoundStore(args.store), args.round, args.repository)
        elif args.action == "status":
            result = {"identity": state.identity, "active": state.active(), "actions": state.actions()}
        elif args.action == "export":
            result = state.record(args.round or state.active()["round_id"])
        else:
            adapter = None
            if args.allow_external:
                if state.identity["mode"] != "production" and transport is None:
                    raise SettlementError("fixture state cannot emit live GitHub actions")
                repository = state.active()["repository"]
                if args.repository and args.repository != repository:
                    raise SettlementError("delivery repository differs from bound repository")
                adapter = GitHubActions(repository, credential_env=args.credential_env, transport=transport)
            result = state.deliver(adapter, hook=hook)
        if args.out:
            atomic_write(args.out, (encoded(result) + "\n").encode())
        print(encoded(result))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"settlement refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
