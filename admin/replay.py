"""Frozen replay from committed settlement evidence. Run `python -m admin.replay --help`."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from admin.artifacts import AuthorityStore, StageError, canonical, content_digest, file_digest, read_record, same_domain
from admin.data_policy import DataPolicy
from validator.aggregate import Episode, admit_episode, aggregate
from validator.intake import Intake, receipt_for_digest
from validator.persistence import atomic_write, locked
from validator.pr_admission import admission_for, round_identity
from validator.score import metrics_from_bytes, policy_hash, policy_record, private_check_required, score
from validator.settlement import SettlementStore
from validator.store import RoundStore

DEFAULT_MIXTURE = {
    "version": "spark-mixture-v1",
    "max_pairs_per_task": 8,
    "max_sft_per_family": 32,
    "max_sft_per_miner": 32,
    "max_pairs_per_family": 32,
    "max_pairs_per_miner": 32,
}


def source_stores(source: dict[str, Any]) -> tuple[RoundStore, SettlementStore, Intake]:
    for key in ("rounds", "settlement", "intake", "receipts"):
        if not isinstance(source.get(key), str) or not Path(source[key]).exists():
            raise StageError(f"configured {key} source does not exist")
    for key in ("rounds", "settlement", "intake"):
        if not (Path(source[key]) / ".identity").is_file():
            raise StageError("learning authority refuses legacy sources without explicit identity")
    return (
        RoundStore(Path(source["rounds"])),
        SettlementStore(Path(source["settlement"])),
        Intake(Path(source["intake"]), Path(source["receipts"])),
    )


def settled_episodes(source: dict[str, Any], round_id: str, policy: DataPolicy) -> tuple[list[Episode], dict[str, Any]]:
    """Check each issuer in its own role and re-score exactly the settlement's bytes."""
    from hermesbench.sink import decode_episodes

    rounds, settlement, intake = source_stores(source)
    identities = {"rounds": rounds.identity, "settlement": settlement.identity, "intake": intake.identity}
    if identities != source.get("identities"):
        raise StageError("configured source issuer changed (round/settlement/intake are distinct roles)")
    record = settlement.record(round_id)
    with rounds.lock(round_id):
        snapshot = read_record(rounds.path_for(round_id))
        if snapshot.get("store_identity") != rounds.identity:
            raise StageError("legacy or foreign round snapshot cannot authorize learning")
        window = rounds.load(round_id)
        expected_scope = {
            "round_id": round_id,
            "round_identity": round_identity(window),
            "task_id": window.task_id,
            "epoch": window.challenge.epoch,
            "origin": rounds.identity,
            "repository": record["scope"]["repository"],
        }
        if (
            record.get("producer") != settlement.identity
            or record.get("origin") != rounds.identity
            or record.get("scope") != expected_scope
            or record.get("schema") != "spark-settlement-v1"
            or any(record.get(k) != settlement.identity[k] for k in ("mode", "namespace"))
        ):
            raise StageError("settlement source/round identity mismatch")
        if record.get("policy") != policy_record() or record.get("policy_hash") != policy_hash(policy_record()):
            raise StageError("settlement score policy is stale or unknown")
        episodes = []
        for entry in record["entries"]:
            miner = entry["miner_id"]
            admission = admission_for(window, miner)
            if entry["admission"] != admission or admission["receipt_origin"] != intake.identity:
                raise StageError("settlement admission or receipt issuer mismatch")
            receipt = receipt_for_digest(
                intake.read_receipts(), round_id=round_id, miner_id=miner, digest=admission["bundle_sha256"]
            )
            if receipt is None or receipt.submission_id != admission["submission_id"]:
                raise StageError("committed contribution has no matching intake receipt")
            intake.verify(receipt)
            log = Path(entry["episodes"]["path"])
            raw = log.read_bytes()
            if "sha256:" + hashlib.sha256(raw).hexdigest() != entry["episodes"]["sha256"]:
                raise StageError("settled episode source bytes changed")
            metrics = metrics_from_bytes(raw, source=str(log))
            regenerated = score(
                window=window,
                miner_id=miner,
                rows=metrics,
                model_revision=window.challenge.epoch["model_revision"],
                harness_digest=window.challenge.epoch["harness_digest"],
            ).to_record()
            if regenerated != entry["scorecard"] or content_digest(regenerated) != entry["scorecard_hash"]:
                raise StageError("settled scorecard disagrees with original evidence")
            version = content_digest(window.challenge.task_pins)
            data_use = policy.provenance(
                task_id=window.task_id,
                repository=admission["repository"],
                version=version,
                contribution=admission["admission_id"],
            )
            for number, (row, metric) in enumerate(zip(decode_episodes(raw, source=str(log)), metrics), 1):
                provenance = {
                    "source": source["name"],
                    "settlement_id": record["settlement_id"],
                    "settlement_hash": content_digest(record),
                    "settlement_producer": settlement.identity,
                    "origin": rounds.identity,
                    "receipt_origin": intake.identity,
                    "admission": admission,
                    "contribution_id": admission["admission_id"],
                    "submission_id": receipt.submission_id,
                    "miner_id": miner,
                    "round_id": round_id,
                    "task_id": window.task_id,
                    "task_version": version,
                    "epoch": window.challenge.epoch,
                    "attempt_id": metric["attempt_id"],
                    "bundle_sha256": admission["bundle_sha256"],
                    "model_revision": metric["model_revision"],
                    "evaluator": {"harness_digest": metric["harness_digest"], "verify_digest": metric["verify_digest"]},
                    "log": {**entry["episodes"], "line": number},
                    "episode_hash": content_digest(row),
                    **data_use,
                }
                episodes.append(
                    admit_episode(
                        row,
                        round_id=round_id,
                        miner_id=miner,
                        private_required=private_check_required(window.challenge),
                        provenance=provenance,
                    )
                )
        # Committed DB state is sufficient even if its GRADED -> SETTLED projection crashed.
        return episodes, record


def _dedup_cap(
    rows: list[dict[str, Any]], *, kind: str, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    fields = ("messages", "tools") if kind == "sft" else ("chosen", "rejected", "tools")
    for row in rows:
        key = content_digest({k: row.get(k) for k in fields})
        if key not in groups:
            groups[key] = {**row, "content_id": key, "provenance": []}
        lineage = {content_digest(p): p for p in groups[key]["provenance"] + row["provenance"]}
        groups[key]["provenance"] = [lineage[k] for k in sorted(lineage)]
    families: dict[str, int] = {}
    miners: dict[str, int] = {}
    chosen = []
    for _, row in sorted(groups.items()):
        fs = {p["membership"]["canonical_family"] for p in row["provenance"]}
        ms = {p["miner_id"] for p in row["provenance"]}
        if any(families.get(f, 0) >= config[f"max_{kind}_per_family"] for f in fs) or any(
            miners.get(m, 0) >= config[f"max_{kind}_per_miner"] for m in ms
        ):
            continue
        chosen.append(row)
        for f in fs:
            families[f] = families.get(f, 0) + 1
        for m in ms:
            miners[m] = miners.get(m, 0) + 1
    return chosen, {"duplicates": len(rows) - len(groups), "capped": len(groups) - len(chosen)}


class ReplayStore:
    def __init__(self, root: Path, *, mode: str | None = None, namespace: str | None = None):
        self.authority = AuthorityStore(root, role="replay", mode=mode, namespace=namespace)
        self.root = self.authority.root
        self.identity = self.authority.identity

    def configure(self, config: dict[str, Any]) -> dict[str, Any]:
        policy = DataPolicy(Path(config["policy"]), identity=self.identity)
        mixture = config.get("mixture", DEFAULT_MIXTURE)
        if (
            set(mixture) != set(DEFAULT_MIXTURE)
            or mixture["version"] != DEFAULT_MIXTURE["version"]
            or any(type(mixture[k]) is not int or mixture[k] < 1 for k in mixture if k != "version")
        ):
            raise StageError("mixture requires the complete versioned positive-integer cap policy")
        sources = []
        for raw in config.get("sources", []):
            if not isinstance(raw.get("name"), str) or not raw["name"]:
                raise StageError("source needs a stable name")
            source = {
                "name": raw["name"],
                **{k: str(Path(raw[k]).resolve()) for k in ("rounds", "settlement", "intake", "receipts")},
            }
            rounds, settlement, intake = source_stores(source)
            source["identities"] = {
                "rounds": rounds.identity,
                "settlement": settlement.identity,
                "intake": intake.identity,
            }
            for identity in source["identities"].values():
                same_domain(self.identity, identity)
            sources.append(source)
        if not sources or len({s["name"] for s in sources}) != len(sources):
            raise StageError("replay requires uniquely named configured settlement sources")
        payload = {
            "policy": str(policy.path),
            "policy_sha256": policy.sha256,
            "sources": sorted(sources, key=lambda s: s["name"]),
            "mixture": mixture,
        }
        with locked(self.root / ".config.lock"):
            prior = self.authority.records(kind="configuration")
            if prior and prior[0]["payload"] != payload:
                raise StageError("replay configuration is immutable; use a new version/root and re-import history")
            return self.authority.put("configuration", payload)

    def configuration(self) -> dict[str, Any]:
        records = self.authority.records(kind="configuration")
        if len(records) != 1:
            raise StageError("configure the replay authority first")
        config = records[0]["payload"]
        if file_digest(Path(config["policy"])) != config["policy_sha256"]:
            raise StageError("configured rights/family/exposure policy changed; create a reviewed corpus version")
        return config

    def read_round(self, source_name: str, round_id: str) -> tuple[list[Episode], dict[str, Any]]:
        config = self.configuration()
        sources = [s for s in config["sources"] if s["name"] == source_name]
        if len(sources) != 1:
            raise StageError("unknown configured settlement source")
        return settled_episodes(sources[0], round_id, DataPolicy(Path(config["policy"]), identity=self.identity))

    def import_round(self, source_name: str, round_id: str) -> dict[str, Any]:
        episodes, record = self.read_round(source_name, round_id)
        payload = {
            "source": source_name,
            "round_id": round_id,
            "settlement_hash": content_digest(record),
            "episodes": [asdict(e) for e in episodes],
        }
        return self.authority.put("settled-experience", payload)

    def experiences(self, identifiers: list[str] | None = None) -> tuple[list[Episode], list[str]]:
        records = (
            self.authority.records(kind="settled-experience")
            if identifiers is None
            else [self.authority.get(i, kind="settled-experience") for i in identifiers]
        )
        episodes = []
        for record in records:
            payload = record["payload"]
            fresh, settlement = self.read_round(payload["source"], payload["round_id"])
            if payload["settlement_hash"] != content_digest(settlement) or payload["episodes"] != [
                asdict(e) for e in fresh
            ]:
                raise StageError("historical settled evidence changed")
            episodes.extend(fresh)
        return episodes, sorted(r["id"] for r in records)

    def freeze(self, workspace: Any, *, identifiers: list[str] | None = None) -> dict[str, Any]:
        from tempfile import TemporaryDirectory

        from hermes.evidence_json import evidence_records

        same_domain(self.identity, workspace.identity)
        config = self.configuration()
        if identifiers is not None and (not identifiers or len(set(identifiers)) != len(identifiers)):
            raise StageError("freeze requires distinct explicit experience IDs")
        episodes, inputs = self.experiences(identifiers)
        if not episodes:
            raise StageError("replay has no admitted executed experience")
        with TemporaryDirectory(prefix="render-", dir=self.root) as scratch:
            out = Path(scratch)
            summary = aggregate(
                episodes, out, max_per_task=config["mixture"]["max_pairs_per_task"], best_only=False
            ).to_record()
            files, selection = {}, {}
            for name, kind in (("sft.jsonl", "sft"), ("preference.jsonl", "pairs")):
                rows = evidence_records((out / name).read_text())
                rows, selection[kind] = _dedup_cap(rows, kind=kind, config=config["mixture"])
                files[name] = b"".join(canonical(r) + b"\n" for r in rows)
                summary["sft_rows" if kind == "sft" else "preference_pairs"] = len(rows)
        manifest = {
            **summary,
            "schema": "spark-replay-mixture-v1",
            "origin": self.identity,
            "inputs": inputs,
            "configuration": config,
            "selection": selection,
            "accepted": sorted({e.task_id for e in episodes}),
            "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
            "authorizes_model_promotion": False,
        }
        committed = self.authority.put("mixture", manifest)
        manifest = {**manifest, "authority": {"root": str(self.root), "identity": self.identity, "id": committed["id"]}}
        with locked(workspace.root / ".corpus.lock"):
            for name, raw in files.items():
                atomic_write(workspace.corpus / name, raw)
            workspace.record("corpus", manifest)
        return manifest


def verify_corpus(workspace: Any, corpus: dict[str, Any]) -> None:
    binding = corpus.get("authority")
    if not isinstance(binding, dict):
        raise StageError("corpus has no committed replay authority; rebuild via corpus/replay")
    if corpus.get("schema") == "spark-operator-corpus-v1":
        authority = AuthorityStore(Path(binding["root"]), role="operator-corpus")
        same_domain(workspace.identity, authority.identity)
        if binding["identity"] != authority.identity or corpus.get("origin") != workspace.identity:
            raise StageError("operator corpus issuer changed")
        committed = authority.get(binding["id"], kind="mixture")["payload"]
        if {k: v for k, v in corpus.items() if k not in {"authority", "summary", "completed_at"}} != committed:
            raise StageError("operator corpus manifest changed")
        for path, expected in committed["inputs"].items():
            if file_digest(Path(path)) != expected:
                raise StageError("operator corpus source/rights/family artifact changed")
        for name, expected in committed["sha256"].items():
            if file_digest(workspace.corpus / name) != expected:
                raise StageError("operator corpus bytes changed")
        return
    replay = ReplayStore(Path(binding["root"]))
    same_domain(workspace.identity, replay.identity)
    if binding["identity"] != replay.identity or corpus.get("origin") != replay.identity:
        raise StageError("corpus issuer changed")
    committed = replay.authority.get(binding["id"], kind="mixture")["payload"]
    if {k: v for k, v in corpus.items() if k not in {"authority", "summary", "completed_at"}} != committed:
        raise StageError("corpus manifest changed from committed mixture")
    replay.configuration()
    replay.experiences(committed["inputs"])
    for name, expected in committed["sha256"].items():
        if file_digest(workspace.corpus / name) != expected:
            raise StageError("frozen replay mixture bytes changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "import", "freeze", "verify"))
    parser.add_argument("--root", type=Path, required=True, help="private replay authority root")
    parser.add_argument("--mode", choices=("production", "fixture"))
    parser.add_argument("--namespace")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--round")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args(argv)
    try:
        replay = ReplayStore(args.root, mode=args.mode, namespace=args.namespace)
        if args.command == "init":
            if args.config is None:
                raise StageError("init requires --config")
            report = replay.configure(read_record(args.config))
        elif args.command == "import":
            if not args.source or not args.round:
                raise StageError("import requires --source and --round")
            report = replay.import_round(args.source, args.round)
            report = {"id": report["id"], "issuer": report["issuer"], "episodes": len(report["payload"]["episodes"])}
        else:
            from admin.pipeline import Workspace, _require

            if args.workspace is None:
                raise StageError("freeze/verify requires --workspace")
            ws = Workspace(args.workspace)
            if args.command == "freeze":
                # Creation is explicit; an existing production root cannot become fixture.
                from validator.persistence import state_identity

                state_identity(ws.root, mode=args.mode, namespace=args.namespace)
                report = replay.freeze(ws)
            else:
                verify_corpus(ws, _require(ws, "corpus"))
                report = {"verified": True, "origin": ws.identity}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
