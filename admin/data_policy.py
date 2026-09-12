"""Versioned operator declarations for data rights, family aliases and exposure.

The configured policy is trusted operator input, never contributor-supplied task text.
Replay freezes its bytes and rechecks the configured file before every consumption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, content_digest, same_domain
from hermes.evidence_json import evidence_object

PARTITIONS = {"public-development", "private-competition", "sealed-release"}
EXPOSURES = {"public", "disclosed", "selection", "training"}


class DataPolicy:
    def __init__(self, path: Path, *, identity: dict[str, str]):
        self.path = path.resolve()
        raw = self.path.read_bytes()
        self.record = evidence_object(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        r = self.record
        if r.get("schema") != "spark-data-policy-v1" or not isinstance(r.get("version"), str) or not r["version"]:
            raise StageError("missing versioned data policy")
        same_domain(identity, r.get("origin", {}))
        self.aliases: Any = r.get("family_aliases")
        if not isinstance(self.aliases, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in self.aliases.items()
        ):
            raise StageError("family_aliases must explicitly map canonical and alias families")
        for family in self.aliases:
            self.family(family)
        self.members: Any = r.get("memberships")
        if not isinstance(self.members, list) or not self.members:
            raise StageError("missing family membership metadata")
        seen = set()
        for member in self.members:
            if not isinstance(member, dict) or any(
                not isinstance(member.get(k), str) or not member[k]
                for k in ("task_id", "repository", "version", "family_id")
            ):
                raise StageError("task membership requires task/family/repository/version")
            self.family(member["family_id"])
            key = tuple(member[k] for k in ("task_id", "repository", "version"))
            if key in seen:
                raise StageError("duplicate task membership")
            seen.add(key)
            if member.get("partition") not in PARTITIONS or not isinstance(member.get("exposure"), list):
                raise StageError("missing partition or explicit exposure state")
            if any(x not in EXPOSURES for x in member["exposure"]):
                raise StageError("unknown exposure state")
            if member["partition"] == "public-development" and "public" not in member["exposure"]:
                raise StageError("public development must record public exposure")
        grants = r.get("rights")
        if not isinstance(grants, list):
            raise StageError("missing explicit data-use rights")
        self.grants: dict[str, dict[str, Any]] = {}
        for grant in grants:
            if (
                not isinstance(grant, dict)
                or any(
                    not isinstance(grant.get(k), str) or not grant[k].strip()
                    for k in ("subject", "license", "attribution")
                )
                or any(type(grant.get(k)) is not bool for k in ("training", "derivatives"))
            ):
                raise StageError("malformed rights grant")
            if grant["subject"] in self.grants:
                raise StageError("duplicate rights subject")
            self.grants[grant["subject"]] = grant

    def family(self, family: str) -> str:
        seen = set()
        while True:
            if family not in self.aliases or family in seen:
                raise StageError("missing or cyclic canonical family mapping")
            target = self.aliases[family]
            if target == family:
                return family
            seen.add(family)
            family = target

    def membership(self, *, task_id: str, repository: str, version: str, purpose: str) -> dict[str, Any]:
        matches = [
            m for m in self.members if (m["task_id"], m["repository"], m["version"]) == (task_id, repository, version)
        ]
        if len(matches) != 1:
            raise StageError("missing exact task/family/repository/version membership")
        member = matches[0]
        family = self.family(member["family_id"])
        related = [m for m in self.members if self.family(m["family_id"]) == family]
        if purpose == "training":
            if any(m["partition"] == "sealed-release" for m in related):
                raise StageError("sealed family or cross-repository alias cannot enter training")
        elif purpose == "release":
            if member["partition"] != "sealed-release" or any(m["exposure"] for m in related):
                raise StageError("disclosed/selection-used family cannot count as fresh release confirmation")
        elif purpose != "curriculum":
            raise StageError("unknown data admission purpose")
        return {**member, "canonical_family": family}

    def rights(self, subject: str) -> dict[str, Any]:
        grant = self.grants.get(subject)
        if not grant or grant["training"] is not True or grant["derivatives"] is not True:
            raise StageError(f"missing or disallowed training/derivative rights for {subject}")
        return grant

    def provenance(self, *, task_id: str, repository: str, version: str, contribution: str) -> dict[str, Any]:
        member = self.membership(task_id=task_id, repository=repository, version=version, purpose="training")
        if member["partition"] == "private-competition" and "selection" not in member["exposure"]:
            raise StageError("settled competition membership must record selection exposure")
        return {
            "membership": member,
            "rights": [self.rights("task:" + version), self.rights(contribution)],
            "policy": {"path": str(self.path), "sha256": self.sha256, "id": content_digest(self.record)},
        }
