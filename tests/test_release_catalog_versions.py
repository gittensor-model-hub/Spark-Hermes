"""Confirmation catalog extensions retain predeclared knowledge across experiments."""

import copy
import json
from pathlib import Path

import pytest
from learning_support import cli
from release_support import setup_release, successor

from admin.artifacts import StageError, content_digest, read_record, write_record
from admin.candidates import file_identity
from admin.evaluation import execute_crossed
from admin.release import ReleaseAuthority


def historical_catalog(root):
    catalog = read_record(root / "confirmation.json")
    catalog["version"] = "extended-history-v2"
    catalog["family_aliases"].update({"historical": "historical", "historical-alias": "historical"})
    catalog["memberships"].append(
        {
            "task_id": "historical-task",
            "repository": "fixture/history",
            "version": "historical-version",
            "family_id": "historical-alias",
            "partition": "private-competition",
            "exposure": ["selection"],
        }
    )
    return catalog


@pytest.mark.parametrize("source", ["configured", "frozen"])
@pytest.mark.parametrize(
    "change", ["alias", "merge", "spent-family", "membership", "exposure", "partition", "identity"]
)
def test_catalog_cannot_forget_knowledge_outside_candidate_corpora(tmp_path, source, change):
    authority, _, _, freeze = setup_release(tmp_path)
    catalog = historical_catalog(tmp_path)
    known = tmp_path / "known.json"
    write_record(known, catalog)
    if source == "configured":
        config = authority.configuration()
        authority = ReleaseAuthority(
            tmp_path / "configured-history", mode="fixture", namespace=authority.identity["namespace"]
        )
        authority.configure(
            candidates=Path(config["candidates"]["root"]),
            incumbent=freeze["old"],
            data_policy=known,
            policy=config["policy"],
        )
    else:
        first = authority.freeze(**freeze, data_policy=known)
        assert authority.freeze(**freeze, data_policy=known)["id"] == first["id"]
    changed = copy.deepcopy(catalog)
    if change == "alias":
        del changed["family_aliases"]["historical-alias"]
        changed["memberships"][-1]["family_id"] = "historical"
    elif change == "merge":
        changed["family_aliases"]["historical"] = "sealed-0"
    elif change == "spent-family":
        changed["family_aliases"].update(
            {"sealed-0": "renamed-confirmation", "renamed-confirmation": "renamed-confirmation"}
        )
    elif change == "membership":
        changed["memberships"].pop()
    elif change == "exposure":
        changed["memberships"][-1]["exposure"] = []
    elif change == "partition":
        changed["memberships"][-1]["partition"] = "sealed-release"
    else:
        changed["memberships"][-1]["repository"] = "fixture/renamed-history"
    replacement = tmp_path / "replacement.json"
    write_record(replacement, changed)
    with pytest.raises(StageError, match="catalog dropped|selection-used"):
        ReleaseAuthority(authority.store.root).freeze(**freeze, data_policy=replacement)


def test_selected_catalog_cli_binding_and_mutation_refusal(tmp_path):
    authority, _, _, freeze = setup_release(tmp_path)
    catalog = tmp_path / "extended.json"
    write_record(catalog, historical_catalog(tmp_path))
    config = tmp_path / "extended-freeze.json"
    write_record(config, {**freeze, "data_policy": str(catalog)})
    issued = json.loads(
        cli(tmp_path, "admin.cotraining", "freeze", "--root", authority.store.root, "--config", config).stdout
    )
    restarted = ReleaseAuthority(authority.store.root)
    assert restarted.plan(issued["id"])["data_policy"] == file_identity(catalog)
    assert restarted.configuration()["data_policy"] == file_identity(tmp_path / "confirmation.json")
    replacement = tmp_path / "unchanged-extension.json"
    replacement.write_bytes(catalog.read_bytes())
    catalog.write_bytes(catalog.read_bytes() + b" ")
    with pytest.raises(StageError, match="artifact bytes/path changed"):
        restarted.plan(issued["id"])
    with pytest.raises(StageError, match="artifact bytes/path changed"):
        restarted.freeze(**freeze, data_policy=replacement)


def test_next_candidate_adds_fresh_confirmation_members_without_retuning(tmp_path):
    authority, candidates, _, freeze = setup_release(tmp_path)
    original_config = authority.configuration()
    first_catalog = tmp_path / "first-catalog.json"
    write_record(first_catalog, historical_catalog(tmp_path))
    first_plan = authority.freeze(**freeze, data_policy=first_catalog)
    evidence = execute_crossed(authority, first_plan["id"], allow_unsandboxed=True)
    approval = authority.decide(evidence["id"])
    assert approval["payload"]["result"] == "accepted"
    authority.activate(approval["id"])
    next_candidate, next_freeze = successor(tmp_path, authority, candidates[1], approval)
    with pytest.raises(StageError, match="missing exact task"):
        authority.freeze(**next_freeze)
    extended = read_record(first_catalog)
    extended["version"] = "next-fresh-families-v3"
    for item in read_record(Path(next_candidate["payload"]["workload"]["path"]))["tasks"]:
        task = item["task"]
        family = "fresh-" + task["task_id"]
        extended["family_aliases"][family] = family
        extended["memberships"].append(
            {
                "task_id": task["task_id"],
                "repository": item["repository"],
                "version": content_digest(task),
                "family_id": family,
                "partition": "sealed-release",
                "exposure": [],
            }
        )
    next_catalog = tmp_path / "next-catalog.json"
    write_record(next_catalog, extended)
    restarted = ReleaseAuthority(authority.store.root)
    issued = restarted.freeze(**next_freeze, data_policy=next_catalog)
    plan = restarted.plan(issued["id"])
    assert plan["data_policy"] == file_identity(next_catalog)
    assert file_identity(first_catalog) in plan["prior_data_policies"]
    assert len({row["family_id"] for row in plan["schedule"]}) == 6
    assert restarted.configuration() == original_config
    assert restarted.status()["candidate"] == candidates[1]["id"]
    first_catalog.write_bytes(first_catalog.read_bytes() + b" ")
    with pytest.raises(StageError, match="artifact bytes/path changed"):
        restarted.plan(issued["id"])
