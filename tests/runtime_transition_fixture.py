"""Installed producer driver; only external CPU task/serving inputs are scripted."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from admin.artifacts import content_digest, read_record, write_record
from admin.cycle_demo import NAMESPACE, SALT, _command, _first, _generated, _replay, _round, _spec
from admin.cycle_fixtures import confirmation_tasks, response_row
from admin.cycles import CycleController
from admin.runtime_transition import capture
from hermesbench.tasks import Task
from validator.persistence import state_identity

EXTRA = [
    ("decimal-sort", "Sort 31,7,12 numerically", "7 12 31", "printf '31\\n7\\n12\\n' | sort -n | xargs"),
    ("hex-decode", "Convert hex 2a to decimal", "42", "printf '%d' 0x2a"),
    (
        "json-field",
        "Read value of k from JSON",
        "17",
        'python -c \'import json; print(json.loads(chr(123)+chr(34)+"k"+chr(34)+":17"+chr(125))["k"])\'',
    ),
    (
        "leap-year",
        "Number of days in February 2024",
        "29",
        "python -c 'import calendar; print(calendar.monthrange(2024,2)[1])'",
    ),
    ("set-union", "Sorted union of ab and bc", "abc", "printf abbc | fold -w1 | sort -u | tr -d '\\n'"),
    ("base-two", "Convert binary 1011 to decimal", "11", "python -c 'print(int(\"1011\",2))'"),
    (
        "json-array",
        "Count members of a JSON array",
        "3",
        "python -c 'import json; print(len(json.loads(\"[1,2,3]\")))'",
    ),
    ("csv-sum", "Sum 10,20,30", "60", "printf '10\\n20\\n30\\n' | awk '{s+=$1} END{print s}'"),
    (
        "path-suffix",
        "Suffix of report.csv",
        ".csv",
        "python -c 'from pathlib import Path; print(Path(\"report.csv\").suffix)'",
    ),
    ("unicode-length", "Length of café", "4", "python -c 'print(len(\"café\"))'"),
    ("dict-sort", "Sort keys z,b,a", "abz", "printf zba | fold -w1 | sort | tr -d '\\n'"),
    ("integer-power", "Calculate 3 to power 4", "81", "python -c 'print(3**4)'"),
    ("reverse-lines", "Reverse order of lines red blue", "blue red", "printf 'red\\nblue\\n' | tac | xargs"),
    ("modulus", "Remainder of 47 divided by 9", "2", "python -c 'print(47%9)'"),
    ("word-count", "Count words in one two three four", "4", "printf 'one two three four' | wc -w | xargs"),
    (
        "date-order",
        "Sort 2024-02-01 and 2023-12-31",
        "2023-12-31 2024-02-01",
        "printf '2024-02-01\\n2023-12-31\\n' | sort | xargs",
    ),
    (
        "url-parse",
        "Get host of https://example.org/a",
        "example.org",
        "python -c 'from urllib.parse import urlparse; print(urlparse(\"https://example.org/a\").hostname)'",
    ),
    ("factorial", "Compute factorial of five", "120", "python -c 'import math; print(math.factorial(5))'"),
]


def tasks():
    result = confirmation_tasks()
    for family, prompt, answer, command in EXTRA:
        result.append(
            (
                Task(
                    task_id="confirm-" + family,
                    prompt=prompt + "; write only the answer to answer.txt.",
                    tools=("terminal",),
                    verify="test -s answer.txt",
                    hidden_verify=f'test "$(cat answer.txt)" = "{answer}"',
                    max_steps=4,
                ),
                family,
                command + " > answer.txt",
            )
        )
    return result


def inputs(root, number):
    selected = tasks()[(number - 1) * 6 : number * 6]
    write_record(
        root / f"workload-{number}.json",
        {
            "schema": "spark-crossed-workload-v1",
            "tasks": [{"task": asdict(t), "repository": "fixture/confirmation"} for t, _, _ in selected],
        },
    )
    return [
        {
            "task_id": t.task_id,
            "repository": "fixture/confirmation",
            "version": content_digest(asdict(t)),
            "family_id": "sealed-" + f,
            "partition": "sealed-release",
            "exposure": [],
        }
        for t, f, _ in selected
    ]


def cycle(root, controller, number, rounds, sealed, *, improving=True, through="activate", inherited=None):
    sealed = [*sealed, *inputs(root, number)]
    r = _round(
        root,
        "transition-round-" + str(number),
        _generated(root, "transition-task-" + str(number)),
        release_root=controller.root,
    )
    rounds = [*rounds, r]
    # Preserve all historical catalog knowledge, even when target replay has new sources.
    historical_members = []
    if inherited:
        for catalog in inherited["catalogs"]:
            p = read_record(Path(catalog["path"]))
            historical_members.extend(p["memberships"])
    members = {content_digest(m): m for m in [*sealed, *historical_members]}
    replay = _replay(root, number, rounds, list(members.values()))
    spec = _spec(root, controller, replay, r, number, history_ids=[])
    policy = read_record(Path(spec["confirmation_policy"]))
    if number == 3:
        policy["family_aliases"]["source-spent-alias"] = "sealed-decimal-sort"
    if inherited:
        for catalog in inherited["catalogs"]:
            policy["family_aliases"].update(read_record(Path(catalog["path"]))["family_aliases"])
    policy["version"] += "-confirmation"
    confirmation = root / f"confirmation-catalog-{number}.json"
    write_record(confirmation, policy)
    spec["confirmation_policy"] = str(confirmation)

    successes = ((2, 5), (4, 9)) if improving else ((9, 8), (8, 7))
    write_record(
        Path(spec["evaluation"]["fixture"]),
        {
            "schema": "spark-cycle-serving-fixture-v1",
            "origin": controller.identity,
            "cells": {
                f"Q{a}{m}": {
                    t.task_id: {
                        str(i): response_row(cmd if i < successes[a][m] else "printf wrong > answer.txt")
                        for i in range(10)
                    }
                    for t, _, cmd in tasks()[(number - 1) * 6 : number * 6]
                }
                for a in range(2)
                for m in range(2)
            },
        },
    )
    if not (controller.root / "driver-configured").exists():
        # New target has no cycle configuration; original source is already configured.
        with controller.store.connect() as db:
            configured = db.execute("SELECT 1 FROM cycle_configuration").fetchone()
        if not configured:
            controller.configure(replay=replay.root)
        (controller.root / "driver-configured").write_text("fixture setup marker\n")
    spec_path = root / f"transition-spec-{number}.json"
    write_record(spec_path, spec)
    started = _command(
        root, "cycle", "start", "--root", controller.root, "--name", "transition-" + str(number), "--spec", spec_path
    )
    result = _command(
        root,
        "cycle",
        "resume",
        "--root",
        controller.root,
        "--id",
        started["id"],
        "--through",
        through,
        "--execute-training",
        "--allow-unsandboxed",
        expected=0 if improving or through != "activate" else 3,
    )
    return result, rounds, sealed


def source(root):
    root.mkdir(parents=True)
    state_identity(root, mode="fixture", namespace=NAMESPACE)
    _first(root)
    progress = read_record(root / "demo-progress.json")
    controller = CycleController(root / "releases")
    first = _command(
        root,
        "cycle",
        "resume",
        "--root",
        controller.root,
        "--id",
        progress["first"],
        "--execute-training",
        "--allow-unsandboxed",
    )
    second, rounds, sealed = cycle(root, controller, 2, progress["rounds"], progress["sealed"])
    third, rounds, sealed = cycle(root, controller, 3, rounds, sealed, through="plan")
    retention = capture(controller.release)
    summary = {
        "first": first,
        "second": second,
        "interrupted": third,
        "retention": retention,
        "rounds": rounds,
        "sealed": sealed,
    }
    write_record(root / "transition-source.json", summary)
    return summary


def continued(root, target):
    root.mkdir(parents=True, exist_ok=True)
    state_identity(root, mode="fixture", namespace=NAMESPACE)
    controller = CycleController(target)
    imported = controller.active_pair()
    history = controller.release.inherited_history()
    accepted, rounds, sealed = cycle(root, controller, 4, [], [], inherited=history)
    failed, _, _ = cycle(root, controller, 5, rounds, sealed, improving=False, inherited=history)
    result = {"imported": imported, "accepted": accepted, "failed": failed, "active": controller.active_pair()}
    assert accepted["status"] == "complete" and failed["status"] == "refused"
    assert accepted["incumbent"] == failed["incumbent"]
    write_record(root / "transition-target.json", result)
    return result


def reserve(root):
    """An actual unexecuted new-family plan under the retained source runtime."""
    from admin.candidates import CandidateStore
    from admin.pipeline import Workspace
    from admin.release import ReleaseAuthority

    summary = read_record(root / "transition-source.json")
    authority = ReleaseAuthority(root / "releases")
    plan_id = next(j["output"]["value"]["id"] for j in summary["interrupted"]["jobs"] if j["stage"] == "plan")
    plan = authority.plan(plan_id)
    previous = authority.candidates().resolve(plan["new"])["payload"]
    new_members = inputs(root, 4)
    policy = read_record(Path(plan["data_policy"]["path"]))
    policy["version"] += "-reservation-extension"
    policy["memberships"].extend(new_members)
    policy["family_aliases"].update({m["family_id"]: m["family_id"] for m in new_members})
    policy["family_aliases"]["renamed-spent-family"] = plan["schedule"][0]["family_id"]
    path = root / "reservation-extension.json"
    write_record(path, policy)
    candidate = CandidateStore(authority.candidates().store.root).register(
        workspace=Workspace(Path(previous["workspace"])),
        merged_record=Path(previous["model"]["record"]),
        agent=Path(previous["agent"]["path"]),
        workload=root / "workload-4.json",
        parent=Path(previous["parent"]["path"]),
    )
    issued = authority.freeze(
        old=plan["old"],
        new=candidate["id"],
        schedule=[{"attempt_id": str(i), "seed": i} for i in range(10)],
        budget=plan["budget"],
        sampling=plan["sampling"],
        serving=plan["serving"],
        data_policy=path,
    )
    write_record(root / "new-reservation.json", issued)
    return issued


def reuse(root, target):
    """Attempt a renamed/aliased historical family after real continued cycles."""
    from admin.artifacts import StageError
    from admin.candidates import CandidateStore
    from admin.pipeline import Workspace
    from admin.release import ReleaseAuthority

    authority = ReleaseAuthority(target)
    result = read_record(root / "transition-target.json")
    failed = result["failed"]
    plan_id = next(j["output"]["value"]["id"] for j in failed["jobs"] if j["stage"] == "plan")
    plan = authority.plan(plan_id)
    previous = authority.candidates().resolve(plan["new"])["payload"]
    policy = read_record(Path(plan["data_policy"]["path"]))
    workload = {"schema": "spark-crossed-workload-v1", "tasks": []}
    for index, (task, family, _) in enumerate(tasks()[12:18]):
        record = asdict(task)
        record["task_id"] = "renamed-" + record["task_id"]
        workload["tasks"].append({"task": record, "repository": "fixture/renamed-confirmation"})
        policy["memberships"].append(
            {
                "task_id": record["task_id"],
                "repository": "fixture/renamed-confirmation",
                "version": content_digest(record),
                "family_id": "source-spent-alias" if index == 0 else "sealed-" + family,
                "partition": "sealed-release",
                "exposure": [],
            }
        )
    policy["version"] += "-renamed-attempt"
    workload_path, policy_path = root / "renamed-workload.json", root / "renamed-catalog.json"
    write_record(workload_path, workload)
    write_record(policy_path, policy)
    candidate = CandidateStore(authority.candidates().store.root).register(
        workspace=Workspace(Path(previous["workspace"])),
        merged_record=Path(previous["model"]["record"]),
        agent=Path(previous["agent"]["path"]),
        workload=workload_path,
        parent=Path(previous["parent"]["path"]),
    )
    before = authority.status()
    try:
        authority.freeze(
            old=before["candidate"],
            new=candidate["id"],
            schedule=[{"attempt_id": str(i), "seed": i} for i in range(10)],
            budget=plan["budget"],
            sampling=plan["sampling"],
            serving=plan["serving"],
            data_policy=policy_path,
        )
    except StageError as exc:
        if "already used" not in str(exc):
            raise
        report = {
            "refused": True,
            "reason": str(exc),
            "unchanged": authority.status() == before,
            "old_plan": plan_id,
            "candidate": candidate["id"],
            "workload": str(workload_path),
            "catalog": str(policy_path),
        }
        write_record(root / "renamed-reuse.json", report)
        assert report["unchanged"]
        return report
    raise AssertionError("renamed historical confirmation families were accepted")


def quiescence(root):
    """Submit an actual next training job, then simulate losing its controller."""
    summary = read_record(root / "transition-source.json")
    controller = CycleController(root / "releases")
    prepared, _, _ = cycle(root, controller, 4, summary["rounds"], summary["sealed"], through="prepare")

    def stop(boundary):
        if boundary == "train:after_submit":
            raise RuntimeError("fixture controller disappeared after durable submission")

    try:
        controller.resume(prepared["id"], through="train", execute_training=True, hook=stop)
    except RuntimeError as exc:
        if "controller disappeared" not in str(exc):
            raise
    else:
        raise AssertionError("controller submission boundary was not reached")
    with controller.store.connect() as db:
        jobs = list(db.execute("SELECT id,status FROM cycle_external_jobs WHERE status='submitted'"))
    assert len(jobs) == 1
    result = {"cycle": prepared["id"], "job": jobs[0][0], "status": jobs[0][1]}
    write_record(root / "quiescence-submission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("source", "continued", "reserve", "reuse", "quiescence"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    os.environ["HERMESBENCH_WITHHELD_SALT"] = SALT
    result = (
        source(args.root)
        if args.command == "source"
        else quiescence(args.root)
        if args.command == "quiescence"
        else reserve(args.root)
        if args.command == "reserve"
        else reuse(args.root, args.target)
        if args.command == "reuse"
        else continued(args.root, args.target)
    )
    print("installed actual producer fixture complete", args.command)
