"""Cycle inputs with actual upstream producers and explicit CPU boundary fixtures."""

from pathlib import Path

from release_support import setup_release

from admin.artifacts import read_record, write_record
from admin.cycles import CycleController
from admin.parents import ParentAuthority
from validator.store import RoundStore


def setup_cycle(root: Path):
    release, candidates, _, freeze = setup_release(root, profile="bf16", pool_families=18)
    controller = CycleController(release.store.root)
    parent = ParentAuthority(root / "bootstrap-parent").store.records(kind="release-decision")[0]
    config = {
        "replay": str(root / "replay"),
        "bootstrap_parent": {"root": str(root / "bootstrap-parent"), "id": parent["id"]},
        "github": {"repository": "example/spark", "credential_env": "GH_TOKEN"},
    }
    controller.configure(
        replay=Path(config["replay"]), bootstrap_parent=config["bootstrap_parent"], github=config["github"]
    )
    scripts = [read_record(root / f"serving-{i}.json") for i in range(2)]
    fixture = {
        "schema": "spark-cycle-serving-fixture-v1",
        "origin": controller.identity,
        "cells": {
            f"Q{a}{m}": scripts[m]["agents"][candidates[a]["payload"]["agent_id"]] for a in range(2) for m in range(2)
        },
    }
    write_record(root / "cycle-serving.json", fixture)
    admission = RoundStore(root / "source/rounds").load("r-1").admissions["alice"]
    spec = {
        "rounds": [
            {
                "source": "source",
                "round_id": "r-1",
                "prs": [
                    {"number": admission["pr_number"], "head": admission["head_sha"], "author": admission["author"]}
                ],
                "scorecards": str(root / "source/cards"),
                "episodes": str(root / "source/episodes"),
            }
        ],
        "agent": candidates[1]["payload"]["agent"]["path"],
        "workload": str(root / "workload.json"),
        "training": {"profile": "bf16", "sequence_len": 65536, "max_steps": 1, "execution": "fixture"},
        "curriculum": {"version": "spark-curriculum-v1", "count": 1, "min_families": 1, "max_per_family": 1},
        "evaluation": {
            **{k: freeze[k] for k in ("schedule", "budget", "sampling")},
            "fixture": str(root / "cycle-serving.json"),
        },
    }
    write_record(root / "cycle-config.json", config)
    write_record(root / "cycle-spec.json", spec)
    return controller, spec, candidates


def run_cycle(controller, spec, name="cycle-one", **kwargs):
    cycle = controller.start(name, spec)
    return controller.resume(cycle["id"], execute_training=True, allow_unsandboxed=True, **kwargs)
