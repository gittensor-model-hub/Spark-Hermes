"""Explicit CPU external-boundary inputs for the installed cycle demonstration.

These are scripted GitHub, synthesis, tokenizer, checkpoint and serving inputs.
They cannot issue admission, corpus, training completion or release authority.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from admin.artifacts import StageError, content_digest, write_record
from hermesbench.tasks import Task


def require_fixture(identity: dict[str, str]) -> None:
    if identity.get("mode") != "fixture":
        raise StageError("CPU demonstration adapters require an immutable fixture namespace")


class GitHubFixture:
    """Authenticated response bytes consumed by GitHubSource.collect; no network."""

    def __init__(self, identity, assignment, commitment, log: Path):
        require_fixture(identity)
        self.assignment, self.commitment, self.log = assignment, commitment, log
        self.author = commitment["miner_id"]
        self.number = 7 if self.author == "alice" else 8
        self.head, self.base = ("1" if self.author == "alice" else "3") * 40, "2" * 40

    def __call__(self, argv, **kwargs):
        if argv[:6] != ["gh", "api", "--hostname", "github.com", "--method", "GET"]:
            raise StageError("fixture permits only metadata GET requests")
        endpoint = argv[-1]
        if endpoint == "user":
            value = {"login": "fixture-validator"}
        elif "/files?" in endpoint:
            value = [{"filename": "datasets/strategies.jsonl", "status": "modified"}]
        elif "/contents/" in endpoint:
            path = endpoint.split("/contents/")[1].split("?")[0]
            body = (
                json.dumps(self.assignment)
                if "/rounds/" in path
                else ""
                if endpoint.endswith(self.base)
                else json.dumps(self.commitment)
            )
            raw = body.encode()
            value = {
                "type": "file",
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(raw).decode(),
                "sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
            }
        else:
            value = {
                "number": self.number,
                "user": {"login": self.author},
                "head": {"sha": self.head, "repo": {"full_name": "fixture/spark"}},
                "base": {"sha": self.base, "repo": {"full_name": "fixture/spark"}},
                "state": "open",
                "draft": False,
                "merged": False,
                "changed_files": 1,
            }
        with self.log.open("a") as stream:
            stream.write(json.dumps({"fixture_only": True, "argv": argv, "response": value, "exit_code": 0}) + "\n")
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")


class SynthesisFixture:
    def __init__(self, identity, log: Path, label: str):
        require_fixture(identity)
        self.log, self.label = log, label

    def __call__(self, messages, **kwargs):
        # The public check catches missing output; withheld execution checks its content.
        # `input.txt` is protected because it is the INPUT the answer is copied from: an agent free
        # to rewrite it could make `cmp` pass by editing the source instead of producing the answer.
        # `answer.txt` is deliberately not protected -- writing it is the task.
        response = f"""=== PROMPT ===
Copy input.txt exactly to answer.txt for the {self.label} task.
=== SETUP ===
printf 'four\\n' > input.txt
=== VERIFY ===
test -s answer.txt
=== WITHHELD ===
cmp input.txt answer.txt
=== REFERENCE ===
cp input.txt answer.txt
=== ALTERNATE ===
cat input.txt > answer.txt
=== CHEAT ===
printf 'wrong\\n' > answer.txt
=== PROTECTED ===
input.txt
"""
        write_record(self.log, {"fixture_only": True, "messages": messages, "response": response})
        return response, {"prompt_tokens": 1, "completion_tokens": 1}


def response_row(command: str, *, tokens: int = 25) -> dict[str, Any]:
    return {
        "responses": [
            "<tool_call>\n<function=terminal>\n<parameter=command>\n"
            + command
            + "\n</parameter>\n</function>\n</tool_call>",
            "CPU serving fixture complete.",
        ],
        "prompt_tokens": tokens,
        "completion_tokens": tokens,
        "latency": 1.0,
    }


def competition_serving(path: Path, *, identity, model_id, agent_id, task_id, successes, tokens):
    require_fixture(identity)
    write_record(
        path,
        {
            "schema": "spark-serving-fixture-v1",
            "origin": identity,
            "model_id": model_id,
            "agents": {
                agent_id: {
                    task_id: {
                        str(i): response_row(
                            "cp input.txt answer.txt" if i < successes else "printf 'wrong\\n' > answer.txt",
                            tokens=tokens,
                        )
                        for i in range(10)
                    }
                }
            },
        },
    )


def confirmation_tasks():
    # Independent fixture task families with different executed objectives. Public and
    # private check bodies are inputs, not evaluation scores or an altered oracle.
    definitions = [
        ("arithmetic", "Compute 7 times 8", "56", "printf '56'"),
        (
            "ordering",
            "Sort the letters c a b ascending, without spaces",
            "abc",
            "printf 'c\\na\\nb\\n' | sort | tr -d '\\n'",
        ),
        ("reverse", "Reverse the text drawer", "reward", "printf drawer | rev"),
        ("filter", "Remove digits from a1b2c3", "abc", "printf a1b2c3 | tr -d '0-9'"),
        ("count", "Count characters in planet", "6", "printf planet | wc -c | tr -d ' '"),
        ("maximum", "Find the largest of 3, 9, 2", "9", "printf '3\\n9\\n2\\n' | sort -n | tail -1 | tr -d '\\n'"),
        ("deduplicate", "Deduplicate adjacent letters in aaabbc", "abc", "printf aaabbc | tr -s abc"),
        ("basename", "Extract the basename from /var/log/a.txt", "a.txt", "basename /var/log/a.txt | tr -d '\\n'"),
        ("case", "Lowercase the text MiXeD", "mixed", "printf MiXeD | tr 'A-Z' 'a-z'"),
        ("remainder", "Compute the remainder of 17 divided by 5", "2", "printf '%s' $((17 % 5))"),
        ("csv", "Extract field two from x,y,z", "y", "printf x,y,z | cut -d, -f2 | tr -d '\\n'"),
        ("encoding", "Decode the base64 text b2s=", "ok", "printf b2s= | base64 -d"),
    ]
    result = []
    for family, prompt, answer, command in definitions:
        task = Task(
            task_id="confirm-" + family,
            prompt=prompt + "; write only the answer to answer.txt.",
            tools=("terminal",),
            verify="test -s answer.txt",
            hidden_verify=f'test "$(cat answer.txt)" = "{answer}"',
            max_steps=4,
        )
        result.append((task, family, command + " > answer.txt"))
    return result


def workloads(root: Path, identity):
    require_fixture(identity)
    members = []
    for index in range(2):
        subset = confirmation_tasks()[index * 6 : (index + 1) * 6]
        write_record(
            root / f"workload-{index + 1}.json",
            {
                "schema": "spark-crossed-workload-v1",
                "tasks": [{"task": asdict(t), "repository": "fixture/confirmation"} for t, _, _ in subset],
            },
        )
        for task, family, _ in subset:
            members.append(
                {
                    "task_id": task.task_id,
                    "repository": "fixture/confirmation",
                    "version": content_digest(asdict(task)),
                    "family_id": "sealed-" + family,
                    "partition": "sealed-release",
                    "exposure": [],
                }
            )
    return members


def matrix_serving(path: Path, identity, *, cycle: int):
    require_fixture(identity)
    successes = ((2, 5), (4, 9)) if cycle == 1 else ((9, 8), (8, 7))
    write_record(
        path,
        {
            "schema": "spark-cycle-serving-fixture-v1",
            "origin": identity,
            "cells": {
                f"Q{a}{m}": {
                    task.task_id: {
                        str(i): response_row(command if i < successes[a][m] else "printf wrong > answer.txt")
                        for i in range(10)
                    }
                    for task, _, command in confirmation_tasks()[(cycle - 1) * 6 : cycle * 6]
                }
                for a in range(2)
                for m in range(2)
            },
        },
    )
