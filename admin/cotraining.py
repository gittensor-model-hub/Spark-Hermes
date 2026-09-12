"""Freeze, execute and inspect a strict four-cell agent/model experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from admin.artifacts import read_record
from admin.evaluation import execute_crossed
from admin.release import ReleaseAuthority


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "show"))
    parser.add_argument("--root", required=True, type=Path, help="configured release authority root")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--allow-unsandboxed", action="store_true")
    args = parser.parse_args(argv)
    try:
        authority = ReleaseAuthority(args.root)
        if args.command == "freeze":
            config = read_record(args.config)
            if config.get("data_policy") is not None:
                config["data_policy"] = Path(config["data_policy"])
            result = authority.freeze(**config)
        elif args.command == "run":
            result = execute_crossed(authority, args.id, allow_unsandboxed=args.allow_unsandboxed)
        else:
            result = authority.evaluation(args.id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"cotraining: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
