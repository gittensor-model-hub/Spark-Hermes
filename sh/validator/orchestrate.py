"""The round loop (spec §9.1) — every stage, unattended, forever.

    open -> window (2 h) -> seal -> evaluate -> close -> crown -> announce -> export -> publish -> next, at once

Tasks are minted ahead by the private queue daemon (`supply.queue`), so a round opens the instant the previous
one closes. Opening publishes the round's tasks and its **submission window**; miners fetch the tasks with the
CLI and submit one signed pull request per hotkey, replacing it as often as they like until the window closes.
The seal takes every strategy PR at its head SHA at that instant, verifies the hotkey's signature over this
round and this digest, evaluates, and pays on the pooled window while crowning on this round alone: the
crowned PR is merged as the incumbent, every other competition PR is closed with the reason, and a PR that
arrives after the seal is closed as outside the window.

Runs wherever the validator's credentials live; the GPU worker is reached over ssh and holds none. Everything
a miner is judged by is published under `rounds/<id>/` and mirrored into `docs/live/live.json` for the board.

Transparency rules the loop enforces, each because its absence would let a validator cheat quietly:

  * the window's open and close times are published with the tasks, before any submission exists;
  * a bundle is sealed by PR head SHA and digest before any evaluation, and the seal is published;
  * the withheld half is committed to in the published task and revealed at close with its salt;
  * future rounds exist only as digests (`rounds/queue.json`) until they open — verifiable, unreadable;
  * the crown is recomputable from the published episodes; the weights from the published close.json.

    python -m sh.validator.orchestrate --queue ../Spark-Hermes-Withheld/queue
    python -m sh.validator.orchestrate --queue ... --mock-miners DIR --mock-keys DIR   # test challengers
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sh.cli import attest
from sh.cli.lint import check_files, collect
from sh.cli.scorecard import render as render_scorecard
from sh.exports.build import build as build_exports
from sh.exports.upload import upload as upload_exports
from sh.scoring.crown import crown as crown_rule
from sh.validator import similarity
from sh.validator.round import close as close_round
from sh.validator.stats import load_episodes
from sh.web.build import render as render_leaderboard

REPO = "gittensor-model-hub/Spark-Hermes"
BRANCH = "main"
LABEL_STRATEGY, LABEL_SCORED = "sh:strategy", "sh:round:scored"
# A hotkey directory name is an ss58 address: base58, 47–48 chars. Everything downstream trusts it as a directory
# name and as a shell word in the worker's `--surfaces` argument; a name like `$(...)` or `../x` must never reach
# there. Anything else in submissions/ is not a submission.
SS58 = re.compile(r"\A[1-9A-HJ-NP-Za-km-z]{47,48}\Z")
# A digest out of a miner's attestation names a directory in the private store. It is attacker-controlled, so it
# is matched strictly before it is ever joined to a path: "../.." must never reach the store lookup.
DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")
_UNRESOLVED: set[tuple[str, str]] = set()  # incumbents already reported unresolved, so it is said once
HF_REPO = "gittensor-model-hub/spark-hermes-rounds"
LIVE = "docs/live/live.json"  # what the dashboard polls; committed on every stage change
# The board calls the loop stale once `updated` is 900 s old (docs/live/index.html). A board whose content has not
# changed is still republished this often, so a quiet window never reads as a dead loop.
BOARD_HEARTBEAT_S = 600
CLAIM_GRACE_S = 30  # after claiming the engine, how long a screen the daemon had just started is given to show up
STAGES = ("open", "window", "seal", "evaluate", "close", "crown", "announce", "export", "publish_close", "done")


@dataclass
class Config:
    """The control plane runs wherever the validator's credentials live; the GPU worker is reached over ssh and
    holds no credentials at all — it receives a round directory, runs episodes, and hands the episodes back."""

    state: Path  # durable validator state: rounds/, archive/, salt secret
    repo: Path  # a checkout of BRANCH the loop commits to
    queue: Path  # the private queue daemon's directory of minted rounds
    pkg: Path  # the public package's parent
    worker: str = "root@162.156.217.154"
    worker_port: int = 40301
    worker_root: str = "/root/sh"  # holds pkg/ (the public package), state/tokens, state/usage
    image: str = "hermes-ubuntu:pin"  # the fallback; an image-defined task names its own
    window: int = 8  # rounds pooled for payment
    window_from: str = "r0001"  # the first round pooled; every round since launch is scored on credit (sh-scoring-v3)
    window_s: int = 2 * 3600  # the submission window
    min_paired: int = 4  # instances a strategy must share with the baseline to be crowned
    concurrency: int = 2
    era: str = "e0"
    canon_every: int = 8  # the reference strategy runs in every n-th round (calibration); 1 = every round
    submit_server: str = ""  # advertised to miners on the board; set it (r0002+) to switch submissions to commit–reveal

    @property
    def rounds(self) -> Path:
        return self.state / "rounds"


def sh(
    cmd: list[str], *, cwd: Path | None = None, env: dict | None = None, check: bool = True, want_err: bool = False
) -> str:
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:4])}… exited {r.returncode}: {r.stderr[-800:]}")
    return (r.stdout + r.stderr) if want_err and r.returncode else r.stdout


def gh(*args: str) -> str:
    return sh(["gh", *args])


def _label(number: int, name: str) -> None:
    """Add a label to a PR, creating the label if the repository has never seen it; idempotent."""
    sh(["gh", "label", "create", name, "--repo", REPO, "--color", "5319e7", "--force"], check=False)
    sh(["gh", "api", "-X", "POST", f"repos/{REPO}/issues/{number}/labels", "-f", f"labels[]={name}"], check=False)


def _worker(cfg: Config, cmd: str) -> str:
    return sh(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-p", str(cfg.worker_port), cfg.worker, cmd],
        check=False,
    )


def _worker_launch(cfg: Config, cmd: str) -> None:
    """Start a long-running command on the worker and come back at once. `-f` backgrounds ssh after
    authentication and `-n` detaches stdin; the polling loop, not this call, decides whether the job runs."""
    try:
        subprocess.run(
            ["ssh", "-f", "-n", "-o", "BatchMode=yes", "-p", str(cfg.worker_port), cfg.worker, cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass


def _rsync(src: str, dst: str, cfg: Config) -> None:
    sh(["rsync", "-az", "--delete", "-e", f"ssh -o BatchMode=yes -p {cfg.worker_port}", src, dst])


# ─── round bookkeeping ─────────────────────────────────────────────────────────────────────────────
def log(rd: Path, stage: str, **fields) -> None:
    rec = {"t": time.time(), "stage": stage, **fields}
    with (rd / "phases.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rd.name}] {stage} {json.dumps(fields)[:200]}", flush=True)


def _commit(cfg: Config, message: str, paths: tuple[str, ...] = ("rounds", "docs/live", "docs/rounds")) -> None:
    paths = tuple(p for p in paths if (cfg.repo / p).exists())  # a fresh checkout has no docs/rounds yet
    if not paths:
        return
    branch = sh(["git", "symbolic-ref", "--short", "-q", "HEAD"], cwd=cfg.repo, check=False).strip()
    if branch != BRANCH:  # the loop commits only on main, in its own clone: never move a developer's branch
        raise RuntimeError(f"the validator checkout is on {branch!r}, not {BRANCH}; refusing to commit")
    sh(["git", "add", *paths], cwd=cfg.repo)
    if sh(["git", "status", "--porcelain", *paths], cwd=cfg.repo).strip():
        sh(["git", "commit", "-q", "-m", message, "--", *paths], cwd=cfg.repo)  # only these paths
        if sh(
            ["git", "pull", "--rebase", "--autostash", "origin", BRANCH], cwd=cfg.repo, check=False, want_err=True
        ).strip():
            # a rebase that could not apply cleanly (a conflict on live.json) leaves the tree mid-rebase and every
            # later parse of it fails: abort, so the next commit retries from a clean HEAD ahead of origin
            if (cfg.repo / ".git" / "rebase-merge").exists() or (cfg.repo / ".git" / "rebase-apply").exists():
                sh(["git", "rebase", "--abort"], cwd=cfg.repo, check=False)
                raise RuntimeError("the validator checkout could not rebase onto origin; aborted, will retry")
        sh(["git", "push", "-q", "origin", f"HEAD:{BRANCH}"], cwd=cfg.repo)


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _trim_scores(closed: dict) -> dict:
    """Per-hotkey score, weight and the paired deltas from a close record — what the board and history show."""
    weights = closed.get("weights", {})
    return {
        h: {
            "score": s.get("score"),
            "weight": weights.get(h, 0.0),
            "mean_d": s.get("mean_d"),
            "delta_c": s.get("delta_c"),
            "n": s.get("n"),
        }
        for h, s in closed.get("scores", {}).items()
    }


def live(
    cfg: Config,
    rd: Path,
    stage: str,
    *,
    progress: dict | None = None,
    submissions: list[dict] | None = None,
    push: bool = True,
) -> None:
    """The dashboard's single source: the round, its window, its stage and progress, the crown and scores once
    it closes, the queue of rounds waiting unread, and the history of closed rounds."""
    phases = (
        [json.loads(line) for line in (rd / "phases.jsonl").read_text().splitlines()]
        if (rd / "phases.jsonl").exists()
        else []
    )
    sealed = _read(rd / "seal.json", {})
    history = _read(cfg.repo / "rounds" / "index.json", {"rounds": []})["rounds"]
    closed = _read(rd / "close" / "close.json")
    crowned = _read(rd / "close" / "crown.json")
    prev = _read(cfg.repo / LIVE, {})
    same = prev.get("round_id") == rd.name
    if progress is None:  # stages after evaluation carry the counts forward rather than blanking the board
        progress = prev.get("progress") if same else None
    if submissions is None and same:
        submissions = prev.get("submissions")
    # Who each hotkey is on GitHub, accumulated across rounds — but only from SEALED entries, whose attestation binds
    # the hotkey to the bundle the PR author signed. An open PR is unverified: anyone can open one touching
    # submissions/<victim>/, and its opener must never be shown, let alone published, as the victim's identity.
    github = dict(prev.get("github") or {})
    for hk, info in sealed.get("active", {}).items():
        if info.get("github"):
            github[hk] = info["github"]
    last = history[-1] if history else None
    scores = None
    if closed:
        weights = closed.get("weights", {})
        scores = {
            "weights": weights,
            "king": crowned.get("king") if crowned else None,
            "per_hotkey": {
                h: {
                    k: s.get(k)
                    for k in ("n", "score", "mean_d", "se", "delta_c", "gate", "overfit_rate", "dq", "reason")
                }
                for h, s in closed.get("scores", {}).items()
            },
            "family_stats": {
                f: {
                    "null_p": r["null"].get("p"),
                    "canon_p": r["canon"].get("p"),
                    "null_credit": r["null"].get("mean_credit"),
                    "canon_credit": r["canon"].get("mean_credit"),
                    "label": r.get("label"),
                }
                for f, r in closed.get("family_stats", {}).items()
            },
            "commitments_ok": closed.get("commitments_ok"),
        }
    state = {
        "schema": "sh-live-v3",
        "updated": time.time(),
        "round_id": rd.name,
        "stage": stage,
        "stages": list(STAGES),
        "started": phases[0]["t"] if phases else time.time(),
        "phases": [{"stage": p["stage"], "t": p["t"]} for p in phases],
        "window": _read(rd / "window.json"),
        "submissions": submissions or [],
        "github": github,  # hotkey -> GitHub login; the board shows the avatar and name beside the hotkey
        "standings": (last or {}).get("scores") or {},  # the pooled standing after the last close: what pays now
        "last_round": (last or {}).get("round_id"),
        "tasks": len(list(shown(rd).glob("*.json"))) if (rd / "tasks").exists() else 0,
        "active": sealed.get("active", {}),
        "rejected": sealed.get("rejected", {}),
        "progress": progress or {},
        "crown": crowned,
        "scores": scores,
        "queue": _read(cfg.repo / "rounds" / "queue.json", {}).get("ready", []),
        "history": history[-20:],
        "repo": REPO,
        "branch": BRANCH,
        "hf_repo": HF_REPO,
        "submit_server": cfg.submit_server or None,  # where the CLI uploads the prose; None = legacy (prose in the PR)
    }
    # Only the clock moved: the loop republishes the board on a timer, and while it waits for the daemon that is
    # every tick. Writing would make `git status` dirty and push a commit, so an idle loop filled the public
    # history with identical "live — waiting" commits (25 in six hours, once). Say nothing rather than that — but
    # only for BOARD_HEARTBEAT_S: the board reads an old `updated` as a dead loop, and a quiet window is not one.
    unchanged = prev and {k: v for k, v in prev.items() if k != "updated"} == {
        k: v for k, v in state.items() if k != "updated"
    }
    if unchanged and state["updated"] - float(prev.get("updated") or 0) < BOARD_HEARTBEAT_S:
        return
    out = cfg.repo / LIVE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=1))
    if push:
        _commit(cfg, f"{rd.name}: live — {stage}", paths=("docs/live",))


# ─── stages ────────────────────────────────────────────────────────────────────────────────────────
def queue_ready(cfg: Config) -> list[str]:
    if not cfg.queue.exists():
        return []
    return sorted(p.name for p in cfg.queue.iterdir() if p.is_dir() and (p / "READY").exists())


def open_round(cfg: Config) -> Path:
    """Take the lowest ready round from the queue and give it a window. Waits — visibly, on the board — while
    the queue is empty. Publishing is a separate, resumable step."""
    waited = 0
    while not queue_ready(cfg):
        if waited % 300 == 0:  # the board shows the wait rather than going stale on the last round's "done"
            print("[loop] queue empty; waiting for the daemon", flush=True)
            previous = sorted(p for p in cfg.rounds.glob("r*") if p.is_dir()) if cfg.rounds.exists() else []
            if previous:
                live(cfg, previous[-1], "waiting")
        time.sleep(60)
        waited += 60
    round_id = queue_ready(cfg)[0]
    rd = cfg.rounds / round_id
    cfg.rounds.mkdir(parents=True, exist_ok=True)
    # The window is written before the move: a round directory, once it exists, always has its window.json, so
    # a restart between the two can resume it (a kill there used to leave a round that no resume could publish).
    now = time.time()
    (cfg.queue / round_id / "window.json").write_text(
        json.dumps({"opens_at": now, "closes_at": now + cfg.window_s, "seconds": cfg.window_s})
    )
    shutil.move(str(cfg.queue / round_id), str(rd))  # consumed: the daemon refills
    (rd / "READY").unlink(missing_ok=True)
    # The task images were built on the worker at mint and are removed there once the round is evaluated.
    log(rd, "start")
    return rd


def shown(rd: Path) -> Path:
    """What miners get when the window opens: the round's tasks, or — for a family that evaluates on hidden bugs
    (swe_fix) — their previews, sibling bugs from the same repositories. The evaluated tasks follow at close."""
    return rd / "preview" if any((rd / "preview").glob("*.json")) else rd / "tasks"


def publish_round(cfg: Config, rd: Path) -> None:
    """The tasks, the window, the queue digests, and each task image's tag and attribution, committed where miners
    can read them. Not the Dockerfile or the fixture tree: both can fingerprint the upstream task."""
    round_id = rd.name
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(shown(rd), dest / "tasks", dirs_exist_ok=True)
    shutil.copy(rd / "window.json", dest / "window.json")
    for ctx in sorted((rd / "images").glob("*/")) if (rd / "images").exists() else []:
        pub = dest / "images" / ctx.name
        pub.mkdir(parents=True, exist_ok=True)
        for name in ("TAG", "SOURCE.json"):  # not the Dockerfile: its build steps can fingerprint the upstream task
            if (ctx / name).exists():
                shutil.copy(ctx / name, pub / name)
    # The daemon rewrites its index only after its next mint, so drop the round just taken and anything not READY.
    waiting = set(queue_ready(cfg)) - {round_id}
    ready = [r for r in _read(cfg.queue / "index.json", {"ready": []}).get("ready", []) if r["round_id"] in waiting]
    (cfg.repo / "rounds" / "queue.json").write_text(
        json.dumps({"schema": "sh-queue-v1", "published_at": time.time(), "ready": ready}, indent=1)
    )
    live(cfg, rd, "window", progress={}, submissions=[], push=False)
    n = len(list(shown(rd).glob("*.json")))
    _commit(cfg, f"{round_id}: open — {n} tasks, window {cfg.window_s // 60} min")
    log(rd, "open", tasks=n, closes_at=_read(rd / "window.json")["closes_at"])


def _image_tags(rd: Path) -> list[str]:
    return [p.read_text().strip() for p in sorted((rd / "images").glob("*/TAG"))] if (rd / "images").exists() else []


def reopened(window: dict, seconds: int, now: float) -> dict:
    """The next window for a round whose window closed with no submission: same round, same tasks, a fresh
    window of the same length. Nothing was sealed, evaluated or revealed, so the tasks are still fair. Pure."""
    n = int(window.get("reopened", 0)) + 1
    return {
        "opens_at": now,
        "closes_at": now + seconds,
        "seconds": seconds,
        "reopened": n,
        "first_opened_at": window.get("first_opened_at", window["opens_at"]),
        "reason": f"no submissions in window {n}",
    }


def window_state(rd: Path, now: float | None = None) -> dict:
    """Where the round is in its window: seconds left, and whether submissions are open."""
    w = _read(rd / "window.json")
    now = now if now is not None else time.time()
    if not w:
        return {"open": False, "remaining": 0}
    return {"open": now < w["closes_at"], "remaining": max(0.0, w["closes_at"] - now), "closes_at": w["closes_at"]}


def _submissions(cfg: Config) -> list[dict]:
    """Open strategy PRs as the board lists them during the window."""
    return [
        {
            "pr": p["number"],
            "hotkey": p["changed"][0],
            "created_at": p.get("createdAt"),
            "updated_at": p.get("updatedAt"),
            "url": p.get("url"),
        }
        for p in _strategy_prs(cfg, f"origin/{BRANCH}")
        if pr_role(p["changed"]) == "strategy"
    ]


def wait_window(cfg: Config, rd: Path, mock: tuple[Path, Path] | None = None) -> None:
    """Hold the round open until the window closes, showing the submissions on the board. Mock challengers
    submit once, at the start of the round's first window, signed for this round, like any miner would."""
    if mock and "mock_submit" not in done_stages(rd):
        from sh.cli.mock_miners import open_prs

        opened = open_prs(mock[0], mock[1], REPO, BRANCH, cfg.repo, rd.name)
        log(rd, "mock_submit", opened=opened)
    last_push = 0.0
    while True:
        st = window_state(rd)
        if not st["open"]:
            break
        if time.time() - last_push > 120:
            live(cfg, rd, "window", submissions=_submissions(cfg))
            last_push = time.time()
        time.sleep(min(60.0, max(1.0, st["remaining"])))


def reopen_window(cfg: Config, rd: Path) -> dict:
    """Give the round a fresh window, publish it, and say why on the board and in the round's history."""
    w = reopened(_read(rd / "window.json"), cfg.window_s, time.time())
    (rd / "window.json").write_text(json.dumps(w))
    dest = cfg.repo / "rounds" / rd.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(rd / "window.json", dest / "window.json")
    live(cfg, rd, "window", submissions=[], push=False)
    _commit(
        cfg, f"{rd.name}: no submissions — window reopened ({w['reopened'] + 1}), closes in {cfg.window_s // 60} min"
    )
    log(rd, "reopen", window=w["reopened"] + 1, closes_at=w["closes_at"])
    return w


def _check_bundle_dir(dest: Path, hotkey: str, round_id: str | None) -> dict:
    """Lint an already-materialised bundle directory — the shared tail of the tree and store paths."""
    files, problems = collect(dest)
    result = check_files(files, problems, hotkey=hotkey, round_id=round_id, require_attestation=round_id is not None)
    return {"problems": result["problems"], "digest": result["bundle_sha256"], "attestation": result["attestation"]}


def _bundle_from_tree(cfg: Config, ref: str, hotkey: str, dest: Path, *, round_id: str | None) -> dict | None:
    """Materialise `submissions/<hotkey>/` as of `ref` into `dest` and lint it; None if there is nothing there.
    With `round_id`, the bundle must carry a valid attestation for that round (a challenger); without, it is an
    incumbent whose attestation was checked when it was sealed."""
    listing = sh(
        ["git", "ls-tree", "-r", "-z", "--name-only", ref, f"submissions/{hotkey}/"], cwd=cfg.repo, check=False
    )
    prefix = f"submissions/{hotkey}/"
    rels = [r for r in listing.split("\0") if r.startswith(prefix)]  # -z: a filename with spaces or newlines is one
    if not rels:
        return None
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for rel in rels:
        out = dest / rel[len(prefix) :]
        if not str(out.resolve()).startswith(str(dest.resolve()) + os.sep):  # a name climbing out of the bundle
            return {"problems": [f"path escapes the bundle: {rel[:60]}"], "digest": "", "attestation": None}
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=cfg.repo, capture_output=True)  # bytes, not text
        out.write_bytes(blob.stdout)
    return _check_bundle_dir(dest, hotkey, round_id)


def _tree_names(cfg: Config, ref: str, hotkey: str) -> list[str]:
    """The file names a PR carries under `submissions/<hotkey>/` at `ref`."""
    prefix = f"submissions/{hotkey}/"
    listing = sh(["git", "ls-tree", "-r", "-z", "--name-only", ref, prefix], cwd=cfg.repo, check=False)
    return [r[len(prefix) :] for r in listing.split("\0") if r.startswith(prefix)]


# All a private-mode PR may add under its `submissions/<hotkey>/`: the signed commitment, and the server's receipt.
PR_FILES = ("attestation.json", "receipt.json")


def _prose_added(cfg: Config, base: str, head: str, hotkey: str) -> list[str]:
    """What a PR adds or changes under `submissions/<hotkey>/` besides its commitment (three-dot, so only the PR's own
    changes; deletions excluded — a king removing its old, already public prose adds none). Raises if git fails."""
    prefix = f"submissions/{hotkey}/"
    names = sh(["git", "diff", "--name-only", "-z", "--diff-filter=d", f"{base}...{head}", "--", prefix], cwd=cfg.repo)
    return [p[len(prefix) :] for p in names.split("\0") if p.startswith(prefix) and p[len(prefix) :] not in PR_FILES]


def _reveal_challenger(
    cfg: Config,
    ref: str,
    hotkey: str,
    dest: Path,
    *,
    round_id: str,
    store: Path,
    base: str = f"origin/{BRANCH}",
    private_only: bool = False,
    defending: tuple[str, Path] | None = None,
) -> dict | None:
    """A challenger's bundle for the seal. The PR carries the signed commitment (`attestation.json`); the prose
    is fetched from the private submission store by the digest that commitment names. With `private_only` (the
    board advertises a submission server) that is the only way in: a PR that adds prose of its own is refused —
    it made the strategy public for every rival to read — and there is no fallback to the PR tree. Without it, a
    PR that still carries the prose seals from its tree. Same shape as `_bundle_from_tree`; None if the PR has
    nothing under `submissions/<hotkey>/`.

    `defending` is `(digest, bundle dir)` of the crown this hotkey already defends with. A PR whose commitment
    names that same digest is a **defense**: the king re-signing its unchanged bundle for this round, so that a
    round it wins has a PR to merge (the reward follows merged PRs). The bundle is the one already defending,
    carrying the PR's new attestation; the result says `defense`."""
    names = _tree_names(cfg, ref, hotkey)
    if not names:
        return None
    att = None
    if "attestation.json" in names:
        raw = subprocess.run(
            ["git", "show", f"{ref}:submissions/{hotkey}/attestation.json"], cwd=cfg.repo, capture_output=True
        )
        try:
            att = json.loads(raw.stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            att = None
    # The PR's own attestation is what claims this hotkey's slot, so it must carry the hotkey's signature over
    # this round and this digest. Without this a one-file PR copying a victim's digest — no signature needed —
    # pointed the seal at the victim's revealed bundle and took over their submission.
    if forged := attest.problems(att, digest=(att or {}).get("bundle_sha256", ""), hotkey=hotkey, round_id=round_id):
        return {"problems": forged, "digest": "", "attestation": None}
    if private_only:
        try:
            added = _prose_added(cfg, base, ref, hotkey)
        except RuntimeError as exc:  # one PR's unreadable diff rejects that PR; it must not stop the seal
            return {
                "problems": [f"the PR could not be compared with {base}: {str(exc)[:120]}"],
                "digest": "",
                "attestation": None,
            }
        if added:
            return {
                "problems": [
                    f"the PR carries strategy files ({', '.join(sorted(added)[:3])[:80]}); submissions are private: the PR "
                    "carries only attestation.json and the prose goes to the submission server (the CLI does both)"
                ],
                "digest": "",
                "attestation": None,
            }
    digest = att.get("bundle_sha256") if isinstance(att, dict) else None
    signed_at = att.get("signed_at") if isinstance(att, dict) else None
    if defending and digest == defending[0] and defending[1].is_dir():
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(defending[1], dest, ignore=shutil.ignore_patterns(attest.FILE, "receipt.json"))
        (dest / attest.FILE).write_text(json.dumps(att))  # this round's signature over the bundle it defends with
        return {**_check_bundle_dir(dest, hotkey, round_id), "defense": True}
    prose_in_pr = [n for n in names if n not in ("attestation.json", "receipt.json")]
    if (
        isinstance(digest, str)
        and DIGEST.match(digest)
        and isinstance(signed_at, int)
        and not isinstance(signed_at, bool)
    ):
        src = store / round_id / hotkey / "uploads" / f"{signed_at}-{digest}"
        if src.is_dir():  # the revealed bundle the commitment points to
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for f in sorted(src.rglob("*")):
                if not f.is_file() or f.name == "receipt.json":  # the receipt is not part of the bundle
                    continue
                out = dest / f.relative_to(src)
                if not str(out.resolve()).startswith(str(dest.resolve()) + os.sep):
                    return {
                        "problems": [f"stored path escapes the bundle: {f.name[:60]}"],
                        "digest": digest,
                        "attestation": None,
                    }
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(f.read_bytes())
            return _check_bundle_dir(dest, hotkey, round_id)
        if not prose_in_pr or private_only:  # committed on the PR but never revealed (or a different digest)
            return {
                "problems": [
                    f"committed digest {str(digest)[:12]}… has no revealed bundle (upload it to the submission server)"
                ],
                "digest": digest,
                "attestation": None,
            }
    if private_only:  # a commitment that names no bundle at all
        return {"problems": ["the commitment names no revealed bundle"], "digest": "", "attestation": None}
    return _bundle_from_tree(cfg, ref, hotkey, dest, round_id=round_id)  # no server advertised: prose from the PR


def _incumbent_bundle(cfg: Config, tip: str, hotkey: str, dest: Path, *, incumbents: Path) -> dict | None:
    """A defending incumbent's bundle. Its prose is not public, so it comes from the private incumbents store;
    a legacy incumbent whose prose is still in `submissions/` comes from the tree. Its attestation was checked
    when it was first sealed, so the round binding is not re-checked (round_id=None)."""
    kept = _settle_incumbent_dir(incumbents, hotkey)
    if kept is not None:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(kept, dest)
        return _check_bundle_dir(dest, hotkey, round_id=None)
    return _bundle_from_tree(cfg, tip, hotkey, dest, round_id=None)


def _staging_hotkey(name: str) -> str | None:
    """`.<hotkey>.new` and `.<hotkey>.old` are a retain's scratch directories, not crowns. A name that is not one
    of those is not a hotkey to recover."""
    for suffix in (".new", ".old"):
        if name.startswith(".") and name.endswith(suffix):
            hotkey = name[1 : -len(suffix)]
            if SS58.match(hotkey):
                return hotkey
    return None


def _settle_incumbent_dir(store: Path, hotkey: str) -> Path | None:
    """The directory this hotkey defends from, finishing a swap a crash left half-done.

    The new bundle is written to `.<hotkey>.new` and the previous one moved to `.<hotkey>.old` before the new one
    takes its place, so a kill never leaves the store with nothing. A finished directory wins; otherwise the new
    bundle is put in place and the old one dropped, and a directory that was only moved aside is put back."""
    dest = store / hotkey
    if not SS58.match(hotkey):  # scratch dirs are only named for a real hotkey; anything else is just itself
        return dest if dest.is_dir() else None
    staging, backup = store / f".{hotkey}.new", store / f".{hotkey}.old"
    if dest.is_dir():
        if staging.is_dir():  # the swap finished; the scratch copy must not be published as its own crown
            shutil.rmtree(staging, ignore_errors=True)
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        return dest
    if staging.is_dir():
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        staging.rename(dest)
        return dest
    if backup.is_dir():
        backup.rename(dest)
        return dest
    return None


def _crown_landed(cfg: Config, hotkey: str) -> bool:
    """Whether `origin/main` has `submissions/<hotkey>/` after a fetch that succeeded.

    A failed fetch, or a listing that does not run, is not an empty tree. The crown may have just merged, and
    reading that failure as "the marker is absent" skips the retain and drops the king out of the next round.
    An empty listing after a fetch that worked means the merge really did not land."""
    sh(["git", "fetch", "-q", "origin"], cwd=cfg.repo)
    prefix = f"submissions/{hotkey}/"
    listing = sh(["git", "ls-tree", "-r", "-z", "--name-only", f"origin/{BRANCH}", "--", prefix], cwd=cfg.repo)
    return any(r.startswith(prefix) for r in listing.split("\0"))


def _retain_incumbent(cfg: Config, hotkey: str, rd: Path) -> None:
    """Keep a freshly crowned bundle in the private incumbents store so it can defend future rounds without its
    prose ever being public. The seal already materialised it under the round's `bundles/`."""
    src = rd / "bundles" / hotkey
    if not src.is_dir():
        return
    store = cfg.state / "incumbents"
    store.mkdir(parents=True, exist_ok=True)
    dest = store / hotkey
    staging, backup = store / f".{hotkey}.new", store / f".{hotkey}.old"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(src, staging)  # copy first: a partial tree must never replace the crown
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    if dest.exists():
        dest.rename(backup)  # the previous crown is still on disk until the new one is in place
    try:
        staging.rename(dest)
    except OSError:
        if not dest.exists() and backup.is_dir():
            backup.rename(dest)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _release_incumbent(cfg: Config, hotkey: str, rd: Path) -> None:
    """A dethroned king leaves the competition, so its bundle is no longer defended: stage it for the round's
    reveal (auditors can match its committed digest) and drop it from the private store."""
    kept = _settle_incumbent_dir(cfg.state / "incumbents", hotkey)
    if kept is None:
        return
    reveal = rd / "reveal" / hotkey
    shutil.rmtree(reveal, ignore_errors=True)
    reveal.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(kept, reveal)
    shutil.rmtree(kept, ignore_errors=True)


def _defending_digest(cfg: Config, hotkey: str) -> str | None:
    """The digest of the bundle this hotkey currently defends with, if it holds a crown privately."""
    kept = _settle_incumbent_dir(cfg.state / "incumbents", hotkey)
    if kept is None:
        return None
    return (_read(kept / attest.FILE, {}) or {}).get("bundle_sha256")


def _reveal_bundles(cfg: Config, rd: Path, dest: Path, king: str | None) -> None:
    """Publish the prose of every bundle that is out of the competition, so anyone can match its committed digest
    against its content. What is in or out is the **bundle**, not the hotkey: a bundle is out unless it is the
    crown, or it is the very bundle its hotkey is still defending with. A reigning king that resubmits and is not
    dethroned therefore keeps defending on its old bundle while the new one it just lost with is revealed. A
    dethroned king's previous bundle is staged under `rd/reveal` by `_release_incumbent`; it is published beside
    this round's bundle only when the two actually differ, so the ordinary dethronement publishes one copy."""
    out = dest / "revealed"
    sealed = _read(rd / "seal.json", {}).get("active") or {}
    published: dict[str, str] = {}
    for hotkey, info in sealed.items():
        if hotkey == king:
            continue  # the crown defends: never revealed
        digest = info.get("bundle_sha256") or ""
        if digest and digest == _defending_digest(cfg, hotkey):
            continue  # this very bundle is the one still defending; anything else of theirs is out
        src = rd / "bundles" / hotkey
        if src.is_dir():
            shutil.copytree(src, out / hotkey, dirs_exist_ok=True)
            published[hotkey] = digest
    if (rd / "reveal").is_dir():
        for staged in (rd / "reveal").iterdir():
            if not staged.is_dir():
                continue
            digest = (_read(staged / attest.FILE, {}) or {}).get("bundle_sha256") or ""
            if staged.name in published and digest and digest == published[staged.name]:
                continue  # the ordinary dethronement: the sealed copy and the staged one are the same bundle
            name = staged.name if staged.name not in published else f"{staged.name}.dethroned"
            shutil.copytree(staged, out / name, dirs_exist_ok=True)


def _changed_paths(cfg: Config, base: str, head: str) -> list[str]:
    """Every path a head changes relative to where it forked from the base (three-dot)."""
    names = sh(["git", "diff", "--name-only", "-z", f"{base}...{head}", "--"], cwd=cfg.repo, check=False)
    return [p for p in names.split("\0") if p]


def _changed_submissions(cfg: Config, base: str, head: str) -> list[str]:
    """The submission directories a head changes relative to where it forked from the base (three-dot: the
    merge base, not the base tip — a crown merged after the miner branched is not the miner's change)."""
    names = sh(["git", "diff", "--name-only", f"{base}...{head}", "--", "submissions/"], cwd=cfg.repo, check=False)
    return sorted({p.split("/")[1] for p in names.split() if p.count("/") >= 2 and p.split("/")[1] != "README.md"})


def pr_role(changed: list[str]) -> str:
    """What a pull request is to the round, from the submission directories it touches: none — maintenance,
    never sealed, never closed by a round; one — a strategy; more — a strategy PR done wrong, rejected."""
    return "maintenance" if not changed else "strategy" if len(changed) == 1 else "malformed"


def _strategy_prs(cfg: Config, tip: str) -> list[dict]:
    """Every open PR against the branch that touches `submissions/`, with the directories it changes. Recognised
    by content, not by label — a miner cannot label a PR — and labelled `sh:strategy` here so the board can see
    it. Fork heads are fetched by their pull ref; `origin` alone carries only same-repository branches."""
    sh(["git", "fetch", "-q", "origin", "+refs/pull/*/head:refs/remotes/origin/pr/*"], cwd=cfg.repo, check=False)
    prs = json.loads(
        gh(
            "pr",
            "list",
            "--repo",
            REPO,
            "--base",
            BRANCH,
            "--state",
            "open",
            "--json",
            "number,headRefOid,headRefName,title,labels,createdAt,updatedAt,url,author",
            "--limit",
            "100",
        )
    )
    out = []
    for pr in prs:
        changed = _changed_submissions(cfg, tip, pr["headRefOid"])
        if not changed:
            continue
        if LABEL_STRATEGY not in {lb["name"] for lb in pr.get("labels", [])}:
            _label(pr["number"], LABEL_STRATEGY)
        out.append({**pr, "changed": changed})
    return out


def one_per_hotkey(prs: list[dict], *, now: float | None = None) -> tuple[dict[str, dict], dict[int, str]]:
    """Of several open PRs for one hotkey, the one whose bundle was **signed** last counts; the others are rejected
    as superseded. Signed bundles are public, so the PR number cannot decide: anyone could reopen a miner's older
    bundle as a newer PR. A signing time in the future is not a submission (it would win every tie). Ties fall to
    the lowest PR number, since a copy of a public attestation can only be opened after the original. Pure: `signed_at` is read from each head's attestation before this is called."""
    now = time.time() if now is None else now
    keep: dict[str, dict] = {}
    superseded: dict[int, str] = {}
    # Ties fall to the *lowest* PR number: a copy of a public attestation can only be opened after the original,
    # so the miner who submitted first keeps the slot. Highest-number-wins handed it to the copier.
    order = lambda p: (p.get("signed_at") if isinstance(p.get("signed_at"), int) else -1, -p["number"])  # noqa: E731
    for pr in sorted(prs, key=order):
        hotkey = pr["changed"][0]
        at = pr.get("signed_at")
        if isinstance(at, int) and at > now + 600:
            superseded[pr["number"]] = "signed_at is in the future"
            continue
        if hotkey in keep:
            superseded[keep[hotkey]["number"]] = (
                f"superseded by #{pr['number']} (one submission per hotkey: the latest signed, "
                "and on a tie the first submitted)"
            )
        keep[hotkey] = pr
    return keep, superseded


def candidates(cfg: Config, round_id: str, bundles: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """What a seal taken now would contain: `(active, rejected)`, with every bundle materialised under `bundles`.

    **Incumbents**: strategies already merged into `submissions/` — a crowned king defends the crown every round
    without resubmitting. **Challengers**: every open PR touching exactly one `submissions/<hotkey>/`, taken at
    its head SHA, whose bundle lints and carries the hotkey's signature over *this* round and *this* digest. One
    submission per hotkey: the latest signed wins. A challenger that reproduces the round's private reference
    answers is refused (S1, `sh/validator/similarity.py`). A challenger for a hotkey supersedes its incumbent."""
    sh(["git", "fetch", "-q", "origin"], cwd=cfg.repo)
    bundles.mkdir(parents=True, exist_ok=True)
    tip = f"origin/{BRANCH}"
    # Same setting the ingestion server reads: pointed apart, uploads succeed and every seal then rejects them.
    store = Path(os.environ.get("SH_SUBMISSION_STORE") or (cfg.state / "submissions"))
    active: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    for entry in sh(["git", "ls-tree", "--name-only", tip, "submissions/"], cwd=cfg.repo, check=False).split():
        hotkey = entry.split("/")[-1]
        if not SS58.match(hotkey):  # README.md and anything not an ss58 directory is not a submission
            continue
        b = _incumbent_bundle(cfg, tip, hotkey, bundles / hotkey, incumbents=cfg.state / "incumbents")
        if b and not b["problems"]:
            active[hotkey] = {"pr": None, "head": tip, "bundle_sha256": b["digest"], "incumbent": True}
        elif (round_id, hotkey) not in _UNRESOLVED:  # said once, not once per window-close poll
            _UNRESOLVED.add((round_id, hotkey))
            why = (b or {}).get("problems") or ["no bundle in the store or the tree"]
            print(f"[{round_id}] incumbent {hotkey[:12]}… unresolved, not sealed: {why[0][:120]}", flush=True)
    prs = _strategy_prs(cfg, tip)
    for pr in prs:
        if pr_role(pr["changed"]) == "malformed":
            rejected[str(pr["number"])] = f"{len(pr['changed'])} changed submission directories (need exactly 1)"
    # Every strategy PR is checked in full *before* choosing one per hotkey. Choosing first let a PR carrying a
    # victim's bundle with a forged, later signed_at win the choice, fail its signature, and take the victim's
    # real submission out of the round with it.
    answers = similarity.load(cfg.rounds / round_id)
    valid: list[dict] = []
    for pr in (p for p in prs if pr_role(p["changed"]) == "strategy"):
        hotkey, staged = pr["changed"][0], bundles / f".pr{pr['number']}"
        if not SS58.match(hotkey):
            rejected[str(pr["number"])] = f"submissions/{hotkey[:16]}…/ is not a hotkey (ss58) directory"
            continue
        outside = [p for p in _changed_paths(cfg, tip, pr["headRefOid"]) if not p.startswith(f"submissions/{hotkey}/")]
        if outside:  # a strategy is prose under one hotkey and nothing else — never code, never another's directory
            rejected[str(pr["number"])] = (
                f"changes {len(outside)} path(s) outside submissions/{hotkey[:8]}…/ (e.g. {outside[0]})"
            )
            continue
        b = _reveal_challenger(
            cfg,
            pr["headRefOid"],
            hotkey,
            staged,
            round_id=round_id,
            store=store,
            base=tip,
            private_only=bool(cfg.submit_server),
            defending=(active[hotkey]["bundle_sha256"], bundles / hotkey) if hotkey in active else None,
        )
        if b is not None and not b["problems"] and answers:
            files, _ = collect(staged)
            if copied := answers.refuse(similarity.bundle_text(files)):
                b["problems"] = [copied]
        if b is None or b["problems"]:
            rejected[str(pr["number"])] = (b or {}).get("problems", ["empty submission"])[0]
            shutil.rmtree(staged, ignore_errors=True)
            continue
        valid.append(
            {
                **pr,
                "signed_at": _read(staged / "attestation.json", {}).get("signed_at"),
                "digest": b["digest"],
                "defense": bool(b.get("defense")),
            }
        )
    keep, superseded = one_per_hotkey(valid)
    rejected.update({str(n): why for n, why in superseded.items()})
    for hotkey, pr in keep.items():
        # A defense is the incumbent itself, now with a PR to merge if it wins. Any other PR from an incumbent's
        # hotkey replaces the bundle it defended with, and that bundle must still be dethroned.
        defense = pr["defense"]
        was_incumbent = (bundles / hotkey).is_dir() and not defense
        shutil.rmtree(bundles / hotkey, ignore_errors=True)  # a challenger supersedes the hotkey's incumbent
        (bundles / f".pr{pr['number']}").rename(bundles / hotkey)
        active[hotkey] = {
            "pr": pr["number"],
            "head": pr["headRefOid"],
            "bundle_sha256": pr["digest"],
            "incumbent": defense,
            "was_incumbent": was_incumbent,
            "defense": defense,
            "github": (pr.get("author") or {}).get("login"),
        }
    for staged in bundles.glob(".pr*"):
        shutil.rmtree(staged, ignore_errors=True)
    return active, rejected


def challenger_count(active: dict[str, dict]) -> int:
    """Submissions that would be sealed: an incumbent alone is nothing to evaluate against. Pure."""
    return sum(1 for info in active.values() if not info.get("incumbent"))


def has_challengers(cfg: Config, round_id: str) -> int:
    """How many valid submissions a seal taken now would carry — checked when a window closes, with no side
    effect on the round (bundles go to a temporary directory; nothing is labelled, written or published)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sh-candidates-") as tmp:
        active, _ = candidates(cfg, round_id, Path(tmp))
    return challenger_count(active)


def seal(cfg: Config, round_id: str, rd: Path) -> dict:
    """Which bundles are in this round, decided the instant the window closes and published before evaluation."""
    if (rd / "seal.json").exists():  # a restart after the seal was taken but before it was logged: never re-seal
        record = json.loads((rd / "seal.json").read_text())  # (a PR pushed after the window would be taken)
        active, rejected = record["active"], record["rejected"]
    else:
        active, rejected = candidates(cfg, round_id, rd / "bundles")
        record = {
            "schema": "sh-seal-v3",
            "round_id": round_id,
            "sealed_at": time.time(),
            "active": active,
            "rejected": rejected,
        }
        (rd / "seal.json").write_text(json.dumps(record, indent=1))
    for number in [i["pr"] for i in active.values() if i.get("pr")] + [int(n) for n in rejected]:
        _label(number, f"sh:round:{round_id}")
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(rd / "seal.json", dest / "seal.json")
    live(cfg, rd, "evaluate", progress={"done": 0, "total": 0, "by_surface": {}}, push=False)
    _commit(cfg, f"{round_id}: seal — {len(active)} active bundles, {len(rejected)} rejected")
    log(
        rd,
        "seal",
        active=len(active),
        incumbents=sum(1 for a in active.values() if a["incumbent"]),
        rejected=len(rejected),
    )
    return record


def _progress(cfg: Config, remote: str, total: int) -> dict:
    """Per-surface counts and mean credit from the worker's episode records so far — what the board shows, plus
    each surface's credit on each instance, so a reader can open a row and see the episodes a score is made of."""
    script = (
        "import glob, json, sys\n"
        "out = {}\n"
        f"for p in glob.glob('{remote}/episodes/*/*/episode.json'):\n"
        "    try:\n"
        "        e = json.load(open(p))\n"
        "    except (OSError, ValueError):\n"
        "        continue\n"
        "    s = out.setdefault(p.split('/')[-3], {'n': 0, 'verified': 0, 'credit': 0.0, 'tasks': {}})\n"
        "    c = e.get('credit')\n"
        "    c = float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else float(bool(e.get('verified_success')))\n"
        "    s['n'] += 1; s['verified'] += bool(e.get('verified_success'))\n"
        "    s['credit'] += c\n"
        "    s['tasks'][p.split('/')[-2]] = {'credit': round(c, 4), 'verified': bool(e.get('verified_success')),\n"
        "                                    'void': bool(e.get('void')), 'dq': bool(e.get('disqualified'))}\n"
        "print(json.dumps(out))\n"
    )
    raw = _worker(cfg, f'python3 -c "$(echo {base64.b64encode(script.encode()).decode()} | base64 -d)"')
    try:
        by = json.loads(raw.strip().splitlines()[-1]) if raw.strip() else {}
    except (ValueError, IndexError):
        by = {}
    for v in by.values():
        v["credit"] = round(v["credit"] / v["n"], 4) if v["n"] else None
    return {"done": sum(v["n"] for v in by.values()), "total": total, "by_surface": by}


def evaluate(cfg: Config, rd: Path, sealed: dict) -> None:
    """Runs on the GPU worker in the background while this side publishes progress every few minutes. The
    worker gets exactly what an episode needs — tasks, withheld halves for the grader, checks, canon, the sealed
    bundles — and hands the episodes back. No credential is ever on it."""
    remote = f"{cfg.worker_root}/rounds/{rd.name}"
    _worker(cfg, f"mkdir -p {remote}")
    for sub in ("tasks", "withheld", "checks", "canon", "bundles"):
        if (rd / sub).exists():
            _rsync(f"{rd / sub}/", f"{cfg.worker}:{remote}/{sub}/", cfg)
    if (rd / "images").exists():  # image-defined tasks: the worker builds each task's image from its context
        _worker(cfg, f"mkdir -p {remote}/images")
        _rsync(f"{rd / 'images'}/", f"{cfg.worker}:{remote}/images/", cfg)
        built = _worker(
            cfg,
            f"for d in {remote}/images/*/; do t=$(cat $d/TAG); docker image inspect $t >/dev/null 2>&1 "
            f'|| docker build -q -t $t $d >/dev/null 2>&1 || echo "FAILED $t"; done; echo built',
        )
        if "FAILED" in built:
            raise RuntimeError(f"task image build failed on the worker: {built.strip()[-300:]}")
        log(rd, "images", built=len(list((rd / "images").iterdir())))
    # The reference strategy (canon) labels a family; nothing is paid by it. On one GPU it runs in calibration
    # rounds only — every `canon_every`-th — and the pooled window carries its measurements between them.
    calibrate = cfg.canon_every <= 1 or int(rd.name[1:]) % cfg.canon_every == 0
    surfaces = (["null"] + ([f"canon={remote}/canon"] if calibrate else [])) + [
        f"{h}={remote}/bundles/{h}" for h in sealed["active"]
    ]
    total = len(surfaces) * len(list((rd / "tasks").glob("*.json")))
    launch = (
        f"cd {cfg.worker_root}/pkg && PYTHONPATH={cfg.worker_root}/pkg setsid nohup python3 -m sh.validator.batch "
        f"--round {remote} --surfaces {','.join(surfaces)} --image {cfg.image} --inference unused "
        f"--out {remote}/episodes --concurrency {cfg.concurrency} --network sh-ep "
        f"--tokens {cfg.worker_root}/state/tokens --usage-dir {cfg.worker_root}/state/usage "
        f">> {remote}/batch.log 2>&1 < /dev/null & echo started"
    )

    def running() -> bool:
        return _worker(cfg, f"pgrep -f '[b]atch --round {remote}' | wc -l").strip() not in ("", "0")

    def screening() -> bool:
        # The daemon's baseline screen holds one of the engine's two long-context slots; launching beside it makes
        # three and the engine refuses everyone. The screen never starts while a batch runs (supply.baseline);
        # this is the other direction.
        return _worker(cfg, f"pgrep -f '[b]atch --round {cfg.worker_root}/screen/' | wc -l").strip() not in ("", "0")

    # Resume-safe: a restarted control plane finds the batch still running and polls it rather than launching a
    # second one; if it is not running, launching is safe — `batch` resumes on its own episode records. A batch
    # that ends short (episodes the provider voided past its own retries) is launched again, a bounded number of
    # times, so a busy engine costs time rather than evidence.
    def wait_for_screen() -> None:
        # Bounded: a screen left by a dead daemon is a process nobody else will end. The board keeps moving.
        waited = 0.0
        if screening():
            log(rd, "evaluate_wait", note="a baseline screen holds the engine; launching when it ends")
        while screening() and waited < 5400:
            live(cfg, rd, "evaluate", progress=_progress(cfg, remote, total))
            time.sleep(60)
            waited += 60
        if waited >= 5400:
            log(rd, "evaluate_wait", note="screen still running after 90 min; ending it")
            _worker(cfg, f"pkill -f 'batch --round {cfg.worker_root}/screen/' ; true")

    def launch_batch() -> None:
        # Claim the engine first. The daemon screens buffered candidates back to back, seconds apart, and a poll
        # alone would rarely see the engine free between two of them; the claim makes the daemon stop starting
        # screens (supply.baseline.Screen.evaluating). The grace lets a screen started just before the claim show
        # up; the claim is held until the batch is visible, so the daemon always sees one or the other. A claim
        # left by a killed loop goes stale after 2 h.
        claim = f"{cfg.worker_root}/state/engine-claim"
        _worker(cfg, f"mkdir -p {cfg.worker_root}/state && touch {claim}")
        try:
            time.sleep(CLAIM_GRACE_S)
            wait_for_screen()
            _worker_launch(cfg, launch)
            for _ in range(12):
                if running():
                    break
                time.sleep(5)
        finally:
            _worker(cfg, f"rm -f {claim}")

    launches = 0
    if running():
        log(rd, "evaluate_resume", note="batch already running on the worker; polling")
    else:
        launch_batch()
        launches = 1
    last_push = 0.0
    while True:
        time.sleep(45)
        alive = running()
        prog = _progress(cfg, remote, total)
        if not alive or time.time() - last_push > 150:
            live(cfg, rd, "evaluate", progress=prog)
            last_push = time.time()
        if alive:
            continue
        if prog["done"] < total and launches < 3:
            log(rd, "evaluate_relaunch", done=prog["done"], total=total)
            launch_batch()
            launches += 1
            continue
        break
    _rsync(f"{cfg.worker}:{remote}/episodes/", f"{rd / 'episodes'}/", cfg)
    if tags := _image_tags(rd):
        _worker(
            cfg,
            "docker image rm -f " + " ".join(tags) + " >/dev/null 2>&1; docker image prune -f >/dev/null 2>&1; "
            "docker builder prune -f --filter until=12h >/dev/null 2>&1; "  # task images leave ~250 GB/day of cache
            "docker builder prune -f --keep-storage 120GB >/dev/null 2>&1; true",  # and a hard cap on what stays
        )
    # Rounds older than the previous one leave the worker: their episodes are archived here.
    _worker(cfg, f"ls -d {cfg.worker_root}/rounds/r* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf")
    n = len(list((rd / "episodes").rglob("episode.json")))
    if not n:
        raise RuntimeError("the worker returned no episodes")
    log(rd, "evaluate", episodes=n, surfaces=len(surfaces))


def window_archive(cfg: Config, round_id: str) -> Path:
    """The last W rounds' episode *records*, pooled: what the scorer sees. Only `episode.json` is archived — the
    scorer reads nothing else, and a round's snapshots and trajectories are gigabytes (measured in testing: 2.9 GB) that stored
    twice and re-copied every close would fill the disk in ~15 rounds. Each round is archived once (a temp dir
    renamed into place, so a kill never leaves a partial archive a later run would skip)."""
    archive = cfg.state / "archive"
    archive.mkdir(exist_ok=True)
    src = cfg.rounds / round_id / "episodes"
    dst = archive / round_id
    if src.exists() and not dst.exists():
        tmp = archive / f".{round_id}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        for ej in src.rglob("episode.json"):
            out = tmp / ej.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ej, out)
        tmp.mkdir(exist_ok=True)  # a round with no episode.json at all still archives (empty), never re-copied
        tmp.rename(dst)
    # The round being closed is always in its own window.
    rounds = sorted(
        p.name
        for p in archive.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p.name >= cfg.window_from or p.name == round_id)
    )
    rounds = rounds[-cfg.window :]
    pooled = cfg.state / "window"
    shutil.rmtree(pooled, ignore_errors=True)
    pooled.mkdir(parents=True)
    for r in rounds:
        shutil.copytree(archive / r, pooled / r)
    (pooled / "rounds.json").write_text(json.dumps(rounds))
    return pooled


def _reclaim_disk(cfg: Config, keep_recent: int = 2) -> None:
    """After a round is done its heavy episode files are dead weight: the king's trajectories are on Hugging Face,
    the scorer keeps only episode.json (archived), and a done round is never resumed. Strip snapshots, trajectories,
    result blobs and container logs from all but the last `keep_recent` closed rounds; episode.json stays for audit."""
    heavy = ("snapshot.tar", "trajectory.json", "result.json", "container.log")
    closed = sorted(p for p in cfg.rounds.glob("r*") if (p / "DONE").exists())
    for rd in closed[:-keep_recent] if keep_recent else closed:
        for name in heavy:
            for f in (rd / "episodes").rglob(name):
                f.unlink(missing_ok=True)


def close(cfg: Config, round_id: str, rd: Path) -> dict:
    pooled = window_archive(cfg, round_id)
    window = json.loads((pooled / "rounds.json").read_text())
    record = close_round(rd, pooled, rd / "close", reveal_dir=rd / "withheld", era=cfg.era, window=window)
    # A family FamilyStats.retirement() has already flagged (in_rotation=False) is written to close.json but
    # never surfaced while the loop runs — an operator only sees it by reading JSON after the fact. Log it now,
    # once per close, so a collapsed or retired family is visible immediately, not discovered days later.
    for family, stats in record.get("family_stats", {}).items():
        if not stats.get("in_rotation", True):
            log(rd, "family_retired", family=family, reason=stats.get("retirement"))
    log(
        rd,
        "close",
        window=window,
        episodes=record["episodes"],
        commitments_ok=record["commitments_ok"],
        weights=record["weights"],
    )
    return record


def crown_round(cfg: Config, rd: Path, record: dict, sealed: dict) -> dict:
    """This round's king, from this round's episodes alone (spec: the crown is a merge, not a payment)."""
    pooled = {h: s.get("delta_c", 0.0) for h, s in record.get("scores", {}).items()}
    incumbent = next((h for h, info in sealed["active"].items() if info.get("incumbent")), None)
    result = crown_rule(
        load_episodes(rd / "episodes"),
        set(sealed["active"]),
        round_id=rd.name,
        pooled_delta_c=pooled,
        min_paired=cfg.min_paired,
        incumbent=incumbent,  # a tie does not dethrone
    )
    (rd / "close" / "crown.json").write_text(json.dumps(result, indent=1))
    ranked = sorted((s["rank"], h) for h, s in result["standings"].items() if "rank" in s)
    log(rd, "crown", king=result["king"], ranked=[h for _, h in ranked])
    return result


def outcome(sealed: dict, king: str | None) -> dict:
    """What happens to each PR the seal named: the king's PR is merged, every other challenger's is closed, the
    rejected ones too. Pure. Only PRs the seal named are ever touched — a maintenance PR is never in the seal."""
    if king is not None and king not in sealed["active"]:
        raise ValueError(f"king {king} is not a sealed strategy")
    king_pr = sealed["active"].get(king, {}).get("pr") if king else None
    close_prs = sorted(
        {info["pr"] for h, info in sealed["active"].items() if info.get("pr") and h != king}
        | {int(n) for n in sealed.get("rejected", {})}
    )
    return {"king": king, "merge": king_pr, "close": close_prs}


def dethroned(
    sealed: dict, king: str | None, scores: dict | None = None, history: list | tuple = (), patience: int = 3
) -> list[str]:
    """Which carried bundles leave `submissions/` this round. An incumbent — or a hotkey whose incumbent bundle its
    own PR replaced this round (`was_incumbent`) — is removed when

      (a) another strategy was crowned: a challenger beat the baseline on these instances while it did not;
      (b) its pooled window fails the correctness gate on enough evidence (`mean_d + z·se < 0`, the same eight
          rounds payment weighs): the evidence says it is worse than the baseline, not merely unlucky;
      (c) it has gone `patience` rounds in a row, this one included, without the crown: a lucky tiebreak king
          cannot squat for free at the validator's expense.

    A round that crowns nobody does not by itself dethrone: on six instances a genuinely better strategy ties the
    baseline by noise in about a third of rounds (seen in testing: an incumbent's Δ of exactly 0.0). `history` is the
    published `rounds/index.json` entries, oldest first, without this round. Pure."""
    scores = scores or {}
    gone = []
    for h, info in sealed["active"].items():
        if not (info.get("incumbent") or info.get("was_incumbent")) or h == king:
            continue
        s = scores.get(h) or {}
        below = s.get("reason") is None and s.get("gate") is False  # a thin window is no evidence either way
        streak = 1
        for entry in reversed(history):
            if entry.get("king") == h:
                break
            streak += 1
        if king is not None or below or streak >= patience:
            gone.append(h)
    return sorted(gone)


def _prune_orphan_incumbents(cfg: Config, rd: Path) -> list[str]:
    """Drop anything in the private store that no marker in `submissions/` claims. Such an entry is a crown that
    never landed, or a marker removed outside the dethrone path: it would defend nothing, and because the reveal
    skips a hotkey that is still in the store it would never be published either. Releasing it does both.

    A listing that did not succeed is not an empty tree. `git ls-tree` prints nothing when it fails, and reading
    that as "nobody defends" would release every crown and publish every bundle. Scratch directories from a retain
    are not crowns either: they are put back in place or dropped, never revealed under their own name."""
    store = cfg.state / "incumbents"
    if store.is_dir():
        for name in sorted(p.name for p in store.iterdir()):
            if hotkey := _staging_hotkey(name):
                _settle_incumbent_dir(store, hotkey)
    try:
        listing = sh(["git", "ls-tree", "--name-only", f"origin/{BRANCH}", "submissions/"], cwd=cfg.repo)
    except RuntimeError as exc:
        log(rd, "incumbent_orphan_skipped", why=str(exc)[:160])
        return []
    markers = {e.split("/")[-1] for e in listing.split()}
    orphans = (
        [
            p.name
            for p in sorted(store.iterdir())
            if p.is_dir() and p.name not in markers and _staging_hotkey(p.name) is None
        ]
        if store.is_dir()
        else []
    )
    for hotkey in orphans:
        _release_incumbent(cfg, hotkey, rd)
        log(rd, "incumbent_orphan", hotkey=hotkey)
    return orphans


def _closing_note(round_id: str, info: dict, *, crowned_over: bool, lost: bool) -> str:
    """The last line on a PR that was not crowned. A defense PR's crown either still stands — a new defense next round
    keeps it paid — or was lost this round, and telling a dethroned king to defend a crown it no longer holds is
    wrong. Pure."""
    if not info.get("defense"):
        return f"Not crowned in `{round_id}`; this PR is closed with the round. Submit again in the next window."
    if lost:
        why = "a challenger was crowned over it" if crowned_over else "its pooled window or its rounds without a win"
        return (
            f"Not crowned in `{round_id}`; this defense PR is closed with the round, and the crown it defended is lost "
            f"({why}). Submit again in the next window to challenge for it."
        )
    return (
        f"Not crowned in `{round_id}`; this defense PR is closed with the round. The crown still defends: open a new "
        "defense PR in the next window to be paid for a round it wins."
    )


def announce(cfg: Config, round_id: str, rd: Path, record: dict, sealed: dict, crowned: dict) -> str | None:
    """Scorecards on every PR; `scored` on every PR; the crown moved to the king; the king's PR merged; every
    other competition PR closed with the reason; PRs that arrived after the seal closed as outside the window."""
    reveal = json.loads((rd / "close" / "reveal.json").read_text())
    (rd / "scorecards").mkdir(exist_ok=True)
    plan = outcome(sealed, crowned["king"])
    king = plan["king"]
    already = {
        pr["number"]
        for pr in json.loads(
            gh(
                "pr",
                "list",
                "--repo",
                REPO,
                "--label",
                LABEL_SCORED,
                "--state",
                "all",
                "--json",
                "number",
                "--limit",
                "200",
            )
        )
    }
    # Who leaves the competition this round — decided now only so each PR's note can say so; the tree changes below.
    leaving = set(
        dethroned(
            sealed, king, record.get("scores"), _read(cfg.repo / "rounds" / "index.json", {"rounds": []})["rounds"]
        )
    )
    for hotkey, info in sealed["active"].items():
        st = crowned["standings"].get(hotkey, {})
        this_round = (
            f"**This round:** {'rank ' + str(st['rank']) if st.get('rank') else 'not ranked'} · Δ vs baseline "
            f"{st['delta']:+.3f} on {st['n']} paired instances · {st['verified']} verified.\n\n"
            if st.get("delta") is not None
            else ""
        )
        card = this_round + render_scorecard(record, hotkey, reveal)
        (rd / "scorecards" / f"{hotkey}.md").write_text(card)
        if not info.get("pr") or info["pr"] in already:
            continue  # an incumbent has no PR to write to; a PR scored before a restart is not written to twice
        body = card
        if hotkey == king:
            body = "👑 **Crowned: best against the baseline on this round's instances. Merging.**\n\n" + body
        else:
            body += "\n\n---\n" + _closing_note(round_id, info, crowned_over=king is not None, lost=hotkey in leaving)
        gh("pr", "comment", str(info["pr"]), "--repo", REPO, "--body", body)
        _label(info["pr"], LABEL_SCORED)
    # The round's crown, applied to the merged PR and never moved: each round's winner keeps its own
    # `sh:<round>:crown` label. A reader — and the SN74 reward, which scores a merged PR by this label — sees
    # exactly which round a PR won, and a defending incumbent (no new merge) keeps the label from when it won.
    if plan["merge"]:
        _label(plan["merge"], f"sh:{round_id}:crown")
        sealed_head = sealed["active"][king]["head"]  # the head the round evaluated; a push after the seal changes it
        merged = subprocess.run(
            [
                "gh",
                "pr",
                "merge",
                str(plan["merge"]),
                "--repo",
                REPO,
                "--squash",
                "--match-head-commit",
                sealed_head,
                "--subject",
                f"crown {round_id}: {king}",
            ],
            capture_output=True,
            text=True,
        )
        if merged.returncode != 0:  # a mismatch (the king pushed after the seal) or a transient failure: retry once
            time.sleep(5)
            merged = subprocess.run(
                ["gh", "pr", "merge", str(plan["merge"]), "--repo", REPO, "--squash", "--match-head-commit",
                 sealed_head, "--subject", f"crown {round_id}: {king}"], capture_output=True, text=True)  # fmt: skip
        log(rd, "merge", pr=plan["merge"], ok=merged.returncode == 0, head=sealed_head[:8], err=merged.stderr[-200:])
    elif king:  # an incumbent that won without a defense PR: it keeps the crown, but there is nothing to merge
        log(rd, "merge", pr=None, ok=True, note="incumbent retains the crown (no defense PR: nothing merged)")
    if king:
        # Retained whenever the crown actually landed in `submissions/`, which is what the next round's seal reads.
        # The exit code alone is the wrong test in both directions: a re-run after a crash fails with "already
        # merged" although the marker is there (that lost the bundle and dropped the king out silently), while a
        # merge that never landed would otherwise store a bundle whose commitment is not public.
        if _crown_landed(cfg, king):
            _retain_incumbent(cfg, king, rd)
        else:
            log(rd, "retain_skipped", king=king, why="no submissions/<king>/ in the tree: the crown did not land")
    history = _read(cfg.repo / "rounds" / "index.json", {"rounds": []})["rounds"]  # closed rounds, this one not yet
    if gone := dethroned(sealed, king, record.get("scores"), history):  # after the merge: the tree is current
        sh(["git", "pull", "-q", "--rebase", "--autostash", "origin", BRANCH], cwd=cfg.repo, check=False)
        for hotkey in gone:
            sh(["git", "rm", "-r", "-q", f"submissions/{hotkey}"], cwd=cfg.repo, check=False)
            _release_incumbent(cfg, hotkey, rd)  # out of the competition: reveal its bundle, drop it from the store
        if sh(["git", "status", "--porcelain", "submissions"], cwd=cfg.repo).strip():
            sh(
                ["git", "commit", "-q", "-m", f"{round_id}: dethroned {', '.join(gone)}", "--", "submissions"],
                cwd=cfg.repo,
            )
            sh(["git", "push", "-q", "origin", BRANCH], cwd=cfg.repo)
        log(rd, "dethroned", hotkeys=gone)
    _prune_orphan_incumbents(cfg, rd)
    for number in plan["close"]:
        reason = sealed.get("rejected", {}).get(str(number))
        why = f"rejected at seal: {reason}" if reason else f"not crowned in `{round_id}`"
        sh(
            [
                "gh",
                "pr",
                "close",
                str(number),
                "--repo",
                REPO,
                "--comment",
                f"Closed with round `{round_id}` — {why}. Submit again in the next window.",
            ],
            check=False,
        )
    sealed_prs = {i["pr"] for i in sealed["active"].values() if i.get("pr")} | {
        int(n) for n in sealed.get("rejected", {})
    }
    sh(["git", "fetch", "-q", "origin"], cwd=cfg.repo)
    late = [pr["number"] for pr in _strategy_prs(cfg, f"origin/{BRANCH}") if pr["number"] not in sealed_prs]
    for number in late:
        sh(
            [
                "gh",
                "pr",
                "close",
                str(number),
                "--repo",
                REPO,
                "--comment",
                f"Arrived after the submission window of `{round_id}` closed, so it was not sealed. "
                "Fetch the next round's tasks when its window opens and submit again.",
            ],
            check=False,
        )
    log(rd, "announce", king=king, merged=plan["merge"], closed=plan["close"], late=late)
    return king


def export_and_upload(cfg: Config, round_id: str, rd: Path, king: str | None) -> dict:
    manifest = build_exports(
        rd,
        rd / "episodes",
        rd / "close" / "close.json",
        rd / "export",
        king=king,
        # The export is public; the king's bundle is not, for as long as it defends. Naming it here lets the
        # builder keep its prose out of the rows (the captured system turn embeds it verbatim).
        bundle_dir=(rd / "bundles" / king) if king else None,
    )
    token = os.environ.get("HF_TOKEN", "")
    if not manifest["sft_rows"] and not manifest["dpo_pairs"]:
        why = "no king this round" if king is None else "the king's episodes yielded no rows"
        log(rd, "export", sft=0, dpo=0, uploaded=False, reason=why)
        return {"manifest": manifest, "upload": {"uploaded": False, "reason": why}}
    if not token:
        log(
            rd, "export", sft=manifest["sft_rows"], dpo=manifest["dpo_pairs"], uploaded=False, reason="HF_TOKEN not set"
        )
        return {"manifest": manifest, "upload": {"uploaded": False, "reason": "HF_TOKEN not set"}}
    try:
        result = upload_exports(rd / "export", HF_REPO, token, round_id=round_id)
    except Exception as exc:  # an HF outage or a bad token must not stall the round; the rows stay on disk to retry
        log(rd, "export", sft=manifest["sft_rows"], dpo=manifest["dpo_pairs"], uploaded=False, reason=repr(exc)[:200])
        return {"manifest": manifest, "upload": {"uploaded": False, "reason": repr(exc)[:200]}}
    log(
        rd,
        "export",
        sft=manifest["sft_rows"],
        dpo=manifest["dpo_pairs"],
        uploaded=result.get("uploaded"),
        url=result.get("url"),
    )
    return {"manifest": manifest, "upload": result}


def publish_close(
    cfg: Config, round_id: str, rd: Path, record: dict, crowned: dict, exported: dict, king: str | None
) -> None:
    """Everything a miner needs to check the round, in the repository, under the round."""
    sh(
        ["git", "pull", "-q", "--rebase", "--autostash", "origin", BRANCH], cwd=cfg.repo, check=False
    )  # the merge just landed
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("close.json", "reveal.json", "crown.json"):
        shutil.copy(rd / "close" / name, dest / name)
    if shown(rd) != rd / "tasks":  # the miners saw previews: the tasks the round was scored on are public now
        shutil.copytree(rd / "tasks", dest / "evaluated", dirs_exist_ok=True)
    if (rd / "checks").exists():
        shutil.copytree(rd / "checks", dest / "checks", dirs_exist_ok=True)  # semantics of `custom` predicates
    shutil.copytree(rd / "scorecards", dest / "scorecards", dirs_exist_ok=True)
    _reveal_bundles(cfg, rd, dest, king)  # every bundle now out of the competition (never one still defending)
    shutil.copy(rd / "export" / "manifest.json", dest / "manifest.json")
    artefacts = sorted(p.name for p in dest.iterdir())
    entry = {
        "round_id": round_id,
        "closed_at": time.time(),
        "window": _read(rd / "window.json"),
        "episodes": record["episodes"],
        "king": king,
        "weights": record["weights"],
        "scores": _trim_scores(record),
        "crown": {h: st for h, st in crowned.get("standings", {}).items() if st.get("rank")},
        "commitments_ok": record["commitments_ok"],
        "sft_rows": exported["manifest"]["sft_rows"],
        "dpo_pairs": exported["manifest"]["dpo_pairs"],
        "hf": exported["upload"].get("url"),
    }
    # hotkey -> GitHub login, published with the round so the page stays recomputable. This round's seal is the
    # authority (attested); the live map, accumulated from earlier seals, covers an incumbent carried without a PR.
    sealed_active = _read(rd / "seal.json", {}).get("active") or {}  # hotkey -> {pr, head, github, ...}
    gh = _read(cfg.repo / LIVE, {}).get("github") or {}
    shown_hk = set(entry["scores"]) | set(crowned.get("standings", {})) | ({king} if king else set())
    entry["github"] = {h: login for h in shown_hk if (login := (sealed_active.get(h) or {}).get("github") or gh.get(h))}
    entry["pr"] = (sealed_active.get(king) or {}).get("pr") if king else None  # the king's PR this round, if any
    index_path = cfg.repo / "rounds" / "index.json"
    index = _read(index_path, {"schema": "sh-rounds-index-v2", "rounds": []})
    index["rounds"] = [r for r in index["rounds"] if r["round_id"] != round_id] + [entry]
    index_path.write_text(json.dumps(index, indent=1))
    page = cfg.repo / "docs" / "rounds" / round_id  # the round's page, where Pages serves it
    page.mkdir(parents=True, exist_ok=True)
    (page / "index.html").write_text(
        render_leaderboard(record, {**entry, "crown": crowned, "artefacts": artefacts, "repo": REPO, "branch": BRANCH})
    )
    live(cfg, rd, "done", push=False)
    _commit(
        cfg,
        f"{round_id}: close — king {king or 'none'}, {record['episodes']} episodes, "
        f"{exported['manifest']['sft_rows']} SFT rows, {exported['manifest']['dpo_pairs']} DPO pairs",
    )
    log(rd, "publish_close", king=king)


# ─── the round, and the loop ───────────────────────────────────────────────────────────────────────
def done_stages(rd: Path) -> set[str]:
    """The stages a round has completed, from its own log — what a restart resumes from (spec §5.7)."""
    p = rd / "phases.jsonl"
    if not p.exists():
        return set()
    return {json.loads(line)["stage"] for line in p.read_text().splitlines() if line.strip()}


def unfinished_round(cfg: Config) -> Path | None:
    """The latest round directory without a DONE marker, if any — the one a restarted loop must pick up."""
    if not cfg.rounds.exists():
        return None
    rounds = sorted(p for p in cfg.rounds.iterdir() if p.is_dir() and p.name.startswith("r"))
    if rounds and not (rounds[-1] / "DONE").exists():
        return rounds[-1]
    return None


def _logged(rd: Path, stage: str, field: str):
    for line in (rd / "phases.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("stage") == stage:
            return rec.get(field)
    return None


def run_round(cfg: Config, mock: tuple[Path, Path] | None = None, resume: Path | None = None) -> dict:
    """One round. With `resume`, the stages that round already logged are skipped, and the seal that was
    published is reused rather than recomputed — a restart must never change what a round sealed. A round
    minted by the previous loop (logged `mint`, never `open`) is taken as already open with its window over."""
    if resume:
        rd, round_id = resume, resume.name
        done = done_stages(rd)
        log(rd, "resume", completed=sorted(done))
    else:
        rd = open_round(cfg)
        round_id, done = rd.name, set()
    if "open" not in done and "seal" not in done:  # a restart between taking a round and publishing it
        publish_round(cfg, rd)
    if "window" not in done and "seal" not in done:
        while True:
            wait_window(cfg, rd, mock)
            if n := has_challengers(cfg, round_id):
                break
            reopen_window(cfg, rd)  # nothing to evaluate: same round, same tasks, a fresh window
        log(rd, "window", closed_at=time.time(), submissions=n)
    if "seal" not in done:
        sealed = seal(cfg, round_id, rd)
    else:
        sealed = json.loads((rd / "seal.json").read_text())
    if "evaluate" not in done:
        evaluate(cfg, rd, sealed)
    live(cfg, rd, "close")
    record = close(cfg, round_id, rd)
    crowned = crown_round(cfg, rd, record, sealed)
    if "announce" not in done:
        live(cfg, rd, "announce", push=False)
        king = announce(cfg, round_id, rd, record, sealed, crowned)
    else:  # announced before the restart: the king is in the round's own log
        king = _logged(rd, "announce", "king")
    live(cfg, rd, "export", push=False)
    exported = export_and_upload(cfg, round_id, rd, king)
    publish_close(cfg, round_id, rd, record, crowned, exported, king)
    (rd / "DONE").write_text(json.dumps({"king": king, "weights": record["weights"]}))
    _reclaim_disk(cfg)  # the round is done: its snapshots and trajectories are no longer needed on this disk
    log(rd, "done", king=king)
    return {"round_id": round_id, "king": king, "weights": record["weights"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.environ.get("SH_STATE", str(Path.home() / ".spark-hermes-state")))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--queue", default=str(Path(__file__).resolve().parents[3] / "Spark-Hermes-Withheld" / "queue"))
    ap.add_argument("--pkg", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument(
        "--worker", default=os.environ.get("SH_WORKER", "root@162.156.217.154"), help="the GPU worker, user@host"
    )
    ap.add_argument("--worker-port", type=int, default=int(os.environ.get("SH_WORKER_PORT", "40301")))
    ap.add_argument("--window", type=int, default=8, help="rounds pooled for payment")
    ap.add_argument("--window-from", default="r0001", help="the first round pooled")
    ap.add_argument("--window-minutes", type=int, default=120, help="the submission window")
    ap.add_argument("--min-paired", type=int, default=4)
    ap.add_argument("--canon-every", type=int, default=8, help="run the reference strategy every n-th round")
    ap.add_argument(
        "--submit-server",
        default=os.environ.get("SH_SUBMIT_SERVER", ""),
        help="the private submission server URL to advertise on the board (r0002+); empty keeps prose-in-PR",
    )
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--pause", type=int, default=0, help="seconds between rounds")
    ap.add_argument("--mock-miners", help="directory of mock miner bundles that submit each window (test only)")
    ap.add_argument("--mock-keys", help="directory holding the mock miners' keypairs (created on demand)")
    a = ap.parse_args(argv)
    if not attest.available():  # every signature check is `available() and verify(...)`: absent, they all pass
        print("substrate-interface is missing; signatures would go unverified. Refusing to run.", file=sys.stderr)
        return 2
    cfg = Config(
        state=Path(a.state),
        repo=Path(a.repo),
        queue=Path(a.queue),
        pkg=Path(a.pkg),
        window=a.window,
        window_from=a.window_from,
        worker=a.worker,
        worker_port=a.worker_port,
        window_s=a.window_minutes * 60,
        min_paired=a.min_paired,
        canon_every=a.canon_every,
        submit_server=a.submit_server,
    )
    mock = (Path(a.mock_miners), Path(a.mock_keys or (cfg.state / "mock-keys"))) if a.mock_miners else None
    resume = unfinished_round(cfg)  # a restart picks up the round it was in the middle of
    while True:
        try:
            result = run_round(cfg, mock, resume=resume)
            resume = None
            print(json.dumps(result), flush=True)
        except Exception as e:  # a failed round is logged, and resumed — never abandoned for a fresh one
            print(f"round failed: {e!r}", flush=True)
            if a.once:
                return 1
            time.sleep(60)
            resume = unfinished_round(cfg)
            continue
        if a.once:
            return 0
        time.sleep(a.pause)


if __name__ == "__main__":
    sys.exit(main())
