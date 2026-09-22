---
name: quiet-proof
description: OPEN before pytest: quiet flags, which file to run, how to tell a regression from a pre-existing failure, and when to revert.
---

# Proving the fix on a short budget

Every line a tool prints is re-sent on every later step. Test output is where the budget goes.

## Flags

`python -m pytest <file> -q --tb=line -x -p no:cacheprovider`

- `-q` even when the project config turns verbosity on.
- `--tb=line` until you care about one failure, then that one test with `--tb=short`.
- `-x` so the run stops at the first failure.
- `-k` or a node id when you want one test, not the file.
- No redirect of a suite into a file in the repo, and no reading that file back.

If pytest is not installed, stop. The `/tmp/repro.py` script is the proof. An install attempt ends the episode.

## What to run, in order

1. `/tmp/repro.py` from `/testbed`, before the edit and after it. If it does not fail the way the issue says, the edit is aimed at nothing.
2. The test file that mirrors the module you changed: `tests/<area>/test_<module>.py` or `tests/test_<module>.py`. On a tree whose tests are a single root `test.py`, run that file with `-k` narrowed to the area you touched. Grading draws mostly from this file.
3. The neighbouring test files in the same area, only if budget remains.
4. A wider directory only when the narrow run is green and you still have steps left. A sweep you read beats a full suite you do not.

## A red test

Ask whether it passed before your edit.

- Your change caused it: the edit is too wide. Narrow it, or `git checkout -- <path>` and re-apply a smaller one. Re-run that one test.
- It was already failing: leave it. Say so. Extra fixes are how a warning, a snapshot, or an expected-failure becomes a new failure.

A new warning can fail unrelated tests on its own in projects that treat warnings as errors. If distant tests go red after a small edit, look for a warning before you look for a logic bug.

## Done

`git diff` and read it. Only source, only lines you meant. The repro is clean. The covering test file is understood. Then stop.
