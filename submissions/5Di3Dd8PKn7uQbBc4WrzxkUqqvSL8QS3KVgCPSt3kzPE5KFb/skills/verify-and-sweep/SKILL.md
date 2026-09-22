---
name: verify-and-sweep
description: How to prove a fix and protect the tests that already pass - the scratch repro script, cheap pytest invocations that keep output small, which tests to run and in what order, reading a failure, and reverting safely. Open this before the first test run.
---

# Proving the fix without spending the budget

Every line a tool prints is re-sent to the model on every later step. Test output is the single largest
thing you will produce, so it is where the budget is won or lost. The defaults below are not style: a
verbose suite run can cost more than the rest of the episode.

## Keep output small

- Always `-q`. Add `--tb=line` for a first look — one line per failure is enough to decide what to do —
  and only re-run the single failing test with `--tb=short` when you need the detail.
- Add `-x` while iterating, so a run stops at the first failure instead of printing fifty.
- Select narrowly with `-k <substring>` or a node id (`path::TestClass::test_name`) rather than running
  a file to see one test.
- `-p no:cacheprovider` avoids writing a cache directory into the repository.
- Never `cat` a whole file. Read in slices around the line you care about. Use `search_files` to find a
  symbol rather than printing a module to look for it.
- Never redirect a suite's output into a file in the repository and then read it back. That pays twice.

## If the test runner is not there

`No module named pytest` is not something to fix. There is no network, an install attempt ends the
episode at zero, and searching the filesystem for another interpreter will eat the budget. Fall back to
your `/tmp` script: import the package out of `/testbed` and call the code path the issue describes.
That shows you the bug before the fix and its absence after, which is the proof that matters. Do not
look for, or compare against, another copy of the project elsewhere on the machine.

## Read the surviving tests before you edit

The tests for this bug are gone, but the ones around them are not. `search_files` the test tree for the
symbol the issue names: what the remaining tests assert is the contract the code must keep, in exact
expected values. This is cheaper than reasoning from the source alone and it is the difference between
guessing a constant and knowing it.

## The order to run things

1. **Your repro, first.** Write the issue's snippet to `/tmp/repro.py` and run it from `/testbed`:
   `cd /testbed && python /tmp/repro.py`. Before the fix it must show the failure described in the
   issue. If it does not, you have not understood the issue yet and the fix will be aimed at nothing.
   After the fix it must be clean. This is your only direct proof, because the tests written for this
   bug are not in the tree — do not go looking for them, and do not add one.
2. **The test file that covers the module you changed.** Mirror the source path into the test tree:
   a change in `<pkg>/<area>/<module>.py` is covered by `tests/<area>/test_<module>.py` or
   `tests/test_<module>.py`. This is the most important run you will make — the tests kept for grading
   are drawn mostly from here.
3. **The neighbours.** The other test files for the same area or subpackage.
4. **The widest set your remaining budget allows**, `-q --tb=line`. If the whole suite is too slow or
   too loud, run the directories nearest your change and stop there. A partial sweep that you actually
   read beats a full sweep that eats the budget.

## Reading a failure

Ask one question first: **did this test pass before my change?**

- If your change caused it, the fix is too broad or wrong. Narrow it. Prefer the smallest edit that
  keeps your repro clean and the neighbours green.
- If it failed before you touched anything, leave it alone. Pre-existing failures are not yours, and
  "fixing" them is extra edits and extra risk. Say so at the end instead.

To tell the two apart, `git diff` shows exactly what you changed, and `git checkout -- <path>` restores
a file to the state you found it in. Revert, re-run the one test, restore your fix. That is cheaper and
more reliable than keeping copies by hand, because the repository is a clean single commit.

A new **warning** can be a failure on its own: some of these projects turn warnings into errors in their
pytest configuration, so a deprecation your change newly raises will fail tests that have nothing to do
with it. If unrelated tests go red after a small edit, suspect this before you suspect the edit's logic.

## What never to do

- Do not add, edit, move or delete a test file, a `conftest.py`, a `pytest.ini`, a `tox.ini`, a
  `setup.cfg`, the pytest section of a `pyproject.toml`, or any data or golden file inside the test
  tree. It scores zero, whatever the intention.
- Do not run a flag that rewrites a suite's expected output. Regenerating golden files rewrites the
  tests.
- Do not install anything. There is no network; a failed install costs turns and a packet that leaves
  the sandbox ends the episode.
- Do not create files inside `/testbed` that the fix does not require. Scratch work goes in `/tmp`.

## Before you say you are done

- `git diff` — and read it. Only source files, only the lines you meant, nothing left in by accident.
- Your repro runs clean.
- The test file covering your change passes, and anything newly red is understood and dealt with.
- If you ran out of budget before sweeping: still say clearly what you changed. A fix that is in the
  tree counts even if you never got to prove it.
