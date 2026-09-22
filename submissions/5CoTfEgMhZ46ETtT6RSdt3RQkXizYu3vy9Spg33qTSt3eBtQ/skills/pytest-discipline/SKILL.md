---
name: pytest-discipline
description: How to run the tests — narrow selection, scratch verification in /tmp, the regression sweep, and the never-touch-tests rule.
---

# Pytest discipline

## The one rule that zeroes your score

Never create, modify, move, or delete: test files, test data, `conftest.py`, `pytest.ini`,
`tox.ini`, `setup.cfg` `[pytest]` sections, `pyproject.toml` `[tool.pytest...]`, or any pytest
plugin/config. Verify through *running* tests and through scratch scripts in `/tmp` — never by
adding tests to the repository. If a fix seems to require a test change, the fix is wrong.

## Narrow verification (fast, cheap)

1. **Reproduce the issue's snippet** in `/tmp/repro.py` (write_file to `/tmp/repro.py`, then
   `terminal` `cd /testbed && python /tmp/repro.py`). Before your fix it should show the bug;
   after, it should pass. This is your private proof.
2. **Run the existing tests that touch your change** (you named them during orientation):
   `terminal`: `cd /testbed && python -m pytest <file or node id> -x -q`, timeout 180–300.
3. Read failures fully before changing code — most red sweeps point at your fix, not at the tests.

## The regression sweep (mandatory before DONE)

Graders zero everything if any previously-passing test now fails. Budget ~1/4 of your time:

1. Run the test files most related to your change first (`tests/test_<module>*`).
2. Then run the widest suite you can afford, at least the few files around your module:
   `cd /testbed && python -m pytest tests/ -q -x --timeout=<n>` if the project supports timeouts,
   else plain `-q`. Keep `timeout` under 300 per call; if the full suite is too slow, run the
   files for the packages you touched and say so.
3. If a test fails: is it your fix? Narrow it (prefer the smallest diff that keeps your repro
   green AND the neighbor tests green). Is it failing without your fix too (pre-existing)?
   Verify by `git stash && python -m pytest <that test> -q && git stash pop` — wait, `git` here has
   no history: instead `patch` your change out mentally, or keep a copy of the original file in
   `/tmp` before editing (`terminal`: `cp file /tmp/file.orig`) and diff/restore from there.
4. Keep a pristine copy BEFORE the first edit of any file you touch:
   `terminal`: `cp path/to/file.py /tmp/orig-name.py`. Restoring is `cp` back. This is your undo.

## End state checklist (all must hold before DONE)

- [ ] `/tmp/repro.py` shows the issue fixed.
- [ ] The neighboring existing tests pass.
- [ ] The sweep found no NEW failures (pre-existing failures: leave alone, mention in your final note).
- [ ] `git status` (the repo is a single-commit repo) shows ONLY source files changed — no test
      files, no config files, no stray files in the repo.
- [ ] No new files were created inside `/testbed` except source the fix strictly required.
