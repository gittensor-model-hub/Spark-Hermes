# Fix the injected bug in /testbed

Working code in the Python project at `/testbed` was edited by a tool to break it. Put it back. The tests that
expose the bug were removed; the rest of the suite is still there. You earn credit for every removed test that
passes after your change, and you score zero if any test that passed before now fails. Nobody will answer
questions.

## This page overrides the general runtime notes that follow it

Those notes tell you to gather prerequisites and to verify before you act. For this task the order is reversed:
**edit first, verify after.** You have about 25 tool calls before the budget ends the run. Runs that keep reading
until they are certain score zero, and most of them had already seen the broken line early and never changed it.
An untouched tree scores zero, so a reasoned patch is never worse than no patch. Think in a few sentences, then
act: every reply contains a tool call until you are done.

## Facts about this machine — already checked

- The project is at `/testbed`. There is no `/workspace`, `/repo` or `/app`, and no history (`git log` shows one
  commit). Work only from `/testbed`: do not look for other copies of the project or compare against them.
- Stay inside `/testbed` and `/tmp`. Do not list, read or run anything anywhere else on the machine, not even with
  `ls`: nothing out there helps, and looking there disqualifies the run.
- The interpreter with the project's dependencies and pytest is `/opt/miniconda3/envs/testbed/bin/python`. The
  `python` on PATH has neither. Write the full path every time.
- Call these tools directly: `terminal`, `read_file`, `search_files`, `patch`, `write_file`. Do not call
  `tool_search` or `tool_call`, and do not open skills: everything you need is on this page.
- If a tool call errors, correct its arguments once. If it errors again, do the same thing with a plain
  `terminal` command.
- Every tool output stays in the conversation and is paid for again on every later call.

## The loop

1. **Find the code (calls 1–2).** Search for the function, class or message the issue names, outside the tests:
   `grep -rn "<name>" /testbed --include=*.py | grep -v /tests/ | head -20`.
2. **See the bug (calls 3–4).** Put the issue's example, or a minimal call of the named function, in
   `/tmp/repro.py` and run `cd /testbed && /opt/miniconda3/envs/testbed/bin/python /tmp/repro.py 2>&1 | tail -25`.
   The last `/testbed` frame of the traceback, or the function that returns the wrong value, is the suspect.
3. **Read the suspect once (call 5).** `read_file` with `offset` from the grep and `limit` at most 60.
4. **Audit it line by line in your reasoning.** For each line ask: does it agree with the function's name,
   docstring, neighbours and callers? Fingerprints of an injected edit:
   - a name used before anything assigns it, or a lone `pass` where work belongs: a line was deleted — write it
     back just before its first use;
   - a `return`, `raise`, `break` or `continue` that makes later lines unreachable;
   - a flipped comparison or boolean (`<` for `<=`, `and` for `or`, a `not` added or dropped), an off-by-one in
     a slice or `range`, a wrong constant or operator;
   - arguments in the wrong order, or the wrong number of them, against the function being called;
   - a method the callers use that no longer exists: restore it, modelled on its siblings;
   - a body rewritten into plausible but wrong code. Its docstring may have been rewritten with it, so trust the
     callers and the surviving tests over the prose.
   Check the edge cases too: empty, `None`, the first and the last item, the fallback branch.
5. **Patch the first line that does not match, in your very next call (call 7 at the latest).** Use `patch`.
   Rerun the repro.
   **Every line looks fine?** Then the whole body was replaced: it reads smoothly but no longer does what its name,
   callers, the surviving tests and the issue require. Stop reading. Write the smallest body that does exactly that,
   using the same helpers its sibling functions use, patch it in now, and let the repro decide.
6. **Still wrong, or the issue lists more than one symptom?** Injected edits often come several at a time: in the
   same function, in sibling functions of the same file, and in **other modules of the same package directory**
   (two formats of one parser, two dialects, two parts of one subpackage). For each symptom still wrong, search that
   directory for the name involved (`grep -rn "<name>" <package dir> | head -20`), audit the hit, and patch it.
   Stuck? Change the angle: another symbol from the issue, a sibling function, a caller. Do not re-read what you
   have already read.
7. **Check.** Run the test file for the module you changed:
   `cd /testbed && /opt/miniconda3/envs/testbed/bin/python -m pytest <test file> -q -x -p no:cacheprovider 2>&1 | tail -15`.
   Make sure it actually collected and passed tests; "no tests ran" proves nothing. A test that fails now but
   passed before means your edit is wrong: narrow it or undo it with `git checkout -- <file>`.
8. **Finish.** Read `git diff`, undo anything you did not mean to change, and stop with a one-line reply.

## Keep outputs small

- Never print a whole file: no `read_file` without a `limit` of at most 60, no `cat` of a source file. Find the
  line with `grep -n` first.
- End every terminal command that can print more than a screen with `| head -40` or `| tail -40`.
- Never run the whole test suite.

## Rules — breaking one scores zero

- Fix the defect where it is, in the smallest way that restores the intended behaviour. Do not refactor,
  reformat, or work around the bug in another file.
- Do not edit, add, move or delete tests or project configuration: nothing under `tests/`, no `conftest.py`,
  `pytest.ini`, `tox.ini`, `setup.cfg`, `setup.py` or `pyproject.toml`, no test data or golden files, and never
  a flag that rewrites expected output.
- No network and no installs: no `pip`, no `curl`, no `git fetch`.
- Write only inside `/testbed` (the fix) and `/tmp` (scratch).

## Where things are

| Project | Source | Tests | Watch out |
|---|---|---|---|
| cantools | `src/cantools/` | `tests/` | command-line tests compare exact stdout |
| python-docx | `src/docx/` | `tests/`, mirrors the source | warnings are errors; `features/` is not pytest |
| python-pptx | `src/pptx/` | `tests/`, mirrors the source | warnings are errors |
| astroid | `astroid/` | `tests/`, mostly flat | warnings are errors; an xfail test that passes fails |
| sqlglot | `sqlglot/` | `tests/` | `tests/fixtures/` holds golden files |
| pygments | `pygments/` | `tests/` | snapshot tests; never regenerate them |
| oauthlib | `oauthlib/` | `tests/`, mirrors the source | exact strings and error codes matter |
| marshmallow | `src/marshmallow/` | `tests/`, flat (`test_fields.py`, `test_schema.py`) | error messages are compared exactly; pytest adds `-v`, so pass `-q` |
| sqlparse | `sqlparse/` (`engine/grouping.py`, `filters/`, `lexer.py`, `keywords.py`) | `tests/`, flat (`test_format.py`, `test_grouping.py`) | formatted SQL is compared as exact strings |
| gpxpy | `gpxpy/` (`gpx.py`, `geo.py`, `gpxfield.py`, `parser.py`) | one file, `test.py` at the root (147 tests; data in `test_files/`) | it is one large file: select tests with `-k` |

Revision r0021.
