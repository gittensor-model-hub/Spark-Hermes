---
name: repo-orientation
description: First 10 steps inside an unknown repository — layout, test config, import name, module map. Run before diagnosing.
---

# Repo orientation ritual (10 steps, ~6 tool calls)

Do these in order and keep the answers in one short mental note.

1. **List the top level.** `search_files` with `target: files`, `pattern: *`, `path: /testbed` — or one
   `terminal` `ls /testbed`. Read `README.rst`/`README.md` title lines only (first ~20 lines).
2. **Find the packaging metadata.** Read `pyproject.toml`, `setup.py`, or `setup.cfg` — get the
   package import name (the `name=` field or the directory with `__init__.py`) and the Python version.
3. **Map the source tree.** `search_files`, `target: files`, `pattern: *.py`, `path: /testbed/<package>`
   to see module names. Module names are the vocabulary of the bug.
4. **Locate the tests.** The test directory (`tests/` usually) and its config: `pytest.ini`,
   `tox.ini`, `setup.cfg` `[tool:pytest]`, or `pyproject.toml` `[tool.pytest.ini_options]`.
   Note `testpaths`, `addopts`, and any `conftest.py` fixtures that matter (parametrization over
   data files is common). **Do not modify any of these files.**
5. **Check the import works.** `terminal`: `python -c "import <package>; print(<package>.__version__ if hasattr(<package>, '__version__') else 'ok')"`.
   If the import itself fails, that failure is your first symptom.
6. **Find the issue's symbols.** Extract every class, function, method, and module name from the
   issue text, then `search_files` (`target: content`, `file_glob: *.py`) each. You now have the
   files that matter. Usually 1–3.
7. **Read the issue once more** with the file list in hand: which module, which operation, what
   breaks, what is the expected behavior. One sentence.
8. **Note the era of the code.** If the issue mentions `DeprecationWarning`s or old APIs, the bug
   may be a migration miss; check for a newer sibling implementation of the same operation.
9. **Plan the verification.** Which existing test files exercise the suspect module
   (`search_files` `pattern: <module>` under the test dir)? Name them now; you will run them later.
10. **Write your one-line diagnosis hypothesis**, even if vague: "the bug is probably in
    `<module>.<function>` because `<reason>`". Then go localize precisely.

Keep every step cheap. If a step takes more than two tool calls, skip ahead — steps 6–7 matter most.

If the repository matches a well-known open-source family (SQL tooling, document generators,
CAN-bus CLIs, syntax highlighters), view the `codebases` reference of this skill first — it has
family-specific layouts, smoke commands, and the mutation patterns typical of each. If the repo
matches nothing here, the 10 steps above are the whole method.
