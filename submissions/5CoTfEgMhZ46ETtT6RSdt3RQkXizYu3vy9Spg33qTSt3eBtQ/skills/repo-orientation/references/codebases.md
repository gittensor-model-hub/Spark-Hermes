# Repo-family notes (debugging recipes for the common codebases)

Load the section matching the repository you are in. Everything here is generic knowledge about
these open-source projects — fall back to the 10-step orientation ritual for anything else.
Layouts: `src/<pkg>/` means the package lives under `src` (imports work because the project is
installed in development mode); a flat layout keeps the package at the repository root.

## SQL tooling / transpilers (e.g. sqlglot-style projects)

- Package at the root; pure Python; tests under `tests/` (unit tests like `test_optimizer.py`,
  per-dialect tests under `tests/dialects/`).
- Smoke: `python -c "import <pkg>; print(<pkg>.parse_one('SELECT 1'))"` — if parsing works, the
  core is importable.
- These projects transform an expression TREE: optimizer rules are small functions in
  `<pkg>/optimizer/` that take and return expression nodes. A mutation in a rule usually breaks
  tree invariants: an undefined local (moved assignment), a branch that now clears instead of
  keeps an attribute, or a `group_by`/`where` clause silently skipped. Trace the quoted variable
  names from the issue inside the named rule function.
- Dialect files (`<pkg>/dialects/*.py`) contain small parser/formatter methods per dialect;
  check for statements stranded after `return` and methods whose body no longer matches the
  tokens it claims to parse.
- Verify with: `python -m pytest tests/test_optimizer.py -q` and the dialect test for the file
  you touched.

## Document generators (python-docx / python-pptx style)

- Layout `src/<pkg>/`; the public API is a thin wrapper over an XML layer (`<pkg>/oxml/`).
- Mutations cluster in `oxml/` property setters/getters (one twisted expression — e.g. a reversed,
  mis-encoded, or wrong-attribute value) and in part factories. The issue usually shows a wrong
  or corrupted property value; find the setter for that property (`search_files` the property
  name) and check its expression against the docstring and the expected output.
- Tests: `python -m pytest tests/ -q` (unittest-style classes run fine under pytest).
- Quick property round-trip in /tmp: build a minimal document object, set the property, read it
  back, compare with the issue's expected value.

## CAN-bus / parsers with CLIs (cantools style)

- Layout `src/<pkg>/`; a CLI subcommand module (`<pkg>/subparsers/`) prints formatted output.
- Mutations love the printing helpers: statements reordered (wrong line order), conditions
  inverted (`strict=not no_strict` becoming its opposite), branch bodies swapped (`if a: X / elif
  b: Y` becoming the reverse), filters dropped, early `return` inserted. When the issue shows
  expected CLI output, compare line ORDER first.
- Smoke: `python -m <pkg> dump <a file from tests/> 2>&1 | head` and compare against the issue.
- Tests: `python -m pytest tests/ -q`; CLI-behavior tests assert captured output.

## Syntax highlighting / lexers (pygments style)

- Package at the root; `tests/` mixes normal test functions with GENERATED items: files under
  `tests/snippets/<lexer>/` and `tests/examplefiles/` are themselves test cases (a custom
  collector runs the lexer over each file and compares tokens). Missing data files under those
  directories are part of the test surface.
- `tests/conftest.py` defines the custom collection — read it if collection errors appear.
- Mutations hit `<pkg>/lexers/<language>.py`: wrong regex boundaries, swapped delegating-lexer
  arguments, dropped `analyse_text` methods (symptom: files detected as the wrong language), or
  changed constants in `analyse_text` (slice widths, version numbers in regexes).
- Smoke: `python -c "from <pkg> import lex; print(list(lex('<sample>', Name.Function)))"` with a
  sample from the issue.

## Static-analysis libraries (astroid / pylint-style)

- Package at the root (`astroid/`); tests under `tests/` using helper utilities that parse code
  snippets into nodes.
- Two hot zones: node classes (attributes, `infer` logic) and the `brain/` modules that teach the
  library about stdlib/framework behavior. Mutations there flip an inferred type/value or break a
  small transformation. The issue usually names the construct it mis-handles; find the node or
  brain function handling that construct (`search_files` the construct's name) and check its
  logic against a minimal example in /tmp.
- Verify: run the test module matching the area (`python -m pytest tests/ -q`, or the single
  test file for the touched module).

## Track/geo parsers (gpxpy style)

- Small package (`<pkg>/`) with one core module (`gpx.py`-style) holding the data classes and
  math (distances, bounds, moving/stopped times, elevation gains).
- Mutations twist the math: wrong constants (radius, thresholds), inverted comparisons
  (`>` vs `>=`), swapped min/max, wrong start/end of a range. The issue shows wrong numbers —
  recompute the expected number by hand from the issue's data in a `/tmp` script, then trace
  which computation step diverges.
- Round-trip test in /tmp: parse the issue's sample, re-serialize, compare fields.

## Protocol/spec libraries (oauthlib style)

- Layout `<pkg>/oauth1|oauth2/...`; behavior is dictated by RFCs — the spec IS the docstring.
- Mutations flip a required check: a missing `is not None`, a wrong required-flag, an error
  raised where a success is expected (or the reverse), wrong attribute read from a request
  object. The issue names the flow (token, authorize, revoke) and the expected error/success —
  mirror the flow in a `/tmp` script with minimal inputs, then walk the code path and compare
  each guard against the expected outcome stated in the issue.
- Verify: the test file for that endpoint/flow (`tests/`), run only that file.

## Anything else

Use the orientation ritual; the taxonomy and localization skills carry the general method.
