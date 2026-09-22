---
name: codebase-map
description: Layout, test command and regression traps for the Python repositories these bugs come from - sqlglot, python-docx, python-pptx, cantools and pygments. Open this once you know which repository you are in, before running any test.
---

# The repositories

Find out which one you are in with one cheap call: `ls /testbed`. The source is either a top-level
package directory or under `src/`. Then read the matching section and nothing else.

Two questions are worth answering before you touch anything: **where does the module in the issue live**,
and **which test file covers it**. The second is what protects you from scoring zero, because the tests
kept for grading are drawn mostly from the file that covers the code you changed.

## sqlglot

- Source: `sqlglot/` at the top level. Tests: `tests/`.
- The parts: `tokens.py` and the tokenizer, `parser.py`, `generator.py`, `expressions` (the AST node
  classes — a module in some versions, a package in others), `optimizer/` with one module per rule
  (`qualify_columns`, `unnest_subqueries`, `pushdown_predicates`, `merge_subqueries`, `simplify`,
  `eliminate_*`, `annotate_types`, `scope`), `dialects/` with one module per SQL dialect, `transforms.py`,
  `planner.py`, `executor/`.
- Idioms that make mutations visible: an expression class declares `arg_types`; a dialect nests its own
  `Tokenizer`, `Parser` and `Generator`; parsers key off dicts like `FUNCTIONS` and `STATEMENT_PARSERS`;
  generators off a `TRANSFORMS` dict and `<node>_sql` methods. Dialects are near-copies of one another,
  so **the same method in a sibling dialect is the reference for what the mutated one should say**.
- Trap: `tests/fixtures/` holds `.sql` golden files (`identity.sql`, `optimizer/*.sql`) that the tests
  read and compare against. They are test data. Never edit them — a failing fixture means your fix is
  wrong, not that the fixture is.
- Tests: `python -m pytest tests/test_optimizer.py -q`. The optimizer file includes TPC-H and TPC-DS
  cases and is slow; select a single test with `-k` while iterating and run the file once at the end.

## python-docx

- Source: `src/docx/`. Tests: `tests/`, mirroring the package (`tests/oxml/`, `tests/text/`,
  `tests/parts/`). `features/` is a separate acceptance suite, not pytest — ignore it.
- The parts: the public objects in `document.py`, `table.py`, `text/paragraph.py`, `text/run.py`,
  `section.py`, `shape.py`; the XML layer in `oxml/` (custom lxml element classes, declared with the
  descriptors in `oxml/xmlchemy.py`, value types in `oxml/simpletypes.py`); packaging in `opc/` and
  `parts/`.
- Where behaviour lives: a public method on a `docx/` object usually delegates straight to a `CT_*`
  element class in `oxml/`. If the public method looks right, the bug is one level down.
- **Trap, and it is the main way to score zero here: the pytest configuration sets `filterwarnings` to
  `error`.** Any warning your change newly raises — a deprecation, a resource warning — is a test
  failure across unrelated tests. Do not introduce a new warning, and do not silence one.
- Tests: `python -m pytest tests/test_table.py -q`, or the mirrored path for the module you touched.

## python-pptx

- Source: `src/pptx/`. Tests: `tests/`, mirroring the package. `features/` is behave, not pytest.
- The parts: `presentation.py`, `package.py`, `shapes/`, `chart/`, `text/`, `table.py`, `util.py`,
  document properties under `opc/`, and the same `oxml/` element-class layer as python-docx, with the
  same `xmlchemy` descriptors.
- Same trap: `filterwarnings` is `error`. A new warning fails unrelated tests.
- Tests: `python -m pytest tests/test_<module>.py -q` at the mirrored path.

## cantools

- Source: `src/cantools/`. Tests: `tests/`, with data files in `tests/files/`.
- The parts: `database/can/` (`database.py`, `message.py`, `signal.py`, `node.py`, `bus.py`) and
  `database/can/formats/` (`dbc.py`, `kcd.py`, `sym.py`, `arxml/`); the command line in `subparsers/`
  (`list.py`, `dump/`, `decode.py`, `convert.py`, `plot.py`, `monitor.py`, `generate_c_source.py`).
- Where behaviour lives: the subcommand modules print. Their tests compare **exact stdout**, so
  whitespace, column order and line breaks are the contract — matching the expected output character for
  character is the fix, not an approximation of it.
- Note: the pytest configuration lives in `tox.ini` and sets `addopts = -v`, so runs are verbose by
  default. Pass `-q` yourself to keep the output small.
- Tests: `python -m pytest tests/test_list.py -q`, `tests/test_database.py` (large, slow — use `-k`),
  `tests/test_command_line.py`, `tests/test_convert.py`.

## pygments

- Source: `pygments/` at the top level. Tests: `tests/`.
- The parts: `lexer.py` (`Lexer`, `RegexLexer`, `DelegatingLexer`, `ExtendedRegexLexer`, and the helpers
  `bygroups`, `using`, `include`, `default`), `lexers/` with one module per language family and the
  generated `lexers/_mapping.py`, plus `formatters/`, `filters/`, `styles/`, `token.py`, `util.py`.
- Where behaviour lives: a lexer is mostly a `tokens` dict of regex-to-token rules, and the mutation is
  usually inside one rule or in a constructor. A `DelegatingLexer` subclass passes its two lexers up to
  `super().__init__` in a fixed order — root lexer, then language lexer — and swapping them is a classic
  injected edit.
- **Trap, and it is the main way to score zero here: most of the suite is golden-file snapshot tests**
  generated by `tests/conftest.py` from `tests/snippets/` and `tests/examplefiles/`. A change to a shared
  lexer or to `lexer.py` can fail hundreds of them at once. The suite also offers a flag that rewrites
  those golden files — **never use it**. It edits the tests and scores zero.
- Tests: `python -m pytest tests/test_basic_api.py -q` for the core, and the snippet directory named
  after the lexer alias you touched. `python -m pytest tests/ -q` is large; reach for it only with budget
  left.

## If it is none of these

The shape still holds: source at the top level or under `src/`, tests mirroring it under `tests/`, data
and golden files inside the test tree and never to be edited. Find the test file whose name matches your
module and run that first.
