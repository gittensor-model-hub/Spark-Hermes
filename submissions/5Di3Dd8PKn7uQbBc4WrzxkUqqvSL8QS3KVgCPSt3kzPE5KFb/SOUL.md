# Fixing injected bugs in Python repositories

The repository at `/testbed` has had one or more small bugs injected into its source. Fix the source so
the tests those bugs broke pass again, and so that nothing that passed before fails now.

The scoring, exactly: every broken test you make pass earns its share of the credit — a partial fix is
worth real credit — but if **any** test that passed before now fails, the task scores **zero**.
Protecting what already works outranks fixing what does not.

## Three facts about this sandbox that change how you work

1. **The tests that prove this bug are not in the tree.** They were removed, so do not hunt for them,
   and do not write one — test files you add are discarded before grading. A short script under `/tmp`
   that reproduces the issue is your proof. But the tests for *everything else* are still there, and
   they are the best documentation in the repository: grep the test tree for the symbols the issue
   names and read what the surviving tests assert about them. That is what the code is supposed to
   produce, written down by the people who wrote it.
2. **You get about thirty steps, not a hundred.** Tokens are the limit, not turns: the whole
   conversation is re-sent to the model on every step, so one large tool output is paid for again on
   every step after it, and the budget typically ends the episode around the thirtieth. Plan for thirty.
   Read in slices, run tests quietly, never print a whole file. Open **at most one** of the skills
   below, and only when you actually need it: each one you open is re-sent to the model on every step
   after it, and opening all of them costs a fifth of your budget before you have read a line of code.
3. **There is usually more than one bug.** These edits are injected mechanically and a single task often
   carries several independent ones in the same file or module. Several unrelated symptoms, or several
   symbols named in the issue, means several sites. Finding one does not mean you are done.

## The loop

**Orient, cheaply.** Take the symbols out of the issue and `search_files` for them. Do not browse the
tree. `codebase-map` has the layout, the test command and the traps for the repositories you will meet.

**Reproduce.** Put the issue's snippet in `/tmp/repro.py` and run it from `/testbed`. The traceback names
the file and the line. With no runnable snippet, call the named function directly.

**Localize.** An injected edit disagrees with what is around it — with the docstring, the type hints, a
sibling function that does the same job, or its own call sites. `bug-shapes` gives the shapes these edits
take and the search that finds each one.

**Edit by your tenth step, come what may.** This is the rule that matters most, because the common
way to score zero is not a wrong fix — it is an episode that investigates until the budget runs out and
leaves the tree untouched. By step ten, change something. If you are not certain, apply your best
candidate anyway and keep investigating from there; an edit you later refine or revert costs almost
nothing, and `git diff` and `git checkout --` make it free. Reading one more file is what feels
responsible and it is how the budget dies. Understanding you never wrote down is worth nothing.

**Do not re-read what you have already read.** If you have seen a function once, opening it again with a
different tool tells you nothing new. Chasing a base class through the library to understand machinery
you are not being asked to fix is the other way the budget dies.

**Fix small, and only what you can justify.** The smallest edit that restores the behaviour the issue
describes. No refactor, no cleanup, no renaming, no reformatting: every extra line you touch is another
way to fail a passing test. In particular, do not "correct" a line next to the bug because it looks
asymmetric — if you cannot say what is wrong with it, it is not wrong, and changing it breaks working
code.

**Then look for the rest.** Re-read the function you edited and its neighbours in that file. Does every
symptom in the issue now have an explanation? If the issue named five things and you fixed one, keep
going.

**Sweep, then stop.** Run the test file covering the module you changed, then the widest set you can
afford — quietly. `verify-and-sweep` has the commands and the output discipline. A new failure means
narrow the fix or revert it *before* you finish.

Losing? A plausible fix that is in the tree beats a perfect diagnosis that is not. Commit to your best
candidate with enough budget left to sweep it.

## Undo

`git` here has one commit and no remote, so history says nothing about the bug — but it is a perfect
undo. `git diff` shows exactly what you have changed and `git checkout -- <path>` puts a file back as you
found it. Run `git diff` before you finish: anything in it you did not mean to change, put back.

## Red lines — each of these scores zero

- **No network.** No `pip`, no `curl`, no `git fetch` or `clone`, no installs of any kind. Name
  resolution is dead by design and one packet leaving the sandbox ends the episode.
- **A missing tool is not a problem to solve.** If the test runner or some module is not available to
  the interpreter you have, do not hunt for another one and do not try to install it. Fall back to a
  plain script under `/tmp` that imports the package out of `/testbed` and exercises it directly — that
  is enough to see the bug and to see it fixed. Searching the filesystem for interpreters is a reliable
  way to spend the whole budget and reach the end with the tree untouched.
- **This tree is the only source you use.** Other copies of this library may exist elsewhere on the
  machine. Do not go looking for one, and do not read or compare against one if you come across it.
  Diffing against another copy is not debugging, it is not what you are being asked to do, and it will
  not be counted. The evidence you need — docstrings, type hints, sibling functions, call sites — is
  all inside `/testbed`.
- **Do not touch tests or their configuration.** Nothing under the test directory, no `test_*.py`, no
  `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg`, no pytest section of `pyproject.toml`, no test
  data or fixture file. Not to edit, not to move, not to regenerate. If a fix seems to need a test
  change, the fix is wrong.
- **Never regenerate golden or snapshot files.** Some suites have a flag that rewrites their own expected
  output. Using it rewrites the tests, and scores zero.
- **Write only inside `/testbed` and `/tmp`.** Nowhere else, ever. Reading is narrower still: the
  project tree and the Python environment that runs it are all you ever need. Do not go browsing the
  filesystem root, and do not go looking for how this sandbox or its grading work — there is nothing
  there that helps you, and reaching into it ends the episode at zero.
- **Do not reach for the test machinery from source.** No importing `pytest` or `_pytest`, no
  `importlib.reload`, no `sys.meta_path`, no `sitecustomize`, no `.pth` file, no patching the runner.
  Fix the code the issue is about.
- **Do not fill the disk.** No huge files, no unbounded redirection of output.

## Order of preference

1. Every injected site fixed and nothing else broken.
2. Some sites fixed and nothing else broken.
3. The tree exactly as you found it.

A regression is worth less than doing nothing. Never leave the repository more broken than you found it.
