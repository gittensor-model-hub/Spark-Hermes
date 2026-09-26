# Fix the injected bug in /testbed

Working code in `/testbed` was mechanically edited to break it. Put it back. No one will answer questions.

## How this task overrides the runtime notes below

The runtime notes after this section talk about prerequisite checks, not guessing, and verifying before
you act. **For this task, the order is reversed: edit first, verify after.** You have about 25 tool calls
before the budget cuts you off. Runs that keep gathering context until they are sure score 0 — almost all
of them had already read the broken line in their first few calls and never changed it. An untouched tree
scores 0 and a wrong edit scores 0, so an informed guess applied as a patch is never worse and often wins.

Think in a few sentences, then act. Every reply contains a tool call until you finish.

## The loop

1. **Locate** (1–2 calls): `grep -rn "<name from the issue>" /testbed --include=*.py | grep -v /tests/ | head -20`.
2. **Reproduce** (1–2 calls): `write_file` the issue's snippet (or a minimal call of the named function)
   to `/tmp/r.py`, run `cd /testbed && /opt/miniconda3/envs/testbed/bin/python /tmp/r.py 2>&1 | tail -n 25`.
   The last `/testbed` frame, or the function returning the wrong value, is the suspect.
3. **Read the suspect function** once: `read_file` with `offset` and `limit` ≤ 60.
4. **Audit it, line by line, in your reasoning:** for each line, "matches" or "does not match" what its
   docstring, name, neighbours and callers imply. Look hard for these fingerprints:
   - a lone `pass` where an assignment or `return` belongs; a variable used before any assignment — the
     assignment was deleted: write it back above its first use (`x = obj.x` style);
   - a `return`/`raise` placed too early with code after it; a docstring that is not the first statement;
   - a flipped operator or comparison, `not` added or dropped, an off-by-one constant, swapped arguments;
   - a method callers use that no longer exists — write it back, modelled on its siblings;
   - a body that returns something else than its docstring promises, or simulates logic ("simulate",
     fake lookups, a stub that should `raise NotImplementedError` but returns a value).
5. **Patch the first "does not match" line in your very next call.** Then rerun `/tmp/r.py`.
   **No line stands out?** Then the whole body was rewritten: it reads plausibly but no longer does what its
   docstring, name, callers and the issue say. Do not keep reading — rewrite the body minimally so it does
   exactly that (reuse the helpers its siblings use), patch it now, and let `/tmp/r.py` judge.
6. Still wrong, or the issue lists more than one symptom? Injected edits often come several at a time —
   in the same function, in sibling functions of the same file, or **in sibling files of the same package
   directory** (e.g. `diagnostics/database.py` and `diagnostics/did.py`, or two dialect modules). Rerun
   `/tmp/r.py`; for each symptom still wrong, `grep -rn "<name>" <that directory> | head -20`, audit the
   hit and patch it.
7. **Check**: `cd /testbed && /opt/miniconda3/envs/testbed/bin/python -m pytest <test file for that module> -q -x -p no:cacheprovider 2>&1 | tail -n 8`
   (gpxpy: `test.py` at the root). A failure you caused: narrow or revert that edit. Then stop with one line.

## Limits

- **Never list, read or run anything under `/ep`, `/runner` or elsewhere outside `/testbed` and `/tmp`**
  — not even `ls`. Touching the grader's paths disqualifies the run.
- Only the file(s) holding the bug are graded; fix it there. If any test that passed before now fails,
  the score is 0 — keep edits minimal.
- Call these tools directly: `terminal`, `read_file`, `write_file`, `patch`, `search_files`. Never
  `tool_call`, `tool_search` or `tool_describe`; there is no `bash`, `Read`, `Grep` or `Edit`.
- Paths are under `/testbed`, scratch in `/tmp`. There is no other copy of the project anywhere
  (`/opt/sh-pristine` is the same broken code), and no history (`git log` shows one commit).
- **Never let a tool print more than ~60 lines.** One whole-file read (30k+ characters) is what makes runs
  lose track of their tools and fail. `read_file` always gets a `limit` of at most 60 — e.g.
  `read_file(path="/testbed/pkg/mod.py", offset=240, limit=60)`, with the offset found by `grep -n` first.
  Every `terminal` command ends in `| head -n 40` or `| tail -n 40`. Never `cat` a file.
- If a result says "is not a deferrable tool" or "Tool loop warning", your next call is plain `terminal`
  with the command you meant — nothing else.
- Never: network, `pip`; editing or adding tests or test config; writing outside `/testbed` and `/tmp`;
  `inspect`, `importlib`, `subprocess`, `pytest`, `exec`, `eval` in the fix.

## Remember

Stay inside `/testbed` and `/tmp`. Keep every output under ~60 lines. Read the suspect once. Audit it line by line. Patch the first mismatch
immediately. Verify after.
