---
name: fault-localization
description: From issue or traceback to the exact faulty lines — traceback grounding, symbol search, sibling consistency, call-site audit.
---

# Fault localization: issue → lines

## Traceback first

If the issue contains a traceback, it names the exact failing frame: `File "<module>", line N, in
<func>`. Open that file at that line (`read_file` around the line, `offset ≈ N−40, limit 80`).
The bug is in that frame's function **or in something it calls with wrong inputs**. Walk up one
frame at a time until you see the line that produced the bad value.

For `UnboundLocalError` / `NameError`, treat the quoted variable names as the treasure map:
`search_files` each quoted name in the failing file. Its assignment was either DELETED or MOVED
below the use / after a `return` / into a sibling branch. Restore the definition to the position
that satisfies every use — the surrounding code shows what it must produce.

If the issue shows only a symptom (wrong output, wrong exception type), reproduce it first with a
scratch script in `/tmp` that imports the package and calls the public path from the issue. Never
write scratch files inside `/testbed`.

## Symbol walk

1. `search_files` each symbol from the issue (`target: content`, `file_glob: *.py`). Distinguish
   **definition** (`def name`, `class name`) from **uses**.
2. Read the definition and its immediate neighbors. Read the docstring: the docstring states the
   contract the code must keep.
3. Grep the exception text and message strings from the issue — message text lives in `raise`
   statements and pinpoints the branch taken.

## Consistency checks that catch injected mutations

Injected bugs are edits to existing code, so the code now disagrees with itself. Check, in order
of hit rate:

- **Sibling symmetry.** Does the same file contain a parallel function (another visitor, another
  branch of the same transform, the `elif` twin)? Diff the logic mentally. A mutated line usually
  has an unmutated twin. Example shape: two functions that build the same kind of expression —
  if one validates an argument and the other does not, that asymmetry is the bug.
- **Call-site audit.** `search_files` every caller of the suspect function. Count and order the
  arguments at call sites; compare with the signature. Swapped or reordered arguments and changed
  arities show up here.
- **Name coherence.** A variable that is computed but never used, or used but never computed in
  that branch, is a classic dropped-line/renamed-name mutation. `search_files` the name within the
  file and count hits.
- **Paired-name swaps.** When a block misbehaves, check whether two similar names are used
  SWAPPED — open/close, start/end, src/dst, key/value, left/right. Exchanging a pair of names
  inside one block reads perfectly naturally and even keeps the comments plausible, so it survives
  a casual read. For every near-symmetrical line, confirm each name is the one its role requires.
- **Guard check.** Conditions that guard index/attr access: is the boundary `len(x)`, `len(x) − 1`,
  `<=` vs `<`, `and` vs `or`, `is` vs `==`, `not` missing? Off-by-one and inverted conditions are
  the most common mutations of all.
- **Constant check.** Default values, slice widths, magic numbers, format strings: compare against
  the same constants elsewhere in the file and against what the issue says is expected.
- **Return-path check.** Does every branch of the function return? A branch that falls through
  returns `None` and matches "used before assignment"/"None has no attribute" symptoms.
- **Docstring/type-hint contradiction.** The signature's annotations and the docstring are pre-bug
  evidence. Code that contradicts them is suspect.

## Widen before touching code

If no line stands out against docstring, siblings, or tests: check the function's constants,
default arguments, and **decorators** against how callers actually invoke it; check any config or
class-level values it reads; check control-flow boundaries (first/last iteration, empty input,
zero). Then **bisect with prints**: add a single temporary `print` at the function's key branch
points, rerun the issue snippet, and see which branch misbehaves. Remove the prints once located.

**Two equal suspects? Do not fix both blind.** Test ONE at a time: revert candidate A, rerun the
snippet plus the nearest tests, keep only the change that actually fixes behavior. Never change
both at once — you cannot tell which fix worked or whether you added a second bug.

## Before you edit

Write one sentence: *the bug is line(s) L in function F of file P; it should X but it does Y.*
If you cannot write that sentence by ~40% of your budget, pick the single best candidate from the
checks above and fix it — partial credit is real. Say DONE only after the regression sweep.
