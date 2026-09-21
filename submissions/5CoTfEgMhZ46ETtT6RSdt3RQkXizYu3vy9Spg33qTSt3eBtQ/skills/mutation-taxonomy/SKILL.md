---
name: mutation-taxonomy
description: The standard injected-bug classes with a targeted search recipe for each — inverted conditions, off-by-one, swapped args, wrong constants, dropped guards, merged blocks.
---

# Mutation taxonomy — what these bugs look like

The bugs in this episode's repositories were injected mechanically. Each class below lists the
symptom it produces and the exact search that finds it. Work the list top-down; the first three
classes cover most injections.

## 1. Inverted / altered condition

- **Symptom:** wrong branch taken; feature silently skipped; "not working" rather than crashing.
- **Search:** in the suspect function, read every `if`/`while` condition. For each, ask: does the
  branch body make sense for THIS condition? Inversions hide in `if not x`, `==` vs `!=`, `and`
  vs `or`, `in` vs `not in`, `is None` vs `is not None`.

## 2. Off-by-one / boundary

- **Symptom:** `IndexError`, first-or-last element mishandled, empty-result bugs.
- **Search:** every slice `x[a:b]`, `range(...)`, index `[i]`, and `len()` use. Check bounds
  against the loop's intent: inclusive or exclusive? `i+1` vs `i`, `len(x)` vs `len(x)−1`,
  `[:−1]` vs `[:]`.

## 3. Swapped / reordered / renamed identifiers

- **Symptom:** `TypeError` (wrong argument), wrong-but-plausible values, `NameError`,
  UnboundLocalError, "None has no attribute".
- **Search:** at call sites, count arguments against the signature and match them by NAME, not
  position. Inside the function, `search_files` each local name; a name assigned in one branch and
  used in another (or a parameter whose value is overwritten before use) is the bug. UnboundLocal
  errors: find the path where the assignment is skipped.

## 4. Wrong constant / operator / literal

- **Symptom:** subtly wrong output; tests fail on exact values.
- **Search:** collect the numbers, strings, and operators in the suspect block. Compare each with
  its twins elsewhere in the file. Wrong defaults, wrong format chars, `+` vs `−`, `//` vs `/`,
  wrong precedence from dropped parentheses.

## 5. Dropped guard / dropped line

- **Symptom:** crash on edge input; missing validation; attribute error on empty/None.
- **Search:** for every access `x.attr` or `x[i]`, ask what happens when `x` is empty/None/short.
  A missing `if not x: return` or missing `is None` check. Also: a `try` that lost its `except`,
  a loop that lost its `break`/`continue`.

## 6. Merged or reordered blocks (the "combine" family)

- **Symptom:** several related tests fail at once; a whole transform behaves wrong; duplicated or
  missing statements inside a long function.
- **Search:** read the whole function top to bottom in two slices. Look for statements that appear
  twice, in the wrong order relative to their dependencies, or missing entirely (a construction
  step absent while its result is used). Compare the function against its sibling implementation:
  the sequence of operations should mirror the sibling's.

## 7. Mutated return value

- **Symptom:** function returns the wrong object/type; callers fail far from the bug.
- **Search:** every `return` in the function: right variable? right order for tuples? `.copy()`
  missing? `None` in one branch?

## 8. Wrong exception handling

- **Symptom:** exceptions swallowed or the wrong type raised; `finally`-style bugs.
- **Search:** `except` clauses: right exception class? re-raise present? cleanup still runs?

## Recipe for the first 10 minutes

1. From the issue, pick the ONE function most likely mutated.
2. Run checks 1, 3, 4 over that function (they need only the function text).
3. If nothing: run the sibling-symmetry and call-site checks (need two more files at most).
4. Fix the single best candidate, verify, sweep, stop. Never shotgun multiple fixes at once —
   if you must try candidates, try them ONE at a time, re-verifying between attempts.
