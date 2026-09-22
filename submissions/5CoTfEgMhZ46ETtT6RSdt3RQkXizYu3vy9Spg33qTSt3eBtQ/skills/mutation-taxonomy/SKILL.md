---
name: mutation-taxonomy
description: The standard injected-bug classes with a targeted search recipe for each — inverted conditions, off-by-one, swapped args, wrong constants, dropped guards, merged blocks.
---

# Mutation taxonomy — what these bugs look like

The bugs in this episode's repositories were injected mechanically. Each class below lists the
symptom it produces and the exact search that finds it. Work the list top-down; the first three
classes cover most injections.

## 1. Inverted / altered condition (and swapped branch bodies)

- **Symptom:** wrong branch taken; feature silently skipped; "not working" rather than crashing.
- **Search:** in the suspect function, read every `if`/`while` condition. For each, ask: does the
  branch body make sense for THIS condition? Inversions hide in `if not x`, `==` vs `!=`, `and`
  vs `or`, `in` vs `not in`, `is None` vs `is not None`.
- **The condition can be innocent.** A favorite trick swaps the two branch BODIES while leaving
  the condition text alone — both branches stay plausible, only their order is wrong. When a
  branch looks fine but behaves inverted, compare each body against what its condition promises,
  and against the sibling implementation's branch order.

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
- **Multi-site inside ONE function.** Dispatch/inference functions with many branches can carry
  the SAME swap in every branch (`left, right` becoming `right, left` three or four times, plus
  each branch's boolean flag flipped). Fixing the first site you find is not the fix — after any
  edit, re-read the WHOLE function and check every remaining branch for the same pattern.

## 4. Wrong constant / operator / literal — and reordered operands

- **Symptom:** subtly wrong output; tests fail on exact values; "right kind of value, wrong one".
- **Search:** collect the numbers, strings, and operators in the suspect block. Compare each with
  its twins elsewhere in the file. Wrong defaults, wrong format chars, `+` vs `−`, `//` vs `/`,
  wrong precedence from dropped parentheses.
- **Operand ORDER is semantics.** In `a or b` / `a + b` / argument lists, swapping the operands
  changes nothing visually but changes everything at runtime: `or`/`and` short-circuit (order
  decides which side runs and which value is returned when both are usable), and non-commutative
  calls silently misbehave. Near the suspect line, read every multi-operand expression and ask
  which operand the caller/cadence expects FIRST.
- **Defused validations.** A `<validate>/<check>_*` call whose argument was replaced with
  `None`, `0`, or a dummy still RUNS — and verifies nothing. For every guard call near the
  suspect line, confirm it receives the real value, not a placeholder.
- **Wrong string constants in registrations.** A module name, key, or identifier string altered
  (e.g. one dotted module name becoming a sibling) attaches a feature to the wrong target —
  silently. Read every string literal in the suspect function and check it against the names
  the tests and callers actually use.

## 5. Dropped guard / dropped line

- **Symptom:** crash on edge input; missing validation; attribute error on empty/None.
- **Search:** for every access `x.attr` or `x[i]`, ask what happens when `x` is empty/None/short.
  A missing `if not x: return` or missing `is None` check. Also: a `try` that lost its `except`,
  a loop that lost its `break`/`continue`.

## 6. Reordered, stranded, or dropped statements (the most common family here)

- **Symptom:** `UnboundLocalError` / `NameError` on a name the issue quotes; "wrong output order";
  several related tests failing at once; a feature silently skipped.
- **Search:** this family SHUFFLES statements. Concretely:
  1. Statements stranded AFTER a `return` (unreachable) — search the suspect function for
     `return` and read what follows it. An early `return` may also have been INSERTED before the
     function's real logic (a branch that now exits before doing anything).
  2. An assignment moved below its first use, or deleted outright — sometimes SEVERAL at once,
     including whole construction blocks (an object built with 5+ keyword arguments vanishing).
     When the issue says a produced result is MISSING a field or property rather than crashing,
     go to the function that CONSTRUCTS that object and read it statement by statement: a
     deleted assignment often leaves a literal `pass` or a suspiciously short branch behind.
     `search_files` the field/attribute name — if nothing assigns it anymore, that construction
     is your missing block. Restore it from its sibling path and the call site's expectations.
     (For the exact name in an error message: find its assignment and check whether that line
     still executes BEFORE the use on every path.)
  3. Statements moved across branches: a line that belonged inside `if A` now runs under `else`,
     or two branch bodies swapped. Read the branch bodies and ask "does this body make sense for
     THIS condition?" for every condition.
  4. Guards deleted: a `continue`/`skip` filter missing, so the loop now processes items it
     should exclude — compare paired positive/negative filters (if one of a complementary pair
     is missing, that is the mutation).
  5. The mutation often touches MORE THAN ONE FILE. If one site is fixed and the repro still
     fails, look for a sibling mutation in the same module family before doubting the fix.

## 7. Mutated return value

- **Symptom:** function returns the wrong object/type; callers fail far from the bug.
- **Search:** every `return` in the function: right variable? right order for tuples? `.copy()`
  missing? `None` in one branch?

## 7b. Deleted methods, removed decorators, twisted casts

- **Symptom:** `AttributeError` on a method that should exist; a behavior hook silently gone; a
  value of subtly wrong type.
- **Search:** for a missing-method symptom, `search_files` the method name in the package — if
  its definition is gone, compare the class against its siblings or its documentation for what
  methods it must expose, and restore it (interface-preserving, derived from how callers use it).
  Check every decorator above the suspect function against its twins (a removed or reordered
  `@property`/`@staticmethod`/custom decorator changes behavior invisibly). Check explicit casts
  and coercions (`int(x)`, `str(x)`, `float(...)`, `dict(...)`) against what the docstring
  promises — a twisted cast passes silently and breaks far away.

## 8. Wrong exception handling

- **Symptom:** exceptions swallowed or the wrong type raised; `finally`-style bugs.
- **Search:** `except` clauses: right exception class? re-raise present? cleanup still runs?

## 9. Whole-function rewrites (the "rewrite" family)

- **Symptom:** one function behaves almost-right — plausible names, plausible structure, wrong
  details. Tests on exact values or exact output fail while related code is fine.
- **Search:** the rewritten body reads fluently but disagrees with its own docstring, type hints,
  call sites, or the expected output in the issue. Rebuild the contract from the docstring and
  the tests' expectations, then check the body against the contract line by line — especially
  order of operations, exact strings, and boundary handling. Trust what the tests demand, not
  what the code claims.

## Output-exactness note

Several hidden checks compare EXACT strings, exact ordering, or exact formatting. When the issue
shows expected output, treat it as a spec: your fixed code must reproduce it character-for-character,
including line order. Before calling DONE, run the exact command the issue describes and diff its
output against the issue's expected block in your head (or in /tmp).

## Recipe for the first 10 minutes

1. From the issue, pick the ONE function most likely mutated.
2. If the issue shows an error naming variables, search each quoted name — the mutation family
   here loves deleting or displacing exactly those assignments.
3. Run checks 1, 3, 4 over the suspect function (they need only the function text).
4. If nothing: run the stranded-after-return scan, then sibling-symmetry and call-site checks.
5. Fix the single best candidate, verify, sweep, stop. If the repro still partially fails after a
   correct-looking fix, hunt a SECOND mutation site in the same module family (two files is
   common) — each site you fix is real credit. Try candidates ONE at a time, re-verifying.
