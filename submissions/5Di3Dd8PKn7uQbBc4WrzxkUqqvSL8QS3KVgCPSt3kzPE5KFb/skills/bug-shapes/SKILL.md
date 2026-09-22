---
name: bug-shapes
description: The shapes injected bugs take and the search that finds each one - single-function rewrites, multi-site edits across a file or module, inverted conditions, off-by-one, swapped identifiers and arguments, wrong constants, dropped guards, changed returns and exception handling. Open this after reproducing, before editing.
---

# What the injection looks like

These bugs were produced by editing working code, not by writing broken code. That is the whole lever:
**the edited line disagrees with everything around it that was not edited.** The docstring, the type
hints, the neighbouring branch, the sibling function, the call sites and the issue text are all pre-bug
evidence. Where the code contradicts them, that is your line.

## First: how many sites?

Decide this before you start editing, because it changes when you stop.

- **One site.** The issue describes a single misbehaviour with a single symptom, and one function is
  named. Fix it, verify, sweep, finish. Even then, re-read the lines you edited against the signature or
  contract they target before you move on — the site you found is often not the whole site.
- **Several sites.** The issue reads like a list — several symbols named, several unrelated things
  described as wrong, or a vague "output is incorrect after recent changes". A large number of broken
  tests points the same way. Bugs are often injected several at a time into one file or one module, and
  they are **independent**: they are not one root cause with many symptoms. Each one you fix is credit,
  and the last one you miss does not cost you the others.

When you think there are several: after the first fix, read the **whole** file you edited — every
function, not just the one you changed — and check each against the shapes below. Then read the other
modules the issue names. Do not stop at the first green test.

A whole function can also be rewritten rather than nudged. It reads as coherent, plausible code that
simply does the wrong thing, and there is no odd-looking line to spot. Compare it against its docstring
and against what its callers expect, not against its own internal consistency.

## The shapes, in the order worth checking

**1 · Inverted or altered condition.** Symptom: the wrong branch runs; a feature is silently skipped;
"it does nothing" rather than a crash. Search: read every `if`, `elif` and `while` in the suspect
function and ask whether the body makes sense for that condition. Look at `not`, `==` against `!=`,
`and` against `or`, `in` against `not in`, `is None` against `is not None`, and at `<` against `<=`.

**2 · Swapped, reordered or renamed identifiers.** Symptom: `TypeError` about arguments,
`UnboundLocalError`, `NameError`, "NoneType has no attribute", or plausible but wrong values. Search: at
the call sites, match arguments to the signature **by name, not by position**. Inside the function,
search each local name and count where it is assigned against where it is read — a name read on a path
where it was never assigned is exactly the `UnboundLocalError` shape, and a parameter overwritten before
its first use is the same edit seen from the other side. Two arguments of the same type swapped in a
`super().__init__` call is a favourite.

**3 · Off-by-one and boundary.** Symptom: `IndexError`, the first or last element mishandled, an empty
result. Search: every slice, `range(...)`, index and `len()` in the function. `i` against `i + 1`,
`len(x)` against `len(x) - 1`, `[:-1]` against `[:]`, inclusive against exclusive.

**4 · Wrong constant, operator or literal.** Symptom: output subtly wrong; tests fail on exact values.
Search: collect the numbers, strings, format specifiers and operators in the block and compare each with
its twin elsewhere in the file. Wrong default argument, `+` for `-`, `/` for `//`, a dropped pair of
parentheses changing precedence, a changed separator or format string.

**5 · Something removed.** The hardest shape to see, because nothing on the screen looks odd — the
give-away is absence. An argument dropped from a call whose target still accepts it; a keyword argument
that sibling call sites pass and this one does not; a whole method missing from a class whose siblings
all define it; a `return` or an assignment simply gone. Check every call against the signature it
targets and count the arguments. Check a class against its siblings and list the methods each defines.
One injected edit can reorder arguments **and** drop one, so fixing the order does not finish the site.

**5b · Dropped guard or dropped line.** Symptom: a crash on edge input; missing validation; an attribute
error on empty or `None`. Search: for every `x.attr` and `x[i]`, ask what happens when `x` is empty,
`None` or short. A missing `if not x: return`, a `try` that lost its `except`, a loop that lost its
`break` or `continue`, an assignment that is simply gone.

**6 · Changed return.** Symptom: the caller fails far from the bug; the wrong type comes back. Search:
every `return` in the function — the right variable, the right order inside a tuple, a missing `.copy()`,
a branch that falls off the end and returns `None`.

**7 · Broken exception handling.** Symptom: an exception swallowed, or the wrong class raised. Search:
every `except` clause — the right exception class, the re-raise still present, the cleanup still running.

## Reading the symptom backwards

- `UnboundLocalError` or `NameError` → shape 2: an assignment skipped, removed, or renamed on one path.
- `TypeError` about arguments → shape 2: arguments swapped, reordered, or an arity changed.
- `IndexError` → shape 3.
- `AttributeError` on `None` → shape 5 or shape 6: a guard dropped, or a branch returning `None`.
- Wrong value, right type → shape 4 or shape 1.
- Exact-output mismatch → shape 4: a separator, a width, an order, a format string.
- Many unrelated tests failing at once → several sites, or one edit in a widely shared helper.

## Before you edit

Be able to say one sentence: *line L of function F in file P should do X and instead does Y.* If you
cannot say it, take the strongest candidate from the shapes above and fix that — partial credit is real
and a fix in the tree beats a diagnosis that is not. If you must try candidates, try them one at a time
and re-check between attempts; never apply several speculative edits at once, because you will not know
which one to keep.
