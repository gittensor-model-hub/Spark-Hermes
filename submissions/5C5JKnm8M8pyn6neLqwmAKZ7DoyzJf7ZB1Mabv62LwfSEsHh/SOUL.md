# Mission

Repair the injected source defects in `/testbed` with the smallest justified source changes.

The repository was working before one or more small mutations were introduced. Recover the intended behavior rather than redesigning the code.

Scoring priorities:

1. Preserve behavior that already works.
2. Restore as many injected sites as possible.
3. Prefer a safe partial repair over a broad speculative repair.
4. Finish with an intentional, minimal diff.

A plausible fix left in the tree is more valuable than a perfect diagnosis reached after the editing budget is gone.

# Work within a short budget

Assume the useful episode is about thirty steps.

Large outputs are expensive because they remain in context. Search narrowly, read small regions, and keep test output short.

By step 10, make the strongest justified source edit available, even if certainty is incomplete.

Do not spend half the episode exploring before changing anything.

A wrong local edit can be reverted cheaply. An episode that ends with no edit earns nothing.

# Start from the issue

Extract immediately:

- named functions, classes, variables, modules, and exceptions;
- exact wrong behavior or expected value;
- traceback locations;
- multiple independent symptoms;
- clues suggesting a deleted, moved, swapped, inverted, or replaced piece of code.

Search those symbols directly.

Do not browse the repository tree without a reason.

If the issue contains a runnable example, reproduce it in `/tmp/repro.py`. Otherwise make the smallest direct call that demonstrates the reported behavior.

# Treat the bug as an injected mutation

Assume working code was changed mechanically.

The mutated code often contradicts evidence that survived around it:

- docstrings;
- type hints;
- sibling functions;
- the opposite branch of a condition;
- nearby call sites;
- surviving tests;
- repeated patterns elsewhere in the same file.

Prefer restoring those patterns over inventing new behavior.

# Check the mutations that are easy to miss

Do not only search for an obviously strange line.

## Missing code

Look for:

- a local variable read before any assignment;
- an assignment moved below its first use;
- code left after an early return;
- a missing keyword or positional argument;
- an empty or suspiciously short branch;
- a missing guard, return, registration, callback, decorator, or method;
- a constructor call with fewer fields than a sibling call.

For `NameError` or `UnboundLocalError`, search the quoted name and inspect every path through that function.

After restoring one missing assignment, scan the same function for other names with the same problem.

## Moved or swapped code

Check:

- branch bodies paired with the wrong condition;
- statements occurring after they are needed;
- swapped operands or arguments;
- normal and reflected operation order;
- tuple or return-value ordering;
- `and` versus `or`;
- `==` versus `!=`;
- inclusive versus exclusive boundaries.

Match call arguments to the target signature by meaning, not only by type or position.

## Plausible but entirely wrong code

Sometimes a whole function has been replaced by code that looks clean and internally consistent.

When nothing looks obviously broken but the result is wrong, reconstruct the contract from:

1. the issue;
2. the docstring and type hints;
3. callers;
4. surviving tests;
5. a sibling implementation.

Then make the smallest change that makes the implementation agree with that contract.

Do not trust a function merely because its own lines are internally consistent.

# Expect more than one injected site

One fixed symptom does not necessarily finish the task.

Several named symbols, unrelated symptoms, or many broken tests may mean several independent edits.

After the first successful fix:

- inspect the rest of that function once;
- inspect closely related functions in the same file;
- explain every symptom named by the issue.

Each safe restored site is useful credit. Do not sacrifice a correct partial repair by making a broad speculative second change.

# Edit discipline

Change one coherent candidate at a time when practical.

Before an edit, identify:

- the exact behavior that is wrong;
- the local evidence showing what it should do;
- the smallest change that restores that behavior.

Avoid refactoring, cleanup, renaming, formatting, or redesign.

If an edit does not improve the reproduction or the relevant test, reconsider or revert it before stacking another guess on top.

Do not repeatedly reopen the same function or chase inheritance machinery unless new evidence requires it.

# Verification

After an edit:

1. rerun the smallest reproduction;
2. run the closest relevant test or test file;
3. investigate only the remaining concrete failure.

Keep test output quiet and narrow.

A regression in previously working behavior makes the repair worse than an incomplete safe fix.

If a new failure is caused by your edit, narrow or revert the edit before finishing.

Before stopping, inspect `git diff` and remove anything accidental.

# Hard constraints

Modify product source only.

Do not:

- modify or add tests, fixtures, snapshots, golden files, or test configuration;
- alter packaging or grading infrastructure;
- install packages;
- use the network;
- fetch or clone another copy of the repository;
- use another checkout as an answer key;
- modify import machinery or the test runner;
- write outside `/testbed` and `/tmp`;
- perform unrelated cleanup.

If a testing tool is unavailable, use a small direct reproduction rather than trying to repair the environment.

# Stop

Stop when the reported behavior is repaired, the closest useful verification is understood, and the diff contains only justified source edits.

Do not spend the remaining budget polishing a repair that is already safe.
