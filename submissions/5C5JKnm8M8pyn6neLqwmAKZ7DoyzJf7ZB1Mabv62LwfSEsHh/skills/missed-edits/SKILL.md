---
name: missed-edits
description: Find injected mutations that look plausible: deleted assignments, moved statements, swapped branches, and coherent but wrong function bodies.
---

# Missed edits

Use this when the obvious-looking line is not enough to explain the failure.

## Deleted or moved code

For every name mentioned by a traceback or issue:

- find where it is read;
- find where it is assigned;
- check that every execution path assigns it before use;
- check for assignments moved below their first use or after a return.

Also look for absence:

- an empty branch beside a complete sibling branch;
- a call missing an argument that parallel calls include;
- a missing guard before indexing or attribute access;
- a missing registration, callback, decorator, or return;
- a constructor missing fields used by its sibling.

After finding one deleted assignment, scan the same function once for others.

## Swapped control flow

A mutation can leave every individual line looking reasonable.

Check whether:

- an `if` body belongs to the opposite condition;
- two branch bodies were exchanged;
- a statement now executes too early or too late;
- operands or call arguments were swapped;
- output or tuple elements are reversed.

Judge the pairing of condition and body, not only whether each looks valid alone.

## Coherent but wrong function

A whole body may have been replaced by plausible code.

Reconstruct the intended contract from:

1. issue text;
2. docstring and type hints;
3. callers;
4. surviving tests;
5. sibling implementations.

Then change only the smallest span that conflicts with that contract.

Do not redesign the function.

## Partial credit

Several independent mutations may exist.

Keep a verified repair even if another site remains unresolved.

Do not risk a known-good partial fix with a broad speculative rewrite.
