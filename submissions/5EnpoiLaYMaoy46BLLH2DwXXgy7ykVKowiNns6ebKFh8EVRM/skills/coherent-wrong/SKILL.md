---
name: coherent-wrong
description: OPEN when the function looks consistent but the result is wrong, or on RecursionError. A whole-function rewrite or shuffled statements. Match the body to the contract.
---

# When the code looks finished and is not

Two injections produce a function with no ugly line. Tell them apart, then edit one candidate.

## Shuffled control flow

Read the function as a sequence of effects, in order.

- Search for `return`. Anything after it is dead. An early `return` inserted above the real work makes the function look short and "successful" while skipping the behaviour the issue describes.
- For every `if`, `elif`, and `else`, ask whether **that body** belongs under **that condition**. Swapped bodies leave both the condition and the body looking plausible. The tell is the pairing, not the line.
- A statement that belongs in one arm now runs in the other, or runs in both. Compare with the sibling function, which usually still has the original order.
- Wrong output order, a feature that silently does nothing, or the right value from the wrong branch: this shape, not a bad constant.
- In generator-style inference code, order is the result. A `yield` turned into a `return`, a missing yield of the "cannot decide" sentinel, or two yields swapped, changes what callers see while the function still reads naturally.

## A rewritten function

The body is coherent, named well, and does the wrong thing. Stop looking for a typo. Build a contract, then check the body against it.

The contract, in this order:

1. The docstring, sentence by sentence. Each sentence should be a behaviour you can point at in the body.
2. Callers. What they pass, and what they do with the return value.
3. Surviving tests. The exact values they assert.
4. The sibling: the same method on a sibling class, a sibling dialect, a sibling grant, or the next property in the same class.

Then check these, because rewrites break them and still look tidy:

- An attribute name the issue says is missing. Search the class the attribute is read on. A sibling property already uses the real name. Use that spelling.
- A method that calls itself. `RecursionError` while copying, cloning, or resolving an attribute on a proxy is this. The sibling class shows the non-recursive access. Restore that, and do not add a cache or a guard the sibling does not have.
- Argument order at `super().__init__` and at any call whose parameters share a type. Match by name.
- A condition flipped, or a default and a class-level constant changed, inside an otherwise faithful rewrite. Compare each literal with the sibling.
- A cast (`int`, `str`, `float`, `list`) the docstring does not ask for.

## Edit

Change the smallest span that makes the body match the contract. If the rewrite replaced the whole function and the contract disagrees in many places, rewrite back toward the sibling and the docstring, not toward a new design. One attempt, then the repro. If the repro moves but does not pass, the shuffle and the rewrite can both be present: scan for a second site in the same file before you undo the first.
