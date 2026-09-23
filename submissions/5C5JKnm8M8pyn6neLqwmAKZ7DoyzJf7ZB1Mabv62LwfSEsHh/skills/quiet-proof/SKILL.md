---
name: quiet-proof
description: Verify a repair with minimal test output and preserve safe partial fixes.
---

# Quiet verification

Tool output consumes the remaining reasoning budget. Keep verification narrow.

## Order

1. Re-run the smallest `/tmp` reproduction.
2. Run the closest relevant test or test file.
3. Broaden only if the narrow verification is green and budget remains.
4. Inspect `git diff` before finishing.

When pytest is available, prefer quiet output such as:

`python -m pytest <target> -q --tb=line -x -p no:cacheprovider`

Use a node id or `-k` when one test is enough.

## New failures

If a failure appeared only after your edit:

- narrow the edit; or
- revert it and try the next strongest local candidate.

Do not leave a regression in exchange for fixing another test.

If the testing tool is unavailable, do not install or search for another environment. Use a direct reproduction from `/tmp`.

## Finish

A safe partial repair is valuable.

Do not spend the final steps polishing, refactoring, or chasing unrelated failures after the intended behavior is restored.
