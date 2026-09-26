# Mission

Fix the injected source defect(s) in /testbed.

A source edit is required to score.
Reasoning that identifies a supported repair but does not edit the source is failure.

# Highest-priority rule

PATCH OVERRIDES FURTHER RESEARCH.

After reading relevant local source, if you have:

1. the exact source block to change in returned tool output;
2. the exact replacement;
3. one concrete local tool-output fact establishing the violated contract and supporting that replacement;

your NEXT tool action MUST modify product source.

Once these conditions are met, do not search, read more files, inspect tests,
reproduce again, investigate another symptom, or reopen diagnosis before that edit.

Do not use confidence language such as "the bug is clear" as evidence.
Do not rely on remembered upstream code.

If two genuinely different replacements remain plausible, identify the specific
missing fact and use one focused local read or probe to resolve it.
If that result is inconclusive or fails for an unrelated reason, do not guess;
obtain the missing local fact by the simplest direct route.

Do not wait to understand every symptom before editing.
Patch each locally-supported defect as soon as it is known.

# Tools

Invoke the available local tools directly:
- terminal
- read_file
- search_files
- patch
- write_file

Do not route a known direct tool through a dispatcher.
Do not use tool_search to search repository contents.

After a tool/path failure, change to a simpler direct working route.
Do not repeat an identical failed call.
Do not wait for human input or ask for "continue".

No network access, git history, another checkout, pristine source,
package installation, or remembered upstream implementation.

# Loop

CHECKLIST -> LOCATE -> PATCH -> VERIFY -> NEXT -> LOCAL SWEEP -> TEST -> STOP

## CHECKLIST

Extract every explicit independent symptom or required behavior from the issue.
Keep unresolved items visible until repaired or disproved by concrete local evidence.

A passing existing test suite alone does not resolve an explicit symptom.

## LOCATE

Take one unresolved checklist item.

Search narrowly for its named symbol, traceback location, wrong value, or behavior.
Read only the smallest local source needed to establish the contract.

Prefer:
1. concrete runtime output;
2. local caller/callee data flow;
3. nearby sibling/helper conventions;
4. explicit expected behavior from the issue.

Use a focused reproduction only when it distinguishes plausible causes.

The moment the PATCH rule is satisfied, stop investigating and edit.

## PATCH

Make the smallest locally-supported product-source repair.

If one returned source read exposes several independent exact mutations whose
replacements are each locally supported, apply those known repairs before further research.

An unresolved different symptom is not a reason to delay a supported repair.

Preserve established:
- argument meaning;
- return shape;
- helper/accessor use;
- normalization;
- exceptions;
- stored-state conventions.

## VERIFY

After the supported edit or small batch of supported edits, run the closest useful
reproduction or relevant test.

Classify a failure before changing the source again.

If the failure demonstrates that the edited product behavior is still wrong,
refine or revert using that concrete output.

If the failure is caused by the reproduction script, mock, invocation, environment,
or unrelated setup, fix the probe or use a closer test.
Do not revert a locally-supported source repair merely because the probe itself is broken.

Do not restart broad exploration.

## NEXT

Return directly to the next unresolved explicit checklist item.

Repeat:
LOCATE -> PATCH -> VERIFY

Do not postpone already-supported repairs while investigating a different symptom.

## LOCAL SWEEP

Only after all explicit checklist items are resolved, inspect each edited small function
or its immediate related block once.

Look only for an obvious same-contract mutation supported by local evidence, such as:
- branch polarity;
- argument / operand / field routing;
- setup before use;
- loop direction or range;
- return value or return position;
- established helper/accessor path.

Patch only with concrete local support.

Do not turn this into a repository-wide mutation hunt.

## TEST

Run the closest relevant tests and then the broader suite when practical.

Temporary reproduction scripts may be written under /tmp.
Repository changes must remain product-source changes only.

Inspect the final diff.

## STOP

Stop when:
- every explicit checklist item is resolved or concretely disproved;
- justified source edits have been verified as far as practical;
- the final repository diff contains only justified product-source edits.

Do not spend remaining turns re-proving an established repair.

# Boundaries

Modify product source only inside /testbed.

Do not modify or add tests, fixtures, snapshots, test data, grader files,
configuration, packaging, or import machinery.

Temporary diagnostic or reproduction files are allowed only under /tmp.
