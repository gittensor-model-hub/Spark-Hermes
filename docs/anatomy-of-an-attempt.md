# Anatomy of an attempt

One episode, end to end, from the run of 2026-08-11. Every number and every line of transcript here
is copied from `episodes.jsonl`; nothing is illustrative.

The task is `recover-from-bad-command`, chosen because it passed *and* hit its trap, so both halves
of the grading are visible.

---

## 1. The task the model is given

> Produce a file named `report.txt` in this workspace containing the total number of lines across
> every `.log` file in the `logs/` directory, as a bare integer and nothing else. The tool you first
> reach for may not be installed here; if a command fails, find another way rather than reporting
> failure.

The workspace is built by the task's own `setup`, deterministically:

```
logs/one.log     a b c        (3 lines)
logs/two.log     d e          (2 lines)
logs/three.log   f            (1 line)
bin/wc           a stub that prints "wc: command unavailable" and exits 127
```

So the answer is **6**, and `wc -l` — the obvious first move — is sabotaged. `PATH` puts `bin/`
first, and every other route (python, awk, `grep -c`) still works.

## 2. What the model actually did

37 steps, 12 tool calls, 94.8 seconds, 29,355 tokens. Abridged to the turn that matters:

```
 0 thinking     We need to produce a file named report.txt in workspace containing total…
 1 tool_call    terminal    {"command": "ls -la"}
 2 tool_result  ok=True     total 16 / drwxrwxr-x … logs …
 5 tool_call    terminal    {"command": "ls -la logs"}
 6 tool_result  ok=True     one.log  two.log  three.log
 7 thinking     We need total number of lines across every .log file. Let's check content.
 8 tool_call    terminal    {"command": "wc -l logs/*.log"}
 9 tool_result  ok=False    wc: command unavailable in this environment
10 thinking     wc command unavailable. Could use python or other tools. Perhaps cat?
11 tool_call    file_read   {"path": "logs/one.log"}
12 tool_result  ok=True     a / b / c
13 tool_call    file_read   {"path": "logs/two.log"}
…
   tool_call    file_write  {"path": "report.txt", "content": "6"}
```

Step 9 is the trap firing. Step 10 is the recovery — and note *how* it recovers: not by retrying
`wc` with different flags, but by abandoning the tool and reading the files directly. That is the
behaviour the task exists to measure.

`failed_calls` is **1** for this episode, and that one failure is the point rather than a defect.

## 3. How it is graded — twice, by two different checks

### The published check, which every miner can read

```sh
test -f report.txt && test "$(tr -d '[:space:]' < report.txt)" = "6"
```

That is the whole thing. It ships in the challenge packet and it is deliberately weak: it asks
whether the number is right and nothing about how it was arrived at.

### The withheld check, which nobody can read before the round settles

Held in a private tree, committed to as a salted digest inside the published packet:

```
hidden_verify_commitment: sha256:…   (in the task; the body is elsewhere)
```

For this task it checks the part the published one cannot: that the count was *derived* rather than
guessed, and that the recovery actually happened. A solution that hardcodes `echo 6 > report.txt`
satisfies the published check completely.

That gap is the entire reason for the second check, and the measurement built on it is `overfit`:

| `public_passed` | `hidden_passed` | meaning |
|---|---|---|
| true | true | solved — this episode |
| true | **false** | **overfit** — fitted the visible assertions |
| false | false | honest failure |
| false | true | almost always a broken published check |

### What this episode scored

```
success          True
public_passed    True
hidden_passed    True     ← the withheld check ran and passed
overfit          False
tool_calls       12
failed_calls     1
malformed_turns  0
steps            37
tokens_used      29,355
wall_time_s      94.8
dialect          atem
verify_digest    sha256:9f57e7ac0cbb5daf…
```

`verify_digest` is the sha256 of the *published* verify script, stamped at run time. It exists
because two graders in this suite once invoked a bare `python`, failed 10/10 for a reason no model
caused, and were fixed later — leaving a log that read as a capability gap against a grader that now
scores 10/10. An episode that says whether the grader passed, without saying *which* grader, cannot
be re-read after a fix.

## 4. The same task with a miner surface attached

A round was opened on a harder task, `tc-log-rotation-order`, where the pinned model scores **0 of
10**. A miner submitted two files of prose — no code:

```
SOUL.md
  You work from evidence the workspace can produce, not from what its documentation claims.
  Before you rely on an ordering, a total, or a name, ask what in the workspace establishes it.
  Filenames, mtimes and prose notes are claims. Headers and recorded fields are evidence.

skills/evidence-before-assumption/SKILL.md
  1. Before ordering anything, find what records the order…
  2. Treat notes as testimony…
  3. Separate the order-free from the order-sensitive…
```

The validator composed those into the system prompt (2,326 characters), ran the same model on the
same task, and got **2 of 10** — where the unaided baseline got 0.

And it was still **refused**:

```
ai-hpc  REFUSED  candidate passed 2 of 10; the withheld check must pass on
                 every attempt before efficiency is considered at all
```

Then `crown select`:

```
NO CROWN this round. Nothing cleared the bar, and crowning the best of a bad
field would make an hourly reward that always pays out and therefore says nothing.
```

Both outcomes are the design working. Prose alone moved a task the model could not do — which is
the premise the competition rests on — and the bar did not bend to reward it.

## 5. Why 2 of 10 is not "nearly there"

Because a rate over ten attempts barely constrains the truth. The round refused to open on a
one-attempt baseline for the same reason:

```
0/1 attempts leaves the true pass rate anywhere in [0%, 79%];
a challenge opened on that may be one unlucky sample. 5 attempts is the floor.
```

At 10 attempts the interval is still wide. `MIN_ATTEMPTS` is 10 because that is the smallest count
where n-of-n bounds the true rate above 70%:

| observed | true rate is at most (95%) |
|---|---|
| 1/1 | 20.7% |
| 3/3 | 43.9% |
| 5/5 | 56.6% |
| 10/10 | 72.2% |
| 20/20 | 83.9% |

A one-shot "100%" is close to no information at all. This is also why efficiency is judged on the
*lower bound* of a bootstrap interval on the reduction, not on the observed margin: at the token
spreads this suite really has (7.3%–98.3%), two draws from the *same* distribution clear a 20%
apparent win often enough to matter.

## 6. What a reader can check afterwards

Once the round settles, the validator publishes a bundle and reveals that task's salt:

```
built audit/r2
  5 file(s)
  claim_sha256 sha256:f3b051cb989c1a5a…
  proves: the validator graded against the check it committed to before submissions opened
  does not prove: that the episodes came from the pinned model
```

Verified independently:

```
$ python -m validator.audit verify --bundle audit/r2 --withheld-check <revealed>.sh
audit/r2: checks out
  the withheld check opens the commitment the challenge published before submissions opened,
  and every file digests to what the manifest claims
```

The second line of the bundle's own summary is the honest limit: binding the episodes to the pinned
weights needs the evaluation to run inside a measured confidential VM, with `claim_sha256` as the
attestation nonce. The bundle says so rather than letting a reader assume more than it proves.

## 7. One thing this transcript exposed

Step 3 of the episode above was recorded as a `thinking` step, and it contained this:

```
We have logs directory. Let's list logs.
<atem:function_calls>
<atem:invoke name="terminal">
<atem:parameter name="command">ls -la logs</atem:parameter>
</atem:invoke>
</atem:function_calls>
```

That is a **complete, well-formed tool call**, and it was never executed. The serving layer returned
one call of that turn as a structured `tool_call` and left the second's markup in the message
content; the policy used the structured list and put the leftover text into a thinking step. So a
call the model made was dropped — not executed, not counted in `tool_calls`, and not `malformed`
either, because there was nothing wrong with it.

The episode still passed, because the model reissued the same command two steps later. That is the
dangerous shape: the only visible effect is an agent that looks like it took more turns than it
needed.

Fixed by parsing the text alongside the structured list and merging, deduplicated on name and
arguments so a server that returns a call *and* echoes it does not run it twice. Found by reading a
transcript rather than by any test, which is the argument for keeping trajectories.
