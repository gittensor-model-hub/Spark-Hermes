# HermesBench

An agent benchmark that measures **work**, not knowledge.

MMLU and HumanEval ask whether a model knows things. A worker that knows everything and
verifies nothing is the exact failure mode this pipeline exists to train out — so
HermesBench scores episodes instead:

| Metric | Question it answers |
|---|---|
| `success_rate` | did the task actually get done, per the task's own check |
| `tool_efficiency` | ok calls / total calls — did the agent flail |
| `recovery_rate` | of episodes where a tool failed, how many still succeeded |
| `self_check_rate` | of episodes that changed something, how many looked afterwards |
| `mean_tokens` | what the answer cost |
| `mean_wall_time_s` | how long it took |
| `objective_completion` | long-horizon: how many sub-objectives held at the end |
| `objective_regression_rate` | long-horizon: how many it satisfied and then **broke** |

**The agent's final message is never read.** Success is the exit status of the task's
`verify` command, run against the workspace the agent actually modified. A model that
writes "Fixed the bug. All tests pass." over a still-red suite scores zero, which is the
entire point.

## Long-horizon tasks (`v1`) and goal drift

Short episodes cannot show the failures that matter at length — goal drift, context
corruption, an agent quietly undoing its own work. A single end-of-episode `verify`
scores "finished objective A then destroyed it" **identically** to "never did A".

`v1` tasks declare **checkpoints**: sub-objectives checkable on their own, sampled
repeatedly while the episode runs. That turns drift from an anecdote into a measurement.

```yaml
checkpoint_every: 1          # sample after every step
checkpoints:
  - checkpoint_id: median-even
    description: median is correct for even-length input
    verify: python -c "..."
```

Two agents on `migrate-and-keep-green`, both ending with a confident "done":

| Agent | verified | objectives | **regressed** |
|---|---|---|---|
| deletes the shim, breaking its own earlier fixes | ✗ | 0/4 | **3** |
| strips the import first, then deletes | ✓ | 4/4 | 0 |

Only the timeline separates them.

**The measurement under-reports, deliberately.** Sampling is periodic, so an objective
broken and repaired between two samples is invisible. It never *over*-reports — a false
accusation of drift would be worse than a missed one. Lower `checkpoint_every` to buy
resolution; each step costs one sweep of every checkpoint command.

## Verification is not charged to the action budget

`max_steps` and `max_verification_steps` are separate, and a task names its read-only
tools in `verification_tools`.

Sharing one budget made the harness contradict itself: `self_check_rate` rewards an agent
for re-running the tests, while a single budget charges it for doing so and can cut an
episode off mid-verification. Models trained to spend tokens checking their work are
exactly the ones that would be penalised. A task that declares no `verification_tools`
keeps the old single-budget behaviour, so every `v0` task is unchanged.

## The four capability categories

A Hermes-native worker is judged on four things, and each is a task tag so
`load_suite(tags=("long_horizon",))` selects one and `suite_metrics` reports a success
rate per category:

| Category | Question |
|---|---|
| `tool_calling` | right tool, right arguments, no flailing |
| `terminal_agent` | multi-command shell work driven to completion |
| `long_horizon` | objective held across many steps without drift |
| `self_verification` | **evidence generated, not a result asserted** |

Scoring per category rather than pooling is deliberate. A model can be strong at picking
tools and hopeless at holding an objective for a hundred steps; one aggregate number hides
exactly that, and lets a category with many easy tasks carry a category with few hard
ones. `category_support` is reported alongside each rate so a thin category is visible.

`self_verification` is the signature category. Its tasks are built so a plausible-looking
naive approach produces a **confident but wrong** answer, and only actually running
something gives the right one — which is the behaviour the whole pipeline is trying to
train. It pairs with the integrity layer: `verification_skipped` disqualifies an agent
that declared success without ever checking, and `unmeasured_claim` flags a number that
appears in no tool output.

## Writing a task

One YAML file per task under `tasks/<version>/`:

```yaml
task_id: fix-failing-test
tags: [swe, python, debug]
prompt: |
  What the agent is asked to do. Never mentions the verification command.
tools: [terminal, file_read, file_write, python]
timeout_s: 300
max_steps: 30
env:                      # optional; expanded, so $PATH prepends rather than clobbers
  PATH: "./bin:$PATH"
setup: |                  # builds the workspace; runs before the agent sees anything
  ...
verify: |                 # exit 0 == the task was accomplished
  ...
```

Rules that keep a task honest:

- **`verify` must fail on the untouched workspace.** A task that passes before the agent
  starts measures nothing.
- **`verify` must pass on a correct solution.** Check both directions before committing a
  task; `tests/test_hermesbench_tasks.py` enforces the first, and the second is on you.
- **Make failure modes real, not narrated.** The recovery task shadows `wc` with a
  failing stub on `PATH` rather than telling the agent to pretend `wc` is missing.
- **Grade the outcome, not the route.** Where the route matters (fix the source, not the
  test), assert it in `verify` — `fix-failing-test` greps that the assertion survived.
- **Keep the margin wide.** `verify-speedup-claim` measures with min-of-repeats and
  refuses to grade if the performance gap collapses below 1.5×, because a task whose
  winner flips between runs grades noise.
- **Stdlib only where possible.** A task that needs `pytest` on `PATH` partly measures
  the operator's environment.

## Running

No served-model policy ships yet — wiring one is the Phase 0 integration point. Implement
`AgentPolicy`:

```python
from hermesbench.runner import LocalToolExecutor, run_suite
from hermesbench.tasks import load_suite


class MyPolicy:
    def next_steps(self, task, history) -> list[Step]:
        """Return thinking/tool_call steps, or a final step to stop."""

    @property
    def tokens_used(self) -> int: ...


metrics, results = run_suite(
    load_suite("v0"),
    policy_factory=lambda task: MyPolicy(),
    executor=LocalToolExecutor(allow_unsandboxed=True),
    workspace_root=Path("/tmp/hermesbench"),
)
print(metrics.to_record())
```

The runner executes each tool call for real and appends the observed result before asking
the policy again, so **a policy never sees a tool result it invented**. `ReplayPolicy`
replays a recorded trajectory's agent-side steps against a live workspace — useful for
exercising the harness itself.

List the suite without running it:

```bash
python -m hermesbench.runner --suite v0 --workspace-root /tmp/hb --list
```

## Sandboxing

`LocalToolExecutor` runs **model-authored shell commands on the host**. It refuses to
start unless you pass `allow_unsandboxed=True`, which asserts the process is already
inside a container, VM, or disposable machine. That flag provides no isolation itself —
it exists so that "the benchmark wiped my home directory" requires someone to have typed
the words. Path arguments to `file_read`/`file_write` are confined to the workspace, but
`terminal` runs a shell and a shell goes wherever the process can.

## Executed vs simulated

Episodes run here are **executed**: their tool results came from real commands, and the
resulting trajectories carry `metadata.executed = true`. Trajectories from
`hermes.generate` are **simulated** — a teacher wrote down what it imagined the tools
would return — and carry `executed = false`. Both are useful; only the first is evidence.
Train verification behavior on executed data.
