# Hermes Router (Phase 4)

Dispatches a task to the specialist best suited to it — or to the generalist when it is
not sure.

```
                    User
                     |
              Hermes Router
                     |
   +----------+------+------+----------+
   |          |             |          |
  CUDA     Firmware       Cyber       SWE        ... else -> generalist
```

Routing is a much easier problem than solving, which is why a small model can do it.

## The cascade: rules first, tiny LLM only when unsure

```
task -> KeywordRouter --confident--> specialist                      (free)
             |
             +--unsure--> tiny router (3B-7B) --> specialist / generalist   (paid)
```

Most real tasks say `cutlass` or `freertos` or `pytest` somewhere, and matching a word
costs nothing. Routing *everything* through an LLM pays full price for the easy majority;
routing *nothing* through one sends every ambiguous task to the generalist. The cascade
pays only for the hard minority.

On `routing_v0` the free tier settles **74.2%** of tasks and the tiny router is consulted
on **25.8%**.

```python
from hermes.router import CascadeRouter, ModelRouter

router = CascadeRouter(escalate_to=ModelRouter(complete=my_served_3b))
router.route("Profile this Triton kernel")  # free tier, never calls the model
router.route("Something is broken. Look.")  # escalates
```

`CascadeRouter()` with no `escalate_to` degrades to the free tier alone — the correct
behavior when no tiny router is deployed yet, not an error.

### Only uncertainty escalates

`KeywordRouter` abstains for two different reasons, and conflating them wastes calls:

| `reason_code` | Meaning | Escalates? |
|---|---|---|
| `no_evidence` | the rules cannot read this task | **yes** — a model might |
| `too_close` | two domains are tied | **yes** |
| `cross_domain` | the task provably spans 3 specialists | **no** — the generalist *is* the answer |
| `empty_task` | nothing to route | **no** |

`cross_domain` is a *decision*, not a doubt. Paying a model to re-litigate it buys
nothing. That distinction rides on `reason_code`, not on matching the human-readable
reason string, so rewording a message cannot silently change what gets billed.

Every decision records the tier that made it (`tier`, `escalated`), which is what lets
`evaluate()` report `escalation_rate` — the number that says whether the cascade is
saving anything. At `0.0` the tiny router is never consulted and earns nothing; at `1.0`
the cheap tier is doing no work and you have an LLM router with extra steps.

> ## Read this before quoting any number from this harness
>
> `KeywordRouter` scores **100% accuracy on `routing_v0`**. That number is close to
> meaningless, and it is reported here so nobody mistakes it for a result.
>
> The keyword lists in `domains.py` and the labeled tasks in `tasks/routing_v0.jsonl`
> were **written by the same author in the same sitting**. The set is fitted to the
> router by construction. It tells you the router still works after a refactor; it tells
> you nothing about real traffic.
>
> Two consequences:
>
> 1. **A learned router has no headroom to prove itself here.** It cannot beat 100%, so
>    `compare()` against this suite cannot justify a model. That needs `routing_v1`:
>    tasks written by someone who has not seen `domains.py`, ideally sampled from real
>    user requests, labeled by more than one person, with disagreements kept rather
>    than resolved into false clarity.
> 2. **Cross-domain labels are judgment calls.** `cross-02` ("audit a firmware driver's
>    DMA path, then optimize it") is labeled `firmware`; a reasonable person could say
>    `cyber`. Where humans disagree, the gold label is an opinion, not ground truth.
>
> Treat `routing_v0` as a **regression harness**. It is not validation.

## Why abstention is a feature

The two routing errors do not cost the same:

| Mistake | Cost |
|---|---|
| firmware task → `general` | some quality; the generalist tries, and the same verification loop checks it |
| firmware task → `cyber` | a specialist works **confidently outside its training** |

Specialists are tuned to act. One acting outside its domain is the expensive failure, so
the router is built to fall back rather than guess, and `RoutingDecision.abstained`
records which happened. `evaluate()` reports `misroute_rate` and `abstention_rate`
separately for the same reason: a router that trades misroutes for abstentions has
improved even when its accuracy has not moved.

`compare()` encodes that — a candidate that gains accuracy while misrouting *more* is
reported as a regression, and one that merely matches the free baseline is reported as
not worth its inference cost.

## Usage

```python
from hermes.router import KeywordRouter, ModelRouter, evaluate, compare, load_suite

baseline = KeywordRouter()
decision = baseline.route("Profile this Triton kernel with Nsight")
decision.target  # 'cuda'
decision.model  # 'Spark-Hermes-CUDA'
decision.abstained  # False

# A model router needs only a `complete(prompt) -> str` callable.
candidate = ModelRouter(complete=my_served_3b)

examples = load_suite("v0")
print(compare(evaluate(baseline, examples), evaluate(candidate, examples))["verdict"])
```

## Guards on the model router

- **A hallucinated target is an abstention.** If the model names
  `Spark-Hermes-Database`, the request goes to the generalist rather than to a worker
  that does not exist.
- **Low self-reported confidence is an abstention**, enforced here rather than trusted
  downstream.
- **A dead router degrades instead of crashing.** The router sits in front of every
  request; if the endpoint throws, the generalist still gets the work.
- **A deliberate `general` is not an abstention.** Both produce `target == "general"`,
  but one is the router saying "this is generalist work" and the other is it saying "I
  don't know". They score differently, so the distinction survives parsing.

## Phase A: route to a worker configuration, not a model

> **Do not route to a model. Route to a verified worker configuration.**

A model id cannot express what makes a route *valid*: which tools the worker holds,
whether the sandbox has a GPU, how much context it takes, which verifier scores it, or
whether we may train on its output. Routing to a bare model name means those constraints
are discovered at execution time, as failures.

`AgentModule` is the routable unit — model + Hermes profile + tools + environment +
verifier + budget. Lookups go through **aliases** (`expert.cuda.optimization`), so
upgrading `spark-hermes-cuda-3.6-27b` to `-3.8-` is a registry edit, not a routing change.
Repointing an alias to a different module raises rather than silently moving production
traffic.

### Hard filters run *before* scoring

Eliminating impossible candidates is a separate stage from ranking good ones. Scoring
first would let a high capability score paper over an impossibility:

| Filter | Blocks |
|---|---|
| `missing_gpu` / `unsupported_arch` | a CUDA expert on a CPU box, or the wrong `sm_` |
| `missing_tool` | the task needs `ncu`, the module has no `ncu` |
| `action_mismatch` | the best kernel *writer* is not automatically the best *reviewer* |
| `context_too_small` / `unsupported_modality` | can't hold it, can't see it |
| `no_training_rights` | **fails closed** — unknown rights are not approved rights |

`NoEligibleModule` carries every exclusion and its reason, so "nothing was eligible" is
always explainable.

### Capability estimates are shrunk, because raw rates lie

A model with **2/2** has a raw success rate of 1.00; one with **800/1000** has 0.80.
Ranking on raw rates routes everything to whichever model we know least about. Every
estimate is a Beta posterior mean, so sparse evidence is pulled toward the prior and a
candidate only climbs once it has earned the samples. Uncertainty is kept, not discarded —
the router needs it to decide whether to widen to a second candidate.

A never-measured pairing scores the **prior**, not 0.0: never having been tried is not
evidence of failure, and scoring it as such would freeze a new expert out of the traffic
it needs to prove itself. Model versions keep separate histories, so a regression in a new
build cannot hide behind the old build's record.

### Routing mode follows the evidence

| Mode | When |
|---|---|
| `single` | clear winner in a well-covered bucket |
| `dual` | close margin, thin coverage, or **no deterministic verifier** |
| `all_eligible` | no coverage at all — the only way a bucket acquires any |

### Harness pinning

A release number is not a pin. The same tag with different tool schemas, system prompt or
container produces different behaviour, so `HarnessPin` records every digest that can
change what the agent observes. `conformance_verified` defaults to **False**: trajectories
from an unconfirmed harness must not become training data, because a harness that mangles
tool results teaches the student to expect mangled observations.

## Files

| Path | What |
|---|---|
| `domains.py` | specialist taxonomy, weighted term lists, `general` fallback |
| `base.py` | `RoutingDecision`, `Router` protocol, `abstain()` |
| `keyword.py` | deterministic baseline — the floor a model must beat, and the cascade's free tier |
| `model.py` | prompt, parse, and guards around a pluggable completion callable |
| `cascade.py` | two-tier router — rules first, tiny LLM only on genuine uncertainty |
| `evaluate.py` | accuracy, misroute vs abstention, per-domain recall, `compare()` |
| `spec.py` | `TaskSpec` — domain / action / horizon / environment / verification taxonomy |
| `manifest.py` | `AgentModule`, `ModuleRegistry` (aliases), `HarnessPin` |
| `capability.py` | `CapabilityDB` — Beta-posterior estimates with shrinkage |
| `plan.py` | hard eligibility filters, `RouteDecision`, deterministic Phase-A planner |
| `tasks/routing_v0.jsonl` | 31 labeled tasks — regression harness, **not** validation |
