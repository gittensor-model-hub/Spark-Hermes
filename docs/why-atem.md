# Was changing the dialect the right call?

The project's goals are **a stable harness, faster learning, and less ecosystem breakage**. This
repository is named for Hermes and now speaks a second wire format. Whether that serves those three
goals is a fair question, and it has measured answers rather than opinions.

## The dialect was discovered, not chosen

`meta-models/Muse-Glimmer-30B` ships its own `chat_template.jinja`. It renders tool calls as
`<atem:function_calls>` / `<atem:invoke>` / `<atem:parameter>`, returns results in
`<tool_output name="…">`, and addresses deliberation to a `self` recipient. It contains no
`<tool_call>` and no `<think>`.

So there was never a decision between two formats for one model. There was a decision between
speaking the pinned model's format and speaking a different one at it. `hermes/base_model.json`
records the dialect as evidence — read off the template and then confirmed by generation — not as a
preference.

## What forcing Hermes actually cost

Measured, not assumed. Instructing the pinned model in Hermes and parsing its output as Hermes:

| dialect at this base | malformed turns |
|---|---|
| `hermes-4` | **15 in 3 episodes** |
| `atem` | **0** across 19 tasks, 1.69 M tokens, ~300 calls |

`malformed_turns` is one of the two numbers the promotion gate bounds. At 15-in-3 the gate is
measuring the mismatch between harness and model, and every miner surface is scored through that
noise. That is not a stylistic argument against Hermes-at-this-base; it is the harness failing.

## Against each goal, honestly

### Stable harness — net better, but it cost two rounds

ATEM took the harness from 15 malformed turns in 3 episodes to 0. It also opened two defect classes
that Hermes would not have, because every adapter around `<tool_call>` JSON is more battle-tested:

- a serving layer's reasoning parser swallowed call markup, and 8 calls across 11 episodes were never
  executed, never counted, and not malformed either
- the training corpus rendered in the Hermes row shape could not be trained by the model's own
  template at all — and its third fault was silent

Both are fixed and both now have tests that assert on the artifact a consumer actually reads.
[`anatomy-of-an-attempt.md §7`](anatomy-of-an-attempt.md) has the numbers. The honest accounting:
ATEM is the more stable choice *and* it cost real debugging that Hermes would not have.

### Faster learning — ATEM wins, and now measurably

Two reasons, one of which only became visible after the corpus fix.

The obvious one: malformed retries cost tokens, and efficiency is what the promotion gate scores.

The one that matters more: an SFT corpus rendered in a format the base does not speak teaches it to
*unlearn* its own template. The corpus fault made this concrete — reasoning was being written as
`<think>` inside `content`, and the model's own template ignores `content` on any turn that carries
tool calls. Every reasoning block on every tool-calling turn was being dropped before the trainer saw
it. Fixing that put reasoning on the `assistant to=self` channel the model already reasons in, so a
cycle now reinforces the base's format instead of fighting it.

### Less ecosystem breakage — this is where ATEM costs

The real price, and it should not be minimised:

- a Hermes-runtime consumer cannot run these weights
- vLLM 0.27.0 has no native support for the architecture (`vllm#51655`, open), and its
  `--model-impl transformers` fallback returns incoherent output for it
- SGLang works from a branch (`sglang#34262`, open), so serving is a branch build today

Part of that is upstream and out of this project's hands. Part of it is the direct consequence of
picking a base whose format is not the one the agent ecosystem standardised on.

## The alternative worth taking seriously, and why not

**Train toward Hermes** — keep ATEM for baseline measurement, but render the SFT and DPO rows in
Hermes so each cycle closes the ecosystem gap instead of widening it.

It is a genuinely attractive idea and the corpus work is what argues against it. To train Hermes rows
against this base you must replace its chat template: the model's own has no `<think>` notion and
drops message content beside tool calls. The moment you fork the template, the fine-tune stops
honouring the base's serving contract — and `hermes.conformance`, which checks that every marker the
parser reads is still present in the *pinned* template, is no longer describing what you serve. You
would trade a serving-stack gap for a template fork, and a template fork is the harder thing to
verify.

There is a cheaper version of the same benefit. `Dialect` is data, not branches, and the harness
already speaks both formats — so presenting an ATEM model behind a Hermes-shaped endpoint is a
translation layer at the boundary, not a retraining. That buys ecosystem reach without asking the
weights to unlearn their own format.

## Where it stands

Keep ATEM as both the measurement and the training dialect: it is the pinned model's own contract,
`hermes.promotion.check_graders` already refuses to compare runs across dialects, and the corpus now
renders into the channel the model's template reads.

Close the ecosystem gap at the boundary rather than in the weights, and treat the upstream serving
PRs as the thing to watch — when either lands, the largest remaining part of goal three resolves
without anything in this repository changing.

One caveat worth stating plainly: `Serving` records precision, device, engine and sampling, and
`check_graders` refuses a cross-dialect comparison — but `Serving` itself does not record the dialect.
The refusal works off the episode metrics rather than off the serving record. That is enough today
because both come from the same run, and it is worth tightening before anyone compares runs recorded
by different operators.
