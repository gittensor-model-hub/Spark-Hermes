# spark-hermes-glimmer-30b

`Spark-Hermes-Glimmer-30B` is the model this pipeline produces. The base is
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B), pinned in
[`hermes/base_model.json`](../../base_model.json) at
`97c77dff50b2797bcc558fa2d909761dbc575c59`, and `tests/test_base_model.py` keeps every stage in
this directory in step with that pin.

| stage | what it teaches | from |
|---|---|---|
| `stage-a-reasoning.yaml` | reasoning traces | the pinned base |
| `stage-b-hermes.yaml` | agentic shape, mixed corpus | the pinned base |
| `stage-c-tools.yaml` | tool mechanics, executed trajectories only | the pinned base |
| `stage-d-preference.yaml` | DPO over the competition's own near misses | `outputs/spark-hermes-glimmer-30b/stage-c/merged` |

```bash
scripts/train.sh     hermes/recipes/spark-hermes-glimmer-30b/stage-c-tools.yaml
scripts/merge_lora.sh hermes/recipes/spark-hermes-glimmer-30b/stage-c-tools.yaml
scripts/train.sh     hermes/recipes/spark-hermes-glimmer-30b/stage-d-preference.yaml
```

## Three things these recipes get right on purpose

**LoRA over a bf16 base, not QLoRA over a 4-bit one.** The served model is bf16, so adapters
fitted to a quantized copy are adapters for a model nobody runs — and the mismatch is silent,
because they merge back into bf16 either way. QLoRA is the right answer when memory is the
constraint; on a 96 GB card holding ~60 GB of bf16 weights it is not.

**Stage D starts from the merge, not from stage C's adapter.** DPO needs a reference policy, and
with LoRA that comes free — disable the adapter and the base is the reference — but only if the
base *is* the policy you are improving. After A/B/C that policy is the merged model, so chaining
onto stage C's adapter would quietly optimise against a policy three stages out of date.

**`chat_template` names the model's own file.** This base speaks ATEM, not Hermes: no `<tool_call>`,
no `<think>`. Naming a Hermes or Qwen template here would train the wrong wire format into the one
place a model cannot be corrected from afterwards.

## What is not pinned yet

`sequence_len` is 8192 in every stage, which is a floor rather than a measurement. The 190-episode
baseline had a median of 34 turns, so agentic episodes are long and this is the number most likely
to need raising — after measuring headroom on the card, not before.
