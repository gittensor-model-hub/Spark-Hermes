# spark-hermes-3.8-27b

For the operator workflow, use the [training runbook](../../../docs/train-spark-hermes.md).
`rtx5090-poc.yaml` selects the separately pinned 4B proof of concept for an RTX 5090.
`admin.cli prepare` connects these settings to the verified corpus and checks token lengths.
Its first SFT pass starts directly from the pinned base; DPO starts from that run's merged SFT
checkpoint. The A/B/C adapter chain below is an optional curriculum requiring separate data.

`Spark-Hermes-3.8-27B` is the intended final derived model, private by default. CPU fixtures
exercise preparation and artifact handling, and do not prove trained weights or quality gains.
Keep Qwen's upstream Apache-2.0 license/NOTICE and modification attribution with applicable
derivatives; keep Hermes MIT notices and separately reviewed dataset/contribution rights.
See the [contribution and release policy](../../../CONTRIBUTING.md).

The first real training target is **Qwen3.5-4B** in `rtx5090-poc.yaml`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16 LoRA with 2048-token sequences and an
RTX 5090 32 GB target. The final `bf16` profile remains 27B on a PRO 6000 96 GB target.
Both need actual memory, optimizer, serving and quality measurements after CPU validation.

The final base is
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), pinned in
[`hermes/base_model.json`](../../base_model.json) at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and `tests/test_base_model.py` keeps every stage in
this directory in step with that pin.

| stage | what it teaches | from |
|---|---|---|
| `stage-a-reasoning.yaml` | reasoning traces | the pinned base |
| `stage-b-hermes.yaml` | agentic shape, mixed corpus | the pinned base |
| `stage-c-tools.yaml` | tool mechanics, executed trajectories only | the pinned base |
| `stage-d-preference.yaml` | DPO over the competition's own near misses | `outputs/spark-hermes-3.8-27b/stage-c/merged` |

```bash
python -m admin.cli prepare --root var/admin/run-1
python -m admin.cli train --root var/admin/run-1
python -m admin.cli merge --root var/admin/run-1
python -m admin.cli prepare --root var/admin/run-1 --training-stage dpo
python -m admin.cli train --root var/admin/run-1 --training-stage dpo
```

## Three things these recipes get right on purpose

**LoRA over a bf16 base, not QLoRA over a 4-bit one.** The served model is bf16, so adapters
fitted to a quantized copy are adapters for a model nobody runs — and the mismatch is silent,
because they merge back into bf16 either way. QLoRA is the right answer when memory is the
constraint; on a 96 GB card holding ~56 GB of bf16 weights it is not.

**Stage D starts from the merge, not from stage C's adapter.** DPO needs a reference policy, and
with LoRA that comes free — disable the adapter and the base is the reference — but only if the
base *is* the policy you are improving. After A/B/C that policy is the merged model, so chaining
onto stage C's adapter would quietly optimise against a policy three stages out of date.

**`chat_template: tokenizer_default` selects the model's own template.** This base does not speak Hermes, and unlike the
previous one it does not look foreign either — it writes `<tool_call>` and `<think>`, exactly the
Hermes tags, and differs only in what goes *inside* the call tag: `<function=NAME>` with one
`<parameter=KEY>` element each, where Hermes puts a JSON object. That dialect is `qwen35`, it is
implemented in [`hermes/qwen35.py`](../../qwen35.py), and naming a Hermes template here would
train the wrong wire format into the one place a model cannot be corrected from afterwards.

The tag overlap is why this is called out rather than assumed. A reviewer checking "does the
template write `<tool_call>`?" gets yes and concludes Hermes, and every symptom afterwards points
at the model: not silence, but a *malformed* call on every single turn.

## What is not pinned yet

`sequence_len` is 8192 in every stage, which is a floor rather than a measurement. The 190-episode
baseline had a median of 34 turns, so agentic episodes are long and this is the number most likely
to need raising — after measuring headroom on the card, not before.
