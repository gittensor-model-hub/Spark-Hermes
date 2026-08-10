# Qwen3.5-4B — Phase 1 recipe notes

Phase 1 target from the SparkDistill roadmap: prove the trajectory-generation →
Axolotl SFT → eval-harness loop end to end on a dense student model that's cheap
to iterate on. **Qwen3.5-4B** fits BF16 LoRA comfortably on RTX PRO 6000 Blackwell
(96 GB), trains faster than larger students, and uses Axolotl's native
`chat_template: qwen3_5` (no `qwen_25` stand-in).

Official model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (hybrid
Gated DeltaNet + attention; text-only SFT is supported like other Qwen3.5 dense
checkpoints).

Requires **Axolotl with Qwen3.5 examples** (see upstream `examples/qwen3.5/`) and
`transformers>=5.2.0`.

## Data mix

SparkDistill's goal is reasoning distillation, not generic instruction
following: the student should learn to reproduce the teacher's step-by-step
reasoning, not just its final answers. `data/prompts/phase1.jsonl` is
reasoning-heavy (multi-step math, logic puzzles, proof-style code
correctness) so teachers actually produce a reasoning trace worth
distilling.

The pipeline runs in two stages:

1. `teacher/generate.py` → `data/processed/phase1_trajectories.jsonl`, a
   basket of raw teacher trajectories from Claude Fable 5 and GPT 5.6
   (see `teacher/providers.py`), each carrying `response` and, when the
   teacher can capture it, a separate `reasoning` field.
2. `teacher/format.py --format messages` → `data/processed/phase1_sft.jsonl`,
   folding `reasoning` into the assistant message as a leading
   `<think>...</think>` block ahead of the final
   response — Qwen3.5's native chat template supports this format.

`sft.yaml` trains on that second file via Axolotl `type: chat_template` and
`roles_to_train: ["assistant"]` (assistant-only loss).

### Reasoning-capture limitation

Not every teacher can produce a `reasoning` value: `AnthropicTeacher` enables
Claude Fable 5's extended thinking and captures the `thinking` block — but GPT 5.6
via `OpenAICompatibleTeacher` may expose no capturable reasoning tokens over chat
completions. Trajectories from such teachers fall back to training on the response
alone (no `<think>` block). Weight the `--provider` mix toward
`anthropic` if reasoning-capture rate matters for a given run.

## Canonical mining dataset (HF)

```bash
scripts/install_train.sh
scripts/prepare_mining_sft.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml
```

`scripts/install_train.sh` installs Qwen3.5's `torchvision` dependency and Axolotl.
On **Blackwell (SM120+)**, FlashAttention 2 source builds are skipped by default
(compile fails); `scripts/train.sh` falls back to **SDPA** automatically. On Hopper
and earlier GPUs it prefers the FlashAttention 3 wheel, then attempts a FlashAttention
2 build. Set `SPARKDISTILL_SKIP_FLASH_ATTN=1` to force SDPA everywhere.
`scripts/train.sh` also disables `sample_packing` on small jsonl mixes and strips
`CutCrossEntropyPlugin` for `qwen3_5` recipes until CCE's Qwen3.5 patch is fixed.

### Fresh Blackwell miner VM — two known gotchas (issue #283)

On a minimal Ubuntu VM, training Qwen3.5 out of the box hits two coupled failures;
`train_prep` now handles both automatically:

- **`python3-dev` is required.** Qwen3.5's hybrid Gated-DeltaNet path binds
  flash-linear-attention, whose Triton CUDA kernels compile against `Python.h`.
  Without the headers the build fails and FLA rolls back to a CPU path that crashes
  mid-train. `install_train.sh` and `train.sh` warn when `Python.h` is missing — fix
  with `sudo apt-get install -y python3-dev`.
- **`sample_packing` crashes Qwen3.5 on stock Axolotl 0.18.** Axolotl's packing
  monkeypatch reads `self.layer_type`, but current transformers `Qwen3_5DecoderLayer`
  stores `self.block_type` → `AttributeError`. `train.sh` therefore **auto-disables
  `sample_packing` for Qwen3.5 recipes** until the upstream fix lands. Once you're on a
  patched/compatible Axolotl, set `SPARKDISTILL_ALLOW_QWEN3_5_PACKING=1` to keep packing.

## Local phase-1 trajectories

```bash
scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl

scripts/prepare_sft_data.sh \
  --in data/processed/phase1_trajectories.jsonl \
  --out data/processed/phase1_sft.jsonl \
  --format messages

scripts/install_train.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

Each line in `phase1_sft.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\n..."}
  ]
}
```

## Hyperparameter rationale

- **BF16 LoRA (`adapter: lora`, no 4-bit)** — 4B fits in VRAM on a single
  Blackwell workstation GPU with headroom for `micro_batch_size: 2` and seq 4096;
  avoids QLoRA dequant overhead.
- **`lora_r: 64`, `lora_alpha: 128`** — attention + MLP projections; ~5–15%
  slower than r=16, not 4×.
- **`micro_batch_size: 2`** — starting point on 96 GB; raise if tokens/sec is
  stable and VRAM allows.
- **`sequence_len: 4096`** — long enough for most teacher trajectories without
  paying quadratic attention cost of much longer context during iteration.
- **`chat_template: qwen3_5`** — native Axolotl Qwen3.5 template; do not use
  `qwen_25` for this checkpoint family.

## Hardware: RTX PRO 6000 Blackwell 96 GB

Suggested starting point (already in `sft.yaml`):

- `micro_batch_size: 2`, `gradient_accumulation_steps: 4` → effective batch 8
- `sequence_len: 4096`, `sample_packing: true`

### Training time (50k examples, pre-generated teachers)

Rough planning at **~1.5–2k tokens/example** (Triton code):

| Epochs | Tokens (50k × 1.75k) | Approx. wall time |
|---:|---:|---:|
| 1 | ~87M | **8–18 h** |
| 2 | ~175M | **16–35 h** |
| 3 | ~262M | **24–50 h** |

Measure tokens/sec after 100–200 steps. For synthetic Triton teacher data, prefer
**1–2 epochs** unless held-out TritonBench composite keeps improving.

At `sequence_len: 8192`, multiply wall time by ~1.5–2× if packing length approaches the cap.

## Known gaps (contributions welcome, see `CONTRIBUTING.md`)

- No data-quality filtering pass on raw teacher trajectories yet (e.g.
  length/format/refusal filtering) — recipe currently trains on whatever
  `teacher/generate.py` emits.
- **`CutCrossEntropyPlugin`** — remove from `sft.yaml` if your Axolotl build lacks it.
- **Linear attention modules** (`linear_attn.*`) — add to `lora_target_modules` only
  if you need to adapt Gated DeltaNet paths; start with standard attn + MLP.
