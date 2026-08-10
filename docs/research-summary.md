# SparkDistill & SparkProof — Research Summary

**Audience:** anyone joining SN74 / Gittensor who needs the full picture in plain language.  
**Scope:** what we build, how trust works, how KernelLLM / TritonForge / agent traces generate data, and a practical recipe for the **best SN74-verifiable dataset**.  
**Date:** July 2026. **Last reviewed for staleness:** 2026-07-29 (§4, §7, §8, §18, Key PRs updated).

---

## Table of contents

1. [One-minute pitch](#1-one-minute-pitch)
2. [The two repos](#2-the-two-repos)
3. [What “verified” means](#3-what-verified-means)
4. [Two mining tracks](#4-two-mining-tracks)
5. [How miners are stopped from cheating](#5-how-miners-are-stopped-from-cheating)
6. [Can miners use cheaper teacher models?](#6-can-miners-use-cheaper-teacher-models)
7. [Frontiers: are they up to date?](#7-frontiers-are-they-up-to-date)
8. [Is TritonBench a fair judge?](#8-is-tritonbench-a-fair-judge)
9. [Multi-turn episodes (better data, not just more rows)](#9-multi-turn-episodes-better-data-not-just-more-rows)
10. [Public datasets: KernelBook, traces, KernelBench](#10-public-datasets-kernelbook-traces-kernelbench)
11. [KernelLLM (Meta) — method in depth](#11-kernelllm-meta--method-in-depth)
12. [TritonForge (RLsys) — method in depth](#12-tritonforge-rlsys--method-in-depth)
13. [Other methods worth knowing](#13-other-methods-worth-knowing)
14. [How to generate the best dataset](#14-how-to-generate-the-best-dataset)
15. [Side-by-side comparison](#15-side-by-side-comparison)
16. [What SparkProof / SparkDistill should take from others](#16-what-sparkproof--sparkdistill-should-take-from-others)
17. [End-to-end workflows (cheat sheets)](#17-end-to-end-workflows-cheat-sheets)
18. [Open work and risks](#18-open-work-and-risks)
19. [Glossary](#19-glossary)
20. [Links](#20-links)

---

## 1. One-minute pitch

**Triton-native AI is the proof of concept.**  
The real goal is a **closed loop that keeps producing expert-level models from a pipeline alone**: teachers write data → data is cryptographically proven → students train → eval raises a public frontier → repeat.

Labs run that loop behind closed doors. We run it **open**, on **SN74 / Gittensor**, with rewards only for **measured, re-checkable** wins.

**Headline claim:** *Expert models on autopilot — an open, attested pipeline that keeps raising the frontier.*

---

## 2. The two repos

Think of two cooperating systems:

| Repo | Job in one sentence |
|---|---|
| **[SparkProof](https://github.com/gittensor-model-hub/SparkProof)** | Generate Triton training **trajectories** and **prove** every kept row ran on a real confidential-compute GPU. |
| **[SparkDistill](https://github.com/gittensor-model-hub/SparkDistill-Hermes)** | Train **student** models on that data, **score** them against a public frontier, and pay SN74 for verified wins. |

**SparkProof answers:** “Is this training row real and policy-clean?”  
**SparkDistill answers:** “Did training actually make a better model?”

Production proving hardware today: **NVIDIA RTX PRO 6000 Blackwell** (and Hopper H100/H200) under **Intel TDX** confidential compute (e.g. Targon / SN4). Verification of proofs is meant to be **cheap on CPU** for third parties.

Pinned teachers (dataset generation): **Claude Fable 5** and **GPT 5.6 Sol**, at reasoning effort **`xhigh`**, via OpenRouter or yunwu — not arbitrary chat models.

---

## 3. What “verified” means

A “verified” dataset row is **not** “a model said this looks good.”

It means, together:

1. **Teacher policy** — requests match pinned models and `xhigh` effort (`request_sha256` recomputed from the pinned request shape).
2. **GPU validation** — the Triton kernel **compiled and ran** on the architecture claimed (Blackwell or Hopper).
3. **Release gate** — decontamination (no eval leakage), novelty accounting, hashes consistent.
4. **Attestation** — GPU confidential-compute token (NRAS) and, in production, **Intel TDX** quote bound to the **dataset content**, not to miner-editable JSON fields.
5. **Merkle / manifest** — leaves and roots match; you cannot silently swap trajectories after gating.

**Training-track proofs** add: recipe + dataset pin + eval claim, then either full retrain or **attested cheap re-score** against the frontier.

**Mental model:** CSV is a claim. `proof/` is a **receipt**.

---

## 4. Two mining tracks

### Dataset track (`dataset:xs` … `dataset:xl`)

Miner flow:

1. Run SparkProof on an allowed CC VM.  
2. Publish HF dataset with rows **and** `proof/`.  
3. Open a **text-only** PR on SparkDistill appending one line to `datasets/registry.jsonl`.  

Reward bands are mostly by **novel verified row count after mix dedupe** (fair label), not by “I uploaded 200 lines.” Rough thresholds: XS ≥ 25, S ≥ 50, … XL ≥ 150 novel rows (see miner guide for live numbers).

### Training track (`eval:xs` … `eval:xl`, plus `eval:BASELINE`)

Miner flow:

1. Train the student on the **canonical mining pin** (same data for everyone).  
2. Beat the **architecture-bucketed frontier** (Blackwell vs Hopper are separate).  
3. Ship a proof bundle + attestation; CI fail-closes forged crypto.

**Training tiers pay 2×** the same letter as dataset tiers — a real frontier win is worth more than adding rows. This is SparkDistill's documented intent (`.gittensor/weights.json`), but as of this update the **live** `entrius/gittensor` `master_repositories.json` still pays both tracks equally (`eval:XL` = `dataset:xl` = 4.0); a PR to double the `eval:*` side to match is open ([entrius/gittensor#1660](https://github.com/entrius/gittensor/pull/1660)), unmerged.

Tooling / docs / refactors without a verified quality win score **0** — and as of the community-PR policy below, a community PR that isn't a training/dataset submission is auto-closed rather than reviewed-and-maybe-merged.

---

## 5. How miners are stopped from cheating

Focus: dataset track (SparkProof → HF → registry).

### Cheats that fail (hard)

| Cheat attempt | Why it fails |
|---|---|
| Hand-write `gpu_attestation.json` with `"passed": true` | Online verify checks **NVIDIA JWKS** signature on NRAS tokens. |
| Steal a valid NRAS token, point it at another bundle | Nonce / claim binding uses **signed** `eat_nonce`, not editable `claims` JSON. |
| Forge TDX `report_data` in JSON | Binding reads **REPORTDATA from `quote_b64`**, not miner-editable fields. |
| Swap rows after release gate | `trajectories.jsonl` must match `dataset_manifest.trajectories_sha256` and the PR hash. |
| Lie about Merkle root | Leaves are rehashed; root must match. |
| Mark failed kernels as gold | Raw → verified consistency checks. |
| Train on TritonBench / KernelBench problems | Release gate + `FORBIDDEN_TRAINING_ORIGINS` + decontam fingerprints. |
| Duplicate prior registry rows for a big tier | Novelty + **fair label from rows after cross-registry dedupe**. |
| Commit giant datasets in the PR | Registry PR is **append-only one line**. |

Validators **re-download HF** and re-run policy + crypto. They do not trust the miner’s word.

### Residual gaps (honest)

1. **Enclave boundary** — we prove “attested guest + policy,” not “miner’s process was morally pure.”  
2. **Quality farming** — many **easy, novel, compiling** kernels can still earn size tiers; attestation ≠ “expert.”  
3. **Teacher ledger** — OpenRouter generation lookup needs the **miner’s API key**; CI usually verifies NRAS/TDX, not every teacher receipt.  
4. **HF mutability** — verify-at-PR-time; the canonical `sparkproof-mining` mix is the long-term pin for training.

---

## 6. Can miners use cheaper teacher models?

**Openly: no.**  
Policy requires Fable / Sol @ `xhigh`. Writing `gpt-4o-mini` (or similar) into the bundle **fails** verify.

**Secretly labeling a cheap model as Fable/Sol:**  
Offline checks enforce **claimed** request shape and recorded gateway model IDs. They do **not** fully prove, without the OpenRouter ledger, that the **response bytes** came from that API call. So teacher identity is **strong policy**, not a complete lab signature on every token.

**Practical takeaway:** Cheap models cannot be admitted honestly. Closing the remaining gap needs miner-key escrow or mandatory teacher-ledger checks — separate from GPU attestation, which remains solid for **kernel execution**.

---

## 7. Frontiers: are they up to date?

Frontiers live in SparkDistill:

- `runs/frontiers.json` — per architecture (`blackwell`, `hopper`)  
- `runs/ledger.jsonl` — every merged training-track run  
- `runs/frontier.json` — legacy Blackwell-only file  

Both frontiers are still at their original baselines as of this update — no verified non-`eval:REJECT` training win has landed since: Blackwell (gsm8k **0.6**, triton **≈0.428**), Hopper (gsm8k **0.74**, triton **≈0.372**).

**Two mechanical bugs found and fixed, both in the merge → ledger → frontier path itself (not the scores):**

1. *Verified training PRs never auto-merged.* The dataset-track gate has always auto-merged a `dataset:xs`+ PR; the training-track gate never grew the equivalent — a fully verified `training:valid` + real `eval:*` PR just sat open until a human merged it by hand. Fixed in [#289](https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/289): `gate_training_pr` now reports `merge_eligible`, and CI merges on pass, mirroring the dataset side.
2. *The post-merge ledger write couldn't land.* `training_track_ledger.yml` raw-pushed the ledger/frontier update straight to `main`, which a branch-protection ruleset rejects (`GH013: changes must be made through a pull request`) — so the frontier was **never actually crownable by automation**, even for a fully verified win. Fixed in [#290](https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/290): the writer now falls back to a squash-merge PR when the direct push is blocked (mirroring how `datasets/canonical.json` refreshes already worked), and a `workflow_dispatch(pr_number)` input lets a specific merged PR be (re)crowned by hand.

**Why this is a good "is the frontier honest" case study:** the two bugs combined to nearly cause a *false* rejection. [#288](https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/288) (`magicrails`, `eval:XS`, the first candidate improvement over the Blackwell baseline) passed the gate with a fresh, valid GPU attestation — but because auto-merge didn't exist yet, the PR sat open for ~2 hours before a maintainer merged it manually, by which time the attestation's 1-hour validity window had expired. The post-merge re-verification correctly fail-closed to `eval:REJECT` (`attestation_integrity_failed: JWKS signature has expired`) rather than crowning an unverifiable claim — the right outcome, but it's also *why* the Blackwell frontier is still `0.428` today rather than reflecting that improvement. With #289 shipped, future wins merge within seconds of the gate passing, so the attestation is still fresh when the ledger re-checks it.

**Rule of thumb:** if `ledger.jsonl` has a BASELINE/win and `frontiers.json` is empty (or stale) for that arch, first check `runs/frontiers.json` was actually updated by the ledger *workflow run*, not just computed — a green gate does not by itself mean the frontier moved.

---

## 8. Is TritonBench a fair judge?

### Fair enough for the domain signal

- Training data is **decontaminated** against TritonBench (and KernelBench fingerprints in the seed importer).  
- **Tier labels** come from **Triton** improvement; general suites (e.g. GSM8K) are **regression guards**, not the reward driver.  
- Some kernels are **compiled and run** — but see below for exactly how little of the score that execution actually decides.  
- Scores are **per GPU architecture** (Blackwell ≠ Hopper).

### Not “exact match” and not yet “expert” — and three concrete gaps found by directly auditing the harness

- Headline metric is **`avg_composite`**, not GSM8K-style exact match — and only **35%** of it (`correctness`) is execution-gated at all. The other **65%** (`api_modernity`, `perf_awareness`, `completeness`, `code_quality`) is static keyword/AST matching on the generated code string — e.g. `perf_awareness` scores a point for the literal substring `"blackwell"` appearing in the code (`tritonbench/core/evaluator.py`). A kernel that never runs can still land ~0.4–0.65 composite by looking right.
- **Correctness is self-graded, not independently verified.** There is no held-out reference kernel: the harness runs the model's own script and checks whether the model's *own* `torch.allclose` assertion (which it also wrote, against inputs it also picked) raised. `exec_pass=True` + the substring `"torch.allclose"` in the code = full correctness credit — a model can pick loose tolerances or inputs that never exercise edge cases (e.g. sizes that never trigger boundary masking) and still score full marks.
- **Only 3 problems exist today** (`level1_basic/vector_add`, `level1_basic/softmax`, `bugfix/wrong_mask`) — the two most generic Triton tutorial kernels on earth. `configs/default.yaml` requests `levels: [1,2,3,4]`, but `level2`/`level3`/`level4` directories don't exist, so a "full" run silently collapses to the same 3-problem "quick" run. **Fixed:** the harness now reports this honestly (`eval.triton_bench.level_coverage` — requested vs. covered vs. silently-missing levels, with an opt-in `SPARKDISTILL_TRITONBENCH_STRICT_LEVELS=1` hard-fail), [#282](https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/282). **Not fixed:** actually populating levels 2–4 with held-out problems and adding independent reference kernels (closing the self-grading gap above) is a real, valuable, but large maintainer-owned project — tracked and scoped in [issue #227](https://github.com/gittensor-model-hub/SparkDistill-Hermes/issues/227), closed as roadmap rather than merged (it changes the reward surface for every miner and needs a design pass, not a drive-by fix).
- **Hopper is structurally unbeatable on execution today**, independent of the above: `tritonbench/bench_config.py`'s `require_blackwell_gpu` hard-rejects any non-Blackwell GPU (`sm_9x` and earlier) before a generated kernel can even run — so `exec_pass`/`correctness` are permanently **0** for every Hopper submission, no matter how good the kernel is. Still present as of this update; not yet fixed (it's inside the vendored `tritonbench/` tree, which per `VENDORED.md` must stay byte-identical between evaluators and miners for decontamination — a fix needs care, not a quick patch).
- Re-verify allows **~5 percentage points** tolerance (vLLM / server drift).

**Bottom line:** the harness genuinely executes some kernels and that execution *does* gate correctness credit — it's not pure prose-judging. But today's 3-problem, self-graded, 65%-text-heuristic setup means the frontier plateau (both architectures sit at `exec_pass=0`, `correctness=0`) is a **memorization ceiling**, not a capability ceiling: the optimal miner strategy is closer to "reproduce the 3 canonical answers and keyword-stuff the modern-API terms" than "write correct kernels." Treat `triton` composite scores as a syntax/completeness signal today, not a correctness signal, until levels 2–4 + independent references land.

---

## 9. Multi-turn episodes (better data, not just more rows)

Shipped in SparkProof ([PR #34](https://github.com/gittensor-model-hub/SparkProof/pull/34)).

### Why

Single Prompt→Answer rows teach shallow behavior. Real skill is:

**task → attempt → validator fail → repair → (optional) optimize → accept**

### What we record

`metadata.episode` (`sparkproof-episode-v1`):

- Real **hardware failure tails** as user turns (not fake critique).  
- Optional **measured optimize** pass when `--benchmark` is on (tier `optimized` if improved).  
- HF / SFT export prefers **full multi-turn chat** when an episode exists.

Flags: `--no-episodes`, `--no-optimize`.

Merkle leaf hashing still keys off the **final** answer fields; the episode sits in metadata so provenance of the gold answer stays stable while training gets the full trajectory.

This is the highest research-value dataset upgrade we shipped in this period: **trajectory quality**, not row count.

---

## 10. Public datasets: KernelBook, traces, KernelBench

We studied:

| Dataset | What it is |
|---|---|
| [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) | ~18k PyTorch modules + Inductor-style Triton pairs |
| [ppbhatt500/kernelbook-opus4.8-multiturn-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-opus4.8-multiturn-traces) | Opus agent loops with GPU feedback (multi-turn) |
| [ppbhatt500/kernelbook-triton-reasoning-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces) | gpt-oss-120b reasoning traces (~170 rows) |
| [ScalingIntelligence/KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) | Held-out eval problems (levels 1–4) |

### Can we paste them into SparkProof as verified data?

**No.** They lack pinned teachers, CC attestation, Merkle policy, and (for KernelBench) they are **eval-only**.

### What we do instead (implemented)

`sparkproof-import-external-tasks` (PR [#35](https://github.com/gittensor-model-hub/SparkProof/pull/35), docs in `docs/EXTERNAL_SEEDS.md`):

```text
External corpora
    → extract PyTorch problems only
    → permissive licenses only (MIT / Apache / BSD …)
    → drop KernelBench-sourced rows
    → decontaminate vs TritonBench + KernelBench fingerprints
    → prompts.jsonl (origin: kernelbook_seed)
    → sparkproof-triton-generate (Fable/Sol, multi-turn, CC)
    → release-gate + attest + registry
```

Optional: **code-only** repair hints from opus failed turns (curriculum), never opus/gpt-oss prose as teacher gold.

**KernelBench never enters training** — only fingerprints for decontam.

---

## 11. KernelLLM (Meta) — method in depth

Source: [facebook/KernelLLM](https://huggingface.co/facebook/KernelLLM).

### Goal

Train a **small specialist** (8B) that translates **PyTorch modules → Triton kernels**, and score it on **KernelBench-Triton**. Vision: make GPU kernel writing more accessible, not run an open mining economy.

### Data construction (the “compiler as teacher” method)

1. **Collect PyTorch** from open GitHub / Stack-style corpora.  
2. Extract `nn.Module` units with runnable `get_inputs` / `get_init_inputs` style tests when possible.  
3. Run **`torch.compile` / Inductor** to emit Triton.  
4. Clean / reshape Inductor output toward a KernelBench-like format.  
5. Publish as **[KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook)** (~18k pairs; they cite ~25k including synthetics).  

So gold labels are **compiler output**, not expert hand-tuned kernels. That is cheap and consistent — and it biases the student toward **compiler-shaped** code.

### Training recipe

| Knob | Value / choice | Why it matters |
|---|---|---|
| Base | Llama 3.1 **8B Instruct** | Instruction-following already present |
| Objective | **SFT only** (next-token on chat/code) | No RL in the card |
| Schedule | ~10 epochs, batch 32 | Short specialist finetune |
| Compute | ~12 h × 16 GPUs (~192 GPU-h) | One-day class run |
| Prompt | Fixed template + **format example** | Same template at **train and eval** |
| Hyperparams | Perplexity on held-out **train** slice | Standard SFT selection |

They argue others inflate scores by **training on KernelBench solutions**; KernelLLM claims **external (torch, triton) pairs only**.

### Inference / evaluation tricks

| Trick | What they do | Effect |
|---|---|---|
| **Pass@k** | Sample k kernels, unit-test each, keep any pass | Pass@1 ≈ 20 → Pass@10 ≈ 52 → Pass@20 ≈ 57 (their L1 table) |
| Sampling | `temperature=1.0`, `top_p=0.97` | Diversity for pass@k |
| Unit tests | Random inputs of known shapes vs PyTorch ref | Correctness filter |
| Hardware | H100 for reported numbers | Timing is hardware-sensitive |

**Pass@k is half the method.** A weak single-shot model can look strong if you afford many samples and a good harness.

### What the student actually learns

- PyTorch → Triton **layout and boilerplate**  
- Inductor idioms (grids, loads/stores, heuristics-shaped code)  
- Format compliance with their prompt template  

### What it does *not* learn well (their own limits)

- Meaningful hand-written scheduling insight  
- Robust API naming / shapes / dtypes / precision  
- Long agentic repair loops (that is TritonForge / our episodes)

### Takeaways for best datasets

| Keep | Drop |
|---|---|
| Diverse **PyTorch tasks** from KernelBook-style sources | Inductor text as *attested teacher* gold |
| Strict **format** for SFT rows | Training on **eval** solutions |
| Plan for **best-of-N** at gen and eval | Assuming one sample = skill |

---

## 12. TritonForge (RLsys) — method in depth

Source: [RLsys-Foundation/TritonForge](https://github.com/RLsys-Foundation/TritonForge).  
Related: forked **[SLIME](https://github.com/THUDM/slime)** RL stack; **KBenchEval** on KernelBench; checkpoints under `JinnP/…` on Hugging Face.

### Goal

Not only imitate Inductor — train a model that **iteratively repairs kernels using live compile / run / speed feedback**, like a junior engineer with a compiler in the loop.

### Architecture (two boxes)

```text
┌─────────────────────┐     ┌──────────────────────┐
│  SLIME (SFT + RL)   │────▶│  KBenchEval (env)     │
│  Qwen3-8B policy    │◀────│  compile / correct /  │
│  multi-turn agent   │     │  speedup rewards      │
└─────────────────────┘     └──────────────────────┘
```

- **SLIME:** unified trainer for SFT and RL (they ship bugfixes for multi-turn kernel loops).  
- **KBenchEval:** problem server + metrics (based on ScalingIntelligence/KernelBench).  
- **Serving:** SGLang-style inference for rollouts.  
- **Hardware:** NVIDIA **H100** and AMD **MI300X** (ROCm) — rare open dual-vendor path.

### Stage 1 — SFT (KernelBook)

| Item | Detail |
|---|---|
| Base | **Qwen3-8B** |
| Data | KernelBook **~18.2k → ~17k filtered** pairs |
| Augmentations | Multi-turn conversations, **thinking tags**, length filtering |
| Parallelism | TP=2, **CP=4** (long contexts), PP=1, DP=1 |
| Batch / LR | Global BS **32**, LR **1e-5 → 1e-6** cosine |
| Precision | BF16, full gradient recomputation (12 layers) |
| Script | `SLIME/scripts/run-qwen3-8B-kernelbook-sft.sh` |
| Output | `JinnP/Qwen3-8B-Kernelbook-SFT-filtered` (+ HF format for eval) |

SFT alone on KernelBench L1/L2 is modest (~**18%** L1 Pass@1, ~**8%** L2) — close to a “KernelBook baseline” (~20% L1). So **SFT is only the warm start**.

### Stage 2 — RL (KernelBench as the environment)

This is the core trick.

| Item | Detail |
|---|---|
| Problems | KernelBench **Level 1–2** (**200** problems) |
| Interaction | Up to **3 turns** per kernel |
| Feedback into context | Compile errors, functional failures, performance signals |
| Reward | **Compile success + functional correctness + speedup** |
| Discount | **γ = 0.4** (strongly prefers early success / short trajectories) |
| Script (NVIDIA) | `SLIME/scripts/run_agent_kbench_qwen3_8B_sft_fixed.sh` |

**What one RL episode looks like (conceptually):**

```text
Turn 1: model emits kernel
        env: compile? run tests? measure speed?
        reward_1, error_tail or metrics → appended to dialogue
Turn 2: model repairs / optimizes using that feedback
        env again → reward_2
Turn 3: final attempt → reward_3
Return: discounted sum with γ=0.4
```

Low γ means the agent is pressured to **get something compiling and correct quickly**, then improve — not wander for ten turns.

### Why multi-turn RL beats single-turn SFT

| Signal | Single-turn SFT | Multi-turn RL |
|---|---|---|
| Sees compile errors? | Only if present in static pairs | **Yes, live** |
| Sees wrong numerics? | Rare in Inductor gold | **Yes, from tests** |
| Sees speed? | No | **Yes, in reward** |
| Learns *repair policy*? | Weakly | **Yes** |

They report roughly **15–20%** improvement from multi-turn refinement on complex patterns, and **>90%** compile rate when error feedback is in the loop. Exact curves live on **WandB** (linked from their README).

### Cross-platform notes

| Platform | Status |
|---|---|
| NVIDIA H100 multi-turn | Primary results path |
| AMD MI300X single-turn | Works (ROCm/HIP pipeline) |
| AMD MI300X multi-turn RL | **Known crash** (CPU 100%, node die within steps) — open issue |

### Roadmap signals (what they optimize toward)

- Larger / MoE students (e.g. Qwen3-30B-A3B)  
- Tool calling (profiling, docs, search)  
- Multi-DSL (CUDA, HIP, OpenCL beyond Triton)  
- GUI for monitoring rollouts  

### Policy conflict with SparkDistill (critical)

| TritonForge | SparkDistill / SparkProof |
|---|---|
| RL **trains on KernelBench** | KernelBench / TritonBench = **held-out scoreboard** |
| Success = higher KernelBench pass | Success = higher **frontier** without ever training on it |

If we copy their RL env **onto KernelBench**, we **poison** our own metric — exactly the inflation KernelLLM warned about.

### Safe way to steal the method

```text
TritonForge idea:  live GPU reward + multi-turn repair policy
Our constraint:    never use TritonBench / KernelBench as RL tasks

Safe RL tasks:     kernelbook_seed problems, native torch_op / mutation /
                   failure-mined / self-evolved prompts, private holdouts
Eval only:         TritonBench (+ GSM8K regression, etc.)
```

### Takeaways for best datasets

| From TritonForge | How we use it |
|---|---|
| Multi-turn with **real** error tails | SparkProof **episodes** (data-side) |
| Reward = compile + correct + speed | Episode optimize pass + future student RL |
| SFT warm start on pairs | SFT on **attested** multi-turn / mining mix |
| γ / short horizons | Prefer short repair chains (we already cap repairs) |
| Open training stack | Optional later; not required for dataset track |

---

## 13. Other methods worth knowing

### A. Opus multi-turn agent traces (ppbhatt500)

[kernelbook-opus4.8-multiturn-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-opus4.8-multiturn-traces)

| Piece | Detail |
|---|---|
| Model | Claude **Opus 4.8**, effort medium, headless agent |
| Loop | Write kernel → **GPU judge** (correctness + speedup vs PyTorch) → read feedback → iterate (up to ~8 evals) |
| Hardware | NVIDIA **GB300**, Triton 3.6, CUDA 12.8 |
| Anti-cheat | Static gate rejects torch-only “solutions”, unlaunched kernels, timer games |
| Stats | ~300 rows, ~99.7% final correct, median ~5.8× speedup, ~3.7 turns/problem |

**Method lesson:** the **judge is the teacher**. Trajectories are valuable because feedback is **grounded**, not because Opus is magical.

**Our use:** task text + optional **failed kernel code** as repair hints — **re-solve** with Fable/Sol on CC. Do not import Opus messages as certified teacher output.

### B. gpt-oss reasoning traces

[kernelbook-triton-reasoning-traces](https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces)

| Piece | Detail |
|---|---|
| Model | **gpt-oss-120b** |
| Shape | Single-shot-ish CoT + Triton; ~85% correct, ~15% wrong-but-formatted for SFT |
| Columns | `model_reasoning`, `triton_code`, correctness / speedup flags |

**Method lesson:** explicit **reasoning traces** help SFT students if format matches (`<think>`). Wrong-but-pretty rows can still teach format — dangerous if you care about correctness priors.

**Our use:** PyTorch tasks only when not KernelBench-sourced; regenerate CoT under pinned teachers (or Fable CoT recovery for Sol).

### C. Classical pass@k / best-of-N (shared by everyone)

All strong systems use some form of:

```text
sample N candidates → filter by tests / compile → rank by speed or composite → keep winner
```

KernelLLM does it at **inference**. SparkProof does it at **generation** (multi-candidate teachers + repair). TritonForge does it **inside RL rollouts**.

**Dataset implication:** store **losers and failure tails**, not only winners — otherwise students never see repair.

### D. Inductor-only distillation (KernelBook raw)

Train only on `(pytorch_code, triton_code)` pairs with no agent loop.

| Pros | Cons |
|---|---|
| Huge volume, cheap | Compiler bias, no repair skill |
| Easy SFT format | Weak when tests fail at deploy |

**Best as stage-0 warm start**, not the final mix for an expert student.

### E. Pure RL from scratch (no SFT)

Rare in this niche: KernelBench rewards are sparse (most random code fails compile). Everyone **SFT first**, then RL. We should too.

---

## 14. How to generate the best dataset

“Best” here means: **maximizes student skill on held-out TritonBench while staying SN74-verifiable and decontam-clean.**

### 14.1 Rank data types by research value

| Rank | Data type | Why |
|---|---|---|
| **1** | **Multi-turn episodes** with real validator fail → repair → accept | Teaches debug policy |
| **2** | Episodes with **measured optimize** (correct → faster, still correct) | Teaches performance taste |
| **3** | Single-turn gold with **plaintext CoT** (Fable / recovered Sol) | Teaches planning |
| **4** | Single-turn gold, code only | Teaches format / syntax |
| **5** | Failed-only or unattested public CoT | Research only, not registry |

Row count without (1)–(3) is a **weak** leaderboard.

### 14.2 Golden generation recipe (SparkProof)

```text
1. Build a diverse task pool
   - Native: api_doc, mutation, torch_op, failure_mining, self_evolution
   - External seeds: KernelBook / opus / gpt-oss → PyTorch only
   - Licenses: permissive only
   - Decontam: TritonBench + KernelBench fingerprints
   - Origin: never kernelbench / tritonbench

2. Generate with pinned teachers (Fable + Sol @ xhigh)
   - Best-of-N / multi-candidate
   - max_repairs ≥ 1 (capture fail→fix)
   - --benchmark for optimize pass when possible
   - Episodes ON (default)

3. Hardware validate every kept row on CC GPU
   - Same arch family you care about (blackwell / hopper)

4. Prefer CoT quality
   - Fable extended thinking
   - Sol→Fable CoT recovery when Sol wins encrypted

5. Release gate
   - decontam, novelty vs accepted registry snapshot (--mining-repo)
   - Merkle + attestation (NRAS + TDX)

6. Publish proof/ + registry line
   - Fair label = novel rows after mix, not raw count
```

### 14.3 Concrete knobs that move quality

| Knob | Prefer | Avoid |
|---|---|---|
| Tasks | Hard ops, adversarial shapes, fusion-like modules | Only ReLU-tier toys |
| Feedback | Real compile/runtime tails in episodes | Fake “try again” text |
| Optimize | Only after **correct**; keep measured metrics | Optimize broken code |
| Teachers | Pinned Fable/Sol xhigh | Cheap models, unpinned gateways |
| Dedup | Novelty vs live `sparkproof-mining` snapshot | Blind XL farming of near-dupes |
| Mix | Multi-turn share ↑ over time | 100% single-turn forever |
| Eval leakage | Strict decontam + origin policy | “It’s fine, different wording” |
| Architecture | Label and attest the GPU you used | Mix Hopper numbers into Blackwell frontier |

### 14.4 What *not* to put in the best dataset

1. **KernelBench / TritonBench problems** as training tasks.  
2. **Inductor / Opus / gpt-oss text** as the teacher response (unless regenerated and attested).  
3. Rows that **compile but never execute correctly** only — OK as early baseline, toxic as majority of mix once students can pass syntax.  
4. **Duplicates** of the accepted registry mix (zero fair reward, wastes GPU, teaches nothing new).  
5. **Unattested** “I ran it on my laptop” CSVs claiming production tiers.

### 14.5 Target mix profile (practical)

For the next high-value mining wave:

| Slice | Share (guidance) | Source |
|---|---|---|
| Multi-turn repair episodes | **40–60%** | `sparkproof-triton-generate` with repairs |
| Optimize-improved episodes | **10–20%** | `--benchmark` winners |
| Single-turn + strong CoT | **20–30%** | Fable-heavy / CoT recovery |
| Hard kernelbook_seed tasks | **10–20%** of *tasks*, all re-proven | `import-external-tasks` |

Adjust by what actually moves **TritonBench exec/correctness**, not only syntax rate.

### 14.6 From dataset → student (two stages)

**Stage A — SFT (ready now)**  
- Input: registry mix / episode exports / mining SFT jsonl (`messages`, optional `<think>`).  
- Recipe: `recipes/qwen3.5-4b-phase1/sft-mining.yaml` (or phase1 SFT).  
- Goal: format + repair patterns + CoT style.

**Stage B — RL (future, TritonForge-inspired, decontam-safe)**  
- Start from SFT checkpoint.  
- Env tasks: **private / seed / native** only — **never** TritonBench/KernelBench.  
- Reward: compile + correctness + speedup (same spirit as TritonForge, same components as episode optimize).  
- Cap turns (2–4). Prefer low-ish γ so short repairs win.  
- Eval: held-out TritonBench only.

### 14.7 Quality checklist before you call a bundle “best”

- [ ] Every kept row has **passed** hardware validation on the claimed arch.  
- [ ] Episode present for repair-capable tasks (`metadata.episode`).  
- [ ] CoT is plaintext where possible (`cot_recovery` if Sol).  
- [ ] `novelty_report.json` shows enough **novel** rows vs mining snapshot.  
- [ ] Release gate + online attestation green.  
- [ ] Zero KernelBench/TritonBench origins.  
- [ ] Permissive licenses on external-derived tasks.  
- [ ] SFT export is multi-turn chat, not only final code.

### 14.8 One-page “best dataset” formula

```text
best_dataset =
    diverse_hard_tasks(no_eval_leak)
  × pinned_teachers(xhigh)
  × best_of_N
  × real_validator_multi_turn
  × optional_measured_optimize
  × plaintext_cot
  × cc_attestation
  × registry_novelty
```

Anything that drops a factor (especially **real_validator_multi_turn** or **no_eval_leak**) is cheaper to produce and usually **worse** for students.

---

## 15. Side-by-side comparison

| | KernelLLM | TritonForge | Opus traces | SparkProof / SparkDistill |
|---|---|---|---|---|
| Goal | Specialist 8B translator | SFT+RL agent trainer | Research agent logs | Attested data + open student economy |
| Stage 1 | KernelBook SFT | KernelBook SFT | — | Attested teacher trajectories (Fable/Sol) |
| Stage 2 | Pass@k at test time | RL on KernelBench | Agent loop at gen time | Frontier eval on TritonBench (+ guards) |
| Multi-turn | Sample-many | Yes (RL, ≤3) | Yes (agent, ~4 turns) | Yes (data episodes) |
| Feedback | Unit tests at eval | Live compile/correct/speed | Live GPU judge | Live CC validator |
| KernelBench in train? | No (claimed) | **Yes (RL)** | Sometimes as source (we drop) | **Never** |
| Trust model | Weights + card | Open recipes | None for SN74 | CC + TDX + Merkle + registry |
| Pays contributors? | No | No | No | **SN74** |

---

## 16. What SparkProof / SparkDistill should take from others

### Adopt (ideas)

1. **KernelBook / public corpora as task factories** — already: import → re-prove.  
2. **Multi-turn + measured optimize** — already: episodes.  
3. **Two-stage student training** (research path):  
   - **SFT** on attested multi-turn / registry data.  
   - **RL (later)** on **non-eval** tasks with compile/correct/speedup rewards — *not* KernelBench.  
4. **Pass@k / best-of-N** at eval and generation (we already use multi-candidate teachers).  
5. **Thinking tags / `<think>`** consistent with Qwen3.5 templates.  
6. **Short-horizon repair** (TritonForge γ=0.4 intuition): prefer 1–3 repair turns, not endless loops.

### Do not adopt blindly

1. Inductor gold as “teacher of record.”  
2. RL **on** KernelBench / TritonBench.  
3. Trusting public CoT (Opus, gpt-oss) without re-generation under pinned teachers.  
4. Optimizing only for Pass@1 syntax while exec stays 0%.

### Suggested student pipeline

```text
SparkProof (episodes + kernelbook_seed)
    → SFT student (Axolotl, Qwen3.5-4B phase-1 recipes)
    → optional RL on private/dev tasks + real GPU rewards
    → training-track PR vs runs/frontiers.json
```

SFT is ready today with existing recipes (`sft-mining.yaml`, etc.). RL needs a careful env that **never** touches held-out benches.

---

## 17. End-to-end workflows (cheat sheets)

### A. Best-effort dataset miner (CC VM)

```bash
# 1) Task pool (native + external seeds)
scripts/import_external_tasks.sh --limit 100
# optional: mix with sparkproof-build-prompts --seed-prompts ...

# 2) Multi-turn attested generation
sparkproof-triton-generate --prompts prompts/kernelbook_seed.jsonl \
  --out bundles/run-001 --decontaminate --orchestrate --benchmark

# 3) Novelty-aware publish
sparkproof-publish-dataset --bundle bundles/run-001 \
  --repo-id YOU/sparkproof-v1 --release-gate --mining-repo

# 4) Text-only SparkDistill registry PR
```

### B. Training miner

```bash
scripts/prepare_mining_sft.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml
scripts/eval.sh --checkpoint outputs/... --compare-frontier
# proof bundle + attestation → training-track PR
```

### C. Watch progress over a month

| Artifact | What “moved” looks like |
|---|---|
| `datasets/registry.jsonl` | More verified dataset entries |
| `runs/ledger.jsonl` | New training merges |
| `runs/frontiers.json` | Higher scores per architecture (esp. exec/correctness) |
| Share of multi-turn rows in mix | Trajectory quality ↑ |
| GitHub releases / CHANGELOG | Shipped protocol upgrades |

No private dashboard required — the **repo is the dashboard**.

---

## 18. Open work and risks

### Done recently (context for this doc)

- Attestation fail-close (NRAS / TDX REPORTDATA binding).  
- Multi-turn episodes.  
- Frontier merge pipeline + Hopper backfill.  
- External seed importer + docs (`EXTERNAL_SEEDS.md`).

### Still important

| Risk / gap | Why it matters |
|---|---|
| Frontiers still low on **exec/correctness** — root causes now identified (§8): only 3 harness problems, self-graded correctness, Hopper hard-rejected from execution entirely | Domain signal is weak, and on Hopper the execution signal is structurally impossible to earn, not just hard |
| Teacher ledger optional in CI | Cheap-model forgery edge case |
| RL student stage not in-repo yet | TritonForge-style gains not captured in weights |
| Fair dataset labels depend on mix freshness | Miners must use `--mining-repo` novelty |
| AMD path (TritonForge) | Interesting, out of scope for current CC pins |

### Research priorities (ordered)

1. Raise **exec pass / correctness** on TritonBench frontiers.  
2. Scale **attested multi-turn** data (seeds + native prompts) using §14 recipe.  
3. Student **SFT** on episode-rich mixes.  
4. **RL** only on private/dev tasks with GPU rewards.  
5. Optional: teacher-ledger escrow for stronger model identity.

---

## 19. Glossary

| Term | Plain meaning |
|---|---|
| **CC** | Confidential compute — hardware-attested guest |
| **TDX** | Intel Trust Domain Extensions (VM attestation) |
| **NRAS** | NVIDIA Remote Attestation Service (GPU token) |
| **Frontier** | Current best verified scores to beat |
| **BASELINE** | First verified run on an empty architecture bucket |
| **Release gate** | Pre-publish checks (decontam, novelty, hashes) |
| **Decontam** | Block training rows that look like eval problems |
| **Episode** | Multi-turn fail/repair/optimize trajectory |
| **KernelBook** | Public PyTorch→Triton pair dataset |
| **KernelBench** | Public **eval** suite — never train on it here |
| **TritonBench** | Our domain eval composite for students |
| **SFT** | Supervised fine-tuning on chat/code pairs |
| **RL** | Reinforcement learning from rewards (compile/correct/speed) |
| **Pass@k** | Success if any of k samples passes tests |
| **γ (gamma)** | RL discount; lower = prefer short successful trajectories |
| **Inductor** | PyTorch compiler backend that can emit Triton |

---

## 20. Links

### Ours

- SparkDistill: https://github.com/gittensor-model-hub/SparkDistill-Hermes  
- SparkProof: https://github.com/gittensor-model-hub/SparkProof  
- Frontiers / ledger: https://github.com/gittensor-model-hub/SparkDistill-Hermes/tree/main/runs  
- Dataset registry: https://github.com/gittensor-model-hub/SparkDistill-Hermes/blob/main/datasets/registry.jsonl  
- External seeds guide: `SparkProof/docs/EXTERNAL_SEEDS.md`  
- Miner guide: `SparkProof/docs/MINER_GUIDE.md`, `SparkDistill/docs/miner-guide.md`  
- This document: `SparkDistill/docs/research-summary.md`  
- Gittensor / SN74: https://gittensor.io/

### Related work & data

- KernelLLM: https://huggingface.co/facebook/KernelLLM  
- TritonForge: https://github.com/RLsys-Foundation/TritonForge  
- SLIME (upstream RL): https://github.com/THUDM/slime  
- KernelBook: https://huggingface.co/datasets/GPUMODE/KernelBook  
- KernelBench: https://huggingface.co/datasets/ScalingIntelligence/KernelBench  
- Opus multi-turn traces: https://huggingface.co/datasets/ppbhatt500/kernelbook-opus4.8-multiturn-traces  
- gpt-oss reasoning traces: https://huggingface.co/datasets/ppbhatt500/kernelbook-triton-reasoning-traces  

### Key PRs (recent, as of this update)

- SparkDistill TritonBench honest level coverage: https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/282  
- SparkDistill Qwen3.5 training hardening (`sample_packing` skew, missing `Python.h`): https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/284 , https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/286  
- SparkDistill training-track auto-merge parity with the dataset gate: https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/289  
- SparkDistill ledger publish via PR fallback (branch-protection fix): https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/290  
- SparkDistill community-PR auto-close policy: https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/295 , https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/296  
- SparkDistill reward-tier bypass fixes (`gpu_architecture` spoof, forged `triton` composite): https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/298  
- Gittensor live weights: double training-track multipliers over dataset track (open, unmerged): https://github.com/entrius/gittensor/pull/1660  

### Earlier key PRs

- SparkProof multi-turn episodes: https://github.com/gittensor-model-hub/SparkProof/pull/34  
- SparkProof external task seeds: https://github.com/gittensor-model-hub/SparkProof/pull/35  
- SparkDistill frontiers on merge: https://github.com/gittensor-model-hub/SparkDistill-Hermes/pull/201  

---

## Closing

SparkDistill / SparkProof are not “another KernelBench fine-tune.” They are an attempt to make the **industrial distillation loop** — teachers, proof, students, frontier — **public, payable, and re-verifiable**.

**KernelLLM** shows compiler-supervised SFT + pass@k.  
**TritonForge** shows SFT warm-start + **multi-turn RL on live GPU rewards** (but trains on the scoreboard we must keep clean).  
**Opus traces** show grounded agent loops.

Our differentiator is **attested provenance** and a **clean eval wall**. The winning path is:

**use their problems and loops, keep our proofs, never train on our scoreboard — and spend GPU budget on multi-turn, measured, novel rows.**

That is how Triton-native PoC becomes a pipeline that continuously raises expert-level models — without trusting a CSV.
