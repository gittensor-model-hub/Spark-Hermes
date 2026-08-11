# Serving Muse-Glimmer-30B

Measured on 2026-08-11 against `meta-models/Muse-Glimmer-30B` at revision
`97c77dff50b2797bcc558fa2d909761dbc575c59`, on one RTX PRO 6000 Blackwell Server Edition (96 GB,
driver 580.126.09, sm_120) in a container with **no system CUDA toolkit**.

Eight startup failures got to a working server. Not one of them was the model, which is the reason
this file exists: every message along the way reads like "this model is unsupported", and none of
them meant that.

## The short answer

**SGLang from the `muse-glimmer` branch, with the `muse` parsers.** vLLM 0.27.0 cannot do it.

| | vLLM 0.27.0 | SGLang @ `sgl-project:muse-glimmer` |
|---|---|---|
| native architecture | none — PR #51655 open | `MuseGlimmerForConditionalGeneration` |
| output | ten tokens of multilingual noise | correct |
| ATEM tool calls | no parser exists | `MuseGlimmerDetector`, registered `muse` |
| reasoning channel | — | reasoning parser, also `muse` |
| speculative decoding | — | `DFlashDraftModel`, `--speculative-dflash-block-size` |

vLLM's `--model-impl transformers` fallback *starts* and *serves*. It returns
` gelten giận giậnะกิน giận gelten gelten`. Plain transformers on the same weights, same revision,
same card returns a correct tool call — so the fallback mishandles this architecture and the model
is fine. Anyone who tries the fallback first will conclude the opposite.

## Install

```bash
python3 -m venv .venv-sglang
export SGLANG_BUILD_RUST_EXTS=none          # (1)
./.venv-sglang/bin/pip install \
  "git+https://github.com/sgl-project/sglang.git@muse-glimmer#subdirectory=python"
./.venv-sglang/bin/pip install ninja        # (4)
apt-get install -y ffmpeg                   # (3)
```

Do **not** upgrade transformers. SGLang pins 5.12.1 and ships its own `muse_glimmer` config, so it
never needed a newer one — and transformers main raises
`'qwen3_asr' is already used by a Transformers config` when SGLang registers its own. (2)

## Serve

```bash
export CUDA_HOME=$PWD/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13   # (5)
export PATH="$CUDA_HOME/bin:$PWD/.venv-sglang/bin:$PATH"                      # (4)
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"                        # (6)
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"                          # (6)

./.venv-sglang/bin/python -m sglang.launch_server \
  --model-path /workspace/glimmer-30b \
  --served-model-name muse-glimmer-30b \
  --context-length 32768 \
  --mem-fraction-static 0.88 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --tool-call-parser muse \
  --reasoning-parser muse \
  --trust-remote-code
```

Ready when the log says `The server is fired up and ready to roll!`. About 88 GB of the 96 resident.

## The eight, and what each one actually was

1. **`cargo is required to discover the Rust extension modules`** — the build wants a Rust
   toolchain. Its own message names the way out; the extensions are an optimisation, not the engine.
2. **`'qwen3_asr' is already used by a Transformers config`** — overriding transformers to main.
   SGLang registers configs for models its pinned transformers lacks; on main they already exist.
   The override was pointless as well as fatal: the `muse_glimmer` config ships with SGLang.
3. **`libavutil.so.56: cannot open shared object file`** — torchcodec, for video input. `ffmpeg`.
4. **`FileNotFoundError: 'ninja'`** — installed in the venv and invisible, because invoking
   `.venv-sglang/bin/python` by absolute path leaves the venv's `bin` off `PATH`. Killed the
   scheduler three times.
5. **`CUDA compiler and CUDA toolkit headers are incompatible`** — flashinfer's JIT. Two causes at
   once: `CUDA_HOME` pointed at a *different venv's* toolchain, and a `CPATH` pointing at the
   toolkit's includes let flashinfer's bundled libcudacxx find headers that disagreed with nvcc.
   Same-venv `CUDA_HOME`, no `CPATH`, and `--attention-backend triton` to stay off flashinfer.
6. **`ld: cannot find -lcudart`** — the wheel ships `libcudart.so.13` and no bare `.so`, which is
   not linkable. `LIBRARY_PATH` plus the symlink.
7. **The server came up, served its warmup, and died on the first real request** — triton covered
   attention, and the *sampling* kernel still went through flashinfer's JIT.
   `--sampling-backend pytorch`. vLLM needed the identical switch
   (`VLLM_USE_FLASHINFER_SAMPLER=0`).
8. **`prompt_tokens: 77`, and the model musing "Use bash tool? Probably exec."** — the quietest
   one. Without `--tool-call-parser`, SGLang does not render the tools into the prompt at all, so
   the model had never been told it had any. It looked like a model that would not call tools.

## What the harness gets back

With `--tool-call-parser muse`, the endpoint parses the wire format itself and **`content` is
empty**:

```
finish_reason    tool_calls
prompt_tokens    424          (77 without the parser -- the tools were never rendered)
tool_calls       [{"function": {"name": "terminal", "arguments": "{\"command\": \"ls -l logs\"}"}}]
reasoning_content "The directory logs/ holds several .log files..."
content          ""
```

`hermesbench.policy` prefers those structured calls and falls back to parsing the text with
`hermes.atem`. Both paths are real — a server without the parser, or a raw `generate()` — and a
policy that read only `content` would see nothing said, score an **abstention**, and report a model
that called a tool correctly as one that declined to, on every turn.

`hermes.atem` is still required regardless: it renders history back to the model and builds training
rows, and no serving layer does that.

## Two traps outside the server

- `AutoModelForCausalLM` refuses `MuseGlimmerConfig`. It is an image-text-to-text model:
  `AutoModelForImageTextToText`. Text hyperparameters are nested under `text_config`, which is the
  shape `tests/test_base_model.py` already warns about.
- `pkill -f "vllm serve"` matches the ssh command's own command line and kills its own shell, so
  three relaunches vanished without a word. `pkill -f "vllm[ ]serve"`.

## Still open

Native vLLM support is PR #51655; SGLang's is PR #34262, on an upstream branch rather than a fork.
Until one merges, the serving recipe above is a branch build and the base-model pin stays on
Qwen3.6-27B.

`Muse-Glimmer-30B-assistant` is a 2.56 B, 5-layer DFlash **drafter**, not an instruct variant.
SGLang's branch carries `DFlashDraftModel` and `--speculative-dflash-block-size`, so speculative
decoding is available and worth measuring: episode throughput is what bounds corpus generation,
challenge supply and any later on-policy RL.
