# `runs/`

The immutable, in-repo record of every verified proof-of-training submission that
was merged. Written by the eval bot at merge time — miners don't write here directly.

- **`ledger.jsonl`** — append-only, one JSON line per merged PR (see `eval/ledger.py`).
  Never edited or reordered; a bad entry is corrected by appending a new one, not by
  rewriting history.
- **`frontiers.json`** — per-GPU-architecture frontier buckets (`blackwell`,
  `hopper`). TritonBench composites are hardware-sensitive, so each architecture
  keeps its own scores — a Hopper run is never tiered against Blackwell numbers.
  `eval.verify` resolves the bundle's architecture from `gpu_architecture` or
  `train_gpu` and loads the matching bucket (empty bucket → `eval:BASELINE`).
  Updated at merge time by `eval.record_training_ledger` / `apply_verified_report_to_frontiers`
  whenever a verified non-REJECT training run raises a per-benchmark high (including
  seeding an empty architecture bucket from `eval:BASELINE`).
- **`frontier.json`** — legacy single-file Blackwell frontier kept for backward
  compatibility; `eval.frontiers` falls back to it when `frontiers.json` is absent.
- **`<run-id>/`** — one directory per merged run, holding the artifacts the ledger
  entry references:
  - `result.json` — the `eval.score` report (tier label, per-benchmark deltas).
  - `attestation.json` — the `eval.attestation` result, if the submission included
    GPU confidential-computing attestation (optional; unattested submissions still
    go through full retrain-verification instead of cheap re-score — see
    `docs/miner-guide.md`).

**What's not tracked here: dataset provenance.** The ledger's schema (`eval/ledger.py`'s
`LedgerEntry`) has no dataset field — it only records the run's eval delta, tier label,
and (optionally) attestation. The dataset a merged run was trained on is cited via
`proof.bundle --dataset-url` pointing at a merged entry in
[`datasets/registry.jsonl`](../datasets/registry.jsonl) (or a small committed file for
non-Triton experiments). Cross-miner dataset mixing is supported via `scripts/mix_registry.sh` — see
[`datasets/README.md`](../datasets/README.md#cross-miner-mixing).

This mirrors `sparkinfer-log`'s public run-log convention, kept inside this repo
instead of a separate sibling repo.
