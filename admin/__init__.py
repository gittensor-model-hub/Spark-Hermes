"""The operator-driven training pipeline: no miners, no rounds, no competition.

    python -m admin.cli status
    python -m admin.cli doctor
    python -m admin.cli generate --count 350 --salt-file /private/salt --allow-unsandboxed
    python -m admin.cli rollout --repeats 8 --salt-file /private/salt --allow-unsandboxed
    python -m admin.cli corpus
    python -m admin.cli prepare
    python -m admin.cli train
    python -m admin.cli merge
    python -m admin.cli evaluate --print-only

The rest of this repository is a competition. Miners submit prose surfaces, a validator executes them
against a published challenge, a withheld check grades them, and an hourly crown pays out. That
machinery assumes an adversary and spends most of its complexity on being checkable by one.

This package assumes the opposite: one operator, one GPU, and a loop that has to produce a better
model rather than a defensible verdict. So there are no submissions, no commit-reveal, no receipts,
no board, and no crown. What is left is the part that actually trains something:

    generate  ->  roll out  ->  aggregate  ->  train  ->  evaluate

## What is kept from the competition side, and why

**The withheld checks stay.** Not to catch a miner -- there isn't one -- but because a corpus built
only on what a published check can see is a corpus of solutions fitted to visible assertions, and
training on those teaches fitting. `overfit_rate` is as useful pointed at your own rollouts as at
somebody else's surface.

**The promotion gate stays.** `hermes.promotion.check_graders` refuses to compare two runs that used
different graders, dialects, or serving precision. An operator comparing their own before and after
is exactly as capable of comparing incomparable things as a stranger is, and rather more likely to
want the answer to come out a particular way.

**The statistics stay.** `MIN_ATTEMPTS = 10`, the bootstrap interval on the reduction, the refusal to
read a one-attempt result. Removing the adversary does not make small samples informative.

## The constraint this package exists to enforce

The nineteen hand-written tasks are the EVALUATION suite. Training on rollouts of them and then
reporting suite success is training on the test set -- and `overfit_rate`, the one instrument that
would notice, is measured on those same tasks. `admin.split` holds that line: generated tasks are
train, hand-written tasks are eval, and a corpus containing an eval task is refused rather than
built.
"""

__all__ = ["pipeline", "split"]
