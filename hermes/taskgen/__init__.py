"""Generate agent tasks at scale, from the shape of real traces rather than from nothing.

    python -m hermes.taskgen.cli --seeds lambda --count 350 --out var/tasks/gen-1

Four stages, each in its own module:

  `dna`    a public trace -> an abstract specification, carrying no verbatim content from the seed
  `synth`  a specification + a model -> a concrete task, its two checks, and two solutions
  `gate`   eleven checks the task must pass before it is allowed to exist -- one static, ten executed
  `cli`    the driver: parallel, resumable, and honest about what it dropped
  `triage` what the gate cannot decide: which accepted tasks the pinned model actually finds hard

The gate is the reason to trust any of it. A generated task nobody executed is indistinguishable
from a broken one, and a broken task produces failed episodes that read as a model that could not do
the work -- which then become `rejected` training examples teaching the model to avoid something
that was never its fault.
"""

__all__ = ["dna", "gate"]
