"""HermesBench -- an agent benchmark that measures work, not knowledge.

MMLU and HumanEval ask whether a model knows things. A worker that knows everything and
verifies nothing is the exact failure mode this pipeline exists to train out, so
HermesBench scores episodes instead: did the task get done, did the agent flail getting
there, did it recover when a tool failed, and did it check its own work before declaring
victory. See docs/roadmap-hermes.md and hermesbench/README.md.
"""

BENCH_VERSION = "v0"

# The capability categories a Hermes-native worker is judged on. These are tags on tasks,
# so `load_suite(tags=(...))` selects a category, and `suite_metrics` reports a success
# rate per category.
#
# The point of scoring per category rather than pooling: a model can be strong at picking
# tools and hopeless at holding an objective for a hundred steps, and one aggregate number
# hides exactly that. It also stops a category with many easy tasks from carrying a
# category with few hard ones.
TOOL_CALLING = "tool_calling"
TERMINAL_AGENT = "terminal_agent"
LONG_HORIZON = "long_horizon"
SELF_VERIFICATION = "self_verification"

HERMES_CATEGORIES = (TOOL_CALLING, TERMINAL_AGENT, LONG_HORIZON, SELF_VERIFICATION)
