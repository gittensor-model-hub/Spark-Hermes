"""Hermes-native agent trajectory pipeline.

SparkDistill trains workers, not chatbots: the training unit is an *execution
trajectory* (observe -> plan -> act -> verify -> recover), not a prompt/response
pair. See docs/roadmap-hermes.md.
"""
