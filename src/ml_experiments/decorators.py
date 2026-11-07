"""Decorator obrazac za mjerenje trajanja eksperimenta."""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from time import perf_counter

from .experiment import Experiment, ExperimentResult


class ExperimentDecorator(Experiment, ABC):
    def __init__(self, wrapped: Experiment) -> None:
        self._wrapped = wrapped


class TimedExperiment(ExperimentDecorator):
    def run(self) -> ExperimentResult:
        started_at = perf_counter()
        result = self._wrapped.run()
        elapsed = perf_counter() - started_at
        return replace(result, duration_seconds=elapsed)
