"""Strategy obrazac za zamjenjive metrike evaluacije."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score


class MetricStrategy(ABC):
    """Zajednički interfejs svih strategija evaluacije."""

    name: str

    @abstractmethod
    def calculate(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Izračunava vrijednost metrike."""


class AccuracyMetric(MetricStrategy):
    name = "accuracy"

    def calculate(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        return float(accuracy_score(y_true, y_pred))


class PrecisionMacroMetric(MetricStrategy):
    name = "precision_macro"

    def calculate(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


class F1MacroMetric(MetricStrategy):
    name = "f1_macro"

    def calculate(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
