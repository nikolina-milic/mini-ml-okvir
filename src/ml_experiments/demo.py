"""Demonstracija svih modela na Iris skupu podataka."""

from __future__ import annotations

from .decorators import TimedExperiment
from .experiment import IrisExperiment
from .factory import ModelFactory
from .metrics import AccuracyMetric
from .observers import ConsoleObserver


def main() -> None:
    print("Mini okvir za ML eksperimente — Iris dataset\n")
    results = []

    for model_name in ModelFactory.available_models():
        experiment = IrisExperiment(
            model_name=model_name,
            metric=AccuracyMetric(),
            observers=[ConsoleObserver()],
        )
        result = TimedExperiment(experiment).run()
        results.append(result)
        print(
            f"REZULTAT: {result.model_name:13} | "
            f"{result.metric_name}={result.metric_value:.4f} | "
            f"vrijeme={result.duration_seconds:.6f}s\n"
        )

    best = max(results, key=lambda item: item.metric_value)
    print(f"Najbolji model: {best.model_name} ({best.metric_value:.4f})")


if __name__ == "__main__":
    main()
