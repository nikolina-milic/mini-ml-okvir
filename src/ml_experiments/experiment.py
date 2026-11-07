"""Template Method za stabilan tok ML eksperimenta."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from .factory import ModelFactory
from .metrics import MetricStrategy
from .observers import TrainingEvent, TrainingObserver, TrainingSubject


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    metric_name: str
    metric_value: float
    train_size: int
    test_size: int
    duration_seconds: float | None = None


class Experiment(ABC):
    @abstractmethod
    def run(self) -> ExperimentResult:
        """Pokreće eksperiment i vraća rezultat."""


class TemplateExperiment(Experiment, TrainingSubject, ABC):
    """Bazna klasa čija metoda ``run`` definiše redosljed koraka."""

    def __init__(
        self,
        model_name: str,
        metric: MetricStrategy,
        *,
        test_size: float = 0.25,
        random_state: int = 42,
        model_params: dict[str, object] | None = None,
        observers: list[TrainingObserver] | None = None,
    ) -> None:
        TrainingSubject.__init__(self)
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size mora biti između 0 i 1.")

        self.model_name = model_name
        self.metric = metric
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or {}
        for observer in observers or []:
            self.attach(observer)

    def run(self) -> ExperimentResult:
        """Template Method: redosljed koraka je definisan na jednom mjestu."""

        self.notify(TrainingEvent("started", "Eksperiment je pokrenut."))
        features, target = self.load_data()
        self.notify(
            TrainingEvent(
                "data_loaded",
                "Podaci su učitani.",
                {"samples": len(target)},
            )
        )

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target,
        )
        self.notify(
            TrainingEvent(
                "data_split",
                "Podaci su podijeljeni na trening i test skup.",
                {"train_size": len(y_train), "test_size": len(y_test)},
            )
        )

        model = ModelFactory.create(self.model_name, **self.model_params)
        self.notify(
            TrainingEvent(
                "training_started",
                f"Počinje treniranje modela '{self.model_name}'.",
            )
        )
        model.fit(x_train, y_train)
        self.notify(TrainingEvent("training_finished", "Treniranje je završeno."))

        predictions = model.predict(x_test)
        score = self.metric.calculate(y_test, predictions)
        self.notify(
            TrainingEvent(
                "evaluated",
                f"{self.metric.name} = {score:.4f}",
                {"score": score},
            )
        )

        return ExperimentResult(
            model_name=self.model_name,
            metric_name=self.metric.name,
            metric_value=score,
            train_size=len(y_train),
            test_size=len(y_test),
        )

    @abstractmethod
    def load_data(self) -> tuple[Any, Any]:
        """Varijabilni korak koji podklasa mora implementirati."""


class IrisExperiment(TemplateExperiment):
    def load_data(self) -> tuple[Any, Any]:
        return load_iris(return_X_y=True)
