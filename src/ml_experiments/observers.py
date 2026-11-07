"""Observer obrazac za praćenje toka treniranja."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingEvent:
    stage: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class TrainingObserver(ABC):
    @abstractmethod
    def update(self, event: TrainingEvent) -> None:
        """Prima događaj iz eksperimenta."""


class ConsoleObserver(TrainingObserver):
    def update(self, event: TrainingEvent) -> None:
        print(f"[{event.stage}] {event.message}")


class HistoryObserver(TrainingObserver):
    """Čuva događaje; koristan za testove, UI i audit trag."""

    def __init__(self) -> None:
        self.events: list[TrainingEvent] = []

    def update(self, event: TrainingEvent) -> None:
        self.events.append(event)


class TrainingSubject:
    def __init__(self) -> None:
        self._observers: list[TrainingObserver] = []

    def attach(self, observer: TrainingObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: TrainingObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: TrainingEvent) -> None:
        for observer in tuple(self._observers):
            observer.update(event)
