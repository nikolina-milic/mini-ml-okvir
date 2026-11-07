"""Factory obrazac za kreiranje scikit-learn modela."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class UnknownModelError(ValueError):
    """Jasna greška za naziv modela koji fabrika ne poznaje."""


class ModelFactory:
    """Kreira modele bez vezivanja klijenta za konkretne klase."""

    _creators: dict[str, Callable[..., ClassifierMixin]] = {
        "linear": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
    }

    _defaults: dict[str, dict[str, object]] = {
        "linear": {"max_iter": 300, "random_state": 42},
        "decision_tree": {"random_state": 42},
        "knn": {"n_neighbors": 5},
    }

    @classmethod
    def create(cls, name: str, **params: object) -> ClassifierMixin:
        normalized_name = name.strip().lower() if isinstance(name, str) else ""
        creator = cls._creators.get(normalized_name)
        if creator is None:
            allowed = ", ".join(cls.available_models())
            raise UnknownModelError(
                f"Nepoznat model '{name}'. Dostupni modeli: {allowed}."
            )

        model_params = {**cls._defaults[normalized_name], **params}
        return creator(**model_params)

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._creators))
