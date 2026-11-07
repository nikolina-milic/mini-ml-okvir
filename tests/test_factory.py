import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_experiments.factory import ModelFactory, UnknownModelError


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("linear", LogisticRegression),
        ("decision_tree", DecisionTreeClassifier),
        ("knn", KNeighborsClassifier),
    ],
)
def test_factory_creates_supported_models(name, expected_type):
    assert isinstance(ModelFactory.create(name), expected_type)


def test_factory_passes_custom_parameters():
    model = ModelFactory.create("knn", n_neighbors=3)
    assert model.n_neighbors == 3


def test_factory_rejects_unknown_model_with_clear_message():
    with pytest.raises(UnknownModelError, match="Nepoznat model 'svm'.*linear"):
        ModelFactory.create("svm")
