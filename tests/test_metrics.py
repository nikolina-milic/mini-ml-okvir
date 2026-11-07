import pytest

from ml_experiments.metrics import AccuracyMetric, F1MacroMetric, PrecisionMacroMetric


def test_accuracy_strategy():
    score = AccuracyMetric().calculate([0, 1, 1, 0], [0, 1, 0, 0])
    assert score == pytest.approx(0.75)


@pytest.mark.parametrize("metric", [PrecisionMacroMetric(), F1MacroMetric()])
def test_macro_metric_strategies_return_valid_scores(metric):
    score = metric.calculate([0, 1, 2, 2], [0, 1, 1, 2])
    assert 0.0 <= score <= 1.0
