from ml_experiments.decorators import TimedExperiment
from ml_experiments.experiment import IrisExperiment
from ml_experiments.metrics import AccuracyMetric
from ml_experiments.observers import HistoryObserver


def test_template_method_runs_complete_iris_experiment():
    result = IrisExperiment("linear", AccuracyMetric()).run()

    assert result.model_name == "linear"
    assert result.metric_name == "accuracy"
    assert 0.0 <= result.metric_value <= 1.0
    assert result.train_size + result.test_size == 150


def test_observer_receives_training_events_in_order():
    history = HistoryObserver()
    IrisExperiment("decision_tree", AccuracyMetric(), observers=[history]).run()

    stages = [event.stage for event in history.events]
    assert stages == [
        "started",
        "data_loaded",
        "data_split",
        "training_started",
        "training_finished",
        "evaluated",
    ]


def test_observer_can_be_detached():
    history = HistoryObserver()
    experiment = IrisExperiment("knn", AccuracyMetric(), observers=[history])
    experiment.detach(history)
    experiment.run()

    assert history.events == []


def test_timing_decorator_adds_nonnegative_duration():
    experiment = IrisExperiment("linear", AccuracyMetric())
    result = TimedExperiment(experiment).run()

    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0.0
