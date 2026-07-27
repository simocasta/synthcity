import pytest

from synthcity.metrics.scores import ScoreEvaluator


class _FailingEvaluator:
    @staticmethod
    def fqdn() -> str:
        return "privacy.test_failure"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def evaluate() -> dict:
        raise ValueError("deliberate metric failure")


def test_score_evaluator_preserves_legacy_fail_soft_behavior() -> None:
    scores = ScoreEvaluator()
    scores.queue(_FailingEvaluator())

    scores.compute()

    assert scores.to_dataframe().empty


def test_score_evaluator_can_propagate_metric_errors() -> None:
    scores = ScoreEvaluator()
    scores.queue(_FailingEvaluator())

    with pytest.raises(
        RuntimeError,
        match="privacy.test_failure: deliberate metric failure",
    ):
        scores.compute(raise_on_error=True)
