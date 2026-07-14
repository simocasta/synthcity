# stdlib
import sys
from pathlib import Path
from typing import Any, Tuple, Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
    IdentifiabilityScore,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    ImageDataLoader,
)


class RecordingDomias(DomiasMIAPrior):
    @staticmethod
    def name() -> str:
        return "DomiasMIA_recording"

    def evaluate_p_R(
        self,
        synth_set: DataLoader,
        synth_val_set: DataLoader,
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.calls = getattr(self, "calls", 0) + 1
        self.recorded_reference = reference_set.copy()
        self.recorded_test = X_test.copy()
        scores = np.arange(1, len(X_test) + 1, dtype=float)
        return scores, np.ones(len(X_test), dtype=float)


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DeltaPresence,
        kAnonymization,
        kMap,
        lDiversityDistinct,
        IdentifiabilityScore,
        DomiasMIABNAF,
        DomiasMIAKDE,
        DomiasMIAPrior,
    ],
)
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluator(evaluator_t: Type, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    Xloader = GenericDataLoader(
        X,
        sensitive_features=["sepal length (cm)", "sepal width (cm)"],
    )
    evaluator = evaluator_t(
        use_cache=False,
    )

    X_train = None
    if "DomiasMIA" in evaluator.name():
        X_train = GenericDataLoader(X.iloc[:50])
        Xloader = GenericDataLoader(
            X.iloc[50:],
            sensitive_features=["sepal length (cm)", "sepal width (cm)"],
        )

    test_plugin.fit(X_train if X_train is not None else Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        score = evaluator.evaluate(
            Xloader,
            X_gen,
            X_train,
            X_ref_syn,
            reference_size=10,
        )
    else:
        score = evaluator.evaluate(Xloader, X_gen)

    for submetric in score:
        assert score[submetric] > 0

    assert evaluator.type() == "privacy"

    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        def_score = evaluator.evaluate_default(
            Xloader,
            X_gen,
            X_train,
            X_ref_syn,
            reference_size=10,
        )
    else:
        def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert isinstance(def_score, (float, int))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_image_support() -> None:
    dataset = datasets.MNIST(".", download=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        IdentifiabilityScore,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict)
        for k in score:
            assert score[k] >= 0
            assert not np.isnan(score[k])


def test_domias_uses_balanced_member_and_nonmember_samples() -> None:
    columns = ["x", "y"]
    X_train = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 1000, columns=columns)
    )
    X_gt = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2), columns=columns)
    )
    X_syn = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 2000, columns=columns)
    )
    X_ref_syn = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 3000, columns=columns)
    )

    evaluator = RecordingDomias(use_cache=False)
    evaluator.evaluate(
        X_gt,
        X_syn,
        X_train,
        X_ref_syn,
        member_size=6,
        reference_size=4,
    )

    assert evaluator.recorded_test.shape == (12, 2)
    np.testing.assert_array_equal(evaluator.recorded_test[:6], X_train.numpy()[:6])
    np.testing.assert_array_equal(evaluator.recorded_test[6:], X_gt.numpy()[:6])
    np.testing.assert_array_equal(evaluator.recorded_reference, X_gt.numpy()[-4:])


def test_domias_rejects_identical_member_and_nonmember_sources() -> None:
    X, _ = load_iris(return_X_y=True, as_frame=True)
    X_loader = GenericDataLoader(X)
    X_syn = GenericDataLoader(X + 1)
    X_ref_syn = GenericDataLoader(X + 2)

    with pytest.raises(ValueError, match="must differ"):
        RecordingDomias(use_cache=False).evaluate(
            X_loader,
            X_syn,
            X_loader,
            X_ref_syn,
            reference_size=10,
        )


def test_domias_cache_includes_member_and_reference_context(tmp_path: Path) -> None:
    columns = ["x", "y"]
    X_gt = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2), columns=columns)
    )
    X_syn = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 2000, columns=columns)
    )
    X_ref_syn = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 3000, columns=columns)
    )
    X_train_a = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 1000, columns=columns)
    )
    X_train_b = GenericDataLoader(
        pd.DataFrame(np.arange(40).reshape(20, 2) + 1100, columns=columns)
    )

    evaluator = RecordingDomias(workspace=tmp_path, use_cache=True)
    kwargs = {"member_size": 6, "reference_size": 4}

    evaluator.evaluate(X_gt, X_syn, X_train_a, X_ref_syn, **kwargs)
    evaluator.evaluate(X_gt, X_syn, X_train_a, X_ref_syn, **kwargs)
    assert evaluator.calls == 1

    evaluator.evaluate(X_gt, X_syn, X_train_b, X_ref_syn, **kwargs)
    assert evaluator.calls == 2
