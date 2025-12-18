"""Linear probe implementations for binary and multiclass classification.

This module provides pre-trained linear probe classes that use fixed coefficients
and intercepts for making predictions without requiring additional training.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

log = logging.getLogger(__name__)


class BinaryLinearProbe(BaseEstimator, ClassifierMixin):
    """Binary linear probe with pre-trained coefficients and intercept.

    This probe does not require fitting; it uses provided coefficients and
    intercept to make predictions.

    Args:
        coef: Coefficient array for the linear model.
        intercept: Intercept value(s) for the linear model.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> BinaryLinearProbe:
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept).ravel()
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.classes_ = [0, 1]
        self.is_fitted_ = True

    def fit(self, X: np.ndarray = None, y: np.ndarray = None) -> BinaryLinearProbe:
        """Fit method (no-op for pre-trained probe)."""
        log.warning("This probe is pre-trained and does not require fitting.")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for each sample in X.
        Args:
            X: (N, d) array of input data
        Returns:
            scores: (N,) array of decision scores
        """
        check_is_fitted(self)
        X = np.asarray(X)
        return (X @ self.coef_.T).ravel() + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability for each sample in X.

        Args:
            X: (N, d) array of input data.

        Returns:
            (N,) array of probabilities.
        """
        return expit(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Args:
            X: (N, d) array of input data.

        Returns:
            (N,) array of predicted class labels.
        """
        return self.predict_proba(X).round()


class MultiClassBaggedProjector(BaseEstimator, ClassifierMixin):
    """Multiclass linear probe for bag-of-instances data.

    This probe uses pre-trained coefficients and intercept to make predictions
    on bags of instances, aggregating predictions across instances.

    Args:
        coef: Coefficient matrix of shape (n_classes, n_features).
        intercept: Intercept array of shape (n_classes,).
        aggregation: Method to aggregate predictions ('max', 'mean', or 'sum').
            Defaults to 'max'.
        classes: Class labels. If None, uses range(n_classes).
    """

    def __init__(
        self,
        coef: np.ndarray,
        intercept: np.ndarray,
        aggregation: str = "max",
        classes=None,
    ) -> None:
        assert (
            coef.shape[0] == intercept.shape[0]
        ), "Number of classes must match between coef and intercept."
        assert aggregation in ["max"], "Aggregation must be either 'max'."
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept).ravel()
        self.aggregation = aggregation

        # Sklearn Attributes
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.classes_ = (
            np.arange(self.coef.shape[0]) if classes is None else np.asarray(classes)
        )
        self.is_fitted_ = True

    def fit(
        self, X: Sequence[np.ndarray] = None, y: np.ndarray = None
    ) -> MultiClassBaggedProjector:
        """Fit method (no-op for pre-trained probe)."""
        log.warning("This probe is pre-trained and does not require fitting.")
        return self

    def _sanitize_(self, X: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Convert input to list of arrays."""
        if type(X) is np.ndarray:
            X = [X]
        X = [np.asarray(bag) for bag in X]
        return X

    def decision_function(self, X: Sequence[np.ndarray], agg: str = None) -> np.ndarray:
        """Compute decision function for each bag in X.

        Args:
            X: Sequence of bags, where each bag is an array of instances.
            agg: Aggregation method override (unused, kept for compatibility).

        Returns:
            (N, C) array of decision scores for N bags and C classes.

        Raises:
            ValueError: If aggregation method is unsupported.
        """
        check_is_fitted(self)
        X = self._sanitize_(X)
        scores = []
        for bag in X:
            logits = bag @ self.coef_.T + self.intercept_
            if logits.ndim == 1:
                agg = logits
            elif self.aggregation == "max":
                agg = np.max(logits, axis=0)
            elif self.aggregation == "mean":
                agg = np.mean(logits, axis=0)
            elif self.aggregation == "sum":
                agg = np.sum(logits, axis=0)
            else:
                raise ValueError(f"Unsupported aggregation method: {self.aggregation}")
            scores.append(agg)

        return np.vstack(scores)

    def predict_proba(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """Predict class scores for each bag in X.

        Args:
            X: Sequence of bags.

        Returns:
            (N, C) array of class scores.
        """
        return self.decision_function(X)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """Predict class labels by argmax over aggregated logits.

        Args:
            X: Sequence of bags.

        Returns:
            (N,) array of predicted class labels.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class MulticlassProbe(BaseEstimator, ClassifierMixin):
    """Multiclass probe using base projector with optional calibration.

    This probe wraps a MultiClassBaggedProjector with a scaler and predictor
    for calibrated multiclass predictions.

    Args:
        base: Base MultiClassBaggedProjector for extracting features.
        scaler: Transformer for scaling features. Defaults to StandardScaler.
        predictor: Classifier for final predictions. Defaults to
            LogisticRegression with balanced weights.
    """

    def __init__(
        self,
        base: MultiClassBaggedProjector,
        scaler: TransformerMixin = StandardScaler(),
        predictor: ClassifierMixin = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=3000,
            class_weight="balanced",
            random_state=0,
        ),
    ) -> MulticlassProbe:
        self.base = base
        self.scaler = scaler
        self.predictor = predictor
        self.classes_ = base.classes_
        self.is_fitted_ = False

    def fit(
        self, X: Sequence[np.ndarray] | np.ndarray = None, y: np.ndarray = None
    ) -> MulticlassProbe:
        """Fit the scaler and predictor on transformed base features.

        Args:
            X: Sequence of bags or array of samples.
            y: Target labels.

        Returns:
            Self.
        """

        scores = self.base.decision_function(X)
        transformed_scores = self.scaler.fit_transform(scores)
        self.predictor.fit(transformed_scores, y)
        self.is_fitted_ = True
        return self

    def decision_function(self, X: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Compute decision function on X.

        Args:
            X: Input data.

        Returns:
            Decision scores.
        """
        check_is_fitted(self)
        scores = self.base.decision_function(X)
        transformed_scores = self.scaler.transform(scores)
        return self.predictor.decision_function(transformed_scores)

    def predict(self, X: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Input data.

        Returns:
            Predicted class labels.
        """
        check_is_fitted(self)
        scores = self.base.decision_function(X)
        transformed_scores = self.scaler.transform(scores)
        return self.predictor.predict(transformed_scores)

    def predict_proba(self, X: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

        Args:
            X: Input data.

        Returns:
            Class probability estimates.
        """
        check_is_fitted(self)
        scores = self.base.decision_function(X)
        transformed_scores = self.scaler.transform(scores)
        return self.predictor.predict_proba(transformed_scores)
