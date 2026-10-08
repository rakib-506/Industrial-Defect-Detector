"""Custom scikit-learn estimators used by the defect-type classifier.

These live in their own module rather than in the training script so that the
pickled pipeline resolves them by a stable import path. A class defined inside a
``python -m`` entry point is recorded as ``__main__.<name>`` and fails to load
from any other process, such as the API server.
"""

from __future__ import annotations

from sklearn.decomposition import PCA


class SafePCA(PCA):
    """PCA that clamps ``n_components`` to what the fold can actually support.

    With only ~63 labelled defect images, an inner cross-validation fold can
    hold fewer samples than the requested component count, which plain PCA
    rejects outright.
    """

    def __init__(self, n_components=None, random_state=None):
        super().__init__(n_components=n_components, random_state=random_state)
        self._requested = n_components

    def _clamp(self, X) -> None:  # noqa: N803 - sklearn signature
        self.n_components = min(self._requested, X.shape[0] - 1, X.shape[1])

    def fit(self, X, y=None):  # noqa: N803 - sklearn signature
        self._clamp(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):  # noqa: N803 - sklearn signature
        self._clamp(X)
        return super().fit_transform(X, y)

    def set_params(self, **params):
        if "n_components" in params:
            self._requested = params["n_components"]
        return super().set_params(**params)
