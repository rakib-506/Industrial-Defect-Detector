

from __future__ import annotations

from sklearn.decomposition import PCA


class SafePCA(PCA):


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
