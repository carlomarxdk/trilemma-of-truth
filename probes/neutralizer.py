import numpy as np
import torch
import pandas as pd
from typing import List, Set, Dict
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
VALID_METHODS = ["svd", "pca", "inlp"]


class DirectionNeutralizer(BaseEstimator, TransformerMixin):
    '''
    A class to neutralize the embeddings by removing the bias components.
    '''

    def __init__(self,  method: str = "svd", verbose: bool = True, random_seed: int = 42,
                 max_n_components: int = 500, threshold: float = 0.95, normalize: bool = True) -> None:
        assert method in VALID_METHODS, f"Method must be one of {VALID_METHODS}"
        self.random_seed = random_seed
        self.method = method
        self.verbose = verbose
        self.normalize = normalize
        self.max_n_components = max_n_components
        self.threshold = threshold

        self.model = None

    def fit(self, X: np.array, df: pd.DataFrame, attribute: str = "negation", frac: float = 0.3, n: int = None,  use_category: bool = False) -> None:
        # assert all(attr in df.columns for attr in attributes), "Attributes must be in the dataframe."
        np.random.seed(self.random_seed)
        assert attribute in df.columns, "Attribute must be in the dataframe."
        assert "object_1" in df.columns
        assert "object_2" in df.columns
        assert frac > 0 and frac <= 1, "Fraction must be in (0, 1]."
        df = df.reset_index(drop=False)  # need to keep ids to assemble pairs
        dfn = df[df[attribute] == 1]
        dfs = df[df[attribute] == 0]

        self.corresponding_obj1_sets = self.find_pairs(
            dfn, dfs, on=["correct", "object_1"], frac=frac)
        if n is None:
            n = len(self.corresponding_obj1_sets)
        if use_category:
            self.corresponding_obj2_sets = self.find_pairs_conditional(
                dfn, dfs, on=["correct"], n=n, condition="category")
            neutralized_embeddings = np.vstack([
                self._mean_removed_embeddings(
                    self.corresponding_obj1_sets, X),
                self._mean_removed_embeddings(
                    self.corresponding_obj2_sets, X)
                # self._mean_removed_embeddings(
                # self.corresponding_random_sets, X)
            ])
        else:
            neutralized_embeddings = np.vstack([
                self._mean_removed_embeddings(
                    self.corresponding_obj1_sets, X),
                # self._mean_removed_embeddings(
                # self.corresponding_random_sets, X)
            ])

        if self.verbose:
            print("Neutralized embeddings have been computed. Shape:",
                  neutralized_embeddings.shape)

        if self.method == "svd":
            self.compute_svd(X=neutralized_embeddings)
        self.is_fitted_ = True

        return self

    def find_pairs(self, dfn: pd.DataFrame, dfs: pd.DataFrame, on: List, frac: float) -> Dict:
        output = {}
        pairs = pd.merge(dfs, dfn, on=on)[["index_x", "index_y"]].sample(
            frac=frac, random_state=self.random_seed).values
        for i in set(pairs[:, 0].tolist()):
            ids = np.where(pairs[:, 0] == i)[0]
            output[i] = pairs[ids, 1].tolist()
        return output

    def find_pairs_conditional(self, dfn: pd.DataFrame, dfs: pd.DataFrame, on: List, n: int, condition: str) -> Dict:
        output = {}
        pairs = pd.merge(dfs, dfn, on=on)
        mask = pairs[f"{condition}_x"] != pairs[f"{condition}_y"]
        weights = pairs[mask].groupby(
            f"{condition}_x")[f"{condition}_y"].count().reset_index()
        weights = weights.rename(
            columns={f"{condition}_y": "count"})
        weights["count"] = 1 - \
            (weights["count"] / weights["count"].sum())
        self.weights = weights.set_index("category_x").to_dict(orient="index")
        _w = np.array([self.weights[c]["count"]
                      for c in pairs[mask].category_x.values])
        pairs = pairs[mask][["index_x", "index_y"]].sample(
            n=n, weights=_w, random_state=self.random_seed).values
        for i in set(pairs[:, 0].tolist()):
            ids = np.where(pairs[:, 0] == i)[0]
            output[i] = pairs[ids, 1].tolist()
        return output

    def compute_svd(self, X) -> None:
        self.model = TruncatedSVD(
            n_components=self.max_n_components, algorithm="randomized", random_state=self.random_seed)
        self.model.fit(X)
        # Determine the number of components to keep
        try:
            self.n_components_to_keep = np.where(
                self.model.explained_variance_ratio_.cumsum() >= self.threshold)[0][0]
            if self.n_components_to_keep == 0:
                self.n_components_to_keep = 1
        except:
            print("No components meet the threshold. Keeping all components.")
            self.n_components_to_keep = self.max_n_components
        self.bias_components = self.model.components_[
            :self.n_components_to_keep]
        if self.verbose:
            print(
                f"Components have been computed. Keeping {self.n_components_to_keep} components.")

    def transform(self, X: np.array) -> np.array:
        """
        Neutralize the embeddings by removing the bias components.
        """
        check_is_fitted(self)
        # Compute the projection of X onto each bias component and subtract it
        bias_component = X @ self.bias_components.T
        X_neutralized = X - bias_component @ self.bias_components

        if self.normalize:
            return self._normalize(X_neutralized)
        else:  # not recommended
            return X_neutralized

    def _normalize(self, X: np.array) -> np.array:
        '''
        Normalize the embeddings to have unit length.
        '''
        return X / np.linalg.norm(X, axis=1)[:, np.newaxis]

    def _mean_removed_embeddings(self, pairs: Dict, activations: torch.Tensor) -> np.ndarray:
        output = list()
        for k, v in pairs.items():
            words_in_set = list()
            words_in_set.append(activations[k])  # append the first word
            for j in v:  # append the rest of the words
                words_in_set.append(activations[j])
            mu = np.mean(words_in_set, axis=0)
            for w in words_in_set:
                output.append(w - mu)
        return np.array(output)
