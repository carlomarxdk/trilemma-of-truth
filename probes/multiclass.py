
import numpy as np
from abc import ABC, abstractmethod
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import sys
sys.path.insert(0, '..')
# should be here
if True:
    from loaders.task import Task
    from utils_hydra import BagProcessor

log = logging.getLogger(__name__)


# Abstract base class for ensemble classifiers
class EnsembleClassifier(BaseEstimator, ClassifierMixin, ABC):
    def fit(self, bags, y):
        # Abstract method to fit the model
        pass

    def predict_proba(self, bags):
        # Abstract method to predict probabilities
        pass

    def predict_scores(self, bags):
        # Abstract method to predict raw scores
        pass

    def _get_scores_for_each_instance(self, bags):
        # Abstract method to scores for each instance in the bag
        pass

    def _get_scores_for_last_instance(self, bags):
        # Abstract method to scores for the last instance in the bag
        pass

    def _decision_function(self, X, direction, bias):
        # Abstract method to compute the decision function
        # for a given input X, direction and bias
        pass


class MulticlassSIL(EnsembleClassifier):
    def __init__(self, readers, max_bag_size=1, layer_id=0):
        # task T:4, F:5, IDK:3
        assert readers.keys() == {0, 1, 2}, 'Invalid readers'
        self.readers = readers
        self.max_bag_size = max_bag_size
        self.layer_id = layer_id

        self.cls = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('predictor', LogisticRegression(solver='lbfgs',
                                             class_weight='balanced', penalty=None,
                                             multi_class='multinomial'))
        ])

    @staticmethod
    def _decision_function(X: np.ndarray, direction: np.ndarray, bias: np.ndarray) -> np.ndarray:
        '''
        Compute the decision function for a given input X, direction and bias.
        Args:
            X: input data
            direction: direction vector
            bias: bias vector
        '''
        return np.dot(X, direction) + bias

    def fit(self, bags, y):
        '''
        Fit the classifier using the provided bags and labels.
        Args:
            bags: List of bags (num bags, bag_i size, hidden size)
            y: Labels (per bag)
        '''
        scores = self.predict_scores(bags, per_instance=False)
        self.cls.fit(scores, y)
        return self

    def predict_proba(self, bags, per_instance=False):
        '''
        Predict probabilities for the given bags.
        Args:
            bags: List of bags
            per_instance: Whether to predict per instance or per bag
        '''
        bags = deepcopy(bags)
        scores = self.predict_scores(bags, per_instance=per_instance)
        if per_instance:
            output = []
            for i in range(len(scores)):
                output.append(self.cls.predict_proba(scores[i]))
            return output
        else:
            return self.cls.predict_proba(scores)

    def predict_scores(self, bags, per_instance=False):
        '''
        Get raw scores for each bag.
        Args:
            bags: list of bags
            per_instance: Whether to predict per instance or per bag
        '''
        if per_instance:
            return self._get_scores_for_each_instance(bags)
        else:
            return self._get_scores_for_last_instance(bags)

    def _get_scores_for_last_instance(self, bags):
        '''
        Get scores based on the last token only.
        '''
        scores = np.zeros((len(bags), 3))
        X = np.vstack([bag[-1] for bag in bags])
        for key, reader in self.readers.items():
            direction = reader.direction(layer_id=self.layer_id)
            bias = reader.bias(layer_id=self.layer_id)
            scaler = reader.scaler(layer_id=self.layer_id)
            Xs = scaler.transform(X)
            s = self._decision_function(Xs, direction, bias)
            scores[:, key] = deepcopy(s)
        return scores  # (num bags, num classes)

    def _get_scores_for_each_instance(self, bags):
        '''
        Get scores based on the each element in the bag.
        '''
        reader_params = []
        for reader in self.readers.values():
            direction = reader.direction(layer_id=self.layer_id)
            bias = reader.bias(layer_id=self.layer_id)
            scaler = reader.scaler(layer_id=self.layer_id)
            reader_params.append((scaler, direction, bias))

        output = []
        for bag in bags:
            # assume `bag` is array‐like shape (bag size, hidden size)
            X = self.sanitize(np.vstack(bag))
            # for each reader, transform the *whole* bag at once and score
            per_reader_scores = []
            for scaler, direction, bias in reader_params:
                Xs = scaler.transform(X)  # (bag size, hidden size)
                s = self._decision_function(
                    Xs, direction, bias)  # (bag size,)
                s = self.sanitize(s)
                per_reader_scores.append(s)
            # stack into shape (bag size, num classes)
            bag_scores = np.column_stack(per_reader_scores)
            output.append(bag_scores)
        return output  # (num bags, bag size, num classes)

    def sanitize(self, X, val=1e4):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(np.nan_to_num(
            X,
            nan=0.0,
            posinf=val,
            neginf=-val
        ), a_min=-val, a_max=val)


class MulticlassMIL(EnsembleClassifier):
    def __init__(self, readers, max_bag_size=100, layer_id=0):
        # task T:4, F:5, IDK:3
        assert readers.keys() == {0, 1, 2}, 'Invalid readers'
        self.readers = readers
        self.max_bag_size = max_bag_size
        self.layer_id = layer_id
        self.cls = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('predictor', LogisticRegression(solver='lbfgs',
             class_weight='balanced', penalty=None, multi_class='multinomial'))
        ])

    def fit(self, bags, y):
        scores = self._get_scores(bags, verbose=True)
        self.cls.fit(scores, y)
        return self

    def predict_proba(self, bags):
        scores = self._get_scores(bags)
        return self.cls.predict_proba(scores)

    def _get_scores(self, bags, verbose=False):
        scores = np.zeros((len(bags), 3))
        for key, reader in self.readers.items():
            if verbose:
                task = Task(reader.task)
            bp = BagProcessor(max_bag_size=self.max_bag_size,
                              pos_labels_in_bag=2,
                              scaler=reader.scaler(layer_id=self.layer_id))
            processed_bags = bp.process(bags)
            s = reader.predict_with_bag(
                bags=processed_bags[0], layer_id=self.layer_id)
            scores[:, key] = deepcopy(s)
        return scores

    def predict_scores(self, bags):
        return self._get_scores(bags)
