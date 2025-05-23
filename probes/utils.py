from copy import deepcopy
import numpy as np
from typing import List, Union


class BagSplitter_pathced:
    def __init__(self, bags, classes, bag_labels=None):
        self._bags = deepcopy(bags)
        self._classes = deepcopy(classes)
        self._bag_labels = deepcopy(bag_labels)

    @property
    def bags(self):
        return self._bags

    @property
    def classes(self):
        return self._classes

    @property
    def bag_labels(self) -> np.ndarray | List:
        return self._bag_labels

    @property
    def pos_bags_with_labels(self):
        return [(bag, label) for bag, cls, label in zip(self.bags, self.classes, self.bag_labels) if cls > 0.0]

    @property
    def pos_bags(self):
        return deepcopy([bag for bag, cls in zip(self.bags, self.classes) if cls > 0.0])

    @property
    def neg_bags(self):
        return deepcopy([bag for bag, cls in zip(self.bags, self.classes) if cls <= 0.0])

    @property
    def neg_instances(self):
        return np.vstack(self.neg_bags)

    @property
    def pos_instances(self):
        return np.vstack(self.pos_bags)

    @property
    def instances(self):
        return np.vstack([self.neg_instances, self.pos_instances])

    @property
    def inst_classes(self):
        return np.vstack([-np.ones((self.L_n, 1)), np.ones((self.L_p, 1))])

    @property
    def pos_groups(self):
        return [len(bag) for bag in self.pos_bags]

    @property
    def neg_groups(self):
        return [len(bag) for bag in self.neg_bags]

    @property
    def L_n(self):
        return len(self.neg_instances)

    @property
    def L_p(self):
        '''Number of positive instances'''
        return len(self.pos_instances)

    @property
    def L(self):
        return self.L_p + self.L_n

    @property
    def X_n(self):
        return len(self.neg_bags)

    @property
    def X_p(self):
        '''Number of positive bags'''
        return len(self.pos_bags)

    @property
    def X(self):
        return self.X_p + self.X_n

    @property
    def neg_inst_as_bags(self):
        return [inst for bag in self.neg_bags for inst in bag]

    @property
    def pos_inst_as_bags(self):
        return [inst for bag in self.pos_bags for inst in bag]

    @property
    def instance_intrabag_labels_pos(self):
        '''Intra-bag labels for positive instances
        Return: a flattened array of intra-bag labels for positive instances (collected over all positive bags)
        '''
        x = [label for _, label in self.pos_bags_with_labels]
        return np.concatenate(x)
