# monkey patchin the miSVM package
# to use the sparse binary SVM
from __future__ import print_function, division
from probes.utils import BagSplitter_pathced
from misvm.sil import SIL
from misvm.smil import sMIL
from misvm.sbmil import sbMIL
from misvm.svm import SVM, _smart_kernel

import numpy as np
from misvm.kernel import by_name as kernel_by_name
from misvm.util import spdiag
from misvm.util import BagSplitter

import numpy as np

import logging

log = logging.getLogger("milSVM_patch")


BagSplitter = BagSplitter_pathced


def __init__(self, kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True,
             verbose=True, sv_cutoff=1e-7, penalty=-0.1):
    """
    @param kernel : the desired kernel function; can be linear, quadratic,
                    polynomial, or rbf [default: linear]
    @param C : the loss/regularization tradeoff constant [default: 1.0]
    @param scale_C : if True [default], scale C by the number of examples
    @param p : polynomial degree when a 'polynomial' kernel is used
                [default: 3]
    @param gamma : RBF scale parameter when an 'rbf' kernel is used
                    [default: 1.0]
    @param verbose : print optimization status messages [default: True]
    @param sv_cutoff : the numerical cutoff for an example to be considered
                        a support vector [default: 1e-7]
    """
    self.kernel = kernel
    self.C = C
    self.gamma = gamma
    self.p = p
    self.scale_C = scale_C
    self.verbose = verbose
    self.sv_cutoff = sv_cutoff
    self.penalty = penalty

    self._X = None
    self._y = None
    self._objective = None
    self._alphas = None
    self._sv = None
    self._sv_alphas = None
    self._sv_X = None
    self._sv_y = None
    self._b = None
    self._predictions = None


def _setup_svm(self, examples, classes, C):
    kernel = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)
    n = len(examples)
    e = np.matrix(np.ones((n, 1)))

    # Kernel and Hessian
    if kernel is None:
        K = None
        H = None
    else:
        K = _smart_kernel(kernel, examples)
        D = spdiag(classes)
        H = D * K * D

    # Incorporate L1 regularization into H
    if self.penalty > 0:
        print("Adding L1 penalty: %s" % self.penalty)
        H += self.penalty * np.eye(n)  # Add L1 penalty (diagonal term)

    # Term for -sum of alphas
    f = -e

    # Sum(y_i * alpha_i) = 0
    A = classes.T.astype(float)
    b = np.matrix([0.0])

    # 0 <= alpha_i <= C
    lb = np.matrix(np.zeros((n, 1)))
    if type(C) == float:
        ub = C * e
    else:
        # Allow for C to be an array
        ub = C
    return K, H, f, A, b, lb, ub


def fit(self, bags, y, in_bag_labels=None):
    """
    @param bags : a sequence of n bags; each bag is an m-by-k array-like
                    object containing m instances with k features
    @param y : an array-like object of length n containing -1/+1 labels
    """
    self._bags = [np.asmatrix(bag) for bag in bags]
    y = np.asmatrix(y).reshape((-1, 1))

    if in_bag_labels is not None:
        self.bs = BagSplitter(self._bags, y, in_bag_labels)
    else:
        raise NotImplementedError("This method is not implemented yet")
    bs = self.bs

    if self.verbose:
        log.warning('Training initial sMIL classifier for sbMIL...')
    # STAGE 1
    init_classifier = sMIL(kernel=self.kernel, C=self.C, p=self.p, gamma=self.gamma,
                           scale_C=self.scale_C, verbose=self.verbose,
                           sv_cutoff=self.sv_cutoff, penalty=self.penalty)
    init_classifier.fit(bags, y)
    # STAGE 2
    if self.verbose:
        log.warning('Training SIL classifier for sbMIL...')
    f_pos = init_classifier.predict(bs.pos_inst_as_bags)
    # Select nth largest value as cutoff for positive instances
    pos_labels, f_cutoff, _ = sort_and_label(self, bs, f_pos)
    # If less than 5% of positives chosen, ignore the intra-bag-labels
    if (pos_labels == 1).sum() < 0.05 * bs.L_p:
        log.warning(
            f'Less than 5% of positives chosen {(pos_labels == 1).sum()}; ignoring the intra-bag labels.')
        pos_labels = -np.matrix(np.ones((bs.L_p, 1)))
        pos_labels[np.nonzero(f_pos >= f_cutoff)] = 1.0
    # Train on all instances
    if self.verbose:
        log.warning('Retraining with top %d%% as positive...' %
                    int(100 * self.eta))
    # Construct the final labels
    labels = np.vstack([-np.ones((bs.L_n, 1)), pos_labels])
    self._labels = labels

    # #sanity check
    # pos_keep_mask = ((pos_labels.flatten() == 1) & (intrabag_labels.flatten() == 1)) | (
    #     (pos_labels.flatten() == -1) & (intrabag_labels.flatten() == 0))
    # pos_keep_mask = pos_keep_mask.reshape(-1, 1)
    # # Construct the final mask
    # mask = np.array(np.vstack([
    #     np.ones((bs.L_n, 1)),  # Include all negative instances
    #     pos_keep_mask  # Include filtered positive instances
    # ])).astype(bool).flatten()
    # print("mask: ", mask.shape)
    # log.warning(
    #     f"Number of positive instances: {np.sum(pos_labels == 1)} out of {labels[mask].shape[0]}")
    # super(SIL, self).fit(
    #     bs.instances[mask], labels[mask])
    if self.verbose:
        log.warning(
            f"Number of positive instances: {np.sum(pos_labels == 1)} out of {labels.shape[0]}")
    super(SIL, self).fit(
        bs.instances, labels)


def sort_and_label(self, bs, f_pos):
    '''Sort the positive instances and label them
    Return:
        labels: a column vector of labels for positive instances (labeled based on the ranking and intrabag labels)
    '''
    # Select nth largest value as cutoff for positive instances
    n = int(round(bs.L_p * self.eta))
    # Double check the number
    n = min(bs.L_p, n)
    n = max(bs.X_p, n)
    # Threshold value
    f_cutoff = sorted((float(f) for f in f_pos), reverse=True)[n - 1]

    # Label all except for n largest as -1
    labels = -np.matrix(np.ones((bs.L_p, 1)))
    mask = bs.instance_intrabag_labels_pos
    labels[(f_pos >= f_cutoff) & (mask == 1)] = 1.0
    return labels, f_cutoff, mask


def linearize(self, normalize: bool = True):
    X = self._sv_X
    alphas = self._sv_alphas[:, 0]
    y = self._sv_y[:, 0]
    # 1. Compute the direction
    coefs = np.einsum("bs, bs -> b", alphas, y)
    w = np.einsum("b, bh -> bh", coefs, X).sum(0)
    if normalize:
        w /= np.linalg.norm(w)
    # 2. Compute the bias
    non_bound_mask = (alphas > self.sv_cutoff).flatten() & (
        alphas < self.C - self.sv_cutoff).flatten()

    mask_pos = non_bound_mask & (y > 0).flatten()
    mask_neg = non_bound_mask & (y < 0).flatten()
    bias_pos = np.mean(y.flatten()[mask_pos] -
                       np.dot(X, w).flatten()[mask_pos])
    bias_neg = np.mean(y.flatten()[mask_neg] -
                       np.dot(X, w).flatten()[mask_neg])
    bias = (bias_pos + bias_neg) / 2.
    return w, bias


SVM.__init__ = __init__
SVM._setup_svm = _setup_svm
sbMIL.fit = fit
sbMIL.linearize = linearize
sMIL.linearize = linearize
SVM.linearize = linearize
sbMIL.sort_and_label = sort_and_label
