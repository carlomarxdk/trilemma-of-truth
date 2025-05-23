# monkey patchin the miSVM package
# to use the sparse binary SVM
from __future__ import print_function, division
from probes.utils import BagSplitter_pathced
from misvm.sil import SIL
from misvm.svm import SVM, _smart_kernel

import numpy as np
from misvm.kernel import by_name as kernel_by_name
from misvm.util import spdiag
from misvm.util import BagSplitter

import numpy as np

import logging

log = logging.getLogger("silSVM_patch")


BagSplitter = BagSplitter_pathced


def __init__(self, kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True,
             verbose=True, sv_cutoff=1e-7):
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
    self._labels = y

    super(SIL, self).fit(
        bs.instances, y)


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


def get_params(self, deep=True):
    """
    return params
    """
    return {
        'status': 'complete'}


SVM.__init__ = __init__
SVM._setup_svm = _setup_svm
SIL.fit = fit
SIL.get_params = get_params
SIL.linearize = linearize
SVM.linearize = linearize
