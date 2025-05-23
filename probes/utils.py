import math
import numpy as np


def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    return sum([(math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q)])


def compute_hellinger_distance(p, q, n_bins: int = 100):
    """
    Compute the Hellinger distance between two distributions.
    """
    amax = max(max(p), max(q))
    amin = min(min(p), min(q))
    def norm(x): return (x - amin) / (amax - amin)
    bins = np.linspace(0, 1, n_bins)
    p = np.array([norm(x) for x in p])
    q = np.array([norm(x) for x in q])
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)
    return hellinger_fast(p_hist, q_hist)
