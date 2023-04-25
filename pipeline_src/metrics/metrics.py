import numpy as np


def mean_reciprocal_rank(r):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    r = np.asarray(r).nonzero()[0]
    return 1.0 / (r[0] + 1) if r.size else 0.0


def precision_at_k(r, k, n):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return (np.mean(r) * k) / min(k, n)
    # Modified from the first version. Now the gold elements are taken into account


def average_precision(r, n):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1, n) for k in range(r.size)]
    # Modified from the first version (removed "if r[k]"). All elements (zero and nonzero) are taken into account
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(r, n):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return average_precision(r, n)