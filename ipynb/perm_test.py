import numpy as np
import pandas as pd
import scipy
from scipy import stats


def perm_kruskal_test(x, y, permutations=1000):
    """ Permutation kruskal test
    Parameters
    ----------
    x : array_like
        Group 1 data
    y : array_like
        Group 2 data
    permutations : int
        Number of permutations

    """
    x = np.ravel(np.array(x.copy()))
    y = np.ravel(np.array(y.copy()))
    n1 = len(x)
    n2 = len(y)
    s_, p_ = stats.kruskal(x, y)
    xy = np.hstack((x, y))
    sstat = 0
    for _ in range(permutations):

        xy_ = np.random.permutation(xy)
        x_ = xy_[:n1]
        y_ = xy_[n1:]
        s, p = stats.kruskal(x_, y_)
        if s > s_:
           sstat += 1
    pval = (sstat + 1) / (permutations + 1)
    return s_, pval


def pw_perm_kruskal_test(data, cats, permutations=1000):
    """ Computes pairwise kruskal test

    Parameters
    ----------
    data : pd.Series
  Datapoints (same order as in cat)
    cats : pd.Series
        Categories denoting class membership for datapoints
    permutations : int
        Number of permutations

    Returns
    -------
    pd.DataFrame
        Pairwise comparisions
    """
    c = cats.unique()
    res = []
    for i in range(len(c)):
        for j in range(i):
            ci, cj = c[i], c[j]
            idxi = cats==ci
            idxj = cats==cj
            s, p = perm_kruskal_test(data[idxi], data[idxj], permutations)
            res.append((ci, cj, s, p))

    res = pd.DataFrame(res)
    res.columns = ['Group1', 'Group2', 'Statistic', 'Pval']
    return res


