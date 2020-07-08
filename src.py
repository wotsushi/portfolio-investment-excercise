from typing import Dict, Callable
from math import sqrt


def expectation(X: Dict[int, float]) -> float:
    """
    確率くじXの期待値を返します。

    Parameters
    ----------
    X: Dict[int, float]
        確率くじ。X[k] = p とすると確率くじXは確率pでk円になることを表す。
        Xのkeyをk_1, ..., k_n とするとき、X[k_1] + ... + X[k_n] = 1を満たす必要がある。

    Returns
    -------
    float
        Xの期待値
    """

    return sum(p * k for k, p in X.items())


def sd(X: Dict[int, float]) -> float:
    """
    確率くじXの標準偏差を返します。

    Parameters
    ----------
    X: Dict[int, float]
        確率くじ。X[k] = p とすると確率くじXは確率pでk円になることを表す。
        Xのkeyをk_1, ..., k_n とするとき、X[k_1] + ... + X[k_n] = 1を満たす必要がある。

    Returns
    -------
    float
        Xの標準偏差
    """

    E_X = expectation(X)
    return sqrt(sum(p * (k - E_X) ** 2 for k, p in X.items()))


def expected_utility(u: Callable[[float], float], X: Dict[int, float]) -> float:
    """
    効用関数uの下で確率くじXのもたらす期待効用を返します。

    Parameters
    ----------
    u: Callable[[float], float]
        効用関数
    X: Dict[int, float]
        確率くじ。X[k] = p とすると確率くじXは確率pでk円になることを表す。
        Xのkeyをk_1, ..., k_n とするとき、X[k_1] + ... + X[k_n] = 1を満たす必要がある。

    Returns
    -------
    float
        u(X)の期待値
    """

    return sum(p * u(k) for k, p in X.items())
