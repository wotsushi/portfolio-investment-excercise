from typing import Dict, Callable
from math import sqrt, exp


def quad_util(a: float) -> Callable[[float], float]:
    """
    二次関数的な効用関数を返します。

    Parameters
    ----------
    a: float
        実係数

    Returns
    -------
    Callable[[float], float]
        以下を満たす効用関数u
        - x <= a/2 について、 u(x) = ax^2 - x^2
        - x > a/2 について、 u(x) = a^2 / 4
    """

    def u(x: float) -> float:
        if x <= a / 2:
            return a * x - x ** 2
        else:
            return a ** 2 / 4

    return u


def exp_util(c: float) -> Callable[[float], float]:
    """
    指数関数的な効用関数を返します。

    Parameters
    ----------
    c: float
        実数

    Returns
    -------
    Callable[[float], float]
        x >= 0 について、u(x) = 1 - exp(-cx) を満たす効用関数u
    """

    def u(x: float) -> float:
        return 1 - exp(-c * x)

    return u


def pow_util(gamma: float) -> Callable[[float], float]:
    """
    べき乗関数的な効用関数を返します。

    Parameters
    ----------
    gamma: float
        実指数

    Returns
    -------
    Callable[[float], float]
        x >= 0 について、u(x) = x^{1 - gamma} / (1 - gamma) を満たす効用関数u
    """

    def u(x: float) -> float:
        return x ** (1 - gamma) / (1 - gamma)

    return u


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


def certainty_equivalent(u: Callable[[float], float], X: Dict[int, float]) -> float:
    """
    効用関数uの下で確率くじXの確実等価額を返します。

    Parameters
    ----------
    u: Callable[[float], float]
        上界が10^100, かつ、単調増加な効用関数
    X: Dict[int, float]
        確率くじ。X[k] = p とすると確率くじXは確率pでk円になることを表す。
        Xのkeyをk_1, ..., k_n とするとき、X[k_1] + ... + X[k_n] = 1を満たす必要がある。

    Returns
    -------
    float
        uの下でのXの確実等価額
    """

    E_u = expected_utility(u, X)
    ok = 1e100
    ng = 0.0
    while ok - ng > 1e-9:
        mid = (ok + ng) / 2
        if u(mid) >= E_u:
            ok = mid
        else:
            ng = mid
    return ok


def risk_discount(u: Callable[[float], float], X: Dict[int, float]) -> float:
    """
    効用関数uの下で確率くじXのリスク・ディスカウント額を返します。

    Parameters
    ----------
    u: Callable[[float], float]
        上界が10^100, かつ、単調増加な効用関数
    X: Dict[int, float]
        確率くじ。X[k] = p とすると確率くじXは確率pでk円になることを表す。
        Xのkeyをk_1, ..., k_n とするとき、X[k_1] + ... + X[k_n] = 1を満たす必要がある。

    Returns
    -------
    float
        uの下でのXのリスク・ディスカウント額
    """

    return expectation(X) - certainty_equivalent(u, X)


def risk_aversion(u: Callable[[float], float], x: float) -> float:
    """
    資産額xにおける効用関数uのリスク回避度を返します。

    Parameters
    ----------
    u: Callable[[float], float]
        単調増加、かつ、限界効用逓減を満たす効用関数
    x: float
        資産額

    Returns
    -------
    float
        xにおけるuのリスク回避度
    """

    h = 1e-3
    du = (u(x + h) - u(x - h)) / (2 * h)
    ddu = (u(x + 2 * h) - 2 * u(x) + u(x - 2 * h)) / (4 * h ** 2)
    return -ddu / du
