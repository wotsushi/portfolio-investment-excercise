from typing import Dict, Callable, List
from src import (
    quad_util,
    exp_util,
    pow_util,
    expectation,
    sd,
    expected_utility,
    certainty_equivalent,
    risk_discount,
    risk_aversion,
    rate_return,
    portfolio_return,
)

import pytest
from pytest import approx


@pytest.mark.parametrize(
    "a,x,expected",
    [(300, 0, 0), (300, 149, 22499), (300, 150, 22500), (300, 151, 22500)],
)
def test_quad_util(a: float, x: float, expected: float):
    u = quad_util(a)
    actual = u(x)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "c,x,expected",
    [
        (1, 0, 0),
        (1, 1, 0.6321205588285577),
        (1, 2, 0.8646647167633873),
        (2, 1, 0.8646647167633873),
    ],
)
def test_exp_util(c: float, x: float, expected: float):
    u = exp_util(c)
    actual = u(x)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "gamma,x,expected", [(0.5, 0, 0), (0.5, 0.25, 1), (0.5, 1, 2), (0.75, 1, 4)],
)
def test_pow_util(gamma: float, x: float, expected: float):
    u = pow_util(gamma)
    actual = u(x)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "X,expected",
    [
        ({100: 0.5, 50: 0.5}, 75),
        ({120: 0.5, 40: 0.5}, 80),
        ({40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 30),
    ],
)
def test_expectation(X: Dict[int, float], expected: float):
    actual = expectation(X)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "X,expected",
    [
        ({100: 0.5, 50: 0.5}, 25),
        ({120: 0.5, 40: 0.5}, 40),
        ({40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 10),
    ],
)
def test_sd(X: Dict[int, float], expected: float):
    actual = sd(X)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "u,X,expected",
    [
        (quad_util(300), {100: 0.5, 50: 0.5}, 16250),
        (quad_util(300), {120: 0.5, 40: 0.5}, 16000),
        (quad_util(500), {100: 0.5, 50: 0.5}, 31250),
        (quad_util(500), {120: 0.5, 40: 0.5}, 32000),
        (quad_util(100), {40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 2000),
    ],
)
def test_expected_utility(
    u: Callable[[float], float], X: Dict[int, float], expected: float
):
    actual = expected_utility(u, X)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "u,X,expected",
    [
        (quad_util(300), {100: 0.5, 50: 0.5}, 70.94305849579051),
        (quad_util(300), {120: 0.5, 40: 0.5}, 69.37742251701451),
        (quad_util(500), {100: 0.5, 50: 0.5}, 73.22330470336311),
        (quad_util(500), {120: 0.5, 40: 0.5}, 75.35750803427021),
        (quad_util(100), {40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 27.639320225002102),
    ],
)
def test_certainty_equivalent(
    u: Callable[[float], float], X: Dict[int, float], expected: float
):
    actual = certainty_equivalent(u, X)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "u,X,expected",
    [
        (quad_util(300), {100: 0.5, 50: 0.5}, 4.05694150420949),
        (quad_util(300), {120: 0.5, 40: 0.5}, 10.622577482985491),
        (quad_util(500), {100: 0.5, 50: 0.5}, 1.7766952966368876),
        (quad_util(500), {120: 0.5, 40: 0.5}, 4.642491965729789),
        (quad_util(100), {40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 2.360679774997898),
    ],
)
def test_risk_discount(
    u: Callable[[float], float], X: Dict[int, float], expected: float
):
    actual = risk_discount(u, X)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "u,x,expected",
    [
        (quad_util(300), 0, 0.006666666666666667),
        (quad_util(300), 50, 0.01),
        (quad_util(300), 149, 1),
        (quad_util(500), 0, 0.004),
        (quad_util(500), 50, 0.005),
        (quad_util(500), 249, 1),
    ],
)
def test_risk_aversion(u: Callable[[float], float], x: float, expected: float):
    actual = risk_aversion(u, x)
    assert actual == approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "X0,X1,expected", [(100, 120, 0.2), (100, 90, -0.1), (3.14, 6.28, 1)]
)
def test_rate_return(X0: float, X1: float, expected: float):
    actual = rate_return(X0, X1)
    assert actual == approx(expected)


@pytest.mark.parametrize(
    "w,mu,expected",
    [([0.12, 0.06], [0.4, 0.6], 0.084), ([0.12, 0.06, 0.03], [1, 0, 0], 0.12)],
)
def test_portfolio_return(w: List[float], mu: List[float], expected: float):
    actual = portfolio_return(w, mu)
    assert actual == approx(expected)
