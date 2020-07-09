from typing import Dict, Callable
from src import (
    quad_util,
    expectation,
    sd,
    expected_utility,
    certainty_equivalent,
    risk_discount,
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
        (quad_util(100), {40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 2.360679774997898),
    ],
)
def test_risk_discount(
    u: Callable[[float], float], X: Dict[int, float], expected: float
):
    actual = risk_discount(u, X)
    assert actual == approx(expected)
