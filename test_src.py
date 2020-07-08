from typing import Dict, Callable
from src import expectation, sd, expected_utility

import pytest
from pytest import approx


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
        (lambda x: 300 * x - x ** 2, {100: 0.5, 50: 0.5}, 16250),
        (lambda x: 300 * x - x ** 2, {120: 0.5, 40: 0.5}, 16000),
        (lambda x: 100 * x - x ** 2, {40: 0.4, 30: 0.3, 20: 0.2, 10: 0.1}, 2000),
    ],
)
def test_expected_utility(
    u: Callable[[float], float], X: Dict[int, float], expected: float
):
    actual = expected_utility(u, X)
    assert actual == approx(expected)
