from typing import Dict
from src import expectation, sd

# from expectation import expectation
import pytest


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
    assert actual == expected


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
    assert actual == expected
