"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input as is."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y. Return 1.0 if true, 0.0 otherwise."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y. Return 1.0 if true, 0.0 otherwise."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close(|x-y| < 1e-2) to y."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the relu of x."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Return the log of x."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Return the exponential of x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Return the reciprocal of x."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log(x) times a second arg d"""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of 1/x times a second arg d"""
    return -1.0 / (x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU(x) times a second arg d"""
    return d if x > 0 else 0.0


def sigmoid_back(x: float, d: float) -> float:
    """Computes the derivative of sigmoid(x) times a second arg d"""
    return d * sigmoid(x) * (1.0 - sigmoid(x))


def exp_back(x: float, d: float) -> float:
    """Computes the derivative of exp(x) times a second arg d"""
    return d * exp(x)


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    Args:
    ----
        fn: A function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns
        a new list.

    """

    def _map(xs: Iterable[float]) -> Iterable[float]:
        ret = [fn(x) for x in xs]
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith.

    Args:
    ----
        fn: A function from two values to one value.

    Returns:
    -------
        A function that takes two equally sized lists xs and ys, produces a
        new list by applying `fn(x, y)` to each pair of elements.

    """

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = [fn(x, y) for x, y in zip(xs, ys)]
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
        fn: Combine two values
        init: Initial value

    Returns:
    -------
        A function that takes a list and reduces it to a single value by
        applying `fn` to each element and the accumulator.

    """

    def _reduce(xs: Iterable[float]) -> float:
        acc = init
        for x in xs:
            acc = fn(acc, x)
        return acc

    return _reduce


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate each element of a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add each element of two lists together."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum all elements of a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Multiply all elements of a list."""
    return reduce(mul, 1.0)(xs)
