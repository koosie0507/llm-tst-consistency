from dataclasses import dataclass, field
from sys import maxsize
from typing import Callable, Any, Iterable


@dataclass(unsafe_hash=True)
class Stats:
    """Stats container."""

    mean: float = field(init=True, hash=True, compare=True)
    """average over a container of numbers"""
    low: float = field(init=True, hash=True, compare=True)
    """minimum value from a container of numbers"""
    high: float = field(init=True, hash=True, compare=True)
    """maximum value from a container of numbers"""
    variance: float = field(init=True, hash=True, compare=True)
    """variance computed over a container of numbers"""

    @classmethod
    def empty(cls) -> "Stats":
        """Return an object that has all stats initialized to 0."""
        return Stats(0, 0, 0, 0)

    @classmethod
    def hlf(cls, docs: list, hlf: Callable[[Any], float]) -> "Stats":
        computer = Welford(hlf)
        return computer(docs)


class Welford:
    def __init__(self, number_extractor: Callable[[Any], float]) -> None:
        self._get_x = number_extractor

    @classmethod
    def _welford_pass(
        cls, result: Stats, x: float, n: int, m2: float
    ) -> tuple[Stats, int, float]:
        """Implement a single pass of the Welford algorithm.

        More details on the algorithm can be found here:
        * https://natural-blogarithm.com/post/variance-welford-vs-numpy/
        * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        result.high = max(result.high, x)
        result.low = min(result.low, x)

        n += 1
        delta = x - result.mean
        result.mean += float(delta) / n
        delta2 = x - result.mean
        m2 += delta * delta2

        return result, n, m2

    @classmethod
    def _compute_final_numeric_result(
        cls, result: Stats, n: int, m2: float
    ) -> Stats:
        if n < 1:
            return Stats.empty()
        if n > 1:
            result.variance = m2 / float(n - 1)
        result.low = round(result.low, 2)
        result.high = round(result.high, 2)
        result.mean = round(result.mean, 2)
        result.variance = round(result.variance, 2)
        return result

    def __call__(self, items: Iterable) -> Stats:
        if not items or not any(item for item in items):
            return Stats.empty()
        result = Stats(0, maxsize, -1, 0)
        count = 0
        m2: float = 0
        for i, item in enumerate(items):
            result, count, m2 = self._welford_pass(
                result, self._get_x(item), count, m2
            )
        return self._compute_final_numeric_result(result, count, m2)
