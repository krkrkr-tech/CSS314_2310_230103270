"""Serial hardware probe and numerical baseline for Task 1."""

from __future__ import annotations

import math
import os
import platform
import time

import numpy as np


STUDENT_ID = 230103270
ELEMENT_COUNT = 20_000_000


def serial_sum(values: np.ndarray) -> float:
    """Compute the sum with one explicit serial pass."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def serial_standard_deviation(values: np.ndarray, mean: float) -> float:
    """Compute population standard deviation with one explicit serial pass."""
    squared_deviation_sum = 0.0
    for value in values:
        difference = float(value) - mean
        squared_deviation_sum += difference * difference
    return math.sqrt(squared_deviation_sum / values.size)


def main() -> None:
    logical_cores = os.cpu_count() or 1
    platform_details = platform.platform()
    cpu_model = platform.processor() or platform.machine()

    start = time.perf_counter()
    rng = np.random.default_rng(STUDENT_ID)
    values = rng.random(ELEMENT_COUNT, dtype=np.float64)
    total = serial_sum(values)
    mean = total / values.size
    standard_deviation = serial_standard_deviation(values, mean)
    elapsed_ms = (time.perf_counter() - start) * 1_000

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform_details}")
    print(f"CPU model/platform: {cpu_model}")
    print(f"Total logical cores: {logical_cores}")
    print(f"Array elements: {values.size}")
    print(f"Mathematical sum: {total:.15f}")
    print(f"Standard deviation: {standard_deviation:.15f}")
    print(f"Serial wall-clock execution time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()