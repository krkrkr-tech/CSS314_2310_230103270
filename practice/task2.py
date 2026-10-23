"""Measure multi-threaded summation scaling against Amdahl's Law."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103270
ELEMENT_COUNT = 20_000_000
PARALLEL_FRACTION = 0.95


def _sum_chunk(chunk: np.ndarray) -> float:
    """Sum one array chunk in a worker thread."""
    return float(np.sum(chunk, dtype=np.float64))


def parallel_sum(values: np.ndarray, worker_count: int) -> tuple[float, float]:
    """Return the sum and wall-clock time for the requested worker count."""
    chunks = np.array_split(values, worker_count)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        partial_sums = list(executor.map(_sum_chunk, chunks))
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return float(np.sum(partial_sums, dtype=np.float64)), elapsed_ms


def amdahl_upper_bound(worker_count: int) -> float:
    """Calculate Amdahl's theoretical maximum speedup for this worker count."""
    return 1.0 / ((1.0 - PARALLEL_FRACTION) + (PARALLEL_FRACTION / worker_count))


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = list(dict.fromkeys([1, 2, 4, max_cores, max_cores * 2]))

    values = np.random.default_rng(STUDENT_ID).random(
        ELEMENT_COUNT,
        dtype=np.float64,
    )
    serial_sum, serial_time_ms = parallel_sum(values, 1)

    results: list[tuple[int, float, float, float]] = [
        (1, serial_time_ms, 1.0, amdahl_upper_bound(1))
    ]
    for worker_count in worker_counts:
        if worker_count == 1:
            continue
        total, elapsed_ms = parallel_sum(values, worker_count)
        if not np.isclose(total, serial_sum, rtol=1e-12, atol=1e-9):
            raise RuntimeError(
                f"Worker count {worker_count} produced an incorrect sum."
            )
        results.append(
            (
                worker_count,
                elapsed_ms,
                serial_time_ms / elapsed_ms,
                amdahl_upper_bound(worker_count),
            )
        )

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array elements: {ELEMENT_COUNT:,}")
    print(f"Logical cores available: {max_cores}")
    print(f"Parallel fraction (P): {PARALLEL_FRACTION:.2f}")
    print(f"Serial reference sum: {serial_sum:.15f}")
    print()
    print(
        f"{'Cores':>8} | {'Wall Time (ms)':>15} | "
        f"{'Speedup':>10} | {'Amdahl Upper Bound':>19}"
    )
    print("-" * 62)
    for worker_count, elapsed_ms, speedup, upper_bound in results:
        print(
            f"{worker_count:>8} | {elapsed_ms:>15.3f} | "
            f"{speedup:>10.3f} | {upper_bound:>19.3f}"
        )


if __name__ == "__main__":
    main()
