from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ARRAY_SIZES = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
STUDENT_ID = 230103270
SCALAR_A = 2.5
SCALAR_B = -0.75
MEASUREMENTS = 3


def serial_vector_operation(x: np.ndarray) -> np.ndarray:
    return (SCALAR_A * x) + SCALAR_B


def _vector_chunk(x_chunk: np.ndarray) -> np.ndarray:
    return (SCALAR_A * x_chunk) + SCALAR_B


def parallel_vector_operation(
    x: np.ndarray,
    worker_count: int,) -> tuple[np.ndarray, float]:
    chunks = np.array_split(x, worker_count)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        output_chunks = list(executor.map(_vector_chunk, chunks))
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return np.concatenate(output_chunks), elapsed_ms


def timed_serial_operation(x: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    output = serial_vector_operation(x)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return output, elapsed_ms


def median_timings(
    x: np.ndarray,
    worker_count: int,) -> tuple[np.ndarray, float, float]:
    serial_vector_operation(x)
    parallel_vector_operation(x, worker_count)

    serial_times: list[float] = []
    parallel_times: list[float] = []
    serial_output = np.empty(0, dtype=np.float64)
    parallel_output = np.empty(0, dtype=np.float64)
    for _ in range(MEASUREMENTS):
        serial_output, serial_time_ms = timed_serial_operation(x)
        parallel_output, parallel_time_ms = parallel_vector_operation(
            x,
            worker_count,
        )
        serial_times.append(serial_time_ms)
        parallel_times.append(parallel_time_ms)

    return (
        serial_output,
        float(np.median(serial_times)),
        float(np.median(parallel_times)),
    )


def crossover_ranges(
    results: list[tuple[int, float, float]],) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for previous, current in zip(results, results[1:]):
        previous_parallel_faster = previous[2] < previous[1]
        current_parallel_faster = current[2] < current[1]
        if not previous_parallel_faster and current_parallel_faster:
            ranges.append((previous[0], current[0]))
    return ranges


def main() -> None:
    worker_count = os.cpu_count() or 1
    rng = np.random.default_rng(STUDENT_ID)
    results: list[tuple[int, float, float]] = []

    for size in ARRAY_SIZES:
        x = rng.random(size, dtype=np.float64)
        serial_output, serial_time_ms, parallel_time_ms = median_timings(
            x,
            worker_count,
        )
        parallel_output, _ = parallel_vector_operation(x, worker_count)
        if not np.array_equal(serial_output, parallel_output):
            raise RuntimeError(f"Vector result mismatch at N={size}.")
        results.append((size, serial_time_ms, parallel_time_ms))

    print("Task 5: Parallel overhead crossover point")
    print("Operation: y[i] = a * x[i] + b")
    print(f"All-core worker count: {worker_count}")
    print(f"Measurements per size: {MEASUREMENTS} (median after warm-up)")
    print()
    print(
        f"{'N':>12} | {'Serial (ms)':>13} | "
        f"{'Parallel (ms)':>15} | {'Parallel/Serial':>16} | Classification"
    )
    print("-" * 78)
    for size, serial_time_ms, parallel_time_ms in results:
        ratio = parallel_time_ms / serial_time_ms
        classification = (
            "FASTER" if parallel_time_ms < serial_time_ms else "SLOWER"
        )
        print(
            f"{size:>12,} | {serial_time_ms:>13.3f} | "
            f"{parallel_time_ms:>15.3f} | {ratio:>16.3f}x | {classification}"
        )

    transitions = crossover_ranges(results)
    print()
    if transitions:
        print("Crossover range(s), slower-to-faster:")
        for lower, upper in transitions:
            print(f"  N > {lower:,} and N <= {upper:,}")
    else:
        print("Crossover range(s), slower-to-faster: none in tested sizes")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
