from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np


ELEMENT_COUNT = 30_000_000
WORKER_COUNTS = (1, 2)
STUDENT_ID = 230103270


def memory_kernel(source: np.ndarray, destination: np.ndarray) -> None:
    np.copyto(destination, source)


def compute_kernel(source: np.ndarray, destination: np.ndarray) -> None:
    np.sin(source, out=destination)
    destination *= np.cos(source)
    destination += np.tan(source)


def _ranges(size: int, worker_count: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, size, worker_count + 1, dtype=np.int64)
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(worker_count)
    ]


def run_kernel(
    source: np.ndarray,
    destination: np.ndarray,
    worker_count: int,
    kernel: Callable[[np.ndarray, np.ndarray], None],
) -> float:
    ranges = _ranges(source.size, worker_count)
    start = time.perf_counter()
    if worker_count == 1:
        kernel(source, destination)
    else:
        def run_range(item_range: tuple[int, int]) -> None:
            begin, end = item_range
            kernel(source[begin:end], destination[begin:end])

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(run_range, ranges))
    return (time.perf_counter() - start) * 1_000


def benchmark_kernel(
    source: np.ndarray,
    worker_count: int,
    kernel: Callable[[np.ndarray, np.ndarray], None],
) -> float:
    destination = np.empty_like(source)
    kernel(source, destination)
    destination.fill(0.0)
    return run_kernel(source, destination, worker_count, kernel)


def verify_kernels(source: np.ndarray) -> None:
    sample = source[:1024]
    expected_memory = sample.copy()
    expected_compute = np.sin(sample) * np.cos(sample) + np.tan(sample)
    actual_memory = np.empty_like(sample)
    actual_compute = np.empty_like(sample)
    memory_kernel(sample, actual_memory)
    compute_kernel(sample, actual_compute)
    if not np.array_equal(actual_memory, expected_memory):
        raise RuntimeError("Memory kernel verification failed.")
    if not np.allclose(actual_compute, expected_compute):
        raise RuntimeError("Compute kernel verification failed.")


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = WORKER_COUNTS + (max_cores,)
    source = np.random.default_rng(STUDENT_ID).uniform(
        -1.0,
        1.0,
        size=ELEMENT_COUNT,
    )
    verify_kernels(source)

    timings: dict[str, dict[int, float]] = {
        "Memory-Bound (Copy)": {},
        "Compute-Bound (sin*cos+tan)": {},
    }
    kernels = {
        "Memory-Bound (Copy)": memory_kernel,
        "Compute-Bound (sin*cos+tan)": compute_kernel,
    }
    for label, kernel in kernels.items():
        for worker_count in worker_counts:
            timings[label][worker_count] = benchmark_kernel(
                source,
                worker_count,
                kernel,
            )

    print("Task 7: Arithmetic intensity versus memory bottlenecks")
    print(f"Array elements: {ELEMENT_COUNT:,}")
    print(f"Worker counts: {', '.join(str(count) for count in worker_counts)}")
    print()
    print(
        f"{'Kernel':<31} {'Cores':>7} {'Wall Time (ms)':>16} "
        f"{'Speedup':>11} {'Efficiency':>13}"
    )
    print("-" * 84)
    for label, kernel_timings in timings.items():
        serial_time = kernel_timings[1]
        for worker_count in worker_counts:
            speedup = serial_time / kernel_timings[worker_count]
            efficiency = speedup / worker_count
            print(
                f"{label:<31} {worker_count:>7} "
                f"{kernel_timings[worker_count]:>16.3f} "
                f"{speedup:>11.3f}x {efficiency:>12.3f}"
            )

    memory_efficiency = timings["Memory-Bound (Copy)"][1] / (
        timings["Memory-Bound (Copy)"][max_cores] * max_cores
    )
    compute_efficiency = timings["Compute-Bound (sin*cos+tan)"][1] / (
        timings["Compute-Bound (sin*cos+tan)"][max_cores] * max_cores
    )
    stalled_kernel = (
        "Memory-Bound (Copy)"
        if memory_efficiency < compute_efficiency
        else "Compute-Bound (sin*cos+tan)"
    )
    print()
    print(
        f"Scaling conclusion: {stalled_kernel} stalled more at max cores "
        "because its scaling efficiency was lower, indicating the limiting "
        "resource was saturated."
    )
    print(f"Memory kernel max-core efficiency: {memory_efficiency:.3f}")
    print(f"Compute kernel max-core efficiency: {compute_efficiency:.3f}")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
