
from __future__ import annotations

import threading
import time

import numpy as np


STUDENT_ID = 230103270
SAMPLE_COUNT = 40_000_000
THREAD_COUNT = 4
BATCH_SIZE = 250_000


def _count_points(
    sample_count: int,
    generator: np.random.Generator,
    generator_lock: threading.Lock | None = None,
) -> int:
    inside = 0
    remaining = sample_count
    while remaining:
        batch_count = min(BATCH_SIZE, remaining)
        if generator_lock is None:
            points = generator.random((batch_count, 2))
        else:
            with generator_lock:
                points = generator.random((batch_count, 2))
        inside += int(np.count_nonzero(np.sum(points * points, axis=1) <= 1.0))
        remaining -= batch_count
    return inside


def _run_shared_generator() -> tuple[float, float]:
    generator = np.random.default_rng(STUDENT_ID)
    generator_lock = threading.Lock()
    counts = [0] * THREAD_COUNT
    samples_per_thread = SAMPLE_COUNT // THREAD_COUNT

    def worker(thread_index: int) -> None:
        counts[thread_index] = _count_points(
            samples_per_thread,
            generator,
            generator_lock,
        )

    threads = [
        threading.Thread(target=worker, args=(thread_index,))
        for thread_index in range(THREAD_COUNT)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    estimate = 4.0 * sum(counts) / SAMPLE_COUNT
    return estimate, elapsed_ms


def _run_independent_generators() -> tuple[float, float]:
    counts = [0] * THREAD_COUNT
    samples_per_thread = SAMPLE_COUNT // THREAD_COUNT

    def worker(thread_index: int) -> None:
        generator = np.random.default_rng((STUDENT_ID * 10) + thread_index)
        counts[thread_index] = _count_points(samples_per_thread, generator)

    threads = [
        threading.Thread(target=worker, args=(thread_index,))
        for thread_index in range(THREAD_COUNT)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    estimate = 4.0 * sum(counts) / SAMPLE_COUNT
    return estimate, elapsed_ms


def main() -> None:
    shared_pi, shared_time_ms = _run_shared_generator()
    independent_pi, independent_time_ms = _run_independent_generators()
    shared_error = abs(shared_pi - np.pi)
    independent_error = abs(independent_pi - np.pi)
    speedup = shared_time_ms / independent_time_ms

    print(f"Student ID: {STUDENT_ID}")
    print(f"Sample points: {SAMPLE_COUNT:,}")
    print(f"Threads: {THREAD_COUNT}")
    print(
        "Shared PRNG: "
        f"Computed pi = {shared_pi:.10f}, "
        f"absolute error = {shared_error:.10f}, "
        f"wall time = {shared_time_ms:.3f} ms"
    )
    print(
        "Independent PRNG: "
        f"Computed pi = {independent_pi:.10f}, "
        f"absolute error = {independent_error:.10f}, "
        f"wall time = {independent_time_ms:.3f} ms"
    )
    print(f"Speedup from eliminating generator state locking: {speedup:.3f}x")


if __name__ == "__main__":
    main()
