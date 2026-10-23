"""Benchmark serial and chunked threaded inclusive prefix scans."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103270
ARRAY_SIZE = 2_000_000
CHUNK_COUNT = 4


def serial_inclusive_scan(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.int64)
    running_total = 0
    for index, value in enumerate(values):
        running_total += int(value)
        result[index] = running_total
    return result


def _chunk_ranges(size: int, chunk_count: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, size, chunk_count + 1, dtype=np.int64)
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(chunk_count)
        if boundaries[index] < boundaries[index + 1]
    ]


def _local_scan(
    values: np.ndarray,
    chunk_range: tuple[int, int],
) -> tuple[np.ndarray, int]:
    start, end = chunk_range
    local_output = np.cumsum(values[start:end], dtype=np.int64)
    return local_output, int(local_output[-1])


def _apply_offset(
    output: np.ndarray,
    chunk_range: tuple[int, int],
    offset: int,
) -> None:
    start, end = chunk_range
    if offset:
        output[start:end] += offset


def parallel_chunked_scan(
    values: np.ndarray,
    executor: ThreadPoolExecutor,
) -> np.ndarray:
    ranges = _chunk_ranges(len(values), CHUNK_COUNT)

    local_results = list(executor.map(lambda item: _local_scan(values, item), ranges))
    local_outputs = [result[0] for result in local_results]
    chunk_sums = [result[1] for result in local_results]

    offsets: list[int] = []
    running_offset = 0
    for chunk_sum in chunk_sums:
        offsets.append(running_offset)
        running_offset += chunk_sum

    output = np.empty_like(values, dtype=np.int64)
    for (start, end), local_output in zip(ranges, local_outputs):
        output[start:end] = local_output
    list(
        executor.map(
            lambda item: _apply_offset(output, item[0], item[1]),
            zip(ranges, offsets),
        )
    )
    return output


def main() -> None:
    rng = np.random.default_rng(STUDENT_ID)
    values = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    serial_start = time.perf_counter()
    serial_output = serial_inclusive_scan(values)
    serial_time_ms = (time.perf_counter() - serial_start) * 1_000

    worker_count = min(CHUNK_COUNT, os.cpu_count() or 1)
    parallel_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        parallel_output = parallel_chunked_scan(values, executor)
    parallel_time_ms = (time.perf_counter() - parallel_start) * 1_000

    try:
        assert parallel_output[-1] == serial_output[-1]
    except AssertionError:
        print(
            "Verification Status: FAIL "
            f"(serial={serial_output[-1]}, parallel={parallel_output[-1]})"
        )
        raise SystemExit(1)

    speedup = serial_time_ms / parallel_time_ms
    print(f"Serial Time (ms): {serial_time_ms:.3f}")
    print(f"Parallel Time (ms): {parallel_time_ms:.3f}")
    print(f"Speedup: {speedup:.3f}x")
    print("Verification Status: PASS")


if __name__ == "__main__":
    main()
