
from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np


STUDENT_ID = 230103270
MATRIX_SIZE = 8_000

_matrix: np.ndarray | None = None
_shared_memory: SharedMemory | None = None


def _physical_core_count() -> int:
    
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None

    return physical or (os.cpu_count() or 1)


def _initialize_worker(shared_memory_name: str, shape: tuple[int, int]) -> None:
    global _matrix, _shared_memory
    _shared_memory = SharedMemory(name=shared_memory_name)
    _matrix = np.ndarray(shape, dtype=np.float64, buffer=_shared_memory.buf)


def _sum_rows(row_range: tuple[int, int]) -> float:
    if _matrix is None:
        raise RuntimeError("Worker matrix is not initialized.")
    start, end = row_range
    total = 0.0
    for row in range(start, end):
        total += float(np.sum(_matrix[row, :], dtype=np.float64))
    return total


def _sum_columns(column_range: tuple[int, int]) -> float:
    if _matrix is None:
        raise RuntimeError("Worker matrix is not initialized.")
    start, end = column_range
    total = 0.0
    for column in range(start, end):
        total += float(np.sum(_matrix[:, column], dtype=np.float64))
    return total


def _ranges(size: int, workers: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, size, workers + 1, dtype=np.int64)
    return [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(workers)
        if boundaries[index] < boundaries[index + 1]
    ]


def _run_mode(
    pool: mp.pool.Pool,
    worker,
    ranges: list[tuple[int, int]],
) -> tuple[float, float]:
    start = time.perf_counter()
    partial_sums = pool.map(worker, ranges)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return float(np.sum(partial_sums, dtype=np.float64)), elapsed_ms


def main() -> None:
    logical_cores = os.cpu_count() or 1
    physical_cores = _physical_core_count()
    seed = STUDENT_ID
    rng = np.random.default_rng(seed)

    matrix = np.asarray(
        rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64),
        dtype=np.float64,
        order="C",
    )
    shared_memory = SharedMemory(create=True, size=matrix.nbytes)
    shared_matrix = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shared_memory.buf)
    shared_matrix[:] = matrix
    del matrix

    try:
        row_ranges = _ranges(MATRIX_SIZE, physical_cores)
        column_ranges = _ranges(MATRIX_SIZE, physical_cores)
        context = mp.get_context("spawn")
        with context.Pool(
            processes=physical_cores,
            initializer=_initialize_worker,
            initargs=(shared_memory.name, (MATRIX_SIZE, MATRIX_SIZE)),
        ) as pool:
            row_sum, row_time_ms = _run_mode(pool, _sum_rows, row_ranges)
            column_sum, column_time_ms = _run_mode(pool, _sum_columns, column_ranges)

        if not np.isclose(row_sum, column_sum, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Row-major and column-major sums do not match.")
    finally:
        shared_memory.close()
        shared_memory.unlink()

    penalty_factor = column_time_ms / row_time_ms
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical core count: {logical_cores}")
    print(f"Physical core count used: {physical_cores}")
    print(f"Mode A (Row-Major) runtime: {row_time_ms:.3f} ms")
    print(f"Mode B (Column-Major / Strided) runtime: {column_time_ms:.3f} ms")
    print(f"Cache penalty factor (Mode B / Mode A): {penalty_factor:.3f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()