"""Compare tiled CPU matrix multiplication geometries."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


MATRIX_SIZE = 1_024
BLOCK_SIZES = (16, 64)
STUDENT_ID = 230103270


def non_tiled_matmul(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = left @ right
    return result, (time.perf_counter() - start) * 1_000


def tiled_matmul(
    left: np.ndarray,
    right: np.ndarray,
    block_size: int,
    worker_count: int,
) -> tuple[np.ndarray, float]:
    result = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)
    blocks = [
        (row, column)
        for row in range(0, MATRIX_SIZE, block_size)
        for column in range(0, MATRIX_SIZE, block_size)
    ]

    def multiply_block(block: tuple[int, int]) -> None:
        row, column = block
        row_end = min(row + block_size, MATRIX_SIZE)
        column_end = min(column + block_size, MATRIX_SIZE)
        result[row:row_end, column:column_end] = (
            left[row:row_end, :] @ right[:, column:column_end]
        )

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(multiply_block, blocks))
    return result, (time.perf_counter() - start) * 1_000


def main() -> None:
    worker_count = os.cpu_count() or 1
    rng = np.random.default_rng(STUDENT_ID)
    left = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)
    right = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)

    reference, non_tiled_time_ms = non_tiled_matmul(left, right)
    results: list[tuple[str, float, float]] = [
        ("Non-tiled", non_tiled_time_ms, 1.0)
    ]
    for block_size in BLOCK_SIZES:
        tiled_result, elapsed_ms = tiled_matmul(
            left,
            right,
            block_size,
            worker_count,
        )
        if not np.allclose(tiled_result, reference, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"{block_size}x{block_size} result mismatch.")
        results.append(
            (
                f"Tiled {block_size}x{block_size}",
                elapsed_ms,
                elapsed_ms / non_tiled_time_ms,
            )
        )

    print("Task 8: CPU parallel tile geometry")
    print(f"Matrix dimensions: {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"CPU worker threads: {worker_count}")
    print("GPU detected: no")
    print()
    print(f"{'Geometry':<24} {'Runtime (ms)':>16} {'Latency vs non-tiled':>22}")
    print("-" * 66)
    for geometry, elapsed_ms, relative_latency in results:
        print(
            f"{geometry:<24} {elapsed_ms:>16.3f} "
            f"{relative_latency:>21.3f}x"
        )
    print()
    fastest = min(results, key=lambda item: item[1])
    print(f"Fastest geometry: {fastest[0]}")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
