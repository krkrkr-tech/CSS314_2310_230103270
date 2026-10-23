"""Measure array transfer overhead for pipe serialization and shared memory."""

from __future__ import annotations

import multiprocessing as mp
import statistics
import time
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np


ARRAY_SIZES = (100_000, 1_000_000, 10_000_000)
REPETITIONS = 3
DTYPE = np.float64


def _pipe_worker(connection: Connection) -> None:
    while True:
        message = connection.recv()
        if message is None:
            return
        values = message
        connection.send(float(np.sum(values, dtype=np.float64)))


def _shared_memory_worker(connection: Connection) -> None:
    while True:
        message = connection.recv()
        if message is None:
            return
        name, size = message
        shared_memory = SharedMemory(name=name)
        try:
            values = np.ndarray(size, dtype=DTYPE, buffer=shared_memory.buf)
            connection.send(float(np.sum(values, dtype=np.float64)))
        finally:
            shared_memory.close()


def _wait_for_worker(process: mp.Process, connection: Connection) -> None:
    if not process.is_alive():
        raise RuntimeError(f"Worker exited before benchmark: exit code {process.exitcode}")
    connection.send(None)
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Worker did not stop within 10 seconds")


def _pipe_timings(values: np.ndarray) -> list[float]:
    parent, child = mp.Pipe()
    process = mp.Process(target=_pipe_worker, args=(child,))
    process.start()
    child.close()
    try:
        timings = []
        expected = float(np.sum(values, dtype=np.float64))
        for _ in range(REPETITIONS + 1):
            start = time.perf_counter()
            parent.send(values)
            received_sum = parent.recv()
            timings.append((time.perf_counter() - start) * 1_000)
            if not np.isclose(received_sum, expected):
                raise RuntimeError("Pipe worker checksum mismatch")
        return timings[1:]
    finally:
        _wait_for_worker(process, parent)
        parent.close()


def _shared_memory_timings(values: np.ndarray) -> list[float]:
    parent, child = mp.Pipe()
    process = mp.Process(target=_shared_memory_worker, args=(child,))
    process.start()
    child.close()
    try:
        timings = []
        expected = float(np.sum(values, dtype=np.float64))
        for _ in range(REPETITIONS + 1):
            start = time.perf_counter()
            shared_memory = SharedMemory(create=True, size=values.nbytes)
            try:
                shared_values = np.ndarray(values.shape, dtype=DTYPE, buffer=shared_memory.buf)
                shared_values[:] = values
                parent.send((shared_memory.name, values.shape))
                received_sum = parent.recv()
            finally:
                shared_memory.close()
                shared_memory.unlink()
            timings.append((time.perf_counter() - start) * 1_000)
            if not np.isclose(received_sum, expected):
                raise RuntimeError("Shared-memory worker checksum mismatch")
        return timings[1:]
    finally:
        _wait_for_worker(process, parent)
        parent.close()


def _benchmark_size(size: int) -> tuple[float, float]:
    values = np.arange(size, dtype=DTYPE)
    pipe_ms = statistics.median(_pipe_timings(values))
    shared_ms = statistics.median(_shared_memory_timings(values))
    return pipe_ms, shared_ms


def main() -> None:
    print("Task 4: CPU array transfer overhead")
    print("Timing excludes worker startup and includes checksum validation.")
    print("Path A includes Pipe/Pickle serialization; Path B includes shared-memory setup and copy.")
    print(f"Repetitions per path: {REPETITIONS} (median reported)")
    print()
    print(f"{'N':>12} {'Path A IPC (ms)':>18} {'Path B shared (ms)':>21} {'A/B ratio':>12}")
    print("-" * 67)
    for size in ARRAY_SIZES:
        pipe_ms, shared_ms = _benchmark_size(size)
        print(f"{size:12,d} {pipe_ms:18.3f} {shared_ms:21.3f} {pipe_ms / shared_ms:12.2f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()
