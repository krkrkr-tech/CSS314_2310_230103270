
from __future__ import annotations

import threading
import time


THREAD_COUNT = 4
INCREMENTS_PER_THREAD = 2_000_000
YIELD_INTERVAL = 1_000


def run_without_lock() -> tuple[int, float]:
    counter = [0]

    def worker() -> None:
        for iteration in range(INCREMENTS_PER_THREAD):
            current = counter[0]
            if iteration % YIELD_INTERVAL == 0:
                time.sleep(0)
            counter[0] = current + 1

    threads = [threading.Thread(target=worker) for _ in range(THREAD_COUNT)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter[0], (time.perf_counter() - start) * 1_000


def run_with_lock() -> tuple[int, float]:
    counter = [0]
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(INCREMENTS_PER_THREAD):
            with lock:
                counter[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(THREAD_COUNT)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counter[0], (time.perf_counter() - start) * 1_000


def main() -> None:
    expected = THREAD_COUNT * INCREMENTS_PER_THREAD
    unlocked_value, unlocked_time_ms = run_without_lock()
    locked_value, locked_time_ms = run_with_lock()
    lost_updates = expected - unlocked_value
    loss_percentage = 100.0 * lost_updates / expected
    time_penalty = locked_time_ms - unlocked_time_ms
    penalty_percentage = 100.0 * time_penalty / unlocked_time_ms

    if locked_value != expected:
        raise RuntimeError(
            f"Locked counter verification failed: expected {expected}, "
            f"got {locked_value}."
        )
    print("Task 6: Synchronization cost and data loss")
    print(f"Threads: {THREAD_COUNT}")
    print(f"Attempts per thread: {INCREMENTS_PER_THREAD:,}")
    print(f"Expected final value: {expected:,}")
    print(f"Actual final value without lock: {unlocked_value:,}")
    print(f"Updates lost: {lost_updates:,}")
    print(f"Percentage of updates lost: {loss_percentage:.3f}%")
    print(f"Unsynchronized wall-clock time: {unlocked_time_ms:.3f} ms")
    print(f"Locked final value: {locked_value:,}")
    print(f"Synchronized wall-clock time: {locked_time_ms:.3f} ms")
    print(f"Wall-clock time penalty from lock: {time_penalty:.3f} ms")
    print(f"Lock time penalty: {penalty_percentage:.3f}%")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
