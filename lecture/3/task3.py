from __future__ import annotations

import os
import threading
import time


INCREMENT_COUNT = 50_000_000


def physical_core_count() -> int:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    return physical or (os.cpu_count() or 1)


def run_unpadded(worker_count: int) -> tuple[list[int], float]:
    shared_counts = [0] * worker_count

    def worker(thread_index: int) -> None:
        for _ in range(INCREMENT_COUNT):
            shared_counts[thread_index] += 1

    threads = [
        threading.Thread(target=worker, args=(thread_index,))
        for thread_index in range(worker_count)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return shared_counts, elapsed_ms


def run_local(worker_count: int) -> tuple[list[int], float]:
    shared_counts = [0] * worker_count

    def worker(thread_index: int) -> None:
        local_count = 0
        for _ in range(INCREMENT_COUNT):
            local_count += 1
        shared_counts[thread_index] = local_count

    threads = [
        threading.Thread(target=worker, args=(thread_index,))
        for thread_index in range(worker_count)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return shared_counts, elapsed_ms


def main() -> None:
    worker_count = physical_core_count()
    expected_count = INCREMENT_COUNT

    unpadded_counts, unpadded_time_ms = run_unpadded(worker_count)
    local_counts, local_time_ms = run_local(worker_count)

    if unpadded_counts != [expected_count] * worker_count:
        raise RuntimeError("Unpadded counters failed verification.")
    if local_counts != [expected_count] * worker_count:
        raise RuntimeError("Local counters failed verification.")

    execution_ratio = unpadded_time_ms / local_time_ms
    print(f"Physical worker threads: {worker_count}")
    print(f"Increments per thread: {INCREMENT_COUNT:,}")
    print(f"Expected count per slot: {expected_count:,}")
    print(f"Unpadded Shared wall time: {unpadded_time_ms:.3f} ms")
    print(f"Padded/Local wall time: {local_time_ms:.3f} ms")
    print(f"Execution ratio (Unpadded / Padded-Local): {execution_ratio:.3f}x")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
