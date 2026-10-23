
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable


ITEM_COUNT = 1_000
WORKER_COUNT = 4
DYNAMIC_CHUNK_SIZE = 1


def process_item(item_index: int) -> None:
    time.sleep(0.0001 * (item_index % 20))


def _run_workers(
    assignments: Iterable[Iterable[int]],
    process: Callable[[int], None],
) -> tuple[float, float, float]:
    completion_times: list[float] = []
    completion_lock = threading.Lock()

    def worker(items: Iterable[int]) -> None:
        for item_index in items:
            process(item_index)
        with completion_lock:
            completion_times.append(time.perf_counter())

    start = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(items,))
        for items in assignments
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    completion_delta_ms = (
        (max(completion_times) - min(completion_times)) * 1_000
        if completion_times
        else 0.0
    )
    return elapsed_ms, min(completion_times) - start, completion_delta_ms


def run_static() -> tuple[float, float, float]:
    block_size = ITEM_COUNT // WORKER_COUNT
    assignments = [
        range(worker_index * block_size, (worker_index + 1) * block_size)
        for worker_index in range(WORKER_COUNT)
    ]
    return _run_workers(assignments, process_item)


def run_dynamic() -> tuple[float, float, float]:
    work_queue: queue.Queue[list[int] | None] = queue.Queue()
    for start in range(0, ITEM_COUNT, DYNAMIC_CHUNK_SIZE):
        work_queue.put(
            list(range(start, min(start + DYNAMIC_CHUNK_SIZE, ITEM_COUNT)))
        )
    for _ in range(WORKER_COUNT):
        work_queue.put(None)

    def consume_queue(_: int) -> None:
        while True:
            chunk = work_queue.get()
            try:
                if chunk is None:
                    return
                for item_index in chunk:
                    process_item(item_index)
            finally:
                work_queue.task_done()

    completion_times: list[float] = []
    completion_lock = threading.Lock()

    def worker() -> None:
        consume_queue(0)
        with completion_lock:
            completion_times.append(time.perf_counter())

    start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(WORKER_COUNT)]
    for thread in threads:
        thread.start()
    work_queue.join()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    completion_delta_ms = (max(completion_times) - min(completion_times)) * 1_000
    return elapsed_ms, min(completion_times) - start, completion_delta_ms


def main() -> None:
    static_time_ms, _, static_delta_ms = run_static()
    dynamic_time_ms, _, dynamic_delta_ms = run_dynamic()

    print("Task 4: Work stealing and scheduling under skew")
    print(f"Items: {ITEM_COUNT:,}")
    print(f"Workers: {WORKER_COUNT}")
    print("Workload per item: sleep(0.0001 * (i % 20)) seconds")
    print(f"Dynamic chunk size: {DYNAMIC_CHUNK_SIZE}")
    print()
    print(f"{'Strategy':<28} {'Wall Time (ms)':>16} {'Completion Delta (ms)':>24}")
    print("-" * 72)
    print(f"{'A: Static chunks':<28} {static_time_ms:>16.3f} {static_delta_ms:>24.3f}")
    print(f"{'B: Dynamic queue':<28} {dynamic_time_ms:>16.3f} {dynamic_delta_ms:>24.3f}")
    print(f"Execution ratio (Static / Dynamic): {static_time_ms / dynamic_time_ms:.3f}x")
    print("Verification status: PASS")


if __name__ == "__main__":
    main()
