from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple


class TransferPipeline:
    """Thread-safe capture transfer bookkeeping and wait helpers."""

    def __init__(
        self,
        *,
        max_inflight: Optional[int] = None,
        inflight_wait_timeout: Optional[float] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._saved_paths: List[str] = []
        self._downloaded_paths: Deque[str] = deque()
        self._download_errors: List[str] = []
        self._completed_transfers = 0
        self._finished_transfers = 0
        self._queued_transfers = 0
        self.max_inflight = max_inflight
        self.inflight_wait_timeout = inflight_wait_timeout

    def reset_paths_and_errors(self) -> None:
        with self._lock:
            self._saved_paths.clear()
            self._downloaded_paths.clear()
            self._download_errors.clear()

    def mark_queued(self) -> None:
        with self._lock:
            self._queued_transfers += 1

    def mark_downloaded(self, path: str) -> None:
        with self._lock:
            self._saved_paths.append(path)
            self._downloaded_paths.append(path)
            self._completed_transfers += 1
            self._finished_transfers += 1

    def mark_download_error(self, exc: Exception) -> None:
        with self._lock:
            self._download_errors.append(str(exc))
            self._finished_transfers += 1

    def capture_baseline(self) -> Tuple[int, int]:
        with self._lock:
            return self._completed_transfers, len(self._download_errors)

    def marker(self) -> int:
        with self._lock:
            return self._completed_transfers

    def wait_for_inflight_slot(
        self,
        pump_messages_once: Callable[[], None],
        timeout: Optional[float] = None,
    ) -> None:
        if self.max_inflight is None:
            return
        if timeout is None:
            timeout = self.inflight_wait_timeout
        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            with self._lock:
                inflight = self._queued_transfers - self._finished_transfers
                if inflight < self.max_inflight:
                    return
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for inflight slot (max_inflight={self.max_inflight})"
                )
            pump_messages_once()
            time.sleep(0.005)

    def wait_for_transfer(
        self,
        pump_messages_once: Callable[[], None],
        timeout: float,
        completed_before: int,
        errors_before: int,
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.01)
            pump_messages_once()
            with self._lock:
                if len(self._download_errors) > errors_before:
                    raise RuntimeError(f"Image download failed: {self._download_errors[-1]}")
                if self._completed_transfers > completed_before:
                    return
        raise TimeoutError("Timed out waiting for image transfer event")

    def wait_for_downloads(
        self,
        pump_messages_once: Callable[[], None],
        expected: int,
        timeout: float,
        *,
        marker: Optional[int] = None,
    ) -> List[str]:
        if expected <= 0:
            return []
        with self._lock:
            start_completed = self._completed_transfers if marker is None else marker
            start_errors = len(self._download_errors)
        target = start_completed + expected
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.01)
            pump_messages_once()
            with self._lock:
                if len(self._download_errors) > start_errors:
                    raise RuntimeError(f"Image download failed: {self._download_errors[-1]}")
                if self._completed_transfers >= target:
                    return self._drain_downloads_locked()
        raise TimeoutError("Timed out waiting for downloaded images")

    def snapshot_for_burst(self) -> Tuple[int, int, int]:
        with self._lock:
            return (
                self._completed_transfers,
                self._queued_transfers,
                len(self._download_errors),
            )

    def burst_progress(self) -> Tuple[int, int]:
        with self._lock:
            return self._queued_transfers, self._queued_transfers - self._finished_transfers

    def has_new_error(self, errors_before: int) -> Optional[str]:
        with self._lock:
            if len(self._download_errors) > errors_before:
                return self._download_errors[-1]
        return None

    def saved_paths_snapshot(self) -> List[str]:
        with self._lock:
            return list(self._saved_paths)

    def drain_downloads(self) -> List[str]:
        with self._lock:
            return self._drain_downloads_locked()

    def _drain_downloads_locked(self) -> List[str]:
        paths = list(self._downloaded_paths)
        self._downloaded_paths.clear()
        return paths
