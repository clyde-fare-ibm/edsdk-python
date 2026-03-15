#!/usr/bin/env python3
"""Benchmark EDSDK capture paths across host vs camera storage.

Runs a scenario matrix including:
- blocking
- async
- burst (non-async) with HighSpeedContinuous and LowSpeedContinuous
- burst_async with HighSpeedContinuous and LowSpeedContinuous
for both host and camera SD save targets.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Ensure local edsdk package import works when launched from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
EDSDK_PYTHON_DIR = REPO_ROOT / "edsdk-python"
if str(EDSDK_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(EDSDK_PYTHON_DIR))

import edsdk  # noqa: E402
from edsdk.camera_controller import CameraController, RateControlledResult  # noqa: E402
from edsdk.constants.commands import CameraCommand, ShutterButton  # noqa: E402
from edsdk.constants.generic import ObjectEvent  # noqa: E402
from edsdk.constants.properties import (  # noqa: E402
    DriveMode,
    ImageQuality,
    PropID,
    SaveTo,
)


@dataclass
class ScenarioResult:
    name: str
    save_target: str
    frames: int
    capture_elapsed_s: Optional[float]
    end_to_end_elapsed_s: Optional[float]
    capture_fps: Optional[float]
    end_to_end_fps: Optional[float]
    ok: bool
    detail: str = ""
    # Rate-controlled extras (None for non-rate scenarios)
    target_fps: Optional[float] = None
    achieved_fps: Optional[float] = None
    mean_interval: Optional[float] = None
    jitter_std: Optional[float] = None
    mean_schedule_error: Optional[float] = None
    skipped: Optional[int] = None


@dataclass
class RunStats:
    frames: int
    capture_elapsed_s: float
    end_to_end_elapsed_s: float


def is_device_busy(exc: Exception) -> bool:
    msg = str(exc).upper()
    code = getattr(exc, "code", None)
    # 0x81 == EDS_ERR_DEVICE_BUSY / PTP device busy.
    return code == 0x00000081 or "DEVICE_BUSY" in msg


def run_with_busy_retry(
    fn: Callable[[], None],
    *,
    retries: int,
    base_delay_s: float,
    on_retry: Optional[Callable[[], None]] = None,
) -> None:
    attempt = 0
    while True:
        try:
            fn()
            return
        except Exception as exc:
            if not is_device_busy(exc) or attempt >= retries:
                raise
            attempt += 1
            if on_retry is not None:
                try:
                    on_retry()
                except Exception:
                    pass
            time.sleep(base_delay_s * attempt)


class ObjectEventCounter:
    """Thread-safe object event counter for waiting on camera events."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._created_count = 0

    def callback(self, event: ObjectEvent, _obj) -> int:
        if event == ObjectEvent.DirItemCreated:
            with self._cv:
                self._created_count += 1
                self._cv.notify_all()
        return 0

    def snapshot(self) -> int:
        with self._cv:
            return self._created_count

    def wait_for_delta(self, start: int, expected_delta: int, timeout_s: float) -> bool:
        target = start + expected_delta
        deadline = time.perf_counter() + timeout_s
        with self._cv:
            while self._created_count < target:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True


def wait_for_event_quiet(
    counter: ObjectEventCounter, quiet_s: float = 1.2, max_wait_s: float = 20.0
) -> int:
    """Wait until DirItemCreated count is stable for quiet_s."""
    deadline = time.perf_counter() + max_wait_s
    last_count = counter.snapshot()
    last_change = time.perf_counter()
    while True:
        time.sleep(0.03)
        now = time.perf_counter()
        current = counter.snapshot()
        if current != last_count:
            last_count = current
            last_change = now
        if now - last_change >= quiet_s:
            return current
        if now >= deadline:
            return current


def count_files_recursive(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for entry in p.rglob("*") if entry.is_file())


def wait_for_file_quiet(path: str, quiet_s: float, max_wait_s: float) -> int:
    """Wait until file count in path stays stable for quiet_s."""
    deadline = time.perf_counter() + max_wait_s
    last_count = count_files_recursive(path)
    last_change = time.perf_counter()
    while True:
        time.sleep(0.05)
        # Keep SDK events flowing while we wait for late transfers.
        try:
            if hasattr(edsdk, "GetEvent"):
                edsdk.GetEvent()
        except Exception:
            pass
        now = time.perf_counter()
        current = count_files_recursive(path)
        if current != last_count:
            last_count = current
            last_change = now
        if now - last_change >= quiet_s:
            return current
        if now >= deadline:
            return current


def set_save_target(controller: CameraController, save_to: SaveTo) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    edsdk.SetPropertyData(controller._cam, PropID.SaveTo, 0, int(save_to))
    controller.save_to = save_to


def set_image_quality_low_jpeg(controller: CameraController) -> str:
    """Set a low-quality JPEG mode with fallback candidates.

    Returns the selected ImageQuality enum member name.
    """
    # Prefer small/normal JPEG when available; fall back to larger normal JPEG.
    candidates = [
        ImageQuality.SJN,
        ImageQuality.S1JN,
        ImageQuality.MJN,
        ImageQuality.LJN,
    ]
    for quality in candidates:
        try:
            controller.set_properties(
                image_quality=int(quality),
                validate=False,
                tolerate_not_supported=True,
            )
            return quality.name
        except Exception:
            continue
    raise RuntimeError("Unable to set low-quality JPEG ImageQuality on this camera")


def restore_image_quality_cr3(controller: CameraController) -> str:
    """Best-effort restore to a CR3-capable raw format."""
    # Prefer cRAW first (commonly used as .CR3), then RAW.
    for quality in (ImageQuality.CR, ImageQuality.LR):
        try:
            controller.set_properties(
                image_quality=int(quality),
                validate=False,
                tolerate_not_supported=True,
            )
            return quality.name
        except Exception:
            continue
    raise RuntimeError("Unable to restore image quality to CR3-compatible raw mode")


def run_host_blocking(controller: CameraController, shots: int, timeout_s: float) -> None:
    paths = controller.capture(shots=shots, timeout=timeout_s)
    if len(paths) < shots:
        raise RuntimeError(f"Expected {shots} downloads, got {len(paths)}")


def run_host_async(controller: CameraController, shots: int, timeout_s: float) -> None:
    # Trigger asynchronously one shot at a time so DEVICE_BUSY can be retried
    # per trigger without discarding the whole scenario.
    first_ticket = None
    expected_total = 0
    for _ in range(shots):
        holder: List[dict] = []

        def _trigger_once() -> None:
            holder.clear()
            holder.append(controller.capture_async(shots=1))

        run_with_busy_retry(
            _trigger_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        ticket = holder[0]
        if first_ticket is None:
            first_ticket = ticket
        expected_total += int(ticket["expected"])

    if first_ticket is None:
        raise RuntimeError("No async capture ticket received")

    paths = controller.wait_for_downloads(
        expected=expected_total,
        marker=first_ticket["marker"],
        timeout=max(timeout_s, shots * 5.0),
    )
    if len(paths) < shots:
        raise RuntimeError(f"Expected {shots} downloads, got {len(paths)}")


def run_host_burst_async(
    controller: CameraController, shots: int, timeout_s: float, drive_mode: DriveMode
) -> None:
    ticket_holder: List[dict] = []

    def _burst_once() -> None:
        ticket_holder.clear()
        ticket_holder.append(
            controller.capture_burst_async(
                shots=shots,
                timeout=max(timeout_s, shots * 2.0),
                drive_mode=drive_mode,
                apply_drive_mode=True,
            )
        )

    run_with_busy_retry(
        _burst_once,
        retries=6,
        base_delay_s=0.08,
        on_retry=lambda: controller.wake_up(),
    )
    ticket = ticket_holder[0]
    paths = controller.wait_for_downloads(
        expected=ticket["expected"],
        marker=ticket["marker"],
        timeout=max(timeout_s, shots * 5.0),
    )
    if len(paths) < shots:
        raise RuntimeError(
            f"Expected at least {shots} downloads during burst, got {len(paths)}"
        )


def run_host_burst_blocking(
    controller: CameraController, shots: int, timeout_s: float, drive_mode: DriveMode
) -> None:
    paths_holder: List[List[str]] = []

    def _burst_once() -> None:
        paths_holder.clear()
        paths_holder.append(
            controller.capture_burst(
                shots=shots,
                timeout=max(timeout_s, shots * 2.0),
                download_timeout=max(timeout_s, shots * 5.0),
                drive_mode=drive_mode,
                apply_drive_mode=True,
            )
        )

    run_with_busy_retry(
        _burst_once,
        retries=6,
        base_delay_s=0.08,
        on_retry=lambda: controller.wake_up(),
    )
    paths = paths_holder[0]
    if len(paths) < shots:
        raise RuntimeError(f"Expected at least {shots} downloads during burst, got {len(paths)}")


def run_camera_blocking(
    controller: CameraController, counter: ObjectEventCounter, shots: int, timeout_s: float
) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    for _ in range(shots):
        start = counter.snapshot()

        def _shot_once() -> None:
            edsdk.SendCommand(controller._cam, CameraCommand.TakePicture, 0)

        run_with_busy_retry(
            _shot_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        if not counter.wait_for_delta(start=start, expected_delta=1, timeout_s=timeout_s):
            raise TimeoutError("Timed out waiting for DirItemCreated in blocking SD mode")


def run_camera_async(
    controller: CameraController, counter: ObjectEventCounter, shots: int, timeout_s: float
) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    start = counter.snapshot()
    for _ in range(shots):
        def _shot_once() -> None:
            edsdk.SendCommand(controller._cam, CameraCommand.TakePicture, 0)

        run_with_busy_retry(
            _shot_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
    if not counter.wait_for_delta(start=start, expected_delta=shots, timeout_s=timeout_s):
        raise TimeoutError("Timed out waiting for DirItemCreated in async SD mode")


def run_camera_burst_async(
    controller: CameraController,
    counter: ObjectEventCounter,
    shots: int,
    timeout_s: float,
    drive_mode: DriveMode,
) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    controller.set_properties(
        drive_mode=drive_mode,
        validate=False,
        tolerate_not_supported=True,
    )
    start = counter.snapshot()
    shutter_pressed = False
    try:
        run_with_busy_retry(
            lambda: edsdk.SendCommand(
                controller._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.Completely),
            ),
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        shutter_pressed = True
        if not counter.wait_for_delta(start=start, expected_delta=shots, timeout_s=timeout_s):
            raise TimeoutError("Timed out waiting for DirItemCreated in burst SD mode")
    finally:
        if shutter_pressed:
            edsdk.SendCommand(
                controller._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.OFF),
            )


def run_camera_burst_blocking(
    controller: CameraController,
    counter: ObjectEventCounter,
    shots: int,
    timeout_s: float,
    drive_mode: DriveMode,
) -> None:
    # There is no dedicated burst-blocking API for SaveTo.Camera without downloads,
    # so this uses the same shutter/event strategy and blocks until `shots` events.
    run_camera_burst_async(
        controller=controller,
        counter=counter,
        shots=shots,
        timeout_s=timeout_s,
        drive_mode=drive_mode,
    )


def run_rate_controlled(
    controller: CameraController,
    target_fps: float,
    duration: float,
) -> Tuple[RunStats, RateControlledResult]:
    """Run rate-controlled capture and return RunStats + detailed result."""
    result = controller.capture_rate_controlled(
        target_fps=target_fps,
        duration=duration,
    )
    stats = RunStats(
        frames=result.total_accepted,
        capture_elapsed_s=result.elapsed,
        end_to_end_elapsed_s=result.elapsed,
    )
    return stats, result


def run_rate_controlled_host(
    controller: CameraController,
    target_fps: float,
    duration: float,
    timeout_s: float,
) -> Tuple[RunStats, RateControlledResult]:
    """Rate-controlled to host -- triggers by clock, then waits for downloads."""
    result = controller.capture_rate_controlled(
        target_fps=target_fps,
        duration=duration,
    )
    if result.total_accepted > 0:
        controller.wait_for_downloads(
            expected=result.total_accepted,
            timeout=max(timeout_s, result.total_accepted * 5.0),
        )
    stats = RunStats(
        frames=result.total_accepted,
        capture_elapsed_s=result.elapsed,
        end_to_end_elapsed_s=result.elapsed,
    )
    return stats, result


def run_rate_controlled_camera(
    controller: CameraController,
    counter: ObjectEventCounter,
    target_fps: float,
    duration: float,
    timeout_s: float,
) -> Tuple[RunStats, RateControlledResult]:
    """Rate-controlled to camera SD -- triggers by clock, verifies via events."""
    start_count = counter.snapshot()
    result = controller.capture_rate_controlled(
        target_fps=target_fps,
        duration=duration,
    )
    if result.total_accepted > 0:
        counter.wait_for_delta(
            start=start_count,
            expected_delta=result.total_accepted,
            timeout_s=max(timeout_s, result.total_accepted * 3.0),
        )
    confirmed = counter.snapshot() - start_count
    stats = RunStats(
        frames=confirmed,
        capture_elapsed_s=result.elapsed,
        end_to_end_elapsed_s=result.elapsed,
    )
    return stats, result


def run_host_blocking_timed(
    controller: CameraController, run_seconds: float, timeout_s: float
) -> RunStats:
    t0 = time.perf_counter()
    deadline = t0 + run_seconds
    frames = 0
    while time.perf_counter() < deadline:
        holder: List[List[str]] = []

        def _shot_once() -> None:
            holder.clear()
            holder.append(controller.capture(shots=1, timeout=timeout_s))

        run_with_busy_retry(
            _shot_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        if holder and holder[0]:
            frames += len(holder[0])
    elapsed = time.perf_counter() - t0
    return RunStats(
        frames=frames,
        capture_elapsed_s=elapsed,
        end_to_end_elapsed_s=elapsed,
    )


def run_host_async_timed(
    controller: CameraController, run_seconds: float, timeout_s: float
) -> RunStats:
    t0 = time.perf_counter()
    deadline = t0 + run_seconds
    first_marker: Optional[int] = None
    expected_total = 0
    while time.perf_counter() < deadline:
        holder: List[dict] = []

        def _trigger_once() -> None:
            holder.clear()
            holder.append(controller.capture_async(shots=1))

        run_with_busy_retry(
            _trigger_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        if not holder:
            continue
        ticket = holder[0]
        if first_marker is None:
            first_marker = int(ticket["marker"])
        expected_total += int(ticket["expected"])
    capture_elapsed = time.perf_counter() - t0
    frames = 0
    if expected_total > 0 and first_marker is not None:
        paths = controller.wait_for_downloads(
            expected=expected_total,
            marker=first_marker,
            timeout=max(timeout_s, expected_total * 5.0),
        )
        frames = len(paths)
    end_to_end_elapsed = time.perf_counter() - t0
    return RunStats(
        frames=frames,
        capture_elapsed_s=capture_elapsed,
        end_to_end_elapsed_s=end_to_end_elapsed,
    )


def run_host_burst_async_timed(
    controller: CameraController,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> RunStats:
    t0 = time.perf_counter()
    ticket_holder: List[dict] = []

    def _burst_once() -> None:
        ticket_holder.clear()
        ticket_holder.append(
            controller.capture_burst_async(
                shots=1,
                duration=run_seconds,
                timeout=max(timeout_s, run_seconds + 2.0),
                drive_mode=drive_mode,
                apply_drive_mode=True,
            )
        )

    run_with_busy_retry(
        _burst_once,
        retries=6,
        base_delay_s=0.08,
        on_retry=lambda: controller.wake_up(),
    )
    ticket = ticket_holder[0]
    capture_elapsed = time.perf_counter() - t0
    paths = controller.wait_for_downloads(
        expected=int(ticket["expected"]),
        marker=int(ticket["marker"]),
        timeout=max(timeout_s, run_seconds + 10.0, int(ticket["expected"]) * 5.0),
    )
    end_to_end_elapsed = time.perf_counter() - t0
    return RunStats(
        frames=len(paths),
        capture_elapsed_s=capture_elapsed,
        end_to_end_elapsed_s=end_to_end_elapsed,
    )


def run_host_burst_blocking_timed(
    controller: CameraController,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> RunStats:
    t0 = time.perf_counter()
    paths_holder: List[List[str]] = []

    def _burst_once() -> None:
        paths_holder.clear()
        paths_holder.append(
            controller.capture_burst(
                shots=1,
                duration=run_seconds,
                timeout=max(timeout_s, run_seconds + 2.0),
                download_timeout=max(timeout_s, run_seconds + 10.0),
                drive_mode=drive_mode,
                apply_drive_mode=True,
            )
        )

    run_with_busy_retry(
        _burst_once,
        retries=6,
        base_delay_s=0.08,
        on_retry=lambda: controller.wake_up(),
    )
    elapsed = time.perf_counter() - t0
    return RunStats(
        frames=len(paths_holder[0]),
        capture_elapsed_s=elapsed,
        end_to_end_elapsed_s=elapsed,
    )


def run_camera_blocking_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
) -> RunStats:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    t0 = time.perf_counter()
    deadline = t0 + run_seconds
    frames = 0
    while time.perf_counter() < deadline:
        start = counter.snapshot()

        def _shot_once() -> None:
            edsdk.SendCommand(controller._cam, CameraCommand.TakePicture, 0)

        run_with_busy_retry(
            _shot_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        if counter.wait_for_delta(start=start, expected_delta=1, timeout_s=timeout_s):
            frames += 1
        else:
            raise TimeoutError("Timed out waiting for DirItemCreated in blocking SD mode")
    elapsed = time.perf_counter() - t0
    return RunStats(
        frames=frames,
        capture_elapsed_s=elapsed,
        end_to_end_elapsed_s=elapsed,
    )


def run_camera_async_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
) -> RunStats:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    t0 = time.perf_counter()
    deadline = t0 + run_seconds
    start_count = counter.snapshot()
    while time.perf_counter() < deadline:
        def _shot_once() -> None:
            edsdk.SendCommand(controller._cam, CameraCommand.TakePicture, 0)

        run_with_busy_retry(
            _shot_once,
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
    capture_elapsed = time.perf_counter() - t0
    end_count = wait_for_event_quiet(counter, quiet_s=1.2, max_wait_s=max(5.0, timeout_s))
    end_to_end_elapsed = time.perf_counter() - t0
    return RunStats(
        frames=max(0, end_count - start_count),
        capture_elapsed_s=capture_elapsed,
        end_to_end_elapsed_s=end_to_end_elapsed,
    )


def run_camera_burst_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> RunStats:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    controller.set_properties(
        drive_mode=drive_mode,
        validate=False,
        tolerate_not_supported=True,
    )
    start_count = counter.snapshot()
    t0 = time.perf_counter()
    shutter_pressed = False
    try:
        run_with_busy_retry(
            lambda: edsdk.SendCommand(
                controller._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.Completely),
            ),
            retries=8,
            base_delay_s=0.05,
            on_retry=lambda: controller.wake_up(),
        )
        shutter_pressed = True
        while (time.perf_counter() - t0) < run_seconds:
            time.sleep(0.01)
    finally:
        if shutter_pressed:
            edsdk.SendCommand(
                controller._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.OFF),
            )
    capture_elapsed = time.perf_counter() - t0
    end_count = wait_for_event_quiet(counter, quiet_s=1.2, max_wait_s=max(5.0, timeout_s))
    end_to_end_elapsed = time.perf_counter() - t0
    return RunStats(
        frames=max(0, end_count - start_count),
        capture_elapsed_s=capture_elapsed,
        end_to_end_elapsed_s=end_to_end_elapsed,
    )


def timed_run(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def print_results(results: List[ScenarioResult], run_seconds: float) -> None:
    regular = [r for r in results if r.target_fps is None]
    rate_rows = [r for r in results if r.target_fps is not None]

    if regular:
        print()
        print(f"Benchmark results (target run window: {run_seconds:.2f}s)")
        print("-" * 138)
        print(
            f"{'Scenario':<26} {'SaveTo':<10} {'Frames':>8} {'Capture(s)':>11} {'E2E(s)':>10} {'CaptureFPS':>11} {'E2EFPS':>9}  {'Status':<8} Detail"
        )
        print("-" * 138)
        for row in regular:
            cap_s = f"{row.capture_elapsed_s:.3f}" if row.capture_elapsed_s is not None else "-"
            e2e_s = (
                f"{row.end_to_end_elapsed_s:.3f}"
                if row.end_to_end_elapsed_s is not None
                else "-"
            )
            cap_fps = f"{row.capture_fps:.3f}" if row.capture_fps is not None else "-"
            e2e_fps = f"{row.end_to_end_fps:.3f}" if row.end_to_end_fps is not None else "-"
            status = "OK" if row.ok else "FAILED"
            print(
                f"{row.name:<26} {row.save_target:<10} {row.frames:>8} {cap_s:>11} {e2e_s:>10} {cap_fps:>11} {e2e_fps:>9}  {status:<8} {row.detail}"
            )
        print("-" * 138)

    if rate_rows:
        print()
        print("Rate-controlled results")
        print("-" * 150)
        print(
            f"{'Scenario':<26} {'SaveTo':<10} {'Frames':>8} {'TgtFPS':>8} {'AchFPS':>8} "
            f"{'MeanInt':>9} {'Jitter':>9} {'SchedErr':>10} {'Skip':>6}  {'Status':<8} Detail"
        )
        print("-" * 150)
        for row in rate_rows:
            ach = f"{row.achieved_fps:.3f}" if row.achieved_fps is not None else "-"
            tgt = f"{row.target_fps:.3f}" if row.target_fps is not None else "-"
            mi = f"{row.mean_interval:.4f}" if row.mean_interval is not None else "-"
            jt = f"{row.jitter_std:.4f}" if row.jitter_std is not None else "-"
            se = f"{row.mean_schedule_error:.4f}" if row.mean_schedule_error is not None else "-"
            sk = f"{row.skipped}" if row.skipped is not None else "-"
            status = "OK" if row.ok else "FAILED"
            print(
                f"{row.name:<26} {row.save_target:<10} {row.frames:>8} {tgt:>8} {ach:>8} "
                f"{mi:>9} {jt:>9} {se:>10} {sk:>6}  {status:<8} {row.detail}"
            )
        print("-" * 150)


def configure_capture_format(controller: CameraController, capture_format: str) -> str:
    """Apply requested capture format and return display label."""
    if capture_format == "current":
        return "current"
    if capture_format == "low_jpeg":
        selected = set_image_quality_low_jpeg(controller)
        return f"low_jpeg({selected})"
    raise ValueError(f"Unknown capture format: {capture_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EDSDK capture timing across capture modes and save targets.",
    )
    parser.add_argument("--index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "data" / "capture_benchmark"),
        help="Directory for host-download scenarios",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=3.0,
        help="How long to run each scenario at maximum throughput (default: 3.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout seconds for waits in each scenario (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from CameraController",
    )
    parser.add_argument(
        "--save-targets",
        choices=("both", "host", "camera"),
        default="both",
        help="Which save targets to benchmark: both, host only, or camera SD only",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.2,
        help="Quiet-window seconds used to detect late frames/transfers (default: 1.2)",
    )
    parser.add_argument(
        "--max-settle-seconds",
        type=float,
        default=20.0,
        help="Max extra wait for late frames/transfers before finalizing counts (default: 20)",
    )
    parser.add_argument(
        "--capture-format",
        choices=("current", "low_jpeg"),
        default="current",
        help="Capture encoding for benchmark: keep current camera format or force low-quality JPEG",
    )
    parser.add_argument(
        "--rate-fps",
        type=float,
        nargs="+",
        default=None,
        help="Target FPS values for rate-controlled scenarios (e.g. --rate-fps 0.5 1.0 1.5). "
        "Omit to skip rate-controlled scenarios.",
    )
    parser.add_argument(
        "--rate-duration",
        type=float,
        default=10.0,
        help="Duration in seconds for each rate-controlled scenario (default: 10.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    results: List[ScenarioResult] = []
    event_counter = ObjectEventCounter()

    scenarios: List[
        Tuple[str, SaveTo, Callable[[CameraController], RunStats]]
    ] = [
        (
            "blocking",
            SaveTo.Host,
            lambda c: run_host_blocking_timed(c, args.run_seconds, args.timeout),
        ),
        (
            "async",
            SaveTo.Host,
            lambda c: run_host_async_timed(c, args.run_seconds, args.timeout),
        ),
        (
            "burst_async_high",
            SaveTo.Host,
            lambda c: run_host_burst_async_timed(
                c, args.run_seconds, args.timeout, DriveMode.HighSpeedContinuous
            ),
        ),
        (
            "burst_async_low",
            SaveTo.Host,
            lambda c: run_host_burst_async_timed(
                c, args.run_seconds, args.timeout, DriveMode.LowSpeedContinuous
            ),
        ),
        (
            "burst_high",
            SaveTo.Host,
            lambda c: run_host_burst_blocking_timed(
                c, args.run_seconds, args.timeout, DriveMode.HighSpeedContinuous
            ),
        ),
        (
            "burst_low",
            SaveTo.Host,
            lambda c: run_host_burst_blocking_timed(
                c, args.run_seconds, args.timeout, DriveMode.LowSpeedContinuous
            ),
        ),
        (
            "blocking",
            SaveTo.Camera,
            lambda c: run_camera_blocking_timed(
                c, event_counter, args.run_seconds, args.timeout
            ),
        ),
        (
            "async",
            SaveTo.Camera,
            lambda c: run_camera_async_timed(
                c, event_counter, args.run_seconds, args.timeout
            ),
        ),
        (
            "burst_async_high",
            SaveTo.Camera,
            lambda c: run_camera_burst_timed(
                c,
                event_counter,
                args.run_seconds,
                args.timeout,
                DriveMode.HighSpeedContinuous,
            ),
        ),
        (
            "burst_async_low",
            SaveTo.Camera,
            lambda c: run_camera_burst_timed(
                c,
                event_counter,
                args.run_seconds,
                args.timeout,
                DriveMode.LowSpeedContinuous,
            ),
        ),
        (
            "burst_high",
            SaveTo.Camera,
            lambda c: run_camera_burst_timed(
                c,
                event_counter,
                args.run_seconds,
                args.timeout,
                DriveMode.HighSpeedContinuous,
            ),
        ),
        (
            "burst_low",
            SaveTo.Camera,
            lambda c: run_camera_burst_timed(
                c,
                event_counter,
                args.run_seconds,
                args.timeout,
                DriveMode.LowSpeedContinuous,
            ),
        ),
    ]
    if args.save_targets == "host":
        scenarios = [s for s in scenarios if s[1] == SaveTo.Host]
    elif args.save_targets == "camera":
        scenarios = [s for s in scenarios if s[1] == SaveTo.Camera]

    with CameraController(
        index=args.index,
        save_dir=args.save_dir,
        save_to=SaveTo.Host,
        auto_capacity=True,
        verbose=args.verbose,
    ) as controller:
        controller.on_object(event_counter.callback)
        selected_format = configure_capture_format(controller, args.capture_format)
        print(f"Capture format: {selected_format}")
        try:
            for mode_name, save_to, runner in scenarios:
                save_label = "host" if save_to == SaveTo.Host else "camera_sd"
                scenario_name = f"{mode_name}"
                try:
                    # Keep each scenario independent and let the camera settle.
                    try:
                        controller.wake_up()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    run_start = time.perf_counter()
                    host_before = 0
                    cam_before = 0
                    if save_to == SaveTo.Host:
                        # Use actual files written as authoritative frame count for host runs.
                        os.makedirs(args.save_dir, exist_ok=True)
                        controller.save_dir = args.save_dir
                        host_before = count_files_recursive(args.save_dir)
                    else:
                        cam_before = event_counter.snapshot()
                    set_save_target(controller, save_to)
                    stats = runner(controller)
                    if save_to == SaveTo.Host:
                        host_after = wait_for_file_quiet(
                            args.save_dir,
                            quiet_s=max(0.2, args.settle_seconds),
                            max_wait_s=max(args.settle_seconds, args.max_settle_seconds),
                        )
                        authoritative_frames = max(0, host_after - host_before)
                        authoritative_e2e = time.perf_counter() - run_start
                        stats = RunStats(
                            frames=authoritative_frames,
                            capture_elapsed_s=stats.capture_elapsed_s,
                            end_to_end_elapsed_s=max(
                                stats.end_to_end_elapsed_s, authoritative_e2e
                            ),
                        )
                    else:
                        # For camera SD, use settled DirItemCreated delta as
                        # authoritative frame count for the scenario.
                        cam_after = wait_for_event_quiet(
                            event_counter,
                            quiet_s=max(0.2, args.settle_seconds),
                            max_wait_s=max(args.settle_seconds, args.max_settle_seconds),
                        )
                        authoritative_frames = max(0, cam_after - cam_before)
                        stats = RunStats(
                            frames=authoritative_frames,
                            capture_elapsed_s=stats.capture_elapsed_s,
                            end_to_end_elapsed_s=max(
                                stats.end_to_end_elapsed_s,
                                time.perf_counter() - run_start,
                            ),
                        )
                    capture_fps = (
                        (stats.frames / stats.capture_elapsed_s)
                        if stats.capture_elapsed_s > 0
                        else None
                    )
                    end_to_end_fps = (
                        (stats.frames / stats.end_to_end_elapsed_s)
                        if stats.end_to_end_elapsed_s > 0
                        else None
                    )
                    results.append(
                        ScenarioResult(
                            name=scenario_name,
                            save_target=save_label,
                            frames=stats.frames,
                            capture_elapsed_s=stats.capture_elapsed_s,
                            end_to_end_elapsed_s=stats.end_to_end_elapsed_s,
                            capture_fps=capture_fps,
                            end_to_end_fps=end_to_end_fps,
                            ok=True,
                        )
                    )
                    print(
                        f"Completed {scenario_name:>11} / {save_label:<9} -> {stats.frames} frames, capture_fps={capture_fps:.3f}, e2e_fps={end_to_end_fps:.3f}"
                    )
                    time.sleep(0.3)
                except Exception as exc:
                    results.append(
                        ScenarioResult(
                            name=scenario_name,
                            save_target=save_label,
                            frames=0,
                            capture_elapsed_s=None,
                            end_to_end_elapsed_s=None,
                            capture_fps=None,
                            end_to_end_fps=None,
                            ok=False,
                            detail=str(exc),
                        )
                    )
                    print(f"Failed    {scenario_name:>11} / {save_label:<9}: {exc}")

            # --- Rate-controlled scenarios ---
            if args.rate_fps:
                rate_targets: List[Tuple[str, SaveTo]] = []
                if args.save_targets in ("both", "host"):
                    rate_targets.append(("host", SaveTo.Host))
                if args.save_targets in ("both", "camera"):
                    rate_targets.append(("camera_sd", SaveTo.Camera))

                for fps_val in args.rate_fps:
                    for save_label, save_to in rate_targets:
                        scenario_name = f"rate_{fps_val:.2f}fps"
                        try:
                            try:
                                controller.wake_up()
                            except Exception:
                                pass
                            time.sleep(0.2)
                            set_save_target(controller, save_to)

                            if save_to == SaveTo.Host:
                                stats, rc = run_rate_controlled_host(
                                    controller, fps_val, args.rate_duration, args.timeout,
                                )
                            else:
                                stats, rc = run_rate_controlled_camera(
                                    controller, event_counter, fps_val,
                                    args.rate_duration, args.timeout,
                                )

                            results.append(
                                ScenarioResult(
                                    name=scenario_name,
                                    save_target=save_label,
                                    frames=stats.frames,
                                    capture_elapsed_s=stats.capture_elapsed_s,
                                    end_to_end_elapsed_s=stats.end_to_end_elapsed_s,
                                    capture_fps=rc.achieved_fps if rc.achieved_fps else None,
                                    end_to_end_fps=None,
                                    ok=True,
                                    target_fps=fps_val,
                                    achieved_fps=rc.achieved_fps,
                                    mean_interval=rc.mean_interval,
                                    jitter_std=rc.jitter_std,
                                    mean_schedule_error=rc.mean_schedule_error,
                                    skipped=len(rc.skipped_indices),
                                )
                            )
                            print(
                                f"Completed {scenario_name:>20} / {save_label:<9} -> "
                                f"{stats.frames} frames, "
                                f"achieved_fps={rc.achieved_fps:.3f}, "
                                f"jitter={rc.jitter_std:.4f}s, "
                                f"sched_err={rc.mean_schedule_error:.4f}s"
                            )
                            time.sleep(0.3)
                        except Exception as exc:
                            results.append(
                                ScenarioResult(
                                    name=scenario_name,
                                    save_target=save_label,
                                    frames=0,
                                    capture_elapsed_s=None,
                                    end_to_end_elapsed_s=None,
                                    capture_fps=None,
                                    end_to_end_fps=None,
                                    ok=False,
                                    detail=str(exc),
                                    target_fps=fps_val,
                                )
                            )
                            print(f"Failed    {scenario_name:>20} / {save_label:<9}: {exc}")
        finally:
            try:
                restored = restore_image_quality_cr3(controller)
                print(f"Restored capture format to CR3-compatible mode: {restored}")
            except Exception as exc:
                print(f"Warning: failed to restore CR3-compatible image quality: {exc}")

    print_results(results, run_seconds=args.run_seconds)
    failures = sum(1 for r in results if not r.ok)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
