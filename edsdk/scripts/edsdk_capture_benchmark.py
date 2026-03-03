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
from edsdk.camera_controller import CameraController  # noqa: E402
from edsdk.constants.commands import CameraCommand, ShutterButton  # noqa: E402
from edsdk.constants.generic import ObjectEvent  # noqa: E402
from edsdk.constants.properties import DriveMode, PropID, SaveTo  # noqa: E402


@dataclass
class ScenarioResult:
    name: str
    save_target: str
    elapsed_s: Optional[float]
    frames: int
    fps: Optional[float]
    ok: bool
    detail: str = ""


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
    counter: ObjectEventCounter, quiet_s: float = 0.4, max_wait_s: float = 5.0
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


def set_save_target(controller: CameraController, save_to: SaveTo) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    edsdk.SetPropertyData(controller._cam, PropID.SaveTo, 0, int(save_to))
    controller.save_to = save_to


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


def run_host_blocking_timed(
    controller: CameraController, run_seconds: float, timeout_s: float
) -> Tuple[int, float]:
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
    return frames, elapsed


def run_host_async_timed(
    controller: CameraController, run_seconds: float, timeout_s: float
) -> Tuple[int, float]:
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
    frames = 0
    if expected_total > 0 and first_marker is not None:
        paths = controller.wait_for_downloads(
            expected=expected_total,
            marker=first_marker,
            timeout=max(timeout_s, expected_total * 5.0),
        )
        frames = len(paths)
    elapsed = time.perf_counter() - t0
    return frames, elapsed


def run_host_burst_async_timed(
    controller: CameraController,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> Tuple[int, float]:
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
    paths = controller.wait_for_downloads(
        expected=int(ticket["expected"]),
        marker=int(ticket["marker"]),
        timeout=max(timeout_s, run_seconds + 10.0, int(ticket["expected"]) * 5.0),
    )
    elapsed = time.perf_counter() - t0
    return len(paths), elapsed


def run_host_burst_blocking_timed(
    controller: CameraController,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> Tuple[int, float]:
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
    return len(paths_holder[0]), elapsed


def run_camera_blocking_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
) -> Tuple[int, float]:
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
    return frames, elapsed


def run_camera_async_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
) -> Tuple[int, float]:
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
    end_count = wait_for_event_quiet(counter, quiet_s=0.4, max_wait_s=max(1.0, timeout_s))
    elapsed = time.perf_counter() - t0
    return max(0, end_count - start_count), elapsed


def run_camera_burst_timed(
    controller: CameraController,
    counter: ObjectEventCounter,
    run_seconds: float,
    timeout_s: float,
    drive_mode: DriveMode,
) -> Tuple[int, float]:
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
    end_count = wait_for_event_quiet(counter, quiet_s=0.4, max_wait_s=max(1.0, timeout_s))
    elapsed = time.perf_counter() - t0
    return max(0, end_count - start_count), elapsed


def timed_run(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def print_results(results: List[ScenarioResult], run_seconds: float) -> None:
    print()
    print(f"Benchmark results (target run window: {run_seconds:.2f}s)")
    print("-" * 104)
    print(
        f"{'Scenario':<26} {'SaveTo':<10} {'Frames':>8} {'Elapsed(s)':>11} {'FPS':>8}  {'Status':<8} Detail"
    )
    print("-" * 104)
    for row in results:
        sec = f"{row.elapsed_s:.3f}" if row.elapsed_s is not None else "-"
        fps = f"{row.fps:.3f}" if row.fps is not None else "-"
        status = "OK" if row.ok else "FAILED"
        print(
            f"{row.name:<26} {row.save_target:<10} {row.frames:>8} {sec:>11} {fps:>8}  {status:<8} {row.detail}"
        )
    print("-" * 104)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    results: List[ScenarioResult] = []
    event_counter = ObjectEventCounter()

    scenarios: List[
        Tuple[str, SaveTo, Callable[[CameraController], Tuple[int, float]]]
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
                set_save_target(controller, save_to)
                frames, elapsed = runner(controller)
                fps = (frames / elapsed) if elapsed > 0 else None
                results.append(
                    ScenarioResult(
                        name=scenario_name,
                        save_target=save_label,
                        elapsed_s=elapsed,
                        frames=frames,
                        fps=fps,
                        ok=True,
                    )
                )
                print(
                    f"Completed {scenario_name:>11} / {save_label:<9} -> {frames} frames in {elapsed:.3f}s ({fps:.3f} fps)"
                )
                time.sleep(0.3)
            except Exception as exc:
                results.append(
                    ScenarioResult(
                        name=scenario_name,
                        save_target=save_label,
                        elapsed_s=None,
                        frames=0,
                        fps=None,
                        ok=False,
                        detail=str(exc),
                    )
                )
                print(f"Failed    {scenario_name:>11} / {save_label:<9}: {exc}")

    print_results(results, run_seconds=args.run_seconds)
    failures = sum(1 for r in results if not r.ok)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
