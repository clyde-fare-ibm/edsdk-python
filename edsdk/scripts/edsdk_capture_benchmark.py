#!/usr/bin/env python3
"""Benchmark EDSDK capture paths across host vs camera storage.

Runs 6 scenarios:
1) blocking capture -> host download
2) async capture    -> host download
3) burst async      -> host download
4) blocking capture -> camera SD
5) async capture    -> camera SD
6) burst async      -> camera SD
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


def run_host_burst_async(controller: CameraController, shots: int, timeout_s: float) -> None:
    ticket_holder: List[dict] = []

    def _burst_once() -> None:
        ticket_holder.clear()
        ticket_holder.append(
            controller.capture_burst_async(
                shots=shots,
                timeout=max(timeout_s, shots * 2.0),
                drive_mode=DriveMode.HighSpeedContinuous,
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
    controller: CameraController, counter: ObjectEventCounter, shots: int, timeout_s: float
) -> None:
    if controller._cam is None:
        raise RuntimeError("Camera session not open")
    controller.set_properties(
        drive_mode=DriveMode.HighSpeedContinuous,
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


def timed_run(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def print_results(results: List[ScenarioResult], shots: int) -> None:
    print()
    print(f"Benchmark results for {shots} successive images")
    print("-" * 82)
    print(f"{'Scenario':<26} {'SaveTo':<10} {'Seconds':>10}  {'Status':<8} Detail")
    print("-" * 82)
    for row in results:
        sec = f"{row.elapsed_s:.3f}" if row.elapsed_s is not None else "-"
        status = "OK" if row.ok else "FAILED"
        print(f"{row.name:<26} {row.save_target:<10} {sec:>10}  {status:<8} {row.detail}")
    print("-" * 82)


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
        "--shots",
        type=int,
        default=6,
        help="Number of successive images per scenario (default: 6)",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    results: List[ScenarioResult] = []
    event_counter = ObjectEventCounter()

    scenarios: List[Tuple[str, SaveTo, Callable[[CameraController], None]]] = [
        (
            "blocking",
            SaveTo.Host,
            lambda c: run_host_blocking(c, args.shots, args.timeout),
        ),
        (
            "async",
            SaveTo.Host,
            lambda c: run_host_async(c, args.shots, args.timeout),
        ),
        (
            "burst_async",
            SaveTo.Host,
            lambda c: run_host_burst_async(c, args.shots, args.timeout),
        ),
        (
            "blocking",
            SaveTo.Camera,
            lambda c: run_camera_blocking(c, event_counter, args.shots, args.timeout),
        ),
        (
            "async",
            SaveTo.Camera,
            lambda c: run_camera_async(c, event_counter, args.shots, args.timeout),
        ),
        (
            "burst_async",
            SaveTo.Camera,
            lambda c: run_camera_burst_async(c, event_counter, args.shots, args.timeout),
        ),
    ]

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
                elapsed = timed_run(lambda: runner(controller))
                results.append(
                    ScenarioResult(
                        name=scenario_name,
                        save_target=save_label,
                        elapsed_s=elapsed,
                        ok=True,
                    )
                )
                print(
                    f"Completed {scenario_name:>11} / {save_label:<9} in {elapsed:.3f}s"
                )
                time.sleep(0.3)
            except Exception as exc:
                results.append(
                    ScenarioResult(
                        name=scenario_name,
                        save_target=save_label,
                        elapsed_s=None,
                        ok=False,
                        detail=str(exc),
                    )
                )
                print(f"Failed    {scenario_name:>11} / {save_label:<9}: {exc}")

    print_results(results, shots=args.shots)
    failures = sum(1 for r in results if not r.ok)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
