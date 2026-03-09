from __future__ import annotations

import os
import json
import io
import asyncio
import time
import uuid
import queue
import threading
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

# External SDK imports
import edsdk
from edsdk import (
    Access,
    CameraCommand,
    EdsObject,
    FileCreateDisposition,
    ObjectEvent,
    PropID,
    PropertyEvent,
)
from edsdk.constants.commands import ShutterButton
from edsdk.constants.generic import CameraStatusCommand
from edsdk.constants.properties import (
    Av as AvTable,
    Tv as TvTable,
    SaveTo,
    AEMode,
    MeteringMode,
    WhiteBalance,
    ImageQuality,
    DriveMode,
    EvfOutputDevice,
    AFMode,
    EvfAFMode,
    FlashFiring,
    FlashTarget,
)
from ._property_codec import (
    enum_code,
    enum_supported_names,
    iso_code_to_string,
    parse_av,
    parse_iso,
    parse_tv,
)
from ._transfer_pipeline import TransferPipeline

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image
    import numpy as np


# Public callback / return type aliases (after imports to satisfy linters)
ObjectCallback = Callable[["ObjectEvent", "EdsObject"], int]
PropertyCallback = Callable[["PropertyEvent", "PropID", int], int]
LiveViewData = Union[bytes, str]
LiveViewFrame = Union[LiveViewData, Tuple[LiveViewData, Dict[str, object]]]

# Windows message pumping for EDSDK callbacks
if os.name == "nt":
    try:
        import pythoncom  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        pythoncom = None  # type: ignore
else:  # pragma: no cover - not required outside Windows
    pythoncom = None  # type: ignore


def _pump_messages_once() -> None:
    if pythoncom is not None:
        pythoncom.PumpWaitingMessages()
        return
    # On non-Windows platforms, use GetEvent to dispatch SDK events if available.
    try:
        if hasattr(edsdk, "GetEvent"):
            edsdk.GetEvent()
    except Exception:
        pass


def _save_directory_item(
    object_handle: EdsObject, save_dir: str, dst_basename: Optional[str] = None
) -> str:
    info = edsdk.GetDirectoryItemInfo(object_handle)
    orig_name = info.get("szFileName") or f"{uuid.uuid4()}.bin"
    filename = dst_basename or orig_name
    # sanitize path separators in provided name
    filename = filename.replace("\\", "_").replace("/", "_")
    dst = os.path.join(save_dir, filename)
    out_stream = edsdk.CreateFileStream(
        dst,
        FileCreateDisposition.CreateAlways,
        Access.ReadWrite,
    )
    edsdk.Download(object_handle, info["size"], out_stream)
    edsdk.DownloadComplete(object_handle)
    return dst


class CameraController:
    """
    A small, ergonomic wrapper around edsdk for property management and capture.

    Contract
    - Inputs: av (e.g., 5.6 or "f/5.6"), tv (e.g., "1/125" or 0.5), iso (int or "auto"), save_dir
    - Output: list of saved file paths from captures
    - Error modes: invalid properties -> ValueError, no camera -> RuntimeError, timeouts -> TimeoutError
    - Success: returns list with at least one valid path when capture completes
    """

    def __init__(
        self,
        index: int = 0,
        save_dir: str = ".",
        save_to: SaveTo = SaveTo.Host,
        auto_capacity: bool = True,
        *,
        verbose: bool = False,
        logger: Optional[Callable[[str], None]] = None,
        register_property_events: bool = True,
        enable_flash_control: bool = False,
        flash_target: FlashTarget = FlashTarget.ExternalFlash,
        flash_firing: FlashFiring = FlashFiring.Fire,
        register_flash_events: bool = True,
        file_pattern: Optional[str] = None,
        seq_start: int = 1,
        max_inflight: Optional[int] = None,
        inflight_wait_timeout: Optional[float] = None,
    ) -> None:
        self.index = index
        self.save_dir = save_dir
        self.save_to = save_to
        self.auto_capacity = auto_capacity
        self.verbose = verbose
        self.ui_locked = False
        self._log = logger or (print if verbose else (lambda *_args, **_kw: None))
        self._cam: Optional[EdsObject] = None
        self._obj_cb: Optional[ObjectCallback] = None
        self._prop_cb: Optional[PropertyCallback] = None
        self._flash_prop_cb: Optional[PropertyCallback] = None
        self._live_view_on: bool = False
        # asyncio event queue support
        self._async_queue: Optional[asyncio.Queue[Dict[str, Union[str, int]]]] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_pumping: bool = False
        self._register_property_events = register_property_events
        self._enable_flash_control = enable_flash_control
        self._flash_target = flash_target
        self._flash_firing = flash_firing
        self._register_flash_events = register_flash_events
        self._file_pattern = file_pattern
        self._seq = int(seq_start)
        self._max_inflight = max_inflight if max_inflight is None else int(max_inflight)
        if self._max_inflight is not None and self._max_inflight <= 0:
            raise ValueError("max_inflight must be > 0 when specified")
        self._inflight_wait_timeout = inflight_wait_timeout
        # One-shot explicit filename (base name); if set, next capture uses this name
        self._next_filename: Optional[str] = None
        self._flash_ref: Optional[EdsObject] = None
        # Background transfer pipeline
        self._download_q: "queue.Queue[Tuple[EdsObject, Optional[str]]]" = queue.Queue()
        self._download_stop = threading.Event()
        self._download_thread: Optional[threading.Thread] = None
        # Non-Windows uses polling GetEvent; keep pumping in background for async capture.
        self._pump_stop = threading.Event()
        self._pump_thread: Optional[threading.Thread] = None
        self._transfers = TransferPipeline(
            max_inflight=self._max_inflight,
            inflight_wait_timeout=self._inflight_wait_timeout,
        )

    # ---------- Lifecycle ----------
    def __enter__(self) -> "CameraController":
        edsdk.InitializeSDK()
        cam_list = edsdk.GetCameraList()
        nr_cameras = edsdk.GetChildCount(cam_list)
        if nr_cameras == 0:
            self.__exit__(None, None, None)
            raise RuntimeError("No cameras connected")
        if self.index >= nr_cameras:
            self.__exit__(None, None, None)
            raise RuntimeError(
                f"Camera index {self.index} out of range (found {nr_cameras})"
            )
        cam = edsdk.GetChildAtIndex(cam_list, self.index)
        edsdk.OpenSession(cam)
      

        # Event handlers (property event can be suppressed to avoid noisy warnings)
        edsdk.SetObjectEventHandler(cam, ObjectEvent.All, self._on_object_event)
        if self._register_property_events:
            try:
                edsdk.SetPropertyEventHandler(
                    cam, PropertyEvent.All, self._on_property_event
                )
            except Exception as e:
                # Non-fatal: log only if verbose
                self._log(f"Skip property events: {e}")

        # Save to host and capacity
        edsdk.SetPropertyData(cam, PropID.SaveTo, 0, int(self.save_to))
        if self.auto_capacity:
            edsdk.SetCapacity(
                cam,
                {
                    "reset": True,
                    "bytesPerSector": 512,
                    "numberOfFreeClusters": 2_147_483_647,
                },
            )
        if self._enable_flash_control:
            try:
                self._flash_ref = edsdk.CreateFlashSettingRef(cam)
                if self._register_flash_events:
                    edsdk.SetPropertyEventHandler(
                        self._flash_ref, PropertyEvent.All, self._on_flash_property_event
                    )
            except Exception as e:
                self._log(f"Flash control unavailable: {e}")
      
        self._cam = cam
        self._start_background_workers()
        self._log("Camera session opened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_background_workers()
        try:
            if self._cam is not None:
                try:
                    edsdk.CloseSession(self._cam)
                except Exception:
                    pass
        finally:
            try:
                edsdk.TerminateSDK()
            except Exception:
                pass
        self._cam = None
        self._flash_ref = None
        self._log("Camera session closed")

    # ---------- Event handlers ----------
    def on_object(self, fn: ObjectCallback) -> None:
        self._obj_cb = fn

    def on_property(self, fn: PropertyCallback) -> None:
        self._prop_cb = fn

    def on_flash_property(self, fn: PropertyCallback) -> None:
        self._flash_prop_cb = fn

    def _start_background_workers(self) -> None:
        if self._download_thread is None or not self._download_thread.is_alive():
            self._download_stop.clear()
            self._download_thread = threading.Thread(
                target=self._download_worker,
                name="edsdk-download-worker",
                daemon=True,
            )
            self._download_thread.start()
        if os.name != "nt" and (self._pump_thread is None or not self._pump_thread.is_alive()):
            self._pump_stop.clear()
            self._pump_thread = threading.Thread(
                target=self._event_pump_worker,
                name="edsdk-event-pump",
                daemon=True,
            )
            self._pump_thread.start()

    def _stop_background_workers(self) -> None:
        self._pump_stop.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=1.0)
        self._pump_thread = None
        self._download_stop.set()
        if self._download_thread is not None:
            self._download_thread.join()
        self._download_thread = None

    def _event_pump_worker(self) -> None:
        while not self._pump_stop.is_set():
            _pump_messages_once()
            time.sleep(0.005)

    def _download_worker(self) -> None:
        while True:
            if self._download_stop.is_set() and self._download_q.empty():
                return
            try:
                object_handle, dst_name = self._download_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                path = _save_directory_item(
                    object_handle, self.save_dir, dst_basename=dst_name
                )
                self._transfers.mark_downloaded(path)
                self._enqueue_async_event(
                    {
                        "kind": "object",
                        "event": "DirItemDownloaded",
                        "path": path,
                    }
                )
            except Exception as exc:
                self._transfers.mark_download_error(exc)
                self._enqueue_async_event(
                    {
                        "kind": "object",
                        "event": "DirItemDownloadError",
                        "message": str(exc),
                    }
                )
            finally:
                self._download_q.task_done()

    def _queue_transfer(
        self, object_handle: EdsObject, dst_name: Optional[str]
    ) -> None:
        self._transfers.mark_queued()
        self._download_q.put((object_handle, dst_name))

    def _wait_for_inflight_slot(self, timeout: Optional[float] = None) -> None:
        self._transfers.wait_for_inflight_slot(_pump_messages_once, timeout=timeout)

    def _on_object_event(self, event: ObjectEvent, object_handle: EdsObject) -> int:
        if event == ObjectEvent.DirItemRequestTransfer:
            # compute custom filename if pattern is provided
            dst_name: Optional[str] = None
            # 1) Highest priority: explicitly specified next filename via capture(filename=...)
            try:
                info = edsdk.GetDirectoryItemInfo(object_handle)
                orig_name = info.get("szFileName") or f"{uuid.uuid4()}.bin"
            except Exception:
                info = {}
                orig_name = f"{uuid.uuid4()}.bin"
            if self._next_filename:
                # preserve original extension; ignore any extension in provided name
                provided = self._next_filename.replace("\\", "_").replace("/", "_")
                self._next_filename = None
                base_prov, _ext_prov = os.path.splitext(provided)
                if not base_prov:
                    base_prov = "image"
                _base_orig, ext_orig = os.path.splitext(orig_name)
                if not ext_orig:
                    ext_orig = ".bin"
                dst_name = f"{base_prov}{ext_orig}"
            # 2) Next: pattern-based naming if provided
            elif self._file_pattern:
                try:
                    base, ext = os.path.splitext(orig_name)
                    if not ext:
                        ext = ".bin"
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    dst_name = self._file_pattern.format(
                        basename=base,
                        ext=ext.lstrip("."),
                        timestamp=ts,
                        seq=self._seq,
                    )
                    self._seq += 1
                except Exception:
                    dst_name = None

            self._queue_transfer(object_handle, dst_name)
            self._enqueue_async_event(
                {
                    "kind": "object",
                    "event": "DirItemQueued",
                }
            )
        else:
            self._enqueue_async_event(
                {
                    "kind": "object",
                    "event": getattr(ObjectEvent, event.name).name
                    if hasattr(event, "name")
                    else int(event),
                }
            )
        if self._obj_cb:
            try:
                return int(self._obj_cb(event, object_handle))
            except Exception:
                return 0
        return 0

    def _on_property_event(
        self, event: PropertyEvent, prop_id: PropID, param: int
    ) -> int:
        if self._prop_cb:
            try:
                return int(self._prop_cb(event, prop_id, param))
            except Exception:
                return 0
        # queue property event (coarse)
        try:
            self._enqueue_async_event(
                {
                    "kind": "property",
                    "event": event.name if hasattr(event, "name") else int(event),
                    "property": prop_id.name
                    if hasattr(prop_id, "name")
                    else int(prop_id),
                    "param": int(param),
                }
            )
        except Exception:
            pass
        return 0

    def _on_flash_property_event(
        self, event: PropertyEvent, prop_id: PropID, param: int
    ) -> int:
        if self._flash_prop_cb:
            try:
                return int(self._flash_prop_cb(event, prop_id, param))
            except Exception:
                return 0
        try:
            self._enqueue_async_event(
                {
                    "kind": "flash_property",
                    "event": event.name if hasattr(event, "name") else int(event),
                    "property": prop_id.name
                    if hasattr(prop_id, "name")
                    else int(prop_id),
                    "param": int(param),
                }
            )
        except Exception:
            pass
        return 0

    def wake_up(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        try:
            edsdk.SendCommand(
                self._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.Halfway),
            )
            time.sleep(0.5)
        except Exception as e:
            self._log(f"Error pressing shutter button: {e}")
        finally:
            try:
                edsdk.SendCommand(
                    self._cam,
                    CameraCommand.PressShutterButton,
                    int(ShutterButton.OFF),
                )
            except Exception as e:
                self._log(f"Error sending shutter button off command: {e}")
            finally:
                time.sleep(0.5)

    def prepare_flash(self) -> None:
        """Configure flash settings via the flash settings object."""
        if self._cam is None:
            raise RuntimeError("Camera session not open")

        self.wake_up()
        self.lock_ui()
        time.sleep(0.5)
        try:
            # Wake flash by simulating a half-press before setting properties.

            for attempt in range(3):
                if self._flash_ref is None or attempt > 0:
                    self._flash_ref = edsdk.CreateFlashSettingRef(self._cam)

                try:
                    edsdk.SetPropertyData(self._flash_ref, PropID.Flash_Target, 0, int(self._flash_target))
                    edsdk.SetPropertyData(self._flash_ref, PropID.Flash_Firing, 0, int(self._flash_firing))
                    break
                except Exception as e:
                    info = classify_error(e)
                    if info.get("code") == 80:  # PROPERTIES_UNAVAILABLE
                        time.sleep(0.3)
                        continue
                    raise
        finally:
            self.unlock_ui()


    # ---------- Properties ----------
    def set_properties(
        self,
        *,
        av: Optional[Union[str, float, int]] = None,
        tv: Optional[Union[str, float, int]] = None,
        iso: Optional[Union[str, int]] = None,
        ae_mode: Optional[Union[str, int]] = None,
        metering: Optional[Union[str, int]] = None,
        white_balance: Optional[Union[str, int]] = None,
        image_quality: Optional[Union[str, int]] = None,
        drive_mode: Optional[Union[str, int]] = None,
        manual_focus: Optional[bool] = None,
        af_mode: Optional[Union[str, int]] = None,
        evf_af_mode: Optional[Union[str, int]] = None,
        validate: bool = True,
        tolerate_not_supported: bool = False,
    ) -> None:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        # Prepare desired values
        to_set: List[Tuple[PropID, int]] = []
        if av is not None:
            to_set.append((PropID.Av, parse_av(av)))
        if tv is not None:
            to_set.append((PropID.Tv, parse_tv(tv)))
        if iso is not None:
            to_set.append((PropID.ISOSpeed, parse_iso(iso)))
        if ae_mode is not None:
            to_set.append((PropID.AEMode, enum_code(AEMode, ae_mode)))
        if metering is not None:
            to_set.append((PropID.MeteringMode, enum_code(MeteringMode, metering)))
        if white_balance is not None:
            to_set.append(
                (PropID.WhiteBalance, enum_code(WhiteBalance, white_balance))
            )
        if image_quality is not None:
            to_set.append(
                (PropID.ImageQuality, enum_code(ImageQuality, image_quality))
            )
        if drive_mode is not None:
            to_set.append((PropID.DriveMode, enum_code(DriveMode, drive_mode)))
        # Manual focus convenience flag takes precedence over af_mode
        if manual_focus is True:
            to_set.append((PropID.AFMode, int(AFMode.ManualFocus)))
        elif af_mode is not None:
            to_set.append((PropID.AFMode, enum_code(AFMode, af_mode)))
        if evf_af_mode is not None:
            to_set.append((PropID.Evf_AFMode, enum_code(EvfAFMode, evf_af_mode)))

        # Validate against camera descriptors; optionally tolerate AF/AEMode unsupported
        if validate:
            filtered: List[Tuple[PropID, int]] = []
            for pid, code in to_set:
                supported = self._get_supported_codes(pid)
                if supported and code not in supported:
                    if tolerate_not_supported and pid in (PropID.AEMode, PropID.AFMode):
                        self._log(
                            f"Skip unsupported {pid.name} during validate: requested {code}"
                        )
                        continue  # drop this setting silently
                    raise ValueError(f"Value {code} not supported for {pid}")
                filtered.append((pid, code))
            to_set = filtered

        # Apply
        for pid, code in to_set:
            self._log(f"Set {pid.name} -> {code}")
            try:
                edsdk.SetPropertyData(self._cam, pid, 0, code)
            except Exception as e:
                # Many Canon bodies do not allow changing AEMode via SDK.
                # Optionally ignore NOT_SUPPORTED for AEMode / AFMode when tolerate flag is set.
                if tolerate_not_supported and pid in (PropID.AEMode, PropID.AFMode):
                    self._log(f"Skip unsupported {pid.name}: {e}")
                    continue
                raise

    def get_properties(self) -> Dict[str, Union[str, int]]:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        av_code = edsdk.GetPropertyData(self._cam, PropID.Av, 0)
        tv_code = edsdk.GetPropertyData(self._cam, PropID.Tv, 0)
        iso_code = edsdk.GetPropertyData(self._cam, PropID.ISOSpeed, 0)

        # additional
        def enum_name(enum_cls, code: int) -> str:
            for name, member in enum_cls.__members__.items():
                if int(member) == int(code):
                    return name
            return str(code)

        props: Dict[str, Union[str, int]] = {
            "Av": AvTable.get(av_code, str(av_code)),
            "Tv": TvTable.get(tv_code, str(tv_code)),
            "ISO": iso_code_to_string(int(iso_code)),
            "SaveTo": str(edsdk.GetPropertyData(self._cam, PropID.SaveTo, 0)),
            "AEMode": enum_name(
                AEMode, edsdk.GetPropertyData(self._cam, PropID.AEMode, 0)
            ),
            "MeteringMode": enum_name(
                MeteringMode, edsdk.GetPropertyData(self._cam, PropID.MeteringMode, 0)
            ),
            "WhiteBalance": enum_name(
                WhiteBalance, edsdk.GetPropertyData(self._cam, PropID.WhiteBalance, 0)
            ),
            "ImageQuality": enum_name(
                ImageQuality, edsdk.GetPropertyData(self._cam, PropID.ImageQuality, 0)
            ),
            "DriveMode": enum_name(
                DriveMode, edsdk.GetPropertyData(self._cam, PropID.DriveMode, 0)
            ),
            "AFMode": enum_name(
                AFMode,
                self._safe_get_property(PropID.AFMode),
            ),
            "EvfAFMode": enum_name(
                EvfAFMode,
                self._safe_get_property(PropID.Evf_AFMode),
            ),
        }
        return props

    # ---------- Profiles ----------
    def save_profile(self, path: str) -> None:
        """Save current properties to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        profile = self.get_properties()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        self._log(f"Profile saved: {path}")

    def load_profile(
        self, path: str, *, apply: bool = True, validate: bool = True
    ) -> Dict[str, Union[str, int]]:
        """Load properties from JSON file and optionally apply to the camera."""
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if apply:
            self.set_properties(
                av=profile.get("Av"),
                tv=profile.get("Tv"),
                iso=profile.get("ISO"),
                ae_mode=profile.get("AEMode"),
                metering=profile.get("MeteringMode"),
                white_balance=profile.get("WhiteBalance"),
                image_quality=profile.get("ImageQuality"),
                drive_mode=profile.get("DriveMode"),
                af_mode=profile.get("AFMode"),
                evf_af_mode=profile.get("EvfAFMode"),
                manual_focus=True if profile.get("AFMode") == "ManualFocus" else None,
                validate=validate,
            )
        self._log(f"Profile loaded: {path}")
        return profile

    # ---------- Supported candidates ----------
    def list_supported(self) -> Dict[str, List[str]]:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        return {
            "Av": [
                AvTable.get(c, str(c)) for c in self._get_supported_codes(PropID.Av)
            ],
            "Tv": [
                TvTable.get(c, str(c)) for c in self._get_supported_codes(PropID.Tv)
            ],
            "ISO": [
                iso_code_to_string(int(c))
                for c in self._get_supported_codes(PropID.ISOSpeed)
            ],
            "AEMode": enum_supported_names(
                AEMode, self._get_supported_codes(PropID.AEMode)
            ),
            "MeteringMode": enum_supported_names(
                MeteringMode,
                self._get_supported_codes(PropID.MeteringMode),
            ),
            "WhiteBalance": enum_supported_names(
                WhiteBalance,
                self._get_supported_codes(PropID.WhiteBalance),
            ),
            "ImageQuality": enum_supported_names(
                ImageQuality,
                self._get_supported_codes(PropID.ImageQuality),
            ),
            "DriveMode": enum_supported_names(
                DriveMode, self._get_supported_codes(PropID.DriveMode)
            ),
            "AFMode": enum_supported_names(
                AFMode, self._get_supported_codes(PropID.AFMode)
            ),
            "EvfAFMode": enum_supported_names(
                EvfAFMode,
                self._get_supported_codes(PropID.Evf_AFMode),
            ),
        }

    def _get_supported_codes(self, pid: PropID) -> List[int]:
        try:
            desc = edsdk.GetPropertyDesc(self._cam, pid)
            return list(desc.get("propDesc", ()))
        except Exception:
            return []

    # ---------- ImageQuality helpers ----------
    def get_image_quality_code(self) -> int:
        """Return current ImageQuality property code."""
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        return int(edsdk.GetPropertyData(self._cam, PropID.ImageQuality, 0))

    # ---------- UI Lock/Unlock ----------
    def lock_ui(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        edsdk.SendStatusCommand(self._cam, CameraStatusCommand.UILock, 1)
        self.ui_locked = True

    def unlock_ui(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        edsdk.SendStatusCommand(self._cam, CameraStatusCommand.UIUnLock, 0)
        self.ui_locked = False

    # ---------- Capture ----------
    def capture(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        interval: float = 0.0,
        retry: int = 0,
        retry_delay: float = 0.3,
        filename: Optional[str] = None,
    ) -> List[str]:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        self._transfers.reset_paths_and_errors()
        if filename is not None:
            if shots != 1:
                raise ValueError("filename can be used only when shots=1")
            # Store provided name for next object transfer (extension will be preserved from camera)
            self._next_filename = filename
        for i in range(max(1, shots)):
            attempt = 0
            while True:
                try:
                    self._log(f"Trigger shot {i + 1}/{shots}")
                    completed_before, errors_before = self._transfers.capture_baseline()
                    self._wait_for_inflight_slot(timeout)
                    edsdk.SendCommand(self._cam, CameraCommand.TakePicture, 0)
                    self._wait_for_transfer(timeout, completed_before, errors_before)
                    break
                except TimeoutError:
                    if attempt >= retry:
                        raise
                    attempt += 1
                    self._log(f"Retry shot {i + 1}/{shots} (attempt {attempt}/{retry})")
                    time.sleep(retry_delay)
            if interval > 0 and i < shots - 1:
                time.sleep(interval)
        return self._transfers.saved_paths_snapshot()

    def capture_async(
        self,
        shots: int = 1,
        *,
        interval: float = 0.0,
        filename: Optional[str] = None,
    ) -> Dict[str, int]:
        """Trigger capture and return immediately.

        Returns a marker dictionary for `wait_for_downloads()`:
            {"marker": <completed_before>, "expected": <shots>}
        """
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        if filename is not None:
            if shots != 1:
                raise ValueError("filename can be used only when shots=1")
            self._next_filename = filename
        marker = self._transfers.marker()
        for i in range(max(1, shots)):
            self._log(f"Trigger async shot {i + 1}/{shots}")
            self._wait_for_inflight_slot()
            edsdk.SendCommand(self._cam, CameraCommand.TakePicture, 0)
            if interval > 0 and i < shots - 1:
                time.sleep(interval)
        return {"marker": marker, "expected": max(1, shots)}

    def capture_burst_async(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        duration: Optional[float] = None,
        drive_mode: Union[str, int] = DriveMode.LowSpeedContinuous,
        apply_drive_mode: bool = True,
        poll_interval: float = 0.005,
    ) -> Dict[str, int]:
        """Trigger continuous burst capture and return immediately.

        Returns a marker dictionary for `wait_for_downloads()`:
            {"marker": <completed_before>, "expected": <queued_during_burst>}
        """
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be > 0 when specified")
        target = max(1, int(shots))
        burst_start = time.time()
        if duration is not None:
            # Keep timeout as safety guard but never shorter than requested duration.
            deadline = burst_start + max(float(timeout), float(duration) + 1.0)
        else:
            deadline = burst_start + float(timeout)
        if apply_drive_mode:
            self.set_properties(
                drive_mode=drive_mode,
                validate=False,
                tolerate_not_supported=True,
            )
        marker, queued_before, errors_before = self._transfers.snapshot_for_burst()
        shutter_pressed = False
        queued_now = queued_before
        try:
            self._wait_for_inflight_slot(timeout)
            edsdk.SendCommand(
                self._cam,
                CameraCommand.PressShutterButton,
                int(ShutterButton.Completely),
            )
            shutter_pressed = True
            while True:
                _pump_messages_once()
                queued_now, inflight = self._transfers.burst_progress()
                latest_error = self._transfers.has_new_error(errors_before)
                if latest_error is not None:
                    raise RuntimeError(f"Image download failed: {latest_error}")
                if duration is not None:
                    if (time.time() - burst_start) >= duration:
                        break
                elif queued_now - queued_before >= target:
                    break
                if time.time() >= deadline:
                    raise TimeoutError("Timed out waiting for burst transfer events")
                # If inflight cap is reached during burst, pause shutter until queue drains.
                if self._max_inflight is not None and inflight >= self._max_inflight:
                    if shutter_pressed:
                        edsdk.SendCommand(
                            self._cam,
                            CameraCommand.PressShutterButton,
                            int(ShutterButton.OFF),
                        )
                        shutter_pressed = False
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for burst transfer events")
                    self._wait_for_inflight_slot(remaining)
                    edsdk.SendCommand(
                        self._cam,
                        CameraCommand.PressShutterButton,
                        int(ShutterButton.Completely),
                    )
                    shutter_pressed = True
                time.sleep(max(0.001, poll_interval))
        finally:
            try:
                edsdk.SendCommand(
                    self._cam,
                    CameraCommand.PressShutterButton,
                    int(ShutterButton.OFF),
                )
            except Exception:
                pass
        if duration is not None:
            # After releasing the shutter, transfer events can still be queued for a short
            # period while the camera drains its internal burst buffer. Wait for queue
            # growth to go quiet before finalizing the expected download count.
            final_queued = queued_now
            settle_quiet_period = max(0.1, poll_interval * 4.0)
            settle_timeout = max(0.5, min(3.0, float(timeout)))
            settle_deadline = time.time() + settle_timeout
            last_queue_change = time.time()
            while time.time() < settle_deadline:
                _pump_messages_once()
                latest_queued, _inflight = self._transfers.burst_progress()
                if latest_queued > final_queued:
                    final_queued = latest_queued
                    last_queue_change = time.time()
                elif (time.time() - last_queue_change) >= settle_quiet_period:
                    break
                time.sleep(max(0.001, poll_interval))
            expected = max(0, final_queued - queued_before)
        else:
            expected = target
        return {"marker": marker, "expected": max(0, expected)}

    def capture_burst(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        duration: Optional[float] = None,
        download_timeout: Optional[float] = None,
        drive_mode: Union[str, int] = DriveMode.LowSpeedContinuous,
        apply_drive_mode: bool = True,
        poll_interval: float = 0.005,
    ) -> List[str]:
        """Capture burst and wait for files to be downloaded."""
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        self._transfers.reset_paths_and_errors()
        expected = max(1, int(shots))
        ticket = self.capture_burst_async(
            shots=expected,
            timeout=timeout,
            duration=duration,
            drive_mode=drive_mode,
            apply_drive_mode=apply_drive_mode,
            poll_interval=poll_interval,
        )
        wait_timeout = (
            float(download_timeout)
            if download_timeout is not None
            else max(30.0, float(expected) * 5.0)
        )
        return self.wait_for_downloads(
            expected=ticket["expected"],
            timeout=wait_timeout,
            marker=ticket["marker"],
        )

    def wait_for_downloads(
        self,
        expected: int,
        timeout: float = 30.0,
        *,
        marker: Optional[int] = None,
    ) -> List[str]:
        """Wait until expected number of downloads complete after marker."""
        return self._transfers.wait_for_downloads(
            _pump_messages_once,
            expected=expected,
            timeout=timeout,
            marker=marker,
        )

    def drain_downloads(self) -> List[str]:
        """Return completed download paths since last drain."""
        return self._transfers.drain_downloads()

    def _wait_for_transfer(
        self, timeout: float, completed_before: int, errors_before: int
    ) -> None:
        self._transfers.wait_for_transfer(
            _pump_messages_once,
            timeout=timeout,
            completed_before=completed_before,
            errors_before=errors_before,
        )

    # ---------- Capture to memory ----------
    def capture_bytes(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        interval: float = 0.0,
        retry: int = 0,
        retry_delay: float = 0.3,
        keep_files: bool = False,
    ) -> List[bytes]:
        """Capture and return image bytes in memory.
        Optionally keeps or removes the saved files from disk (default: remove).
        """
        paths = self.capture(
            shots=shots,
            timeout=timeout,
            interval=interval,
            retry=retry,
            retry_delay=retry_delay,
        )
        data_list: List[bytes] = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    data_list.append(f.read())
            finally:
                if not keep_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        return data_list

    # ---------- Live View ----------
    def start_live_view(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        # Enable LV to PC
        edsdk.SetPropertyData(self._cam, PropID.Evf_Mode, 0, int(1))
        edsdk.SetPropertyData(
            self._cam, PropID.Evf_OutputDevice, 0, int(EvfOutputDevice.PC)
        )
        self._live_view_on = True
        self._log("Live view started")

    def stop_live_view(self) -> None:
        if self._cam is None:
            return
        try:
            edsdk.SetPropertyData(
                self._cam, PropID.Evf_OutputDevice, 0, int(EvfOutputDevice.TFT)
            )
            edsdk.SetPropertyData(self._cam, PropID.Evf_Mode, 0, int(0))
        except Exception:
            pass
        self._live_view_on = False
        self._log("Live view stopped")

    def grab_live_view_frame(
        self, save_path: Optional[str] = None, include_metadata: bool = False
    ) -> LiveViewFrame:
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        if not self._live_view_on:
            self.start_live_view()
            # give camera a brief moment to deliver first frame
            time.sleep(0.1)
        # Retry loop for transient OBJECT_NOTREADY / DEVICE_BUSY conditions
        MAX_ATTEMPTS = 10
        RETRY_DELAY = 0.07  # ~70ms between attempts
        ERR_OBJECT_NOT_READY = 0x0000A102
        ERR_DEVICE_BUSY = 0x00000081
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    out_stream = edsdk.CreateFileStream(
                        save_path, FileCreateDisposition.CreateAlways, Access.ReadWrite
                    )
                    evf_image = edsdk.CreateEvfImageRef(out_stream)
                    edsdk.DownloadEvfImage(self._cam, evf_image)
                    meta_data: Dict[str, object] = {}
                    if include_metadata:
                        meta_data = self._get_live_view_metadata(evf_image)
                    # too many logs when we stream the live view to ffmpeg
                    #self._log(
                    #    f"Live view saved: {save_path} (attempt {attempt}/{MAX_ATTEMPTS})"
                    #)
                    if include_metadata:
                        return save_path, meta_data
                    return save_path
                # Fallback: save to temp file and read bytes
                tmp_path = os.path.join(self.save_dir, f"evf_{uuid.uuid4().hex}.jpg")
                out_stream = edsdk.CreateFileStream(
                    tmp_path, FileCreateDisposition.CreateAlways, Access.ReadWrite
                )
                evf_image = edsdk.CreateEvfImageRef(out_stream)
                edsdk.DownloadEvfImage(self._cam, evf_image)
                meta_data = {}
                if include_metadata:
                    meta_data = self._get_live_view_metadata(evf_image)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                self._log(
                    f"Live view grabbed: {len(data)} bytes (attempt {attempt}/{MAX_ATTEMPTS})"
                )
                if include_metadata:
                    return data, meta_data
                return data
            except Exception as e:  # Catch SDK error
                code = getattr(e, "code", None)
                msg = str(e)
                # Detect transient errors
                is_transient = False
                if code in (ERR_OBJECT_NOT_READY, ERR_DEVICE_BUSY):
                    is_transient = True
                elif "OBJECT_NOTREADY" in msg or "DEVICE_BUSY" in msg:
                    is_transient = True
                if not is_transient or attempt >= MAX_ATTEMPTS:
                    last_exc = e
                    break
                # Backoff and allow Windows message pump to progress
                self._log(
                    f"Live view retry {attempt}/{MAX_ATTEMPTS} after transient error: {msg}"
                )
                _pump_messages_once()
                time.sleep(RETRY_DELAY)
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected live view failure without exception")

    def grab_live_view_pil(self) -> "Image.Image":
        """Grab one live-view frame and return as PIL Image (requires Pillow)."""
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Pillow (PIL) is required for grab_live_view_pil()"
            ) from e
        data = self.grab_live_view_frame()
        if isinstance(data, str):
            with open(data, "rb") as f:
                raw = f.read()
            img = Image.open(io.BytesIO(raw))
        else:
            img = Image.open(io.BytesIO(data))
        img.load()
        return img

    def grab_live_view_numpy(self) -> "np.ndarray":
        """Grab one live-view frame and return as numpy array (requires numpy)."""
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("numpy is required for grab_live_view_numpy()") from e
        pil_img = self.grab_live_view_pil()
        return np.array(pil_img)

    # ---------- asyncio event queue ----------
    def enable_async(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> asyncio.Queue:
        """Enable async event queue; returns asyncio.Queue for events."""
        if loop is None:
            loop = asyncio.get_event_loop()
        self._async_loop = loop
        self._async_queue = asyncio.Queue()
        self._log("Async event queue enabled")
        return self._async_queue

    def disable_async(self) -> None:
        self._async_queue = None
        self._async_loop = None
        self._async_pumping = False
        self._log("Async event queue disabled")

    async def pump_events(self, interval: float = 0.01) -> None:
        """Run message pumping periodically in asyncio task (Windows required)."""
        self._async_pumping = True
        try:
            while self._async_pumping:
                _pump_messages_once()
                await asyncio.sleep(interval)
        finally:
            self._async_pumping = False

    def _enqueue_async_event(self, evt: Dict[str, Union[str, int]]) -> None:
        if self._async_queue is None or self._async_loop is None:
            return
        try:
            self._async_loop.call_soon_threadsafe(self._async_queue.put_nowait, evt)
        except Exception:
            pass

    # ---------- Helpers ----------
    def _safe_get_property(self, pid: PropID) -> int:
        """Return property value or -1 if unsupported (to avoid raising)."""
        try:
            return int(edsdk.GetPropertyData(self._cam, pid, 0))  # type: ignore[arg-type]
        except Exception:
            return -1

    def _get_live_view_metadata(self, evf_image: EdsObject) -> Dict[str, object]:
        """Extract extra EVF metadata from the EVF image ref."""
        meta: Dict[str, object] = {}
        evf_props = [
            ("evf_zoom", PropID.Evf_Zoom),
            ("evf_zoom_position", PropID.Evf_ZoomPosition),
            ("evf_zoom_rect", PropID.Evf_ZoomRect),
            ("evf_image_position", PropID.Evf_ImagePosition),
            ("evf_image_clip_rect", PropID.Evf_ImageClipRect),
            ("evf_coordinate_system", PropID.Evf_CoordinateSystem),
            ("evf_histogram_status", PropID.Evf_HistogramStatus),
            ("evf_histogram_y", PropID.Evf_HistogramY),
            ("evf_histogram_r", PropID.Evf_HistogramR),
            ("evf_histogram_g", PropID.Evf_HistogramG),
            ("evf_histogram_b", PropID.Evf_HistogramB),
            ("evf_visible_rect", PropID.Evf_VisibleRect),
        ]
        for key, pid in evf_props:
            try:
                meta[key] = edsdk.GetPropertyData(evf_image, pid, 0)  # type: ignore[arg-type]
            except Exception:
                self._log(f"Error getting live view metadata: {key}")
        # Some models expose focus info on the camera rather than the EVF image.
        if self._cam is not None:
            try:
                meta["focus_info"] = edsdk.GetPropertyData(
                    self._cam, PropID.FocusInfo, 0
                )
            except Exception:
                self._log("Error getting focus info from camera")
        return meta

def classify_error(exc: Exception) -> Dict[str, Union[int, str, None]]:
    """Return a structured error info for EdsError exceptions.
    Includes SDK error code and human-readable message from edsdk_utils.
    """
    try:
        if isinstance(exc, getattr(edsdk, "EdsError", Exception)):
            code = getattr(exc, "code", None)
            return {
                "code": int(code) if code is not None else None,
                "message": str(exc),
            }
    except Exception:
        pass
    return {"message": str(exc)}
