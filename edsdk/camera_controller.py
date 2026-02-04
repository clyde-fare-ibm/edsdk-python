from __future__ import annotations

import os
import json
import io
import asyncio
import time
import uuid
import inspect
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TYPE_CHECKING,
    Type,
    runtime_checkable,
)

# Only imported for type checking to avoid runtime cost if deps not installed
if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image
    import numpy as np


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
from edsdk.constants.generic import CameraStatusCommand
from edsdk.constants.properties import (
    Av as AvTable,
    Tv as TvTable,
    ISOSpeedCamera,
    SaveTo,
    AEMode,
    MeteringMode,
    WhiteBalance,
    ImageQuality,
    DriveMode,
    EvfOutputDevice,
    PropID as _PropIDEnum,
    AFMode,
    EvfAFMode,
    FlashFiring,
    FlashTarget,
)


# Public callback / return type aliases (after imports to satisfy linters)
ObjectCallback = Callable[["ObjectEvent", "EdsObject"], int]
PropertyCallback = Callable[["PropertyEvent", "PropID", int], int]
LiveViewData = Union[bytes, str]


@runtime_checkable
class RawProcessor(Protocol):
    """Protocol for RAW image development callbacks.

    Implement a function or callable object with this signature to process
    RAW image data from the camera.

    Args:
        raw_bytes: Binary data of the RAW file (e.g., CR2, CR3).

    Returns:
        Developed image as PIL.Image.Image (RGB mode recommended).

    Example using rawpy:
        ```python
        import io
        import rawpy
        from PIL import Image

        def develop_raw(raw_bytes: bytes) -> Image.Image:
            with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                )
            return Image.fromarray(rgb)

        # Usage
        with CameraController() as cam:
            images = cam.capture_pil(raw_processor=develop_raw)
        ```

    Example with custom settings:
        ```python
        class MyRawProcessor:
            def __init__(self, gamma: float = 2.2):
                self.gamma = gamma

            def __call__(self, raw_bytes: bytes) -> Image.Image:
                import io
                import rawpy
                import numpy as np
                from PIL import Image

                with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
                    rgb = raw.postprocess(gamma=(1, 1), output_bps=16)

                # Apply custom gamma
                rgb_float = rgb.astype(np.float32) / 65535.0
                rgb_gamma = np.power(rgb_float, 1.0 / self.gamma)
                rgb_8bit = (rgb_gamma * 255).clip(0, 255).astype(np.uint8)

                return Image.fromarray(rgb_8bit)

        # Usage
        processor = MyRawProcessor(gamma=2.4)
        images = cam.capture_pil(raw_processor=processor)
        ```
    """

    def __call__(self, raw_bytes: bytes) -> "Image.Image":
        """Process RAW bytes and return developed PIL Image."""
        ...


def _validate_raw_processor(processor: object) -> None:
    """Validate that processor conforms to RawProcessor protocol.

    Args:
        processor: Object to validate.

    Raises:
        TypeError: If processor is not callable or has wrong signature.
    """
    if not callable(processor):
        raise TypeError(
            "raw_processor must be callable.\n"
            "Expected signature: (raw_bytes: bytes) -> PIL.Image.Image\n"
            "See RawProcessor docstring for implementation examples."
        )

    # Check signature if possible
    try:
        sig = inspect.signature(processor)
        params = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        # Should have exactly 1 required positional parameter
        if len(params) != 1:
            raise TypeError(
                f"raw_processor must accept exactly 1 required argument (raw_bytes), "
                f"but got {len(params)} required argument(s).\n"
                "Expected signature: (raw_bytes: bytes) -> PIL.Image.Image"
            )
    except (ValueError, TypeError):
        # Some built-in callables don't support signature inspection
        pass


# RAW type codes in upper 16 bits of ImageQuality
# 0x0064: RAW (CR2/CR3), 0x0164: MRAW (SRAW1), 0x0264: SRAW (SRAW2), 0x0063: CRAW
_RAW_QUALITY_PREFIXES = frozenset({0x0064, 0x0164, 0x0264, 0x0063})

# File extensions for RAW images (fallback check)
RAW_EXTENSIONS = frozenset(
    {
        ".cr2",
        ".cr3",  # Canon
        ".nef",  # Nikon
        ".arw",  # Sony
        ".dng",  # Adobe DNG
        ".orf",  # Olympus
        ".rw2",  # Panasonic
        ".pef",  # Pentax
        ".srw",  # Samsung
        ".raf",  # Fujifilm
    }
)


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


def _reverse_lookup(table: Dict[int, str]) -> Dict[str, int]:
    # Normalize keys to a canonical string for robust matching
    rev: Dict[str, int] = {}
    for k, v in table.items():
        key = str(v).strip().lower()
        rev[key] = k
        # For Av allow prefix like f/5.6
        if (
            key.replace(" ", "").replace("(1/3)", "")
            and "/" not in key
            and "bulb" not in key
        ):
            try:
                fnum = float(key)
                rev[f"f/{fnum:g}"] = k
                rev[f"{fnum:g}"] = k
            except Exception:
                pass
        # For Tv allow variants like 0.5s, 1/125s, integers without quotes
        if any(ch in key for ch in ['"', "/"]) or key.isdigit():
            cleaned = key.replace('"', "s").replace(" ", "")
            rev[cleaned] = k
    return rev


_AV_STR_TO_CODE = _reverse_lookup(AvTable)
_TV_STR_TO_CODE = _reverse_lookup(TvTable)


def _parse_av(value: Union[str, float, int]) -> int:
    if isinstance(value, (int, float)):
        key = f"{float(value):g}"
        if key in _AV_STR_TO_CODE:
            return _AV_STR_TO_CODE[key]
        key2 = f"f/{float(value):g}"
        if key2 in _AV_STR_TO_CODE:
            return _AV_STR_TO_CODE[key2]
        raise ValueError(f"Unsupported Av value: {value}")
    key = str(value).strip().lower()
    key = key.replace("f ", "f/") if key.startswith("f ") else key
    if key.startswith("f/") and key[2:] in _AV_STR_TO_CODE:
        return _AV_STR_TO_CODE[key]
    if key in _AV_STR_TO_CODE:
        return _AV_STR_TO_CODE[key]
    # Try removing trailing 'f' or spaces
    key_alt = key.rstrip("f ")
    if key_alt in _AV_STR_TO_CODE:
        return _AV_STR_TO_CODE[key_alt]
    raise ValueError(f"Unsupported Av value: {value}")


def _parse_tv(value: Union[str, float, int]) -> int:
    # Accept formats: "1/125", 0.5, "0.5", 2 (seconds), "bulb"
    if isinstance(value, (int, float)):
        seconds = float(value)
        # Build candidate keys
        candidates = [
            f"{seconds:g}s",
            f"{int(seconds)}",
            f"{int(seconds)}s",
        ]
        for c in candidates:
            c = c.lower()
            if c in _TV_STR_TO_CODE:
                return _TV_STR_TO_CODE[c]
        # Try to find nearest by computing numeric seconds of table
        best: Optional[Tuple[int, float]] = None
        for code, disp in TvTable.items():
            try:
                s = _tv_display_to_seconds(disp)
            except Exception:
                continue
            err = abs(s - seconds)
            if best is None or err < best[1]:
                best = (code, err)
        if best is not None and best[1] < 1e-6:  # exact or very close
            return best[0]
        raise ValueError(f"Unsupported Tv value: {value}")
    key = str(value).strip().lower()
    if key == "bulb":
        return _TV_STR_TO_CODE.get("bulb", 0x0C)
    # Normalize variants like 1/125s, 0.5s, 2s, 2
    key = key.replace('"', "s")
    if key.endswith("sec"):
        key = key[:-3] + "s"
    if key in _TV_STR_TO_CODE:
        return _TV_STR_TO_CODE[key]
    # Remove trailing 's'
    if key.endswith("s") and key[:-1] in _TV_STR_TO_CODE:
        return _TV_STR_TO_CODE[key[:-1]]
    raise ValueError(f"Unsupported Tv value: {value}")


def _tv_display_to_seconds(display: str) -> float:
    import re

    disp = str(display).strip()
    if disp.lower() == "bulb":
        raise ValueError("Bulb has no fixed seconds")
    # Canon style: 0"5 -> 0.5s, 3"2 -> 3.2s, 30" -> 30s
    if '"' in disp:
        m = re.fullmatch(r"(\d+)\"(\d)", disp)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
        # pure seconds like 30"
        if disp.endswith('"') and disp[:-1].isdigit():
            return float(disp[:-1])
    # Normalize a few patterns
    d = disp.replace('"', "s")
    if d.endswith("s"):
        # 0.5s, 3s, 10s
        return float(d[:-1])
    if "/" in d:
        num, den = d.split("/", 1)
        return float(num) / float(den)
    # plain number means seconds
    return float(d)


def _parse_iso(value: Union[str, int]) -> int:
    if isinstance(value, int):
        if value == 0:
            return int(ISOSpeedCamera.ISOAuto)
        name = f"ISO{value}"
        if hasattr(ISOSpeedCamera, name):
            return int(getattr(ISOSpeedCamera, name))
        raise ValueError(f"Unsupported ISO value: {value}")
    key = str(value).strip().lower()
    if key in ("auto", "isoauto"):
        return int(ISOSpeedCamera.ISOAuto)
    if key.startswith("iso"):
        tail = key[3:]
        if tail.isdigit():
            return _parse_iso(int(tail))
    if key.isdigit():
        return _parse_iso(int(key))
    raise ValueError(f"Unsupported ISO value: {value}")


def _image_quality_includes_raw(quality_code: int) -> bool:
    """Check if the ImageQuality setting includes RAW capture.

    Args:
        quality_code: The ImageQuality property value from camera.

    Returns:
        True if RAW (including CRAW/MRAW/SRAW) is part of the capture.
    """
    upper = (quality_code >> 16) & 0xFFFF
    return upper in _RAW_QUALITY_PREFIXES


def _image_quality_is_raw_only(quality_code: int) -> bool:
    """Check if the ImageQuality setting is RAW-only (no JPEG/HEIF).

    RAW-only values have 0xFF0F in lower 16 bits.
    """
    if not _image_quality_includes_raw(quality_code):
        return False
    lower = quality_code & 0xFFFF
    return lower == 0xFF0F


def _is_raw_file(path: str) -> bool:
    """Check if the file is a RAW image based on extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in RAW_EXTENSIONS


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
    ) -> None:
        self.index = index
        self.save_dir = save_dir
        self.save_to = save_to
        self.auto_capacity = auto_capacity
        self.verbose = verbose
        self.ui_locked = False
        self._log = logger or (print if verbose else (lambda *_args, **_kw: None))
        self._cam: Optional[EdsObject] = None
        self._saved_paths: List[str] = []
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
        # One-shot explicit filename (base name); if set, next capture uses this name
        self._next_filename: Optional[str] = None
        self._flash_ref: Optional[EdsObject] = None

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
        self._cam = cam

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
                time.sleep(0.1)
                self.prepare_flash()
            except Exception as e:
                self._log(f"Flash control unavailable: {e}")
      
        self._log("Camera session opened")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
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

            path = _save_directory_item(
                object_handle, self.save_dir, dst_basename=dst_name
            )
            self._saved_paths.append(path)
            self._enqueue_async_event(
                {
                    "kind": "object",
                    "event": getattr(ObjectEvent, "DirItemRequestTransfer").name,
                    "path": path,
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

    def prepare_flash(self) -> None:
        """Configure flash settings via the flash settings object."""
        if self._cam is None:
            raise RuntimeError("Camera session not open")
        if not self.ui_locked:
            raise RuntimeError("UI must be locked to prepare flash")

        if self._flash_ref is None:
            self._flash_ref = edsdk.CreateFlashSettingRef(self._cam)

        try:
            edsdk.SetPropertyData(
                self._flash_ref,
                PropID.Flash_Target,
                0,
                int(self._flash_target),
            )
            edsdk.SetPropertyData(
                self._flash_ref,
                PropID.Flash_Firing,
                0,
                int(self._flash_firing),
            )
            time.sleep(1)
            
        except Exception as e:
            self._log(f"Error setting flash target: {e}")
            

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
            to_set.append((PropID.Av, _parse_av(av)))
        if tv is not None:
            to_set.append((PropID.Tv, _parse_tv(tv)))
        if iso is not None:
            to_set.append((PropID.ISOSpeed, _parse_iso(iso)))
        if ae_mode is not None:
            to_set.append((PropID.AEMode, _enum_code(AEMode, ae_mode)))
        if metering is not None:
            to_set.append((PropID.MeteringMode, _enum_code(MeteringMode, metering)))
        if white_balance is not None:
            to_set.append(
                (PropID.WhiteBalance, _enum_code(WhiteBalance, white_balance))
            )
        if image_quality is not None:
            to_set.append(
                (PropID.ImageQuality, _enum_code(ImageQuality, image_quality))
            )
        if drive_mode is not None:
            to_set.append((PropID.DriveMode, _enum_code(DriveMode, drive_mode)))
        # Manual focus convenience flag takes precedence over af_mode
        if manual_focus is True:
            to_set.append((PropID.AFMode, int(AFMode.ManualFocus)))
        elif af_mode is not None:
            to_set.append((PropID.AFMode, _enum_code(AFMode, af_mode)))
        if evf_af_mode is not None:
            to_set.append((PropID.Evf_AFMode, _enum_code(EvfAFMode, evf_af_mode)))

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
            "ISO": _iso_code_to_string(int(iso_code)),
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
                _iso_code_to_string(int(c))
                for c in self._get_supported_codes(PropID.ISOSpeed)
            ],
            "AEMode": _enum_supported_names(
                PropID.AEMode, AEMode, self._get_supported_codes(PropID.AEMode)
            ),
            "MeteringMode": _enum_supported_names(
                PropID.MeteringMode,
                MeteringMode,
                self._get_supported_codes(PropID.MeteringMode),
            ),
            "WhiteBalance": _enum_supported_names(
                PropID.WhiteBalance,
                WhiteBalance,
                self._get_supported_codes(PropID.WhiteBalance),
            ),
            "ImageQuality": _enum_supported_names(
                PropID.ImageQuality,
                ImageQuality,
                self._get_supported_codes(PropID.ImageQuality),
            ),
            "DriveMode": _enum_supported_names(
                PropID.DriveMode, DriveMode, self._get_supported_codes(PropID.DriveMode)
            ),
            "AFMode": _enum_supported_names(
                PropID.AFMode, AFMode, self._get_supported_codes(PropID.AFMode)
            ),
            "EvfAFMode": _enum_supported_names(
                PropID.Evf_AFMode,
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

    def includes_raw(self) -> bool:
        """Check if current ImageQuality setting includes RAW capture."""
        return _image_quality_includes_raw(self.get_image_quality_code())

    def is_raw_only(self) -> bool:
        """Check if current ImageQuality setting is RAW-only (no JPEG/HEIF)."""
        return _image_quality_is_raw_only(self.get_image_quality_code())

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
        self._saved_paths.clear()
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
                    edsdk.SendCommand(self._cam, CameraCommand.TakePicture, 0)
                    self._wait_for_transfer(timeout)
                    break
                except TimeoutError:
                    if attempt >= retry:
                        raise
                    attempt += 1
                    self._log(f"Retry shot {i + 1}/{shots} (attempt {attempt}/{retry})")
                    time.sleep(retry_delay)
            if interval > 0 and i < shots - 1:
                time.sleep(interval)
        return list(self._saved_paths)

    def _wait_for_transfer(self, timeout: float) -> None:
        deadline = time.time() + timeout
        already = len(self._saved_paths)
        while time.time() < deadline:
            time.sleep(0.01)
            _pump_messages_once()
            if len(self._saved_paths) > already:
                return
        raise TimeoutError("Timed out waiting for image transfer event")

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

    def capture_pil(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        interval: float = 0.0,
        retry: int = 0,
        retry_delay: float = 0.3,
        keep_files: bool = False,
        raw_processor: Optional[RawProcessor] = None,
    ) -> List["Image.Image"]:
        """Capture and return a list of PIL Images (requires Pillow).

        Args:
            shots: Number of shots to capture.
            timeout: Timeout in seconds for each shot transfer.
            interval: Interval in seconds between shots.
            retry: Number of retries on timeout.
            retry_delay: Delay in seconds between retries.
            keep_files: If True, keep captured files on disk.
            raw_processor: Callback to develop RAW images.
                           Must conform to RawProcessor protocol:
                           (raw_bytes: bytes) -> PIL.Image.Image
                           Required if camera is set to RAW or RAW+JPEG mode.
                           See RawProcessor docstring for examples.

        Returns:
            List of PIL Image objects.

        Raises:
            ValueError: If RAW capture is enabled but no raw_processor provided.
            TypeError: If raw_processor has invalid signature.
            RuntimeError: If Pillow is not installed or camera session not open.
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError("Pillow (PIL) is required for capture_pil()") from e

        # Check current ImageQuality setting before capture
        if self._cam is None:
            raise RuntimeError("Camera session not open")

        quality_code = self.get_image_quality_code()
        includes_raw = _image_quality_includes_raw(quality_code)

        # Validate raw_processor if provided
        if raw_processor is not None:
            _validate_raw_processor(raw_processor)

        if includes_raw and raw_processor is None:
            # Get human-readable name for error message
            quality_name = "Unknown"
            for name, member in ImageQuality.__members__.items():
                if int(member) == quality_code:
                    quality_name = name
                    break
            raise ValueError(
                f"Camera ImageQuality is set to '{quality_name}' which includes RAW, "
                "but no raw_processor was provided.\n"
                "Either pass a raw_processor callback or change camera to JPEG-only mode.\n"
                "See RawProcessor docstring for implementation examples."
            )

        # Capture files
        paths = self.capture(
            shots=shots,
            timeout=timeout,
            interval=interval,
            retry=retry,
            retry_delay=retry_delay,
        )

        images: List["Image.Image"] = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    raw_bytes = f.read()

                # Determine if this specific file is RAW by extension
                if _is_raw_file(p):
                    # RAW file: use processor
                    if raw_processor is not None:
                        self._log(
                            f"Processing RAW file with raw_processor: {os.path.basename(p)}"
                        )
                        img = raw_processor(raw_bytes)
                        # Validate return type
                        if not isinstance(img, Image.Image):
                            raise TypeError(
                                f"raw_processor must return PIL.Image.Image, "
                                f"but got {type(img).__name__}.\n"
                                "See RawProcessor docstring for correct implementation."
                            )
                    else:
                        # This shouldn't happen if we checked above, but safety fallback
                        raise RuntimeError(
                            f"RAW file detected ({os.path.basename(p)}), "
                            "but no raw_processor provided."
                        )
                else:
                    # JPEG/HEIF: let Pillow handle it
                    self._log(f"Loading image with Pillow: {os.path.basename(p)}")
                    img = Image.open(io.BytesIO(raw_bytes))
                    img.load()  # fully load to detach from BytesIO

                images.append(img)
            except Exception as e:
                # Re-raise with more context
                if _is_raw_file(p) and raw_processor is None:
                    raise RuntimeError(
                        f"Failed to process {os.path.basename(p)}. "
                        "If this is a RAW file, provide a raw_processor callback."
                    ) from e
                raise
            finally:
                if not keep_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        return images

    def capture_numpy(
        self,
        shots: int = 1,
        timeout: float = 5.0,
        *,
        interval: float = 0.0,
        retry: int = 0,
        retry_delay: float = 0.3,
        keep_files: bool = False,
        raw_processor: Optional[RawProcessor] = None,
    ) -> List["np.ndarray"]:
        """Capture and return a list of numpy arrays (requires numpy).

        Args:
            shots: Number of shots to capture.
            timeout: Timeout in seconds for each shot transfer.
            interval: Interval in seconds between shots.
            retry: Number of retries on timeout.
            retry_delay: Delay in seconds between retries.
            keep_files: If True, keep captured files on disk.
            raw_processor: Callback to develop RAW images.
                           Must conform to RawProcessor protocol:
                           (raw_bytes: bytes) -> PIL.Image.Image
                           Required if camera is set to RAW or RAW+JPEG mode.
                           See RawProcessor docstring for examples.

        Returns:
            List of numpy arrays (RGB format).

        Raises:
            ValueError: If RAW capture is enabled but no raw_processor provided.
            TypeError: If raw_processor has invalid signature.
            RuntimeError: If numpy is not installed or camera session not open.
        """
        try:
            import numpy as np  # type: ignore
        except Exception as e:
            raise RuntimeError("numpy is required for capture_numpy()") from e
        pil_images = self.capture_pil(
            shots=shots,
            timeout=timeout,
            interval=interval,
            retry=retry,
            retry_delay=retry_delay,
            keep_files=keep_files,
            raw_processor=raw_processor,
        )
        return [np.array(im) for im in pil_images]

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

    def grab_live_view_frame(self, save_path: Optional[str] = None) -> LiveViewData:
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
                    # too many logs when we stream the live view to ffmpeg
                    #self._log(
                    #    f"Live view saved: {save_path} (attempt {attempt}/{MAX_ATTEMPTS})"
                    #)
                    return save_path
                # Fallback: save to temp file and read bytes
                tmp_path = os.path.join(self.save_dir, f"evf_{uuid.uuid4().hex}.jpg")
                out_stream = edsdk.CreateFileStream(
                    tmp_path, FileCreateDisposition.CreateAlways, Access.ReadWrite
                )
                evf_image = edsdk.CreateEvfImageRef(out_stream)
                edsdk.DownloadEvfImage(self._cam, evf_image)
                with open(tmp_path, "rb") as f:
                    data = f.read()
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                self._log(
                    f"Live view grabbed: {len(data)} bytes (attempt {attempt}/{MAX_ATTEMPTS})"
                )
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


def _iso_code_to_string(code: int) -> str:
    try:
        if code == int(ISOSpeedCamera.ISOAuto):
            return "Auto"
        for name in ISOSpeedCamera.__members__:
            if int(getattr(ISOSpeedCamera, name)) == code:
                return name.replace("ISO", "")
    except Exception:
        pass
    return str(code)


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


def _enum_code(enum_cls: Type[object], value: Union[str, int]) -> int:
    if isinstance(value, int):
        return int(value)
    key = str(value).strip()
    # Friendly aliases for some enums
    alias_key = key.lower().replace(" ", "").replace("-", "").replace("_", "")
    try:
        enum_name = enum_cls.__name__
    except Exception:
        enum_name = ""
    # MeteringMode aliases
    if enum_name == "MeteringMode":
        aliases = {
            "evaluative": "EvaluativeMetering",
            "spot": "PartialMetering",
            "partial": "PartialMetering",
            "centerweighted": "CenterWeightedAveragingMetering",
            "centerweightedaverage": "CenterWeightedAveragingMetering",
            "average": "CenterWeightedAveragingMetering",
        }
        if alias_key in aliases:
            key = aliases[alias_key]
    # Accept case-insensitive and some friendly aliases
    for name, member in enum_cls.__members__.items():
        if name.lower() == key.lower():
            return int(member)
    # Also accept numeric string
    if key.isdigit():
        return int(key)
    raise ValueError(f"Unsupported value '{value}' for {enum_cls.__name__}")


def _enum_supported_names(
    pid: _PropIDEnum, enum_cls: Type[object], codes: List[int]
) -> List[str]:
    names: List[str] = []
    if not codes:
        # if descriptors not available, return all enum names as hint
        return list(enum_cls.__members__.keys())
    for code in codes:
        matched = False
        for name, member in enum_cls.__members__.items():
            if int(member) == int(code):
                names.append(name)
                matched = True
                break
        if not matched:
            names.append(str(code))
    return names
