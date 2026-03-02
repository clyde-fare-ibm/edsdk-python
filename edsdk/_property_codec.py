from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

from edsdk.constants.properties import Av as AvTable
from edsdk.constants.properties import ISOSpeedCamera
from edsdk.constants.properties import MeteringMode
from edsdk.constants.properties import Tv as TvTable


def _reverse_lookup(table: Dict[int, str]) -> Dict[str, int]:
    rev: Dict[str, int] = {}
    for code, display in table.items():
        key = str(display).strip().lower()
        rev[key] = code
        # For Av allow prefix like f/5.6
        if key.replace(" ", "").replace("(1/3)", "") and "/" not in key and "bulb" not in key:
            try:
                fnum = float(key)
                rev[f"f/{fnum:g}"] = code
                rev[f"{fnum:g}"] = code
            except Exception:
                pass
        # For Tv allow variants like 0.5s, 1/125s, integers without quotes
        if any(ch in key for ch in ['"', "/"]) or key.isdigit():
            cleaned = key.replace('"', "s").replace(" ", "")
            rev[cleaned] = code
    return rev


_AV_STR_TO_CODE = _reverse_lookup(AvTable)
_TV_STR_TO_CODE = _reverse_lookup(TvTable)


def parse_av(value: Union[str, float, int]) -> int:
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
    key_alt = key.rstrip("f ")
    if key_alt in _AV_STR_TO_CODE:
        return _AV_STR_TO_CODE[key_alt]
    raise ValueError(f"Unsupported Av value: {value}")


def parse_tv(value: Union[str, float, int]) -> int:
    if isinstance(value, (int, float)):
        seconds = float(value)
        candidates = [f"{seconds:g}s", f"{int(seconds)}", f"{int(seconds)}s"]
        for candidate in candidates:
            key = candidate.lower()
            if key in _TV_STR_TO_CODE:
                return _TV_STR_TO_CODE[key]
        best: Optional[Tuple[int, float]] = None
        for code, disp in TvTable.items():
            try:
                seconds_val = tv_display_to_seconds(disp)
            except Exception:
                continue
            err = abs(seconds_val - seconds)
            if best is None or err < best[1]:
                best = (code, err)
        if best is not None and best[1] < 1e-6:
            return best[0]
        raise ValueError(f"Unsupported Tv value: {value}")

    key = str(value).strip().lower()
    if key == "bulb":
        return _TV_STR_TO_CODE.get("bulb", 0x0C)
    key = key.replace('"', "s")
    if key.endswith("sec"):
        key = key[:-3] + "s"
    if key in _TV_STR_TO_CODE:
        return _TV_STR_TO_CODE[key]
    if key.endswith("s") and key[:-1] in _TV_STR_TO_CODE:
        return _TV_STR_TO_CODE[key[:-1]]
    raise ValueError(f"Unsupported Tv value: {value}")


def tv_display_to_seconds(display: str) -> float:
    import re

    disp = str(display).strip()
    if disp.lower() == "bulb":
        raise ValueError("Bulb has no fixed seconds")
    if '"' in disp:
        decimal_match = re.fullmatch(r"(\d+)\"(\d)", disp)
        if decimal_match:
            return float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
        if disp.endswith('"') and disp[:-1].isdigit():
            return float(disp[:-1])
    normalized = disp.replace('"', "s")
    if normalized.endswith("s"):
        return float(normalized[:-1])
    if "/" in normalized:
        num, den = normalized.split("/", 1)
        return float(num) / float(den)
    return float(normalized)


def parse_iso(value: Union[str, int]) -> int:
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
            return parse_iso(int(tail))
    if key.isdigit():
        return parse_iso(int(key))
    raise ValueError(f"Unsupported ISO value: {value}")


def iso_code_to_string(code: int) -> str:
    try:
        if code == int(ISOSpeedCamera.ISOAuto):
            return "Auto"
        for name in ISOSpeedCamera.__members__:
            if int(getattr(ISOSpeedCamera, name)) == code:
                return name.replace("ISO", "")
    except Exception:
        pass
    return str(code)


def enum_code(enum_cls: Type[object], value: Union[str, int]) -> int:
    if isinstance(value, int):
        return int(value)
    key = str(value).strip()
    alias_key = key.lower().replace(" ", "").replace("-", "").replace("_", "")
    enum_name = getattr(enum_cls, "__name__", "")
    if enum_name == MeteringMode.__name__:
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
    for name, member in enum_cls.__members__.items():
        if name.lower() == key.lower():
            return int(member)
    if key.isdigit():
        return int(key)
    raise ValueError(f"Unsupported value '{value}' for {enum_cls.__name__}")


def enum_supported_names(enum_cls: Type[object], codes: List[int]) -> List[str]:
    if not codes:
        return list(enum_cls.__members__.keys())
    names: List[str] = []
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
