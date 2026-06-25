# device_resolution.py

import re
from pathlib import Path

import yaml
from serial.tools import list_ports

from data_fusion_project.core.paths import DEVICES_CONFIG_FILE


def _normalize_identifier(value: str | None) -> str:
    """
    Normalizes IDs such as 'D0:CF:13:27:F8:CC' and 'D0CF1327F8CC'
    to the same comparable format.
    """
    if not value:
        return ""
    return re.sub(r"[^0-9A-Fa-f]", "", value).upper()


def _extract_serial_from_hwid(hwid: str | None) -> str | None:
    if not hwid:
        return None
    # pyserial-Format: "USB VID:PID=303A:1001 SER=XXXXX LOCATION=..."
    match = re.search(r"\bSER=([^ ]+)", hwid)
    if match:
        return match.group(1)
    # Windows-Gerätepfad-Format: "USB\VID_303A&PID_1001&MI_00\6&155ED617&0&0000"
    # → Seriennummer ist das letzte \-getrennte Segment
    parts = hwid.split("\\")
    if len(parts) >= 3:
        return parts[-1]
    return None


def load_device_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = DEVICES_CONFIG_FILE

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Device config not found: {config_path}\n"
            "Create it from config/devices.example.yml and add your board serial numbers."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def available_serial_ports() -> list[dict]:
    ports = []

    for port in sorted(list_ports.comports(), key=lambda p: p.device):
        serial_number = getattr(port, "serial_number", None) or _extract_serial_from_hwid(port.hwid)
        location = getattr(port, "location", None)

        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
                "serial_number": serial_number,
                "location": location,
            }
        )

    return ports


def print_available_serial_ports() -> None:
    print("\nAvailable serial ports:")
    for port in available_serial_ports():
        print(f"  device:        {port['device']}")
        print(f"  description:   {port['description']}")
        print(f"  serial_number: {port['serial_number']}")
        print(f"  location:      {port['location']}")
        print(f"  hwid:          {port['hwid']}")
        print()


def _target_serials(spec: dict) -> set[str]:
    """
    Liest serial_number aus dem Config-Eintrag — entweder als einzelner String
    oder als Liste (für verschiedene Betriebssysteme).

    Beispiel devices.yml:
        imu1:
          serial_number: "6&155ED617&0&0000"          # nur Windows
          # oder plattformübergreifend:
          serial_number:
            - "6&155ED617&0&0000"    # Windows
            - "E6632038A34E2B38"     # macOS
    """
    value = spec.get("serial_number")
    if value is None:
        return set()
    entries = value if isinstance(value, list) else [value]
    return {_normalize_identifier(str(v)) for v in entries if v}


def resolve_device_port(device_key: str, config_path: str | Path | None = None) -> str:
    """
    Löst imu1/imu2 zum aktuellen seriellen Port auf (z.B. COM10 unter Windows,
    /dev/cu.usbmodem2101 unter macOS) anhand der konfigurierten Seriennummer(n).
    """
    config = load_device_config(config_path)

    if device_key not in config:
        raise KeyError(f"Missing device key '{device_key}' in devices.yml")

    spec = config[device_key]

    # Direkter Port als Fallback (wird genutzt wenn serial_number nicht gesetzt)
    configured_port = spec.get("port")
    if configured_port:
        return configured_port

    targets = _target_serials(spec)
    target_location = spec.get("location")

    ports = available_serial_ports()

    if targets:
        for port in ports:
            if _normalize_identifier(port["serial_number"]) in targets:
                return port["device"]

    if target_location:
        for port in ports:
            if port["location"] == target_location:
                return port["device"]

    print_available_serial_ports()

    raise RuntimeError(
        f"Could not resolve port for '{device_key}'.\n"
        f"Expected serial_number={spec.get('serial_number')!r}, location={spec.get('location')!r}.\n"
        "Prüfe ob das Board verbunden ist und ob config/devices.yml die richtige\n"
        "Seriennummer für dieses Betriebssystem enthält.\n"
        "Tipp: python code/device_resolution.py  listet alle verbundenen Ports."
    )


if __name__ == "__main__":
    print_available_serial_ports()