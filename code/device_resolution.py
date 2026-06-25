# device_resolution.py

import re
from pathlib import Path

import yaml
from serial.tools import list_ports


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
    match = re.search(r"\bSER=([^ ]+)", hwid)
    return match.group(1) if match else None


def _find_project_root() -> Path:
    """
    Finds the project root by walking upwards until it finds .git or config/.
    Works whether this file is in the repo root or in a src/scripts folder.
    """
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists() or (parent / "config").exists():
            return parent

    return current.parent


def load_device_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = _find_project_root() / "config" / "devices.yml"

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


def resolve_device_port(device_key: str, config_path: str | Path | None = None) -> str:
    """
    Resolves imu1 / imu2 to the current macOS device path, e.g.
    /dev/cu.usbmodem2101, by matching the configured hardware serial number.
    """
    config = load_device_config(config_path)

    if device_key not in config:
        raise KeyError(f"Missing device key '{device_key}' in devices.yml")

    spec = config[device_key]

    # Optional emergency fallback:
    # Allows config/devices.yml to contain `port: /dev/cu.usbmodem2101`,
    # but serial_number is strongly preferred.
    configured_port = spec.get("port")
    if configured_port:
        return configured_port

    target_serial = _normalize_identifier(spec.get("serial_number"))
    target_location = spec.get("location")

    ports = available_serial_ports()

    if target_serial:
        for port in ports:
            port_serial = _normalize_identifier(port["serial_number"])
            if port_serial == target_serial:
                return port["device"]

    if target_location:
        for port in ports:
            if port["location"] == target_location:
                return port["device"]

    print_available_serial_ports()

    raise RuntimeError(
        f"Could not resolve port for '{device_key}'.\n"
        f"Expected serial_number={spec.get('serial_number')!r}, "
        f"location={spec.get('location')!r}.\n"
        "Check that the XIAO is connected and that config/devices.yml contains the correct serial number."
    )