# src/data_fusion_project/control/shortcuts.py
"""
Keyboard shortcut backends for the PowerPoint control interface.

A *backend* is the component that actually turns a shortcut string (e.g.
``"ctrl+shift+f5"``) into key presses on the host operating system. Backends are
pluggable so the control logic stays independent of the concrete key-sending
library:

- :class:`DryRunBackend`    -> logs the shortcut without sending anything (safe, testable,
                               works on headless machines and without extra dependencies).
- :class:`PyAutoGuiBackend` -> sends real key presses via the optional ``pyautogui`` package.

``pyautogui`` is imported lazily inside :class:`PyAutoGuiBackend` so that importing this
module never requires a display or the optional dependency.

Usage Example:
    from data_fusion_project.control.shortcuts import PyAutoGuiBackend
    backend = PyAutoGuiBackend()
    backend.send("ctrl+shift+f5")
"""


# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


# ======================================================================================================================
# constants
# ======================================================================================================================
# Recognized modifier tokens (canonical, lower-case) used in shortcut strings.
MODIFIERS: frozenset[str] = frozenset({"ctrl", "alt", "shift", "win"})

# Aliases mapping user-friendly tokens to the canonical token names understood by the
# backends (and by pyautogui). Keys are always lower-case.
_KEY_ALIASES: dict[str, str] = {
    # modifiers
    "control": "ctrl",
    "cmd": "win",
    "command": "win",
    "super": "win",
    "meta": "win",
    "windows": "win",
    "option": "alt",
    "opt": "alt",
    # navigation / control keys
    "escape": "esc",
    "return": "enter",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "page_up": "pageup",
    "page_down": "pagedown",
    "spacebar": "space",
    " ": "space",
    "bksp": "backspace",
    # arrow aliases
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "leftarrow": "left",
    "rightarrow": "right",
    "uparrow": "up",
    "downarrow": "down",
}

# Pretty labels for human-readable display of a shortcut (e.g. in CLI menus / logs).
_PRETTY_LABELS: dict[str, str] = {
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
    "esc": "Esc",
    "enter": "Enter",
    "space": "Space",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "Page Up",
    "pagedown": "Page Down",
    "left": "←",
    "right": "→",
    "up": "↑",
    "down": "↓",
}


# ======================================================================================================================
# shortcut parsing helpers
# ======================================================================================================================
def normalize_shortcut(shortcut: str) -> List[str]:
    """
    Parse a shortcut string into a list of canonical key tokens.

    Tokens are separated by '+', trimmed, lower-cased and resolved through the alias
    table. The result preserves the original order so modifiers are pressed before the
    final key (which is what ``pyautogui.hotkey`` expects).

    :param: shortcut (str): shortcut string, e.g. ``"Ctrl + Shift + F5"`` or ``"right"``.
    :return: tokens (List[str]): canonical tokens, e.g. ``["ctrl", "shift", "f5"]``.
    :raises: ValueError: if the string is empty or contains no usable tokens.
    """
    if shortcut is None or not str(shortcut).strip():
        raise ValueError("Shortcut string is empty.")

    raw_tokens = [t.strip().lower() for t in str(shortcut).split("+") if t.strip()]
    if not raw_tokens:
        raise ValueError(f"Shortcut string '{shortcut}' contains no usable key tokens.")

    tokens = [_KEY_ALIASES.get(token, token) for token in raw_tokens]
    if all(token in MODIFIERS for token in tokens):
        raise ValueError(f"Shortcut '{shortcut}' must contain at least one non-modifier key (got only modifiers).")

    return tokens



def format_shortcut(shortcut: str) -> str:
    """
    Render a shortcut string as a human-readable label, e.g. ``"Ctrl + Shift + F5"``.

    :param: shortcut (str): shortcut string to format.
    :return: label (str): pretty, display-friendly representation.
    """
    parts: list[str] = []
    for token in normalize_shortcut(shortcut):
        if token in _PRETTY_LABELS:
            parts.append(_PRETTY_LABELS[token])
        elif len(token) == 1:
            parts.append(token.upper())
        elif token.startswith("f") and token[1:].isdigit():
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return " + ".join(parts)



# ======================================================================================================================
# backends
# ======================================================================================================================
class KeyboardBackend(ABC):
    """
    Abstract base class for shortcut backends.

    A backend only needs to translate a normalized shortcut into actual (or simulated)
    key presses. The control layer never talks to a key-sending library directly.
    """

    name: str = "abstract"

    @abstractmethod
    def send(self, shortcut: str) -> None:
        """
        Send (or simulate) the given shortcut.

        :param: shortcut (str): shortcut string, e.g. ``"ctrl+shift+f5"``.
        :raises: ValueError: if the shortcut cannot be parsed.
        """
        raise NotImplementedError



class DryRunBackend(KeyboardBackend):
    """
    Backend that only logs the shortcut it would send, without touching the keyboard.

    Ideal for tests, headless environments and dry runs. It still parses the shortcut,
    so malformed shortcuts are reported exactly as they would be by a real backend.
    """

    name = "dry-run"

    def send(self, shortcut: str) -> None:
        """
        Log the shortcut that would be sent.

        :param: shortcut (str): shortcut string to simulate.
        :raises: ValueError: if the shortcut cannot be parsed.
        """
        tokens = normalize_shortcut(shortcut)
        logger.info("[DRY-RUN] Would press: %s  (tokens=%s)", format_shortcut(shortcut), "+".join(tokens))



class PyAutoGuiBackend(KeyboardBackend):
    """
    Backend that sends real key presses to the currently focused window via ``pyautogui``.

    ``pyautogui`` is imported lazily so that importing this module has no hard dependency
    on it (and never requires a display). Install it with ``pip install pyautogui`` or via
    the project's optional extra: ``pip install -e .[control]``.

    :param: pause_s (float): delay pyautogui inserts after each key action (seconds).
    :param: failsafe (bool): if True, slamming the mouse into a screen corner aborts pyautogui.
    :raises: ImportError: if ``pyautogui`` is not installed.
    """

    name = "pyautogui"

    def __init__(self, pause_s: float = 0.05, failsafe: bool = True) -> None:
        try:
            import pyautogui  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("pyautogui is required for PyAutoGuiBackend. Install it with 'pip install pyautogui' or 'pip install -e .[control]'.") from exc

        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = max(0.0, float(pause_s))
        self._pyautogui.FAILSAFE = bool(failsafe)
        logger.debug("Initialized PyAutoGuiBackend (pause=%.3fs, failsafe=%s).", pause_s, failsafe)


    def send(self, shortcut: str) -> None:
        """
        Press the key combination described by ``shortcut`` on the active window.

        Uses ``pyautogui.hotkey`` which presses the tokens in order and releases them in
        reverse order (correct behaviour for both single keys and modifier combos).

        :param: shortcut (str): shortcut string, e.g. ``"ctrl+shift+f5"``.
        :raises: ValueError: if the shortcut cannot be parsed.
        """
        tokens = normalize_shortcut(shortcut)
        logger.info("Sending shortcut via pyautogui: %s", format_shortcut(shortcut))
        self._pyautogui.hotkey(*tokens)
