# src/data_fusion_project/control/config.py
"""
Configuration for the PowerPoint control interface.

The configuration has two independent, freely editable maps:

1. ``actions``           -> maps a semantic action name (e.g. ``"next_slide"``) to a
                            keyboard shortcut string (e.g. ``"right"``).
2. ``gesture_bindings``  -> maps a recognized gesture (e.g. ``"swipe_right"``) to an
                            action name (e.g. ``"next_slide"``), or ``None`` to ignore it.

This separation means gestures are bound to *actions*, and actions own the concrete
shortcut. Re-mapping a gesture or changing a shortcut are therefore two independent edits.

Sensible defaults are baked into the dataclass (:meth:`ControlConfig.defaults`), so the
interface is fully usable without any config file. A YAML file may override/extend the
defaults and is the recommended way to add actions or change shortcuts.

Input (optional):
config/
└── powerpoint_control.yml

Output (when saved):
config/
└── powerpoint_control.yml
"""


# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.control.shortcuts import normalize_shortcut

logger = get_logger(__name__)


# ======================================================================================================================
# default mappings
# ======================================================================================================================
# Default action -> shortcut map. PowerPoint slide-show shortcuts (Windows).
# Add your own actions here or in config/powerpoint_control.yml.
DEFAULT_ACTIONS: Dict[str, str] = {
    "next_slide": "right",            # advance to next slide / animation
    "previous_slide": "left",         # go back to previous slide / animation
    "start_presentation": "f5",       # start slide show from the beginning
    "start_from_current": "shift+f5",  # start slide show from the current slide
    "end_presentation": "esc",        # end the slide show
    "first_slide": "home",            # jump to the first slide
    "last_slide": "end",              # jump to the last slide
    "toggle_black_screen": "b",       # toggle a black screen (pause)
    "toggle_white_screen": "w",       # toggle a white screen
    "laser_pointer": "ctrl+l",        # activate the laser pointer
    "pen_tool": "ctrl+p",             # activate the pen / ink tool
    "arrow_pointer": "ctrl+a",        # switch back to the normal arrow pointer
    "erase_ink": "ctrl+e",            # switch to the eraser
}

# Default gesture -> action bindings for the gestures the model already recognizes.
# ``None`` means "recognized but intentionally not bound to any action".
DEFAULT_GESTURE_BINDINGS: Dict[str, Optional[str]] = {
    "none": None,                          # stillness / no gesture -> do nothing
    "swipe_right": "next_slide",           # swipe right -> next slide
    "swipe_left": "previous_slide",        # swipe left  -> previous slide
    "circle_cw": "start_presentation",     # clockwise circle -> start show (F5)
    "circle_ccw": "end_presentation",      # counter-clockwise circle -> end show (Esc)
    "fist": "toggle_black_screen",         # fist -> black screen (pause)
    "jerk_up": "first_slide",              # jerk up -> first slide (Home)
    "jerk_down": "last_slide",             # jerk down -> last slide (End)
}


# ======================================================================================================================
# config dataclass
# ======================================================================================================================
@dataclass
class ControlConfig:
    """
    Holds the action/shortcut map and the gesture->action bindings for the controller.

    :param: actions (Dict[str, str]): action name -> shortcut string.
    :param: gesture_bindings (Dict[str, Optional[str]]): gesture name -> action name (or None).
    :param: send_pause_s (float): delay pyautogui inserts after each key action (seconds).
    :param: focus_delay_s (float): optional delay before sending, to let a window gain focus (seconds).
    """
    actions: Dict[str, str] = field(default_factory=dict)
    gesture_bindings: Dict[str, Optional[str]] = field(default_factory=dict)
    send_pause_s: float = 0.05
    focus_delay_s: float = 0.0


    @classmethod
    def defaults(cls) -> "ControlConfig":
        """
        Build a configuration populated with the built-in default actions and bindings.

        :return: config (ControlConfig): a ready-to-use default configuration.
        """
        return cls(
            actions=dict(DEFAULT_ACTIONS),
            gesture_bindings=dict(DEFAULT_GESTURE_BINDINGS),
        )


    @classmethod
    def from_dict(cls, data: dict) -> "ControlConfig":
        """
        Build a configuration from a plain dict (e.g. parsed YAML), filling in defaults.

        Any key missing from ``data`` falls back to the built-in default, so a partial
        config file only needs to specify what it wants to override.

        :param: data (dict): mapping with optional keys ``actions``, ``gesture_bindings``,
                ``send_pause_s`` and ``focus_delay_s``.
        :return: config (ControlConfig): the resulting configuration.
        """
        data = data or {}
        actions = dict(DEFAULT_ACTIONS)
        actions.update(data.get("actions") or {})

        bindings = dict(DEFAULT_GESTURE_BINDINGS)
        bindings.update(data.get("gesture_bindings") or {})

        return cls(
            actions={str(k): str(v) for k, v in actions.items()},
            gesture_bindings={str(k): (str(v) if v is not None else None) for k, v in bindings.items()},
            send_pause_s=float(data.get("send_pause_s", 0.05)),
            focus_delay_s=float(data.get("focus_delay_s", 0.0)),
        )


    def to_dict(self) -> dict:
        """
        Serialize the configuration to a plain, YAML-friendly dict.

        :return: data (dict): serializable representation of this configuration.
        """
        return {
            "send_pause_s": self.send_pause_s,
            "focus_delay_s": self.focus_delay_s,
            "actions": dict(self.actions),
            "gesture_bindings": dict(self.gesture_bindings),
        }


    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ControlConfig":
        """
        Load the configuration from a YAML file, falling back to defaults if absent.

        :param: path (Optional[Path]): path to the YAML file. Defaults to
                ``config/powerpoint_control.yml`` resolved from the project root.
        :return: config (ControlConfig): the loaded (or default) configuration.
        """
        if path is None:
            from data_fusion_project.core.paths import POWERPOINT_CONTROL_CONFIG_FILE
            path = POWERPOINT_CONTROL_CONFIG_FILE

        path = Path(path)
        if not path.exists():
            logger.warning("Control config '%s' not found; using built-in defaults.", str(path))
            return cls.defaults()

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        logger.info("Loaded PowerPoint control config from: %s", str(path))
        return cls.from_dict(data)


    def save(self, path: Optional[Path] = None) -> None:
        """
        Write the configuration to a YAML file (creating parent directories as needed).

        :param: path (Optional[Path]): target path. Defaults to ``config/powerpoint_control.yml``.
        """
        if path is None:
            from data_fusion_project.core.paths import POWERPOINT_CONTROL_CONFIG_FILE
            path = POWERPOINT_CONTROL_CONFIG_FILE

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info("Saved PowerPoint control config to: %s", str(path))


    def validate(self) -> None:
        """
        Validate the configuration, raising on hard errors and warning on soft issues.

        Hard error: an action whose shortcut string cannot be parsed.
        Soft issues (warning only): a gesture bound to an unknown action, or a gesture
        name that is not part of the official project gesture list.

        :raises: ValueError: if any action has an unparsable shortcut string.
        """
        for action, shortcut in self.actions.items():
            try:
                normalize_shortcut(shortcut)
            except ValueError as exc:
                raise ValueError(f"Action '{action}' has an invalid shortcut '{shortcut}': {exc}") from exc

        known_gestures = self._known_gestures()
        for gesture, action in self.gesture_bindings.items():
            if known_gestures is not None and gesture not in known_gestures:
                logger.warning("Gesture binding '%s' is not in the official project gesture list.", gesture)
            if action is not None and action not in self.actions:
                logger.warning("Gesture '%s' is bound to unknown action '%s'.", gesture, action)


    @staticmethod
    def _known_gestures() -> Optional[frozenset[str]]:
        """
        Return the official project gesture list, or None if it cannot be imported.

        :return: gestures (Optional[frozenset[str]]): set of known gesture names, or None.
        """
        try:
            from data_fusion_project.core.paths import GESTURES
            return frozenset(GESTURES)
        except Exception:
            return None
