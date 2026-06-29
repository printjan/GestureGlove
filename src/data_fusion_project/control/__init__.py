# src/data_fusion_project/control/__init__.py
"""
PowerPoint control interface for the DataFusionProject.

This package turns high-level *actions* (and the *gestures* bound to them) into keyboard
shortcuts sent to PowerPoint. It is independent of the gesture-recognition model: the
:class:`GestureDispatcher` is the seam where the model will be connected later.

Public API:
    - PowerPointController : the interface (action/gesture -> shortcut execution).
    - GestureDispatcher    : de-bounced bridge from a prediction stream to the controller.
    - ControlConfig        : action/shortcut + gesture-binding configuration.
    - KeyboardBackend      : abstract key-sending backend.
    - DryRunBackend        : logs shortcuts without sending them.
    - PyAutoGuiBackend     : sends real key presses via pyautogui.
"""

from data_fusion_project.control.config import (
    ControlConfig,
    DEFAULT_ACTIONS,
    DEFAULT_GESTURE_BINDINGS,
)
from data_fusion_project.control.shortcuts import (
    KeyboardBackend,
    DryRunBackend,
    PyAutoGuiBackend,
    normalize_shortcut,
    format_shortcut,
)
from data_fusion_project.control.powerpoint_controller import PowerPointController
from data_fusion_project.control.dispatcher import GestureDispatcher

__all__ = [
    "PowerPointController",
    "GestureDispatcher",
    "ControlConfig",
    "DEFAULT_ACTIONS",
    "DEFAULT_GESTURE_BINDINGS",
    "KeyboardBackend",
    "DryRunBackend",
    "PyAutoGuiBackend",
    "normalize_shortcut",
    "format_shortcut",
]
