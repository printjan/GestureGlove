# src/data_fusion_project/control/powerpoint_controller.py
"""
PowerPoint control interface.

:class:`PowerPointController` is the interface that the rest of the system talks to. It
receives *actions* (or *gestures*, which it resolves to actions) and executes the matching
keyboard shortcut through a pluggable :class:`~data_fusion_project.control.shortcuts.KeyboardBackend`.

Two entry points form the public "send an action to the interface" surface:

- :meth:`execute_action` -> run a named action's shortcut directly.
- :meth:`trigger_gesture` -> resolve a gesture to its bound action, then run it.

Actions and gesture bindings can be changed at runtime (``add_action``, ``set_shortcut``,
``remove_action``, ``bind_gesture``, ``unbind_gesture``) and persisted with :meth:`save_config`.

This module is intentionally NOT connected to the gesture-recognition model. Feeding live
predictions is the job of :class:`~data_fusion_project.control.dispatcher.GestureDispatcher`.

Usage Example:
    from data_fusion_project.control import PowerPointController, DryRunBackend
    controller = PowerPointController(backend=DryRunBackend())
    controller.execute_action("next_slide")
    controller.trigger_gesture("swipe_left")
"""


# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

import time
from typing import Dict, Optional

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.control.config import ControlConfig
from data_fusion_project.control.shortcuts import (
    KeyboardBackend,
    DryRunBackend,
    format_shortcut,
    normalize_shortcut,
)

logger = get_logger(__name__)


# ======================================================================================================================
# controller
# ======================================================================================================================
class PowerPointController:
    """
    Maps actions/gestures to keyboard shortcuts and executes them via a backend.

    :param: config (Optional[ControlConfig]): action/binding configuration. Loaded from the
            default config file (or built-in defaults) when omitted.
    :param: backend (Optional[KeyboardBackend]): key-sending backend. Defaults to a
            :class:`DryRunBackend` so nothing is sent unless a real backend is provided.
    """

    def __init__(self, config: Optional[ControlConfig] = None, backend: Optional[KeyboardBackend] = None) -> None:
        self.config = config if config is not None else ControlConfig.load()
        self.config.validate()
        self.backend = backend if backend is not None else DryRunBackend()
        logger.info("PowerPointController ready (backend=%s, %d actions, %d bindings).", self.backend.name, len(self.config.actions), len(self.config.gesture_bindings))


    # ==================================================================================================================
    # action management
    # ==================================================================================================================
    def list_actions(self) -> Dict[str, str]:
        """
        Return a copy of the action -> shortcut map.

        :return: actions (Dict[str, str]): current action/shortcut mapping.
        """
        return dict(self.config.actions)


    def get_shortcut(self, action: str) -> str:
        """
        Return the shortcut string bound to an action.

        :param: action (str): action name.
        :return: shortcut (str): the shortcut string.
        :raises: KeyError: if the action does not exist.
        """
        if action not in self.config.actions:
            raise KeyError(f"Unknown action '{action}'. Known actions: {sorted(self.config.actions)}")
        return self.config.actions[action]


    def add_action(self, name: str, shortcut: str, *, overwrite: bool = False) -> None:
        """
        Add a new action with its shortcut (or update an existing one if ``overwrite``).

        :param: name (str): action name to add.
        :param: shortcut (str): shortcut string (validated immediately).
        :param: overwrite (bool): allow replacing an existing action of the same name.
        :raises: ValueError: if the action exists and ``overwrite`` is False, or the shortcut is invalid.
        """
        if name in self.config.actions and not overwrite:
            raise ValueError(f"Action '{name}' already exists. Pass overwrite=True to replace it.")
        normalize_shortcut(shortcut)
        self.config.actions[name] = shortcut
        logger.info("Action '%s' set to shortcut %s.", name, format_shortcut(shortcut))


    def set_shortcut(self, action: str, shortcut: str) -> None:
        """
        Change the shortcut of an existing action.

        :param: action (str): existing action name.
        :param: shortcut (str): new shortcut string (validated immediately).
        :raises: KeyError: if the action does not exist.
        :raises: ValueError: if the shortcut is invalid.
        """
        if action not in self.config.actions:
            raise KeyError(f"Unknown action '{action}'. Use add_action() to create it.")
        normalize_shortcut(shortcut)
        self.config.actions[action] = shortcut
        logger.info("Shortcut for action '%s' changed to %s.", action, format_shortcut(shortcut))


    def remove_action(self, action: str) -> None:
        """
        Remove an action and unbind any gestures that referenced it.

        :param: action (str): action name to remove.
        :raises: KeyError: if the action does not exist.
        """
        if action not in self.config.actions:
            raise KeyError(f"Unknown action '{action}'.")
        del self.config.actions[action]
        for gesture, bound in list(self.config.gesture_bindings.items()):
            if bound == action:
                self.config.gesture_bindings[gesture] = None
                logger.info("Gesture '%s' unbound because action '%s' was removed.", gesture, action)
        logger.info("Action '%s' removed.", action)


    # ==================================================================================================================
    # gesture binding management
    # ==================================================================================================================
    def list_bindings(self) -> Dict[str, Optional[str]]:
        """
        Return a copy of the gesture -> action binding map.

        :return: bindings (Dict[str, Optional[str]]): current gesture/action bindings.
        """
        return dict(self.config.gesture_bindings)


    def action_for_gesture(self, gesture: str) -> Optional[str]:
        """
        Resolve the action bound to a gesture, if any.

        :param: gesture (str): gesture name.
        :return: action (Optional[str]): the bound action name, or None if unbound/unknown.
        """
        return self.config.gesture_bindings.get(gesture)


    def bind_gesture(self, gesture: str, action: Optional[str]) -> None:
        """
        Bind a gesture to an action (or to ``None`` to ignore the gesture).

        :param: gesture (str): gesture name.
        :param: action (Optional[str]): action to bind, or None to clear the binding.
        :raises: KeyError: if ``action`` is given but does not exist.
        """
        if action is not None and action not in self.config.actions:
            raise KeyError(f"Cannot bind gesture '{gesture}': unknown action '{action}'.")
        self.config.gesture_bindings[gesture] = action
        logger.info("Gesture '%s' bound to action '%s'.", gesture, action)


    def unbind_gesture(self, gesture: str) -> None:
        """
        Remove the action binding for a gesture (it will be ignored afterwards).

        :param: gesture (str): gesture name to unbind.
        """
        self.config.gesture_bindings[gesture] = None
        logger.info("Gesture '%s' unbound.", gesture)


    # ==================================================================================================================
    # execution (the interface entry points)
    # ==================================================================================================================
    def execute_action(self, action: str) -> bool:
        """
        Execute the shortcut bound to a named action.

        :param: action (str): action name to execute.
        :return: executed (bool): True if a shortcut was sent.
        :raises: KeyError: if the action does not exist.
        """
        if action not in self.config.actions:
            raise KeyError(f"Unknown action '{action}'. Known actions: {sorted(self.config.actions)}")

        shortcut = self.config.actions[action]
        if self.config.focus_delay_s > 0:
            time.sleep(self.config.focus_delay_s)

        logger.info("Executing action '%s' -> %s", action, format_shortcut(shortcut))
        self.backend.send(shortcut)
        return True


    def trigger_gesture(self, gesture: str) -> Optional[str]:
        """
        Resolve a gesture to its bound action and execute it.

        Unbound gestures (including ``"none"``) are ignored and return ``None``.

        :param: gesture (str): recognized gesture name.
        :return: action (Optional[str]): the executed action name, or None if the gesture was ignored.
        """
        action = self.action_for_gesture(gesture)
        if action is None:
            logger.debug("Gesture '%s' has no bound action; ignoring.", gesture)
            return None

        self.execute_action(action)
        return action


    # ==================================================================================================================
    # persistence
    # ==================================================================================================================
    def save_config(self, path=None) -> None:
        """
        Persist the current actions and bindings to the YAML config file.

        :param: path: optional target path. Defaults to ``config/powerpoint_control.yml``.
        """
        self.config.save(path)
