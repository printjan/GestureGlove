# src/data_fusion_project/control/dispatcher.py
"""
Gesture-to-action dispatcher: the integration seam for the recognition model.

The real-time inference loop produces a prediction (gesture + confidence) several times
per second. Feeding every prediction straight to the controller would fire the same
shortcut many times for a single physical gesture. :class:`GestureDispatcher` sits between
the model and the controller and decides *when* a prediction should actually trigger an
action, using two simple mechanisms:

- a confidence threshold (ignore low-confidence predictions and the idle gesture), and
- de-bouncing: after firing, a "release" (an idle / low-confidence reading) and/or a
  minimum cool-down must occur before the next action can fire.

IMPORTANT: this dispatcher is deliberately NOT wired to the neural network. Its
:meth:`feed` method is called manually (e.g. by the test script). Connecting it to the
live inference loop is a future step — it only needs the loop to call ``feed(gesture, prob)``.

Usage Example:
    from data_fusion_project.control import PowerPointController, GestureDispatcher
    dispatcher = GestureDispatcher(PowerPointController())
    fired = dispatcher.feed("swipe_right", 0.93)   # -> "next_slide" (or None if suppressed)
"""


# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

import time
from typing import Callable, Optional

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.control.powerpoint_controller import PowerPointController

logger = get_logger(__name__)


# ======================================================================================================================
# dispatcher
# ======================================================================================================================
class GestureDispatcher:
    """
    Convert a stream of gesture predictions into de-bounced controller actions.

    :param: controller (PowerPointController): the controller that executes the actions.
    :param: confidence_threshold (float): minimum probability for a prediction to count.
    :param: cooldown_s (float): minimum time between two fired actions (seconds).
    :param: require_release (bool): if True, an idle/low-confidence reading must occur
            between two fired actions (prevents one long gesture from firing repeatedly).
    :param: idle_gesture (str): the gesture label that represents "no gesture".
    :param: clock (Callable[[], float]): monotonic time source (override for testing).
    """

    def __init__(
        self,
        controller: PowerPointController,
        *,
        confidence_threshold: float = 0.80,
        cooldown_s: float = 1.0,
        require_release: bool = True,
        idle_gesture: str = "none",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self.confidence_threshold = float(confidence_threshold)
        self.cooldown_s = float(cooldown_s)
        self.require_release = bool(require_release)
        self.idle_gesture = idle_gesture
        self._clock = clock
        self._armed = True            # whether a new action may fire (release seen)
        self._last_fire_time = float("-inf")


    def _is_actionable(self, gesture: str, confidence: float) -> bool:
        """
        Decide whether a raw prediction could, in principle, trigger an action.

        :param: gesture (str): predicted gesture label.
        :param: confidence (float): predicted probability for that label.
        :return: actionable (bool): True if the gesture is non-idle, confident and bound.
        """
        if gesture == self.idle_gesture or confidence < self.confidence_threshold:
            return False
        return self.controller.action_for_gesture(gesture) is not None


    def feed(self, gesture: str, confidence: float) -> Optional[str]:
        """
        Feed one prediction; fire the bound action if de-bounce conditions allow it.

        :param: gesture (str): predicted gesture label.
        :param: confidence (float): predicted probability for that label.
        :return: action (Optional[str]): the action that was fired, or None if suppressed.
        """
        if not self._is_actionable(gesture, confidence):
            # Idle or low-confidence reading counts as a "release": re-arm the trigger.
            self._armed = True
            return None

        now = self._clock()
        if self.require_release and not self._armed:
            return None
        if now - self._last_fire_time < self.cooldown_s:
            return None

        action = self.controller.trigger_gesture(gesture)
        self._last_fire_time = now
        self._armed = False
        logger.debug("Dispatcher fired action '%s' for gesture '%s' (conf=%.2f).", action, gesture, confidence)
        return action


    def reset(self) -> None:
        """
        Reset the de-bounce state (re-arm and clear the cool-down timer).
        """
        self._armed = True
        self._last_fire_time = float("-inf")
