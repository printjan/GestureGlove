#!/usr/bin/env python
# scripts/test_powerpoint_control.py
"""
Standalone test bench for the PowerPoint control interface.

This script exercises the control interface
(src/data_fusion_project/control) WITHOUT the gesture-recognition model. It is the way to
verify that the configured shortcuts actually do the right thing in PowerPoint before the
neural network is connected.

Backends:
    - live (default): sends real key presses via pyautogui. Falls back to dry-run if
      pyautogui is not installed.
    - --dry-run: only logs the shortcuts it would send (safe, no key presses).

Modes:
    --list                 Print the action map and gesture bindings, then exit.
    --action NAME          Execute a single action, then exit.
    --gesture NAME         Trigger a single gesture (gesture -> action), then exit.
    --sequence             Run hands-free through every bound gesture (great for a live demo).
    --simulate-stream      Feed a synthetic prediction stream through the GestureDispatcher
                           to show the de-bounce logic (how a 10 Hz stream fires once).
    (no mode flag)         Interactive menu.

Live usage example (try it in PowerPoint):
    1. Open your presentation in PowerPoint.
    2. Run:  python scripts/test_powerpoint_control.py --sequence --countdown 6
    3. Alt-Tab to PowerPoint within the countdown; the shortcuts fire automatically.

Input (optional):
config/
└── powerpoint_control.yml
"""


# ======================================================================================================================
# imports
# ======================================================================================================================
import sys
import time
import argparse
from pathlib import Path

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.core.cli_ui import ui, Style
from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.control import (
    PowerPointController,
    GestureDispatcher,
    ControlConfig,
    DryRunBackend,
    PyAutoGuiBackend,
    format_shortcut,
)

logger = get_logger(__name__)


# ======================================================================================================================
# argument parsing
# ======================================================================================================================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the test bench.

    :return: args (argparse.Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Standalone PowerPoint control interface test (no model).")
    parser.add_argument("--config", type=str, default=None, help="Path to a powerpoint_control.yml config file.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send real keys; only log what would be sent.")
    parser.add_argument("--countdown", type=float, default=5.0, help="Seconds to switch to PowerPoint before a live send.")
    parser.add_argument("--step-delay", type=float, default=1.3, help="Seconds between steps in --sequence (live mode).")
    parser.add_argument("--pause", type=float, default=None, help="Override pyautogui per-key pause (seconds).")

    mode = parser.add_argument_group("modes (pick at most one; default is interactive)")
    mode.add_argument("--list", action="store_true", help="Print actions and gesture bindings, then exit.")
    mode.add_argument("--action", type=str, default=None, help="Execute a single action by name, then exit.")
    mode.add_argument("--gesture", type=str, default=None, help="Trigger a single gesture by name, then exit.")
    mode.add_argument("--sequence", action="store_true", help="Run hands-free through all bound gestures.")
    mode.add_argument("--simulate-stream", action="store_true", help="Feed a synthetic prediction stream through the dispatcher.")
    return parser.parse_args()



# ======================================================================================================================
# backend / setup helpers
# ======================================================================================================================
def build_controller(args: argparse.Namespace) -> tuple[PowerPointController, bool]:
    """
    Build the controller with the requested backend.

    :param: args (argparse.Namespace): parsed arguments.
    :return: result (tuple[PowerPointController, bool]): the controller and a flag that is
             True when a real (live) key-sending backend is active.
    """
    config_path = Path(args.config) if args.config else None
    config = ControlConfig.load(config_path)
    pause = args.pause if args.pause is not None else config.send_pause_s

    if args.dry_run:
        return PowerPointController(config=config, backend=DryRunBackend()), False

    try:
        backend = PyAutoGuiBackend(pause_s=pause)
        return PowerPointController(config=config, backend=backend), True
    except ImportError:
        ui.warning("pyautogui is not installed - falling back to DRY-RUN (no keys will be sent).")
        ui.hint("Install it with:  pip install -e .[control]    (or:  pip install pyautogui)")
        return PowerPointController(config=config, backend=DryRunBackend()), False



def print_overview(controller: PowerPointController) -> None:
    """
    Print the current action map and gesture bindings.

    :param: controller (PowerPointController): the controller to inspect.
    """
    ui.hr(title="Actions (name -> shortcut)")
    for name, shortcut in controller.list_actions().items():
        ui.info(f"  {name:<20} {format_shortcut(shortcut)}")

    ui.hr(title="Gesture bindings (gesture -> action -> shortcut)")
    for gesture, action in controller.list_bindings().items():
        if action is None:
            ui.info(f"  {gesture:<14} (ignored)")
        else:
            shortcut = controller.list_actions().get(action, "?")
            ui.info(f"  {gesture:<14} -> {action:<20} {format_shortcut(shortcut)}")



def focus_countdown(seconds: float) -> None:
    """
    Block for ``seconds`` while prompting the user to switch to PowerPoint.

    :param: seconds (float): countdown duration in seconds (skipped if <= 0).
    """
    if seconds <= 0:
        return
    ui.info("Switch to your PowerPoint window now...")
    ui.progress_bar(seconds, label="Focusing: ", color=Style.WARNING)



# ======================================================================================================================
# run modes
# ======================================================================================================================
def run_single_action(controller: PowerPointController, action: str, live: bool, countdown: float) -> None:
    """
    Execute a single action (with a focus countdown in live mode).

    :param: controller (PowerPointController): the controller.
    :param: action (str): action name to execute.
    :param: live (bool): whether a real backend is active.
    :param: countdown (float): focus countdown seconds for live mode.
    """
    if live:
        focus_countdown(countdown)
    try:
        controller.execute_action(action)
        ui.success(f"Executed action '{action}'.")
    except KeyError as exc:
        ui.error(str(exc))



def run_single_gesture(controller: PowerPointController, gesture: str, live: bool, countdown: float) -> None:
    """
    Trigger a single gesture, resolving it to its bound action.

    :param: controller (PowerPointController): the controller.
    :param: gesture (str): gesture name to trigger.
    :param: live (bool): whether a real backend is active.
    :param: countdown (float): focus countdown seconds for live mode.
    """
    if live:
        focus_countdown(countdown)
    action = controller.trigger_gesture(gesture)
    if action is None:
        ui.warning(f"Gesture '{gesture}' is not bound to any action (ignored).")
    else:
        ui.success(f"Gesture '{gesture}' -> action '{action}'.")



def run_sequence(controller: PowerPointController, live: bool, countdown: float, step_delay: float) -> None:
    """
    Run hands-free through every bound gesture, one step at a time.

    :param: controller (PowerPointController): the controller.
    :param: live (bool): whether a real backend is active.
    :param: countdown (float): initial focus countdown seconds for live mode.
    :param: step_delay (float): seconds to wait between steps (live mode only).
    """
    bound = [(g, a) for g, a in controller.list_bindings().items() if a is not None]
    if not bound:
        ui.warning("No gestures are bound to actions; nothing to run.")
        return

    ui.hr(title="Scripted gesture sequence")
    ui.info(f"Will trigger {len(bound)} bound gestures: {', '.join(g for g, _ in bound)}")
    if live:
        focus_countdown(countdown)

    for gesture, action in bound:
        ui.info(f"Gesture '{gesture}' -> action '{action}'")
        controller.trigger_gesture(gesture)
        if live and step_delay > 0:
            time.sleep(step_delay)

    ui.success("Sequence finished.")



def run_simulated_stream(controller: PowerPointController) -> None:
    """
    Feed a synthetic prediction stream through a GestureDispatcher to show de-bouncing.

    A deterministic fake clock (0.1 s per frame) is used so the cool-down behaviour is
    reproducible and the demo runs instantly. This mirrors how the real inference loop
    would call ``dispatcher.feed(gesture, confidence)`` ~10 times per second.

    :param: controller (PowerPointController): the controller actions are dispatched to.
    """
    # Deterministic clock: each frame advances virtual time by 0.1 s.
    state = {"t": 0.0}

    def fake_clock() -> float:
        return state["t"]

    # Cooldown is shortened to 0.5 s here purely to keep the demo compact; the controller
    # default is 1.0 s.
    cooldown_s = 0.5
    dispatcher = GestureDispatcher(controller, confidence_threshold=0.80, cooldown_s=cooldown_s, require_release=True, clock=fake_clock)

    # (gesture, confidence) frames. Held gestures span several frames on purpose.
    frames = [
        ("none", 0.99), ("none", 0.99),
        ("swipe_right", 0.62),                                   # below threshold -> ignored
        ("swipe_right", 0.90), ("swipe_right", 0.93), ("swipe_right", 0.91),  # held -> fires ONCE
        ("none", 0.98), ("none", 0.97), ("none", 0.99),          # release (re-arms the trigger)
        ("swipe_left", 0.94), ("swipe_left", 0.92),              # different gesture -> fires ONCE
        ("none", 0.99),                                          # release
        ("fist", 0.96), ("fist", 0.95),                          # too soon after last fire -> cooldown suppresses
        ("fist", 0.97),                                          # cooldown elapsed -> fires ONCE
        ("none", 0.99),
    ]

    ui.hr(title="Simulated prediction stream (dispatcher de-bounce)")
    ui.info(f"threshold=0.80, cooldown={cooldown_s:.1f}s (shortened for demo), require_release=True, frame step=0.1s\n")
    fired_count = 0
    for gesture, conf in frames:
        action = dispatcher.feed(gesture, conf)
        state["t"] += 0.1
        if action is not None:
            fired_count += 1
            ui.success(f"  t={state['t']:.1f}s  {gesture:<12} conf={conf:.2f}  ==> FIRED '{action}'")
        else:
            marker = "(idle)" if gesture == "none" else "(suppressed)"
            ui.styled(f"  t={state['t']:.1f}s  {gesture:<12} conf={conf:.2f}      {marker}", Style.MUTED)

    ui.info("")
    ui.success(f"Stream finished: {fired_count} actions fired from {len(frames)} predictions.")



def run_interactive(controller: PowerPointController, live: bool, countdown: float) -> None:
    """
    Interactive menu to trigger actions/gestures by name.

    :param: controller (PowerPointController): the controller.
    :param: live (bool): whether a real backend is active.
    :param: countdown (float): focus countdown seconds applied before each live send.
    """
    print_overview(controller)
    ui.hr(title="Interactive mode")
    ui.info("Commands:")
    ui.info("  <action>          run an action by name        (e.g. next_slide)")
    ui.info("  g <gesture>       trigger a gesture by name     (e.g. g swipe_left)")
    ui.info("  seq               run the scripted gesture sequence")
    ui.info("  stream            run the simulated prediction stream")
    ui.info("  list              reprint actions and bindings")
    ui.info("  save              save the current config to YAML")
    ui.info("  q                 quit")
    if live:
        ui.warning(f"LIVE backend active: a {countdown:.0f}s focus countdown runs before each send so you can switch to PowerPoint.")
    else:
        ui.hint("DRY-RUN backend active: shortcuts are only logged, not sent.")

    while True:
        raw = ui.ask("control> ")
        if raw is None:
            break
        cmd = raw.strip()
        if cmd == "":
            continue
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if cmd.lower() == "list":
            print_overview(controller)
            continue
        if cmd.lower() == "save":
            controller.save_config()
            ui.success("Config saved.")
            continue
        if cmd.lower() == "seq":
            run_sequence(controller, live, countdown, step_delay=1.3)
            continue
        if cmd.lower() == "stream":
            run_simulated_stream(controller)
            continue
        if cmd.lower().startswith("g ") or cmd.lower().startswith("g:"):
            gesture = cmd[2:].strip()
            run_single_gesture(controller, gesture, live, countdown)
            continue
        # Otherwise treat the input as an action name.
        run_single_action(controller, cmd, live, countdown)

    ui.success("Bye.")



# ======================================================================================================================
# main
# ======================================================================================================================
def main() -> None:
    """
    Entry point: build the controller and run the selected mode.
    """
    args = parse_args()
    ui.hr(title="PowerPoint Control Interface - Test Bench")

    controller, live = build_controller(args)
    ui.info(f"Backend: {controller.backend.name}  ({'LIVE - real key presses' if live else 'dry-run - logging only'})")

    if args.list:
        print_overview(controller)
        return
    if args.action:
        run_single_action(controller, args.action, live, args.countdown)
        return
    if args.gesture:
        run_single_gesture(controller, args.gesture, live, args.countdown)
        return
    if args.sequence:
        run_sequence(controller, live, args.countdown, args.step_delay)
        return
    if args.simulate_stream:
        run_simulated_stream(controller)
        return

    run_interactive(controller, live, args.countdown)


if __name__ == "__main__":
    main()
