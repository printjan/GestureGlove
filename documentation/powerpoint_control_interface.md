# PowerPoint Control Interface

Interface that turns recognized gestures into PowerPoint keyboard shortcuts. It is the
"Demonstration: Control PowerPoint with hand gestures" goal from the project README, built
as a standalone, model-independent component.

The gesture-recognition neural network is **not** wired to this interface yet. The seam
where it will plug in later is the `GestureDispatcher` (see below); for now it is fed
manually by the test script.


## Concept

The flow has three layers, each independently configurable:

```
gesture  ──(gesture_bindings)──►  action  ──(actions)──►  shortcut  ──(backend)──►  PowerPoint
swipe_right                       next_slide              right                     (key press)
```

- An **action** is a semantic command (`next_slide`) mapped to a keyboard shortcut (`right`).
- A **gesture** is bound to an action. To change what a gesture does, repoint its binding;
  to change the key an action sends, edit the action's shortcut. The two are separate edits.
- A **backend** performs the actual key press. It is pluggable, so the control logic does
  not depend on any specific key-sending library.


## Default gesture mapping

The gestures the model already recognizes each ship with a fitting shortcut:

| Gesture       | Action                | Shortcut | Effect in PowerPoint        |
|---------------|-----------------------|----------|-----------------------------|
| `swipe_right` | `next_slide`          | →        | Next slide / animation      |
| `swipe_left`  | `previous_slide`      | ←        | Previous slide / animation  |
| `circle_cw`   | `start_presentation`  | F5       | Start slide show            |
| `circle_ccw`  | `end_presentation`    | Esc      | End slide show              |
| `fist`        | `toggle_black_screen` | B        | Black screen (pause)        |
| `jerk_up`     | `first_slide`         | Home     | Jump to first slide         |
| `jerk_down`   | `last_slide`          | End      | Jump to last slide          |
| `none`        | — (ignored)           | —        | Nothing                     |

Further actions are predefined but unbound, ready to bind to a gesture: `start_from_current`
(Shift+F5), `toggle_white_screen` (W), `laser_pointer` (Ctrl+L), `pen_tool` (Ctrl+P),
`arrow_pointer` (Ctrl+A), `erase_ink` (Ctrl+E).


## Package layout

```
src/data_fusion_project/control/
├── __init__.py                 # public API
├── shortcuts.py                # KeyboardBackend (abstract), DryRunBackend, PyAutoGuiBackend + parsing
├── config.py                   # ControlConfig dataclass, defaults, YAML load/save, validation
├── powerpoint_controller.py    # PowerPointController: the interface (action/gesture -> shortcut)
└── dispatcher.py               # GestureDispatcher: de-bounced bridge for the model (NOT wired yet)
```

The configuration file (optional; built-in defaults are used if it is missing):

```
config/
└── powerpoint_control.yml
```


## Configuration

Add actions, change shortcuts and rebind gestures by editing `config/powerpoint_control.yml`.
Shortcut syntax is tokens joined by `+`, e.g. `right`, `shift+f5`, `ctrl+l`. Recognized
modifiers: `ctrl`, `alt`, `shift`, `win`. A shortcut must contain at least one non-modifier
key. Everything can also be changed at runtime through the controller API and persisted with
`save_config()`.

```yaml
actions:
  next_slide: right
  my_custom_action: ctrl+shift+f5    # add new actions freely
gesture_bindings:
  swipe_right: next_slide
  fist: my_custom_action             # rebind a gesture to any action
  jerk_up: null                      # null = recognized but ignored
```


## Programmatic use

```python
from data_fusion_project.control import PowerPointController, PyAutoGuiBackend

controller = PowerPointController(backend=PyAutoGuiBackend())

controller.execute_action("next_slide")        # run an action directly
controller.trigger_gesture("swipe_left")        # resolve gesture -> action -> shortcut

controller.add_action("blank", "b")             # add a new action
controller.set_shortcut("next_slide", "pagedown")  # change a shortcut
controller.bind_gesture("fist", "blank")         # rebind a gesture
controller.save_config()                         # persist to YAML
```

Use `DryRunBackend` (the default) to log shortcuts without sending any key presses.


## Independent test

`scripts/test_powerpoint_control.py` exercises the interface **without** the model. It is
the way to confirm the shortcuts actually drive PowerPoint.

```bash
# inspect the current actions and bindings
python scripts/test_powerpoint_control.py --list

# safe, no key presses (just logs what would be sent)
python scripts/test_powerpoint_control.py --dry-run --sequence

# show the dispatcher de-bounce logic on a synthetic prediction stream
python scripts/test_powerpoint_control.py --dry-run --simulate-stream

# LIVE in PowerPoint: open your deck, run this, Alt-Tab into PowerPoint during the countdown
python scripts/test_powerpoint_control.py --sequence --countdown 6
python scripts/test_powerpoint_control.py --action next_slide --countdown 5
python scripts/test_powerpoint_control.py            # interactive menu
```

Live key sending requires `pyautogui`:

```bash
pip install -e .[control]      # or: pip install pyautogui
```

If `pyautogui` is not installed, the test falls back to dry-run automatically.


## Connecting the model later (not done yet)

`GestureDispatcher` is the integration point. The real-time inference loop produces a
prediction several times per second; the dispatcher decides when a prediction should fire,
using a confidence threshold, a cool-down, and a "release" requirement so one physical
gesture fires exactly once.

```python
from data_fusion_project.control import PowerPointController, GestureDispatcher

dispatcher = GestureDispatcher(PowerPointController(backend=PyAutoGuiBackend()))

# inside the inference loop, replace the display call with:
dispatcher.feed(class_name, prob)   # fires the bound action when conditions are met
```

Until that line is added to `scripts/run_realtime_inference_test.py`, the interface stays
fully decoupled from the network.
