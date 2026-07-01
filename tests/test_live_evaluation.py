#!/usr/bin/env python
# tests/test_live_evaluation.py
"""
Unit tests for the objection-based live-performance evaluator.

Covers:
- TriggerDetector de-bounce semantics (one fire per held gesture; re-fire after release + cooldown).
- LivePerformanceEvaluator TP/FP/FN bookkeeping, precision/recall, idle false-triggers and JSON export.

All tests are deterministic and hardware-free using an injected fake clock.
"""

import json
import sys
from pathlib import Path

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.inference.live_evaluation import (
    TriggerDetector,
    LivePerformanceEvaluator,
)


class FakeClock:
    """Deterministic monotonic clock; call to read, ``advance`` to move time forward."""
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ======================================================================================================================
# TriggerDetector
# ======================================================================================================================
def test_trigger_detector_fires_once_for_held_gesture():
    clk = FakeClock()
    det = TriggerDetector(confidence_threshold=0.8, cooldown_s=1.0, require_release=True, clock=clk)

    fires = []
    # 10 Hz stream of the same confident gesture with no idle in between.
    for _ in range(6):
        event = det.feed("swipe_right", 0.95)
        if event is not None:
            fires.append(event)
        clk.advance(0.1)

    assert len(fires) == 1
    assert fires[0][0] == "swipe_right"


def test_trigger_detector_ignores_idle_and_low_confidence():
    clk = FakeClock()
    det = TriggerDetector(confidence_threshold=0.8, cooldown_s=1.0, clock=clk)

    assert det.feed("none", 0.99) is None          # idle never fires
    assert det.feed("swipe_left", 0.5) is None      # below threshold


def test_trigger_detector_refires_after_release_and_cooldown():
    clk = FakeClock()
    det = TriggerDetector(confidence_threshold=0.8, cooldown_s=1.0, require_release=True, clock=clk)

    assert det.feed("fist", 0.9) is not None        # first fire at t=0
    assert det.feed("fist", 0.9) is None            # still held -> not armed
    det.feed("none", 0.99)                          # release re-arms
    clk.advance(1.5)                                 # move past cooldown
    assert det.feed("fist", 0.9) is not None        # fires again


# ======================================================================================================================
# LivePerformanceEvaluator
# ======================================================================================================================
def _run_scripted_session(clk: FakeClock) -> LivePerformanceEvaluator:
    classes = ["none", "swipe_left", "swipe_right", "circle_cw"]
    ev = LivePerformanceEvaluator(classes, objection_window_s=1.0, enable_fn=True, clock=clk)
    # digit hotkeys: 1=swipe_left, 2=swipe_right, 3=circle_cw

    # Event 1: fire swipe_right, no objection -> committed as TP once the window elapses.
    ev.on_fire("swipe_right", 0.95)
    clk.advance(1.1)
    ev.tick()

    # Event 2: fire swipe_left, user says it was actually swipe_right (digit 2) -> FP for swipe_left.
    ev.on_fire("swipe_left", 0.90)
    ev.poll("2")

    # Event 3: fire circle_cw during idle, user says it was nothing (0) -> idle false-trigger FP.
    ev.on_fire("circle_cw", 0.88)
    ev.poll("0")

    # Miss: user performed swipe_left but nothing fired -> 'm' then digit 1 -> FN for swipe_left.
    ev.poll("m")
    ev.poll("1")
    return ev


def test_evaluator_tallies_tp_fp_fn():
    ev = _run_scripted_session(FakeClock())
    report = ev.compute_report()
    pc = report["per_class"]

    assert pc["swipe_right"]["tp"] == 1
    assert pc["swipe_left"]["fp"] == 1
    assert pc["circle_cw"]["fp"] == 1
    assert pc["swipe_left"]["fn"] == 1

    totals = report["totals"]
    assert totals["tp"] == 1
    assert totals["fp"] == 2
    assert totals["fn"] == 1
    assert totals["idle_false_triggers"] == 1


def test_evaluator_precision_recall():
    ev = _run_scripted_session(FakeClock())
    pc = ev.compute_report()["per_class"]

    # swipe_right: 1 TP, 0 FP -> precision 1.0
    assert pc["swipe_right"]["precision"] == 1.0
    # swipe_left: 0 TP, 1 FN -> recall 0.0; 0 TP, 1 FP -> precision 0.0
    assert pc["swipe_left"]["recall"] == 0.0
    assert pc["swipe_left"]["precision"] == 0.0
    # circle_cw was never actually performed correctly -> recall undefined (None)
    assert pc["circle_cw"]["recall"] is None


def test_evaluator_confusion_matrix():
    ev = _run_scripted_session(FakeClock())
    cm = ev.compute_report()["confusion_matrix"]
    labels = cm["labels"]
    matrix = cm["matrix"]
    idx = {label: i for i, label in enumerate(labels)}

    # TP on the diagonal.
    assert matrix[idx["swipe_right"]][idx["swipe_right"]] == 1
    # FP: actual swipe_right, predicted swipe_left.
    assert matrix[idx["swipe_right"]][idx["swipe_left"]] == 1
    # Idle false trigger: actual none, predicted circle_cw.
    assert matrix[idx["none"]][idx["circle_cw"]] == 1
    # Miss represented as actual swipe_left, predicted none.
    assert matrix[idx["swipe_left"]][idx["none"]] == 1


def test_evaluator_finalize_writes_json(tmp_path):
    ev = _run_scripted_session(FakeClock())
    report = ev.finalize(tmp_path, plot=False)

    json_path = tmp_path / "live_evaluation.json"
    assert json_path.exists()

    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["totals"]["tp"] == report["totals"]["tp"] == 1
    assert on_disk["totals"]["fp"] == 2
    assert on_disk["session_meta"]["n_fire_events"] == 3
    assert on_disk["session_meta"]["n_miss_events"] == 1


def test_evaluator_quit_hotkey():
    ev = LivePerformanceEvaluator(["none", "fist"], clock=FakeClock())
    assert ev.quit_requested is False
    ev.poll("q")
    assert ev.quit_requested is True
