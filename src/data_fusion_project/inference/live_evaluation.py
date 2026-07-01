# src/data_fusion_project/inference/live_evaluation.py
"""
Objection-based live-performance evaluation for real-time gesture inference.

The real-time inference loop emits a prediction (gesture + confidence) several times per
second. :class:`TriggerDetector` collapses that stream into discrete *fire* events using the
same de-bounce semantics as :class:`~data_fusion_project.control.dispatcher.GestureDispatcher`
(confidence threshold, release-gating, cool-down) but decoupled from PowerPoint bindings, so
that *every* non-idle gesture (and idle false-triggers) can be evaluated.

:class:`LivePerformanceEvaluator` turns those fire events into a real-world performance report
using an *objection* interaction model that keeps user effort minimal:

- The user trusts every fire by default. If no objection key is pressed within the objection
  window, the event is committed as a **True Positive** (``actual = predicted``).
- If the fire was wrong, the user presses the digit of the gesture they *actually* performed
  (or ``0``/``n`` for "I did nothing" / idle false-trigger), producing a **False Positive**.
- If a performed gesture was *not* detected at all, the user presses ``m`` followed by the
  gesture digit, producing a **False Negative** (optional, gated by ``enable_fn``).

**True Negatives** (idle correctly staying silent) are deliberately not counted: they cannot be
measured without continuous ground-truth logging.

Design note: report and confusion-matrix rendering happen exactly once in :meth:`finalize`
(session end). The hot loop only ever does a minimal terminal alert per fire, so inference
throughput is not affected.
"""

# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.core.cli_ui import ui, Style, style_text
from data_fusion_project.core.json_writer import write_json

logger = get_logger(__name__)

IDLE_GESTURE = "none"


# ======================================================================================================================
# Trigger detection (de-bounce)
# ======================================================================================================================
class TriggerDetector:
    """
    Collapse a stream of per-window predictions into discrete de-bounced fire events.

    Mirrors the de-bounce logic of
    :class:`~data_fusion_project.control.dispatcher.GestureDispatcher` but is independent of the
    PowerPoint controller: it fires for *any* non-idle gesture above the confidence threshold.

    :param: confidence_threshold (float): minimum probability for a prediction to fire.
    :param: cooldown_s (float): minimum time between two fires (seconds).
    :param: require_release (bool): if True, an idle/low-confidence reading must occur between
            two fires (prevents one long gesture from firing repeatedly).
    :param: idle_gesture (str): the gesture label representing "no gesture".
    :param: clock (Callable[[], float]): monotonic time source (override for testing).
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.80,
        cooldown_s: float = 1.0,
        require_release: bool = True,
        idle_gesture: str = IDLE_GESTURE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.cooldown_s = float(cooldown_s)
        self.require_release = bool(require_release)
        self.idle_gesture = idle_gesture
        self._clock = clock
        self._armed = True
        self._last_fire_time = float("-inf")

    def _is_actionable(self, gesture: str, confidence: float) -> bool:
        """Decide whether a raw prediction could, in principle, fire."""
        if gesture == self.idle_gesture or confidence < self.confidence_threshold:
            return False
        return True

    def feed(self, gesture: str, confidence: float) -> Optional[tuple[str, float]]:
        """
        Feed one prediction; return a fire event if de-bounce conditions allow it.

        :param: gesture (str): predicted gesture label.
        :param: confidence (float): predicted probability for that label.
        :return: event (Optional[tuple]): ``(gesture, confidence)`` on a new fire, else None.
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

        self._last_fire_time = now
        self._armed = False
        return (gesture, confidence)

    def reset(self) -> None:
        """Reset the de-bounce state (re-arm and clear the cool-down timer)."""
        self._armed = True
        self._last_fire_time = float("-inf")


# ======================================================================================================================
# Objection-based evaluator
# ======================================================================================================================
class LivePerformanceEvaluator:
    """
    Accumulate objection-based ground truth for fire events and build a real-world report.

    :param: class_names (list[str]): full class list of the model (usually incl. ``"none"``).
    :param: objection_window_s (float): grace period after a fire during which an objection may
            arrive; if none does, the fire is committed as a True Positive.
    :param: enable_fn (bool): enable the miss (False-Negative) hotkey (``m`` + digit).
    :param: idle_gesture (str): label representing "no gesture".
    :param: clock (Callable[[], float]): monotonic time source (override for testing).
    :param: session_meta (dict | None): extra metadata copied verbatim into the JSON report.
    """

    def __init__(
        self,
        class_names: list[str],
        *,
        objection_window_s: float = 1.5,
        enable_fn: bool = True,
        idle_gesture: str = IDLE_GESTURE,
        clock: Callable[[], float] = time.monotonic,
        session_meta: dict | None = None,
    ) -> None:
        self.class_names = list(class_names)
        self.objection_window_s = float(objection_window_s)
        self.enable_fn = bool(enable_fn)
        self.idle_gesture = idle_gesture
        self._clock = clock
        self.session_meta = dict(session_meta or {})

        # Active (fireable) gestures get digit hotkeys 1..N in declaration order.
        self.active_gestures = [c for c in self.class_names if c != idle_gesture]
        self._digit_to_gesture = {str(i + 1): g for i, g in enumerate(self.active_gestures)}

        # Committed fire events: each is {"predicted", "actual", "confidence", "correct", "t"}.
        self.events: list[dict] = []
        # Committed miss (False-Negative) events: each is {"actual", "t"}.
        self.fn_events: list[dict] = []

        self._pending: Optional[dict] = None   # fire awaiting confirmation/objection
        self._await_miss_digit = False          # True after 'm' was pressed
        self.quit_requested = False
        self._start_time = self._clock()

    # ------------------------------------------------------------------------------------------------------------------
    # Hotkey helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_gesture_digit(self, key: str) -> Optional[str]:
        """Map a digit key to an active gesture (``None`` if it is not a valid gesture digit)."""
        return self._digit_to_gesture.get(key)

    def _resolve_actual_key(self, key: str) -> Optional[str]:
        """Map an objection key to a ground-truth label (gesture digit, or ``0``/``n`` -> idle)."""
        if key in ("0", "n", "N"):
            return self.idle_gesture
        return self._resolve_gesture_digit(key)

    def hotkey_legend(self) -> list[str]:
        """Return human-readable hotkey legend lines (for a startup ``ui.box``)."""
        lines = [f"[{i + 1}] {g}" for i, g in enumerate(self.active_gestures)]
        lines.append("[0]/[n] none (idle false-trigger)")
        lines.append("[SPACE] confirm last fire as correct")
        if self.enable_fn:
            lines.append("[m] + digit  report a MISSED gesture (false negative)")
        lines.append("[q] finish evaluation")
        return lines

    # ------------------------------------------------------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------------------------------------------------------
    def _commit_pending(self, actual: Optional[str] = None) -> None:
        """
        Commit the pending fire event. ``actual=None`` means "no objection" -> True Positive.
        """
        if self._pending is None:
            return
        p = self._pending
        predicted = p["predicted"]
        resolved_actual = predicted if actual is None else actual
        self.events.append({
            "predicted": predicted,
            "actual": resolved_actual,
            "confidence": float(p["confidence"]),
            "correct": bool(resolved_actual == predicted),
            "t": p["fire_time"] - self._start_time,
        })
        self._pending = None

    def on_fire(self, gesture: str, confidence: float) -> None:
        """
        Register a new fire event. Any still-pending prior event is committed as a True Positive
        (commit-on-next-fire), then a minimal terminal alert is emitted.
        """
        self._commit_pending(actual=None)
        self._pending = {"predicted": gesture, "confidence": float(confidence), "fire_time": self._clock()}
        self._alert(gesture, confidence)

    def poll(self, key: Optional[str]) -> None:
        """
        Interpret a single keystroke (from :func:`get_key_nonblocking`). Safe to call with
        ``None`` (no key pressed).
        """
        if not key:
            return

        # Second half of a miss sequence: 'm' was pressed, now expect the gesture digit.
        if self._await_miss_digit:
            self._await_miss_digit = False
            actual = self._resolve_gesture_digit(key)
            if actual is not None:
                self.fn_events.append({"actual": actual, "t": self._clock() - self._start_time})
                self._notify(f"MISS recorded: {actual} (false negative)", Style.WARNING)
            else:
                self._notify("Miss cancelled (no valid gesture digit).", Style.HINT)
            return

        if key in ("q", "Q"):
            self.quit_requested = True
            return

        if self.enable_fn and key in ("m", "M"):
            self._await_miss_digit = True
            self._notify("Miss mode: press the digit of the gesture that was NOT detected...", Style.HINT)
            return

        if key == " ":
            # Explicit confirm: commit the pending fire as correct immediately.
            self._commit_pending(actual=None)
            return

        actual = self._resolve_actual_key(key)
        if actual is None:
            return  # unmapped key -> ignore
        if self._pending is None:
            self._notify("Objection ignored: no recent fire to correct.", Style.HINT)
            return

        predicted = self._pending["predicted"]
        self._commit_pending(actual=actual)
        if actual == predicted:
            self._notify(f"Confirmed: {predicted}", Style.SUCCESS)
        else:
            self._notify(f"Correction: fired '{predicted}', actual '{actual}' (false positive)", Style.WARNING)

    def tick(self, now: Optional[float] = None) -> None:
        """
        Commit the pending fire as a True Positive once the objection window has elapsed.
        Call once per loop iteration.
        """
        if self._pending is None:
            return
        current = self._clock() if now is None else now
        if current - self._pending["fire_time"] > self.objection_window_s:
            self._commit_pending(actual=None)

    # ------------------------------------------------------------------------------------------------------------------
    # Terminal feedback (intentionally minimal)
    # ------------------------------------------------------------------------------------------------------------------
    def _alert(self, gesture: str, confidence: float) -> None:
        """Emit a single salient line + bell so the user notices every fire."""
        msg = style_text(f"  ▶ FIRE: {gesture} ({confidence * 100:.0f}%) — object with a key or ignore",
                         Style.INFO)
        sys.stdout.write("\n\a" + msg + "\n")
        sys.stdout.flush()

    def _notify(self, text: str, style) -> None:
        sys.stdout.write("\n" + style_text("  " + text, style) + "\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------------------------------------------------------
    def compute_report(self) -> dict:
        """
        Build the evaluation report dict (counts, per-class precision/recall, confusion matrix).
        Commits any still-pending fire as a True Positive first.
        """
        self._commit_pending(actual=None)

        labels = list(self.class_names)
        idx = {label: i for i, label in enumerate(labels)}
        n = len(labels)

        tp = {c: 0 for c in labels}
        fp = {c: 0 for c in labels}
        fn = {c: 0 for c in labels}
        matrix = [[0 for _ in range(n)] for _ in range(n)]  # rows = actual, cols = predicted

        for ev in self.events:
            predicted, actual = ev["predicted"], ev["actual"]
            if ev["correct"]:
                tp[predicted] += 1
            else:
                fp[predicted] += 1
            if actual in idx and predicted in idx:
                matrix[idx[actual]][idx[predicted]] += 1

        for ev in self.fn_events:
            actual = ev["actual"]
            if actual in fn:
                fn[actual] += 1
            # Represent a miss as (actual, predicted=idle) in the confusion matrix.
            if actual in idx and self.idle_gesture in idx:
                matrix[idx[actual]][idx[self.idle_gesture]] += 1

        per_class = {}
        for c in labels:
            precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else None
            recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else None
            per_class[c] = {
                "tp": tp[c], "fp": fp[c], "fn": fn[c],
                "precision": precision, "recall": recall,
            }

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        idle_false_triggers = sum(
            1 for ev in self.events if not ev["correct"] and ev["actual"] == self.idle_gesture
        )

        return {
            "session_meta": {
                **self.session_meta,
                "duration_s": round(self._clock() - self._start_time, 2),
                "objection_window_s": self.objection_window_s,
                "enable_fn": self.enable_fn,
                "n_fire_events": len(self.events),
                "n_miss_events": len(self.fn_events),
            },
            "totals": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None,
                "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None,
                "idle_false_triggers": idle_false_triggers,
            },
            "per_class": per_class,
            "confusion_matrix": {"labels": labels, "matrix": matrix},
        }

    def finalize(self, out_dir: str | Path | None = None, *, plot: bool = True) -> dict:
        """
        Commit the last pending event, compute the report, print it, and (optionally) write
        ``live_evaluation.json`` + ``live_confusion_matrix.png`` into ``out_dir``.

        :param: out_dir (str | Path | None): destination folder; skip file output if None.
        :param: plot (bool): render the confusion-matrix PNG (lazy matplotlib import).
        :return: report (dict): the computed report.
        """
        report = self.compute_report()
        self._print_report(report)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "live_evaluation.json"
            write_json(json_path, report)
            ui.success(f"Live evaluation report saved to {json_path}")
            if plot:
                try:
                    png_path = out_dir / "live_confusion_matrix.png"
                    _plot_confusion_matrix(report["confusion_matrix"], png_path)
                    ui.success(f"Confusion matrix saved to {png_path}")
                except Exception as exc:  # matplotlib missing / headless issues must not lose the JSON
                    logger.warning("Could not render confusion matrix plot: %s", exc)

        return report

    def _print_report(self, report: dict) -> None:
        """Render the report to the terminal via the CliUI helpers."""
        ui.hr(title="Live Performance Evaluation")
        totals = report["totals"]
        prec = totals["precision"]
        rec = totals["recall"]
        ui.kv([
            ("Fire events", str(report["session_meta"]["n_fire_events"])),
            ("True Positives", str(totals["tp"])),
            ("False Positives", str(totals["fp"])),
            ("False Negatives", str(totals["fn"])),
            ("Idle false-triggers", str(totals["idle_false_triggers"])),
            ("Overall precision", f"{prec * 100:.1f}%" if prec is not None else "n/a"),
            ("Overall recall", f"{rec * 100:.1f}%" if rec is not None else "n/a"),
        ])

        rows = []
        for c in self.class_names:
            m = report["per_class"][c]
            p = f"{m['precision'] * 100:.1f}%" if m["precision"] is not None else "-"
            r = f"{m['recall'] * 100:.1f}%" if m["recall"] is not None else "-"
            rows.append([c, str(m["tp"]), str(m["fp"]), str(m["fn"]), p, r])
        ui.table(["Gesture", "TP", "FP", "FN", "Precision", "Recall"], rows)


# ======================================================================================================================
# Plotting (lazy matplotlib import; kept local to avoid the keras import chain in train.py)
# ======================================================================================================================
def _plot_confusion_matrix(confusion: dict, save_path: Path) -> None:
    """
    Render a partial confusion-matrix heatmap (rows = actual, cols = predicted) to ``save_path``.
    Only labels that appear in at least one fired/miss event are shown, to keep the plot clean.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = confusion["labels"]
    cm = np.array(confusion["matrix"], dtype=int)

    active = [i for i in range(len(labels)) if cm[i, :].sum() > 0 or cm[:, i].sum() > 0]
    if not active:
        active = list(range(min(2, len(labels))))
    cm_active = cm[np.ix_(active, active)]
    active_names = [labels[i] for i in active]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_active, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_active.shape[1]),
        yticks=np.arange(cm_active.shape[0]),
        xticklabels=active_names,
        yticklabels=active_names,
        title="Live Confusion Matrix (actual vs. predicted)",
        ylabel="Actual (ground truth)",
        xlabel="Predicted (fired)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_active.max() / 2.0 if cm_active.size else 0
    for i in range(cm_active.shape[0]):
        for j in range(cm_active.shape[1]):
            ax.text(j, i, format(cm_active[i, j], "d"), ha="center", va="center",
                    color="white" if cm_active[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
