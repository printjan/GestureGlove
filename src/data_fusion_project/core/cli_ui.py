# src/data_fusion_project/core/cli_ui.py
"""
CLI UI primitives: terminal awareness, ANSI styling, layout helpers, interactive prompts.

This module provides building blocks for modern interactive CLIs without external dependencies.
It integrates with the existing logger via `logger.write(...)` for non-prefixed output.

Usage:
    from data_fusion_project.core.cli_ui import ui, Style, is_interactive

    # Styled text
    ui.styled("Success!", style=Style.SUCCESS)

    # Layout
    ui.hr(title="Section")
    ui.banner("Welcome", subtitle="v1.0")
    ui.box(["Line 1", "Line 2"], title="Summary")
    ui.kv([("Key", "Value"), ("Name", "Alice")])

    # Interactive
    name = ui.ask("Enter name: ")
    age = ui.ask_int("Enter age: ", min_val=0, max_val=120)
    confirm = ui.confirm("Continue?")
    choice = ui.ask_choice("Pick one:", ["A", "B", "C"])
    password = ui.ask_secret("Password: ")

    # Spinners
    with ui.spinner("Loading..."):
        do_work()
"""

from __future__ import annotations

import getpass
import itertools
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Sequence,
    TextIO,
)

if TYPE_CHECKING:
    from types import TracebackType


# ======================================================================================================================
# Terminal Capabilities
# ======================================================================================================================

def get_terminal_width(fallback: int = 500) -> int:
    """
    Get current terminal width in columns.

    :param fallback: Width to use if detection fails or not a TTY.
    :return: Terminal width in characters.
    """
    try:
        size = shutil.get_terminal_size((fallback, 24))
        return size.columns
    except Exception:
        return fallback


def is_interactive() -> bool:
    """
    Check if both stdin and stdout are connected to a TTY.

    :return: True if interactive terminal, False if piped/redirected.
    """
    try:
        return sys.stdout.isatty() and sys.stdin.isatty()
    except Exception:
        return False


def is_stdout_tty() -> bool:
    """
    Check if stdout is a TTY.

    :return: True if stdout is a TTY.
    """
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def ansi_enabled() -> bool:
    """
    Determine if ANSI escape codes should be used.

    Respects NO_COLOR env var (https://no-color.org/) and LOG_COLORS.
    Falls back to TTY detection.

    :return: True if ANSI codes are allowed.
    """
    # NO_COLOR takes precedence (standard)
    if os.getenv("NO_COLOR") is not None:
        return False

    # Project-specific override
    log_colors = os.getenv("LOG_COLORS", "").strip().lower()
    if log_colors in {"0", "false", "no", "off"}:
        return False
    if log_colors in {"1", "true", "yes", "on"}:
        return True

    # Default: only if TTY
    return is_stdout_tty()


# ======================================================================================================================
# ANSI Escape Codes
# ======================================================================================================================

_ANSI_RESET = "\x1b[0m"

# Basic formatting
_ANSI_BOLD = "\x1b[1m"
_ANSI_DIM = "\x1b[2m"
_ANSI_ITALIC = "\x1b[3m"
_ANSI_UNDERLINE = "\x1b[4m"
_ANSI_BLINK = "\x1b[5m"
_ANSI_INVERSE = "\x1b[7m"
_ANSI_HIDDEN = "\x1b[8m"
_ANSI_STRIKETHROUGH = "\x1b[9m"

# Basic foreground colors (30-37)
_FG_COLORS: dict[str, str] = {
    "default": "39",
    "black": "30",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "37",
    # Bright variants (90-97)
    "bright_black": "90",
    "bright_red": "91",
    "bright_green": "92",
    "bright_yellow": "93",
    "bright_blue": "94",
    "bright_magenta": "95",
    "bright_cyan": "96",
    "bright_white": "97",
}

# Basic background colors (40-47)
_BG_COLORS: dict[str, str] = {
    "default": "49",
    "black": "40",
    "red": "41",
    "green": "42",
    "yellow": "43",
    "blue": "44",
    "magenta": "45",
    "cyan": "46",
    "white": "47",
    # Bright variants (100-107)
    "bright_black": "100",
    "bright_red": "101",
    "bright_green": "102",
    "bright_yellow": "103",
    "bright_blue": "104",
    "bright_magenta": "105",
    "bright_cyan": "106",
    "bright_white": "107",
}

# Cursor control
_CURSOR_UP = "\x1b[{n}A"
_CURSOR_DOWN = "\x1b[{n}B"
_CURSOR_FORWARD = "\x1b[{n}C"
_CURSOR_BACK = "\x1b[{n}D"
_CURSOR_SAVE = "\x1b[s"
_CURSOR_RESTORE = "\x1b[u"
_CURSOR_HIDE = "\x1b[?25l"
_CURSOR_SHOW = "\x1b[?25h"

# Line/screen control
_CLEAR_LINE = "\x1b[2K"
_CLEAR_LINE_TO_END = "\x1b[0K"
_CLEAR_LINE_TO_START = "\x1b[1K"
_CLEAR_SCREEN = "\x1b[2J"
_CLEAR_SCREEN_TO_END = "\x1b[0J"
_MOVE_TO_START = "\r"


# ======================================================================================================================
# Style System
# ======================================================================================================================

@dataclass(frozen=True)
class CliStyle:
    """
    ANSI style descriptor combining colors and formatting.

    :param fg: Foreground color name or 256-color int (0-255) or RGB tuple.
    :param bg: Background color name or 256-color int (0-255) or RGB tuple.
    :param bold: Enable bold.
    :param dim: Enable dim/faint.
    :param italic: Enable italic (terminal support varies).
    :param underline: Enable underline.
    :param inverse: Swap foreground and background.
    :param strikethrough: Enable strikethrough.
    """

    fg: str | int | tuple[int, int, int] | None = None
    bg: str | int | tuple[int, int, int] | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False
    inverse: bool = False
    strikethrough: bool = False

    def __add__(self, other: "CliStyle") -> "CliStyle":
        """Merge two styles, with `other` taking precedence for colors."""
        return CliStyle(
            fg=other.fg if other.fg is not None else self.fg,
            bg=other.bg if other.bg is not None else self.bg,
            bold=self.bold or other.bold,
            dim=self.dim or other.dim,
            italic=self.italic or other.italic,
            underline=self.underline or other.underline,
            inverse=self.inverse or other.inverse,
            strikethrough=self.strikethrough or other.strikethrough,
        )


class Style:
    """Predefined semantic styles for CLI output."""

    # Text formatting
    BOLD = CliStyle(bold=True)
    DIM = CliStyle(dim=True)
    ITALIC = CliStyle(italic=True)
    UNDERLINE = CliStyle(underline=True)
    INVERSE = CliStyle(inverse=True)
    STRIKETHROUGH = CliStyle(strikethrough=True)

    # Semantic colors
    SUCCESS = CliStyle(fg="green", bold=True)
    ERROR = CliStyle(fg="red", bold=True)
    WARNING = CliStyle(fg="yellow", bold=True)
    INFO = CliStyle(fg="blue")
    HINT = CliStyle(fg="cyan", dim=True)
    MUTED = CliStyle(dim=True)

    # UI elements
    TITLE = CliStyle(fg="white", bold=True)
    SECTION = CliStyle(fg="cyan", bold=True)
    KEY = CliStyle(fg="blue", bold=True)
    VALUE = CliStyle(fg="white")
    HIGHLIGHT = CliStyle(fg="yellow", bold=True)
    LINK = CliStyle(fg="cyan", underline=True)

    # Boxes and borders
    BORDER = CliStyle(fg="bright_black")
    BORDER_ACCENT = CliStyle(fg="cyan")

    # Status indicators
    OK = CliStyle(fg="green")
    FAIL = CliStyle(fg="red")
    SKIP = CliStyle(fg="yellow")
    PENDING = CliStyle(fg="blue")


def _color_code(color: str | int | tuple[int, int, int], is_bg: bool = False) -> str:
    """
    Convert color specification to ANSI code.

    :param color: Color name, 256-color index, or RGB tuple.
    :param is_bg: True for background color.
    :return: ANSI color code string.
    """
    if isinstance(color, str):
        table = _BG_COLORS if is_bg else _FG_COLORS
        return table.get(color, table["default"])

    if isinstance(color, int):
        # 256-color mode: \x1b[38;5;{n}m (fg) or \x1b[48;5;{n}m (bg)
        prefix = "48" if is_bg else "38"
        return f"{prefix};5;{color}"

    if isinstance(color, tuple) and len(color) == 3:
        # Truecolor: \x1b[38;2;{r};{g};{b}m (fg) or \x1b[48;2;{r};{g};{b}m (bg)
        prefix = "48" if is_bg else "38"
        r, g, b = color
        return f"{prefix};2;{r};{g};{b}"

    return "39" if not is_bg else "49"


def style_text(text: str, style: CliStyle | None = None) -> str:
    """
    Apply ANSI styling to text if ANSI is enabled.

    :param text: Text to style.
    :param style: Style to apply.
    :return: Styled text (or plain text if ANSI disabled).
    """
    if style is None or not ansi_enabled():
        return text

    codes: list[str] = []

    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.underline:
        codes.append("4")
    if style.inverse:
        codes.append("7")
    if style.strikethrough:
        codes.append("9")

    if style.fg is not None:
        codes.append(_color_code(style.fg, is_bg=False))
    if style.bg is not None:
        codes.append(_color_code(style.bg, is_bg=True))

    if not codes:
        return text

    return f"\x1b[{';'.join(codes)}m{text}{_ANSI_RESET}"


def strip_ansi(text: str) -> str:
    """
    Remove all ANSI escape codes from text.

    :param text: Text potentially containing ANSI codes.
    :return: Plain text without ANSI codes.
    """
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def visible_len(text: str) -> int:
    """
    Get visible length of text (excluding ANSI codes).

    :param text: Text potentially containing ANSI codes.
    :return: Number of visible characters.
    """
    return len(strip_ansi(text))


# ======================================================================================================================
# Text Wrapping with ANSI Support
# ======================================================================================================================

def wrap_text(
    text: str,
    width: int,
    *,
    initial_indent: str = "",
    subsequent_indent: str = "",
    preserve_ansi: bool = True,
) -> list[str]:
    """
    Word-wrap text to specified width, preserving ANSI codes.

    :param text: Text to wrap.
    :param width: Maximum line width.
    :param initial_indent: Indent for first line.
    :param subsequent_indent: Indent for continuation lines.
    :param preserve_ansi: If True, ANSI codes are preserved across line breaks.
    :return: List of wrapped lines.
    """
    if width <= 0:
        return [text]

    # Extract ANSI codes and plain text
    ansi_pattern = re.compile(r"(\x1b\[[0-9;]*[A-Za-z])")
    parts = ansi_pattern.split(text)

    lines: list[str] = []
    current_line = initial_indent
    current_len = visible_len(initial_indent)
    active_codes: list[str] = []  # Track active ANSI codes for continuation

    for part in parts:
        if ansi_pattern.match(part):
            # This is an ANSI code
            if preserve_ansi:
                if part == _ANSI_RESET:
                    active_codes.clear()
                else:
                    active_codes.append(part)
            current_line += part
            continue

        # This is plain text - wrap it
        words = part.split(" ")
        for i, word in enumerate(words):
            word_len = len(word)

            # Add space before word (except at line start)
            space = " " if (current_len > visible_len(initial_indent if not lines else subsequent_indent)) else ""
            space_len = len(space)

            if current_len + space_len + word_len > width and current_len > visible_len(
                initial_indent if not lines else subsequent_indent
            ):
                # Start new line
                if preserve_ansi and active_codes:
                    current_line += _ANSI_RESET
                lines.append(current_line)

                # New line with continuation indent
                current_line = subsequent_indent
                if preserve_ansi and active_codes:
                    current_line += "".join(active_codes)
                current_len = visible_len(subsequent_indent)
                space = ""

            current_line += space + word
            current_len += len(space) + word_len

    # Don't forget the last line
    if current_line.strip() or (preserve_ansi and active_codes):
        if preserve_ansi and active_codes:
            current_line += _ANSI_RESET
        lines.append(current_line)

    return lines if lines else [""]


# ======================================================================================================================
# Word-Level Highlighting
# ======================================================================================================================

@dataclass
class HighlightSpan:
    """A span of text to highlight."""

    start: int
    end: int
    style: CliStyle


def highlight_spans(text: str, spans: Sequence[HighlightSpan]) -> str:
    """
    Apply styles to specific spans of text.

    :param text: Original text.
    :param spans: List of spans to highlight.
    :return: Text with ANSI styling applied to spans.
    """
    if not spans or not ansi_enabled():
        return text

    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda s: s.start)

    result = []
    pos = 0

    for span in sorted_spans:
        if span.start < pos:
            continue  # Skip overlapping spans
        if span.start > pos:
            result.append(text[pos : span.start])
        result.append(style_text(text[span.start : span.end], span.style))
        pos = span.end

    if pos < len(text):
        result.append(text[pos:])

    return "".join(result)


def highlight_regex(
    text: str,
    pattern: str | re.Pattern[str],
    style: CliStyle,
    *,
    flags: int = 0,
) -> str:
    """
    Highlight all regex matches in text.

    :param text: Text to search.
    :param pattern: Regex pattern.
    :param style: Style to apply to matches.
    :param flags: Regex flags.
    :return: Text with matches highlighted.
    """
    if not ansi_enabled():
        return text

    if isinstance(pattern, str):
        pattern = re.compile(pattern, flags)

    def replacer(m: re.Match[str]) -> str:
        return style_text(m.group(0), style)

    return pattern.sub(replacer, text)


def highlight_words(text: str, words: Sequence[str], style: CliStyle) -> str:
    """
    Highlight specific words in text (case-insensitive).

    :param text: Text to search.
    :param words: Words to highlight.
    :param style: Style to apply.
    :return: Text with words highlighted.
    """
    if not words or not ansi_enabled():
        return text

    # Build pattern that matches any of the words
    escaped = [re.escape(w) for w in words]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return highlight_regex(text, pattern, style)


# ======================================================================================================================
# Cursor and Screen Control
# ======================================================================================================================

class Cursor:
    """Low-level cursor and screen control (only when TTY)."""

    @staticmethod
    def _write(code: str) -> None:
        """Write escape code if TTY."""
        if is_stdout_tty() and ansi_enabled():
            sys.stdout.write(code)
            sys.stdout.flush()

    @staticmethod
    def up(n: int = 1) -> None:
        """Move cursor up n lines."""
        Cursor._write(_CURSOR_UP.format(n=n))

    @staticmethod
    def down(n: int = 1) -> None:
        """Move cursor down n lines."""
        Cursor._write(_CURSOR_DOWN.format(n=n))

    @staticmethod
    def forward(n: int = 1) -> None:
        """Move cursor forward n columns."""
        Cursor._write(_CURSOR_FORWARD.format(n=n))

    @staticmethod
    def back(n: int = 1) -> None:
        """Move cursor back n columns."""
        Cursor._write(_CURSOR_BACK.format(n=n))

    @staticmethod
    def save() -> None:
        """Save cursor position."""
        Cursor._write(_CURSOR_SAVE)

    @staticmethod
    def restore() -> None:
        """Restore cursor position."""
        Cursor._write(_CURSOR_RESTORE)

    @staticmethod
    def hide() -> None:
        """Hide cursor."""
        Cursor._write(_CURSOR_HIDE)

    @staticmethod
    def show() -> None:
        """Show cursor."""
        Cursor._write(_CURSOR_SHOW)

    @staticmethod
    def to_start() -> None:
        """Move cursor to start of line."""
        Cursor._write(_MOVE_TO_START)

    @staticmethod
    def clear_line() -> None:
        """Clear entire current line."""
        Cursor._write(_MOVE_TO_START + _CLEAR_LINE)

    @staticmethod
    def clear_to_end() -> None:
        """Clear from cursor to end of line."""
        Cursor._write(_CLEAR_LINE_TO_END)

    @staticmethod
    def clear_screen() -> None:
        """Clear entire screen."""
        Cursor._write(_CLEAR_SCREEN + "\x1b[H")  # Clear and move to top-left


# ======================================================================================================================
# Spinner
# ======================================================================================================================

class Spinner:
    """
    Animated spinner for long-running operations.

    Usage:
        with Spinner("Loading..."):
            do_work()

        # Or manual control:
        spinner = Spinner("Processing...")
        spinner.start()
        try:
            do_work()
        finally:
            spinner.stop()
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_ASCII = ["|", "/", "-", "\\"]

    def __init__(
        self,
        message: str = "",
        *,
        style: CliStyle | None = Style.INFO,
        frames: Sequence[str] | None = None,
        interval: float = 0.1,
        stream: TextIO | None = None,
    ) -> None:
        """
        Initialize spinner.

        :param message: Message to display next to spinner.
        :param style: Style for the spinner character.
        :param frames: Custom animation frames.
        :param interval: Seconds between frame updates.
        :param stream: Output stream (default: stdout).
        """
        self.message = message
        self.style = style
        self.interval = interval
        self.stream = stream or sys.stdout

        # Use ASCII frames if not interactive or ANSI disabled
        if frames is not None:
            self.frames = list(frames)
        elif is_interactive() and ansi_enabled():
            self.frames = self.FRAMES
        else:
            self.frames = self.FRAMES_ASCII

        self._frame_iter: Iterator[str] = itertools.cycle(self.frames)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def _spin(self) -> None:
        """Animation loop (runs in background thread)."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._stop_event.is_set():
                    break
                frame = next(self._frame_iter)
                styled_frame = style_text(frame, self.style)
                line = f"\r{styled_frame} {self.message}"
                self.stream.write(line)
                self.stream.flush()
            self._stop_event.wait(self.interval)

    def start(self) -> "Spinner":
        """Start the spinner animation."""
        if not is_interactive():
            # Non-interactive: just print message once
            self.stream.write(f"{self.message}...\n")
            self.stream.flush()
            return self

        Cursor.hide()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, final_message: str | None = None, style: CliStyle | None = None) -> None:
        """
        Stop the spinner and optionally show a final message.

        :param final_message: Message to show after stopping.
        :param style: Style for the final message.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if is_interactive():
            Cursor.clear_line()
            Cursor.show()

            if final_message is not None:
                styled = style_text(final_message, style)
                self.stream.write(f"{styled}\n")
                self.stream.flush()

    def update(self, message: str) -> None:
        """Update the spinner message."""
        with self._lock:
            self.message = message

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        if exc_type is not None:
            self.stop(f"✗ {self.message} - Failed", Style.ERROR)
        else:
            self.stop(f"✓ {self.message}", Style.SUCCESS)


# ======================================================================================================================
# Status Line
# ======================================================================================================================

class StatusLine:
    """
    A persistent status line that can be updated in place.

    Usage:
        status = StatusLine()
        status.update("Step 1/3: Loading...")
        do_step_1()
        status.update("Step 2/3: Processing...")
        do_step_2()
        status.clear()
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._active = False

    def update(self, message: str, style: CliStyle | None = None) -> None:
        """Update the status line."""
        if not is_interactive():
            # Non-interactive: print each update on new line
            self.stream.write(f"{strip_ansi(message)}\n")
            self.stream.flush()
            return

        styled = style_text(message, style)
        # Truncate to terminal width
        width = get_terminal_width()
        if visible_len(styled) > width:
            styled = strip_ansi(styled)[: width - 3] + "..."

        Cursor.clear_line()
        self.stream.write(f"\r{styled}")
        self.stream.flush()
        self._active = True

    def clear(self) -> None:
        """Clear the status line."""
        if self._active and is_interactive():
            Cursor.clear_line()
            self.stream.write("\r")
            self.stream.flush()
        self._active = False


# ======================================================================================================================
# Layout Helpers
# ======================================================================================================================

# Box drawing characters
class BoxChars:
    """Unicode box drawing characters."""

    # Single line
    HORIZONTAL = "─"
    VERTICAL = "│"
    TOP_LEFT = "┌"
    TOP_RIGHT = "┐"
    BOTTOM_LEFT = "└"
    BOTTOM_RIGHT = "┘"
    T_DOWN = "┬"
    T_UP = "┴"
    T_RIGHT = "├"
    T_LEFT = "┤"
    CROSS = "┼"

    # Double line
    D_HORIZONTAL = "═"
    D_VERTICAL = "║"
    D_TOP_LEFT = "╔"
    D_TOP_RIGHT = "╗"
    D_BOTTOM_LEFT = "╚"
    D_BOTTOM_RIGHT = "╝"

    # Rounded
    R_TOP_LEFT = "╭"
    R_TOP_RIGHT = "╮"
    R_BOTTOM_LEFT = "╰"
    R_BOTTOM_RIGHT = "╯"

    # Heavy
    H_HORIZONTAL = "━"
    H_VERTICAL = "┃"


@dataclass
class BoxStyle:
    """Box border style configuration."""

    horizontal: str = BoxChars.HORIZONTAL
    vertical: str = BoxChars.VERTICAL
    top_left: str = BoxChars.TOP_LEFT
    top_right: str = BoxChars.TOP_RIGHT
    bottom_left: str = BoxChars.BOTTOM_LEFT
    bottom_right: str = BoxChars.BOTTOM_RIGHT
    color: CliStyle | None = Style.BORDER


# Predefined box styles
class BoxStyles:
    """Predefined box styles."""

    SINGLE = BoxStyle()
    DOUBLE = BoxStyle(
        horizontal=BoxChars.D_HORIZONTAL,
        vertical=BoxChars.D_VERTICAL,
        top_left=BoxChars.D_TOP_LEFT,
        top_right=BoxChars.D_TOP_RIGHT,
        bottom_left=BoxChars.D_BOTTOM_LEFT,
        bottom_right=BoxChars.D_BOTTOM_RIGHT,
    )
    ROUNDED = BoxStyle(
        top_left=BoxChars.R_TOP_LEFT,
        top_right=BoxChars.R_TOP_RIGHT,
        bottom_left=BoxChars.R_BOTTOM_LEFT,
        bottom_right=BoxChars.R_BOTTOM_RIGHT,
    )
    HEAVY = BoxStyle(
        horizontal=BoxChars.H_HORIZONTAL,
        vertical=BoxChars.H_VERTICAL,
    )
    ASCII = BoxStyle(
        horizontal="-",
        vertical="|",
        top_left="+",
        top_right="+",
        bottom_left="+",
        bottom_right="+",
    )


def hr(
    *,
    char: str = BoxChars.HORIZONTAL,
    title: str | None = None,
    width: int | None = None,
    style: CliStyle | None = Style.BORDER,
    title_style: CliStyle | None = Style.SECTION,
) -> str:
    """
    Create a horizontal rule.

    :param char: Character to use for the line.
    :param title: Optional centered title.
    :param width: Line width (default: terminal width).
    :param style: Style for the line.
    :param title_style: Style for the title.
    :return: Formatted horizontal rule string.
    """
    w = width or get_terminal_width()
    if w <= 0:
        w = 80

    if not title:
        line = char * w
        return style_text(line, style)

    # Title with padding
    title_text = f" {title} "
    title_visible_len = len(title_text)

    if title_visible_len >= w:
        return style_text(title_text[:w], title_style)

    left_len = (w - title_visible_len) // 2
    right_len = w - title_visible_len - left_len

    left = style_text(char * left_len, style)
    center = style_text(title_text, title_style)
    right = style_text(char * right_len, style)

    return left + center + right


def banner(
    title: str,
    *,
    subtitle: str | None = None,
    width: int | None = None,
    style: CliStyle | None = Style.TITLE,
    border_style: BoxStyle | None = None,
    padding: int = 1,
) -> list[str]:
    """
    Create a banner/heading block.

    :param title: Main title text.
    :param subtitle: Optional subtitle.
    :param width: Banner width (default: terminal width).
    :param style: Style for the title text.
    :param border_style: Box style for the border.
    :param padding: Vertical padding lines.
    :return: List of formatted lines.
    """
    w = width or get_terminal_width()
    bs = border_style or BoxStyles.DOUBLE

    lines: list[str] = []

    # Top border
    top = bs.top_left + bs.horizontal * (w - 2) + bs.top_right
    lines.append(style_text(top, bs.color))

    # Padding
    empty = bs.vertical + " " * (w - 2) + bs.vertical
    for _ in range(padding):
        lines.append(style_text(empty, bs.color))

    # Title (centered)
    title_stripped = strip_ansi(title)
    title_padded = title_stripped.center(w - 4)
    title_line = bs.vertical + " " + style_text(title_padded, style) + " " + bs.vertical
    # Fix: need to handle the border styling separately
    title_line = (
        style_text(bs.vertical, bs.color)
        + " "
        + style_text(title_stripped.center(w - 4), style)
        + " "
        + style_text(bs.vertical, bs.color)
    )
    lines.append(title_line)

    # Subtitle (centered, dimmer)
    if subtitle:
        sub_stripped = strip_ansi(subtitle)
        sub_line = (
            style_text(bs.vertical, bs.color)
            + " "
            + style_text(sub_stripped.center(w - 4), Style.MUTED)
            + " "
            + style_text(bs.vertical, bs.color)
        )
        lines.append(sub_line)

    # Padding
    for _ in range(padding):
        lines.append(style_text(empty, bs.color))

    # Bottom border
    bottom = bs.bottom_left + bs.horizontal * (w - 2) + bs.bottom_right
    lines.append(style_text(bottom, bs.color))

    return lines


def box(
    content: Sequence[str],
    *,
    title: str | None = None,
    width: int | None = None,
    border_style: BoxStyle | None = None,
    padding: int = 0,
) -> list[str]:
    """
    Create a bordered box around content.

    :param content: Lines of content to box.
    :param title: Optional title in top border.
    :param width: Box width (default: fits content or terminal width).
    :param border_style: Box style for the border.
    :param padding: Horizontal padding inside box.
    :return: List of formatted lines.
    """
    bs = border_style or BoxStyles.SINGLE
    term_width = get_terminal_width()

    # Calculate width
    if width is None:
        max_content_len = max((visible_len(line) for line in content), default=0)
        width = min(max_content_len + 4 + padding * 2, term_width)

    inner_width = width - 2 - padding * 2
    lines: list[str] = []

    # Top border (with optional title)
    if title:
        title_part = f" {title} "
        remaining = width - 2 - len(title_part)
        if remaining > 0:
            top = bs.top_left + title_part + bs.horizontal * remaining + bs.top_right
        else:
            top = bs.top_left + bs.horizontal * (width - 2) + bs.top_right
    else:
        top = bs.top_left + bs.horizontal * (width - 2) + bs.top_right
    lines.append(style_text(top, bs.color))

    # Content lines
    pad = " " * padding
    for line in content:
        stripped = strip_ansi(line)
        if len(stripped) > inner_width:
            # Truncate
            display = stripped[: inner_width - 3] + "..."
        else:
            display = stripped.ljust(inner_width)
        content_line = (
            style_text(bs.vertical, bs.color)
            + pad
            + display
            + pad
            + style_text(bs.vertical, bs.color)
        )
        lines.append(content_line)

    # Bottom border
    bottom = bs.bottom_left + bs.horizontal * (width - 2) + bs.bottom_right
    lines.append(style_text(bottom, bs.color))

    return lines


def text_box(
    text: str,
    *,
    title: str | None = None,
    width: int | None = None,
    border_style: BoxStyle | None = None,
    padding: int = 1,
    content_style: CliStyle | None = None,
) -> list[str]:
    """
    Create a bordered box with word-wrapped text content.

    Unlike ``box()``, which takes a list of pre-formatted lines and truncates
    those exceeding the box width, ``text_box()`` accepts a raw text string
    (potentially containing newlines) and word-wraps it so the full content is
    always displayed within the box borders.

    This is ideal for displaying long-form text such as LLM answers, document
    excerpts, or any content that should not be truncated.

    :param text: Text content to display. May contain newlines (``\\n``).
    :param title: Optional title in the top border.
    :param width: Box width (default: terminal width, capped at reasonable limit).
    :param border_style: Box style for the border.
    :param padding: Horizontal padding (spaces) inside the box on each side.
    :param content_style: Optional style applied to the content text.
    :return: List of formatted lines ready for output.
    """
    bs = border_style or BoxStyles.SINGLE
    term_width = get_terminal_width()

    if width is None:
        width = min(term_width, 120)
    width = max(width, 20)  # enforce a minimum width

    inner_width = width - 2 - padding * 2  # subtract borders + padding
    pad = " " * padding

    lines: list[str] = []

    # ── Top border (with optional title) ────────────────────────────
    if title:
        title_part = f" {title} "
        remaining = width - 2 - len(title_part)
        if remaining > 0:
            top = bs.top_left + title_part + bs.horizontal * remaining + bs.top_right
        else:
            top = bs.top_left + bs.horizontal * (width - 2) + bs.top_right
    else:
        top = bs.top_left + bs.horizontal * (width - 2) + bs.top_right
    lines.append(style_text(top, bs.color))

    # ── Word-wrap the text into content lines ───────────────────────
    # Split by existing newlines first, then word-wrap each paragraph.
    raw_paragraphs = text.split("\n")
    wrapped_lines: list[str] = []

    for paragraph in raw_paragraphs:
        paragraph = paragraph.rstrip()
        if not paragraph:
            # Preserve blank lines from the original text
            wrapped_lines.append("")
            continue

        plain = strip_ansi(paragraph)
        if len(plain) <= inner_width:
            wrapped_lines.append(paragraph)
        else:
            # Word-wrap the paragraph
            remaining_text = plain
            while remaining_text:
                if len(remaining_text) <= inner_width:
                    wrapped_lines.append(remaining_text)
                    break
                # Find a word boundary to break at
                break_at = remaining_text.rfind(" ", 0, inner_width)
                if break_at <= inner_width // 4:
                    # No good word boundary; hard break
                    break_at = inner_width
                wrapped_lines.append(remaining_text[:break_at].rstrip())
                remaining_text = remaining_text[break_at:].lstrip()

    # ── Render each content line ────────────────────────────────────
    for wl in wrapped_lines:
        plain_wl = strip_ansi(wl)
        padded = plain_wl.ljust(inner_width)
        if content_style is not None:
            padded = style_text(padded, content_style)
        content_line = (
            style_text(bs.vertical, bs.color)
            + pad
            + padded
            + pad
            + style_text(bs.vertical, bs.color)
        )
        lines.append(content_line)

    # ── Bottom border ───────────────────────────────────────────────
    bottom = bs.bottom_left + bs.horizontal * (width - 2) + bs.bottom_right
    lines.append(style_text(bottom, bs.color))

    return lines


def columns(
    items: Sequence[tuple[str, str]],
    *,
    separator: str = ": ",
    key_style: CliStyle | None = Style.KEY,
    value_style: CliStyle | None = Style.VALUE,
    min_key_width: int = 0,
) -> list[str]:
    """
    Format key-value pairs in aligned columns.

    :param items: List of (key, value) tuples.
    :param separator: String between key and value.
    :param key_style: Style for keys.
    :param value_style: Style for values.
    :param min_key_width: Minimum width for key column.
    :return: List of formatted lines.
    """
    if not items:
        return []

    # Calculate key column width
    key_width = max(min_key_width, max(len(k) for k, _ in items))

    lines: list[str] = []
    for key, value in items:
        key_padded = key.ljust(key_width)
        styled_key = style_text(key_padded, key_style)
        styled_value = style_text(value, value_style)
        lines.append(f"{styled_key}{separator}{styled_value}")

    return lines


# ======================================================================================================================
# Interactive Prompts
# ======================================================================================================================

def ask(
    prompt: str,
    *,
    default: str | None = None,
    strip: bool = True,
    style: CliStyle | None = None,
) -> str:
    """
    Prompt for text input.

    :param prompt: Prompt message.
    :param default: Default value if empty input.
    :param strip: Strip whitespace from input.
    :param style: Style for the prompt.
    :return: User input or default.
    """
    styled_prompt = style_text(prompt, style)
    if default is not None:
        styled_prompt += style_text(f" [{default}]", Style.MUTED)
    styled_prompt += " "

    sys.stdout.write(styled_prompt)
    sys.stdout.flush()

    try:
        value = sys.stdin.readline()
        if value == "":
            return default or ""
        value = value.rstrip("\n").rstrip("\r\n")
        if strip:
            value = value.strip()
        return value if value else (default or "")
    except (EOFError, KeyboardInterrupt):
        print()
        return default or ""


def ask_int(
    prompt: str,
    *,
    default: int | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
    style: CliStyle | None = None,
) -> int | None:
    """
    Prompt for integer input with validation.

    :param prompt: Prompt message.
    :param default: Default value.
    :param min_val: Minimum allowed value.
    :param max_val: Maximum allowed value.
    :param style: Style for the prompt.
    :return: Valid integer or None on cancel.
    """
    while True:
        value = ask(prompt, default=str(default) if default is not None else None, style=style)
        if not value and default is not None:
            return default
        if not value:
            return None

        try:
            num = int(value)
        except ValueError:
            print(style_text(f"  Invalid number: {value}", Style.ERROR))
            continue

        if min_val is not None and num < min_val:
            print(style_text(f"  Value must be at least {min_val}", Style.ERROR))
            continue
        if max_val is not None and num > max_val:
            print(style_text(f"  Value must be at most {max_val}", Style.ERROR))
            continue

        return num


def ask_float(
    prompt: str,
    *,
    default: float | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    style: CliStyle | None = None,
) -> float | None:
    """
    Prompt for float input with validation.

    :param prompt: Prompt message.
    :param default: Default value.
    :param min_val: Minimum allowed value.
    :param max_val: Maximum allowed value.
    :param style: Style for the prompt.
    :return: Valid float or None on cancel.
    """
    while True:
        value = ask(prompt, default=str(default) if default is not None else None, style=style)
        if not value and default is not None:
            return default
        if not value:
            return None

        try:
            num = float(value)
        except ValueError:
            print(style_text(f"  Invalid number: {value}", Style.ERROR))
            continue

        if min_val is not None and num < min_val:
            print(style_text(f"  Value must be at least {min_val}", Style.ERROR))
            continue
        if max_val is not None and num > max_val:
            print(style_text(f"  Value must be at most {max_val}", Style.ERROR))
            continue

        return num


def confirm(
    prompt: str,
    *,
    default: bool | None = None,
    style: CliStyle | None = None,
) -> bool:
    """
    Prompt for yes/no confirmation.

    :param prompt: Prompt message.
    :param default: Default value (True=yes, False=no, None=require input).
    :param style: Style for the prompt.
    :return: True for yes, False for no.
    """
    if default is True:
        hint = "[Y/n]"
    elif default is False:
        hint = "[y/N]"
    else:
        hint = "[y/n]"

    full_prompt = f"{prompt} {hint}"

    while True:
        value = ask(full_prompt, style=style).lower()

        if not value and default is not None:
            return default
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False

        print(style_text("  Please enter 'y' or 'n'", Style.HINT))


def ask_choice(
    prompt: str,
    choices: Sequence[str],
    *,
    default: int | None = None,
    style: CliStyle | None = None,
) -> int | None:
    """
    Prompt to select from a list of choices.

    :param prompt: Prompt message.
    :param choices: List of choices.
    :param default: Default choice index (1-based).
    :param style: Style for the prompt.
    :return: Selected index (0-based) or None on cancel.
    """
    if not choices:
        return None

    # Display choices
    print(style_text(prompt, style))
    for i, choice in enumerate(choices, 1):
        marker = style_text("→", Style.HIGHLIGHT) if i == default else " "
        num = style_text(f"  {i}.", Style.KEY)
        print(f"{marker}{num} {choice}")

    # Get selection
    while True:
        value = ask(
            "Enter number:",
            default=str(default) if default is not None else None,
            style=Style.HINT,
        )
        if not value:
            return (default - 1) if default is not None else None

        try:
            num = int(value)
        except ValueError:
            print(style_text(f"  Invalid number: {value}", Style.ERROR))
            continue

        if 1 <= num <= len(choices):
            return num - 1

        print(style_text(f"  Please enter a number between 1 and {len(choices)}", Style.ERROR))


def ask_secret(
    prompt: str,
    *,
    style: CliStyle | None = None,
) -> str:
    """
    Prompt for secret input (password, API key, etc.) without echoing.

    :param prompt: Prompt message.
    :param style: Style for the prompt.
    :return: Secret string.
    """
    styled_prompt = style_text(prompt, style)

    try:
        return getpass.getpass(styled_prompt + " ")
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def ask_masked(
    prompt: str,
    *,
    mask_char: str = "*",
    style: CliStyle | None = None,
) -> str:
    """
    Prompt for input with masked echo (shows * for each character).

    Note: This only works on TTY. Falls back to ask_secret() otherwise.

    :param prompt: Prompt message.
    :param mask_char: Character to display for each typed character.
    :param style: Style for the prompt.
    :return: Input string.
    """
    if not is_interactive():
        return ask_secret(prompt, style=style)

    styled_prompt = style_text(prompt, style) + " "
    sys.stdout.write(styled_prompt)
    sys.stdout.flush()

    # Try to use termios for character-by-character reading
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            chars: list[str] = []
            while True:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    break
                elif ch == "\x7f":  # Backspace
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                elif ch == "\x03":  # Ctrl+C
                    sys.stdout.write("\n")
                    raise KeyboardInterrupt
                elif ch >= " ":  # Printable
                    chars.append(ch)
                    sys.stdout.write(mask_char)
                sys.stdout.flush()
            return "".join(chars)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, OSError, AttributeError):
        # Fall back to getpass (termios not available or failed)
        return ask_secret(prompt, style=style)


# ======================================================================================================================
# Big Text (font-size approximation)
# ======================================================================================================================


def big_text(
    text: str,
    *,
    scale: int = 2,
    style: "CliStyle | None" = None,
) -> list[str]:
    """Render a 'bigger' version of a string for terminals.

    Terminals don't support per-line font sizes. This function approximates
    different sizes with a small built-in block font.

    Parameters
    ----------
    scale:
        1 -> single line
        2 -> 5-line block font
        3+ -> block font with additional vertical scaling
    """

    plain = strip_ansi(str(text))
    if scale <= 1 or plain.strip() == "":
        line = plain
        if style is not None:
            line = style_text(line, style)
        return [line]

    font: dict[str, list[str]] = {
        "A": ["  ██  ", " █  █ ", " ████ ", " █  █ ", " █  █ "],
        "B": [" ███  ", " █  █ ", " ███  ", " █  █ ", " ███  "],
        "C": ["  ███ ", " █    ", " █    ", " █    ", "  ███ "],
        "D": [" ███  ", " █  █ ", " █  █ ", " █  █ ", " ███  "],
        "E": [" ████ ", " █    ", " ███  ", " █    ", " ████ "],
        "F": [" ████ ", " █    ", " ███  ", " █    ", " █    "],
        "G": ["  ███ ", " █    ", " █ ██ ", " █  █ ", "  ███ "],
        "H": [" █  █ ", " █  █ ", " ████ ", " █  █ ", " █  █ "],
        "I": [" ███ ", "  █  ", "  █  ", "  █  ", " ███ "],
        "J": ["  ███ ", "   █  ", "   █  ", " █ █  ", "  ██  "],
        "K": [" █  █ ", " █ █  ", " ██   ", " █ █  ", " █  █ "],
        "L": [" █    ", " █    ", " █    ", " █    ", " ████ "],
        "M": [" █   █ ", " ██ ██ ", " █ █ █ ", " █   █ ", " █   █ "],
        "N": [" █   █ ", " ██  █ ", " █ █ █ ", " █  ██ ", " █   █ "],
        "O": ["  ██  ", " █  █ ", " █  █ ", " █  █ ", "  ██  "],
        "P": [" ███  ", " █  █ ", " ███  ", " █    ", " █    "],
        "Q": ["  ██  ", " █  █ ", " █  █ ", " █ ██ ", "  ███ "],
        "R": [" ███  ", " █  █ ", " ███  ", " █ █  ", " █  █ "],
        "S": ["  ███ ", " █    ", "  ██  ", "    █ ", " ███  "],
        "T": [" ████ ", "  ██  ", "  ██  ", "  ██  ", "  ██  "],
        "U": [" █  █ ", " █  █ ", " █  █ ", " █  █ ", "  ██  "],
        "V": [" █   █ ", " █   █ ", " █   █ ", "  █ █  ", "   █   "],
        "W": [" █   █ ", " █   █ ", " █ █ █ ", " ██ ██ ", " █   █ "],
        "X": [" █   █ ", "  █ █  ", "   █   ", "  █ █  ", " █   █ "],
        "Y": [" █   █ ", "  █ █  ", "   █   ", "   █   ", "   █   "],
        "Z": [" ████ ", "    █ ", "   █  ", "  █   ", " ████ "],
        "0": ["  ██  ", " █  █ ", " █ ██ ", " ██ █ ", "  ██  "],
        "1": ["  █  ", " ██  ", "  █  ", "  █  ", " ███ "],
        "2": [" ███ ", "    █", "  ██ ", " █   ", " ████"],
        "3": [" ███ ", "    █", "  ██ ", "    █", " ███ "],
        "4": [" █  █", " █  █", " ████", "    █", "    █"],
        "5": [" ████", " █   ", " ███ ", "    █", " ███ "],
        "6": ["  ███", " █   ", " ███ ", " █  █", "  ██ "],
        "7": [" ████", "    █", "   █ ", "  █  ", "  █  "],
        "8": ["  ██ ", " █  █", "  ██ ", " █  █", "  ██ "],
        "9": ["  ██ ", " █  █", "  ███", "    █", " ███ "],
        "-": ["      ", "      ", " ███ ", "      ", "      "],
        ".": [" ", " ", " ", " ", "█"],
        ":": [" ", "█", " ", "█", " "],
        " ": ["  ", "  ", "  ", "  ", "  "],
    }

    chars = [c.upper() for c in plain]
    lines = ["" for _ in range(5)]
    for c in chars:
        glyph = font.get(c)
        if glyph is None:
            glyph = [f" {c} "] * 5
        for i in range(5):
            lines[i] += glyph[i] + " "

    vrep = max(1, int(scale) - 1)
    out: list[str] = []
    for ln in lines:
        base = ln.rstrip()
        if style is not None:
            base = style_text(base, style)
        for _ in range(vrep):
            out.append(base)
    return out


# ======================================================================================================================
# Lightweight CLI UI Manager (restored)
# ======================================================================================================================


class CliUI:
    """High-level CLI UI manager.

    This is intentionally dependency-free and is used both standalone (`ui = CliUI()`)
    and via `logger.ui`.
    """

    def __init__(self, *, max_width: int = 120, stream: TextIO | None = None) -> None:
        self.max_width = max_width
        self.stream = stream or sys.stdout
        self._write_func: Callable[[str], None] | None = None

    def set_writer(self, writer: Callable[[str], None]) -> None:
        self._write_func = writer

    def _write(self, text: str, end: str = "\n") -> None:
        if self._write_func is not None:
            self._write_func(text + end if end else text)
        else:
            self.stream.write(text + end)
            try:
                self.stream.flush()
            except Exception:
                pass

    def _effective_width(self) -> int:
        term_width = get_terminal_width()
        return min(term_width, self.max_width) if self.max_width > 0 else term_width

    # Styled output
    def styled(self, text: str, style: CliStyle | None = None) -> None:
        self._write(style_text(text, style))

    def success(self, text: str) -> None:
        self._write(style_text(f"✓ {text}", Style.SUCCESS))

    def error(self, text: str) -> None:
        self._write(style_text(f"✗ {text}", Style.ERROR))

    def warning(self, text: str) -> None:
        self._write(style_text(f"⚠ {text}", Style.WARNING))

    def info(self, text: str) -> None:
        self._write(style_text(f"ℹ {text}", Style.INFO))

    def hint(self, text: str) -> None:
        self._write(style_text(f"💡 {text}", Style.HINT))

    # Layout
    def hr(
        self,
        *,
        char: str = "─",
        title: str | None = None,
        style: CliStyle | None = Style.BORDER,
        title_style: CliStyle | None = Style.SECTION,
    ) -> None:
        self._write(hr(char=char, title=title, width=self._effective_width(), style=style, title_style=title_style))

    def banner(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        width: int | None = None,
        style: CliStyle | None = Style.TITLE,
        border_style: "BoxStyle | None" = None,
        padding: int = 1,
    ) -> None:
        for line in banner(
            title,
            subtitle=subtitle,
            width=width or self._effective_width(),
            style=style,
            border_style=border_style,
            padding=padding,
        ):
            self._write(line)

    def box(
        self,
        content: Sequence[str],
        *,
        title: str | None = None,
        border_style: "BoxStyle | None" = None,
    ) -> None:
        for line in box(content, title=title, width=self._effective_width(), border_style=border_style):
            self._write(line)

    def text_box(
        self,
        text: str,
        *,
        title: str | None = None,
        border_style: "BoxStyle | None" = None,
        padding: int = 1,
        content_style: "CliStyle | None" = None,
    ) -> None:
        """Print word-wrapped text in a bordered box (no truncation)."""
        for line in text_box(
            text,
            title=title,
            width=self._effective_width(),
            border_style=border_style,
            padding=padding,
            content_style=content_style,
        ):
            self._write(line)

    def kv(
        self,
        items: Sequence[tuple[str, str]],
        *,
        separator: str = ": ",
        key_style: CliStyle | None = Style.KEY,
        value_style: CliStyle | None = Style.VALUE,
    ) -> None:
        for line in columns(items, separator=separator, key_style=key_style, value_style=value_style):
            self._write(line)

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        border_style: "BoxStyle | None" = None,
    ) -> None:
        """Print a simple table."""
        if not headers and not rows:
            return

        bs = border_style or BoxStyles.SINGLE

        col_count = len(headers) if headers else (len(rows[0]) if rows else 0)
        widths = [0] * col_count

        for i, h in enumerate(headers):
            widths[i] = max(widths[i], visible_len(str(h)))
        for row in rows:
            for i, cell in enumerate(row):
                if i < col_count:
                    widths[i] = max(widths[i], visible_len(str(cell)))

        if headers:
            header_cells = [style_text(str(h).ljust(widths[i]), Style.KEY) for i, h in enumerate(headers)]
            self._write(f" {' │ '.join(header_cells)} ")
            sep_parts = [bs.horizontal * (w + 2) for w in widths]
            self._write(style_text("─".join(sep_parts), bs.color))

        for row in rows:
            cells = [str(row[i]).ljust(widths[i]) if i < len(row) else " " * widths[i] for i in range(col_count)]
            self._write(f" {' │ '.join(cells)} ")

    # Interactive wrappers
    def ask(self, prompt: str, **kwargs) -> str:
        return ask(prompt, **kwargs)

    def ask_int(self, prompt: str, **kwargs):
        return ask_int(prompt, **kwargs)

    def ask_float(self, prompt: str, **kwargs):
        return ask_float(prompt, **kwargs)

    def confirm(self, prompt: str, **kwargs) -> bool:
        return confirm(prompt, **kwargs)

    def ask_choice(self, prompt: str, choices: Sequence[str], **kwargs):
        return ask_choice(prompt, choices, **kwargs)

    def ask_secret(self, prompt: str, **kwargs) -> str:
        return ask_secret(prompt, **kwargs)

    def spinner(self, message: str = "", **kwargs) -> Spinner:
        return Spinner(message, stream=self.stream, **kwargs)

    def status(self) -> StatusLine:
        return StatusLine(stream=self.stream)

    # Big text
    def big(self, text: str, *, scale: int = 2, style: CliStyle | None = Style.TITLE) -> None:
        for line in big_text(text, scale=scale, style=style):
            self._write(line)


# Global UI instance
ui = CliUI()


# ======================================================================================================================
# Logger Integration Mixin
# ======================================================================================================================

class LoggerUIMixin:
    """
    Mixin class that adds UI methods to a logger.

    This is designed to be mixed into _CliLogger to provide a `logger.ui` namespace.
    """

    _ui: CliUI | None = None

    @property
    def ui(self) -> CliUI:
        """Get the UI instance attached to this logger."""
        if self._ui is None:
            self._ui = CliUI()
            # Connect to logger.write if available
            if hasattr(self, "write"):
                self._ui.set_writer(lambda text: getattr(self, "write")(text, end=""))
        return self._ui
