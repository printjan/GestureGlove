# src/data_fusion_project/core/logger_setup.py
"""
Centralized STDOUT logger setup for the entire project.
- Format: [filename] [funcName] [LEVEL]: message
- Honors LOG_LEVEL env var (DEBUG, INFO, WARNING, ERROR, CRITICAL)
Usage Example:
from data_fusion_project.core.logger_setup import get_logger
logger = get_logger(__name__)
"""



# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations

import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Import CLI UI components for integration
from data_fusion_project.core.cli_ui import (
    CliUI,
    LoggerUIMixin,
    get_terminal_width,
    strip_ansi,
    ansi_enabled as cli_ansi_enabled,
)


# ======================================================================================================================
# constants
# ======================================================================================================================
_APP_LOGGER_NAME = "app"  # root name for the application's logger hierarchy

# Optional file logging (rotating)
_ENV_LOG_FILE = "LOG_FILE"
_ENV_LOG_FILE_MAX_BYTES = "LOG_FILE_MAX_BYTES"
_ENV_LOG_FILE_BACKUP_COUNT = "LOG_FILE_BACKUP_COUNT"

_DEFAULT_LOG_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_LOG_FILE_BACKUP_COUNT = 3

# Default maximum line width for log output (0 = no limit, use terminal width)
_DEFAULT_MAX_WIDTH = 500


# ANSI colors (kept minimal, no external dependency)
_ANSI_RESET = "\x1b[0m"
_ANSI_COLORS_BY_LEVEL: dict[int, str] = {
    logging.DEBUG: "\x1b[34m",  # blue
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[35m",  # magenta
}


@dataclass(frozen=True)
class _LogColumns:
    filename: int = 22
    funcname: int = 24
    level: int = 9


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean-ish env var."""

    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_level(level: str | int | None) -> int:
    """
    Resolve an integer logging level from input or LOG_LEVEL env var.
    """

    if isinstance(level, int):
        return level
    env = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    return getattr(logging, env, logging.INFO)


def _truncate_ellipsis(value: str, width: int) -> str:
    """Left-trim to `width`, using '...' suffix when truncated."""

    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    return value[: width - 3] + "..."


def _resolve_use_colors(use_colors: bool | None) -> bool:
    """
    Decide whether ANSI colors should be used.

    Precedence:
    1) explicit `use_colors` argument (True/False)
    2) LOG_COLORS env var (truthy/falsy)
    3) default: True

    Note: We intentionally default to True to improve readability in modern terminals.
    If your environment doesn't support ANSI, set LOG_COLORS=0 or pass use_colors=False.
    """


    if use_colors is not None:
        return bool(use_colors)
    return _env_flag("LOG_COLORS", True)


def _tqdm_write(text: str, *, end: str = "\n") -> None:
    """Best-effort tqdm-safe writer; falls back to plain stdout."""

    try:
        # tqdm is an optional dependency
        from tqdm import tqdm  # type: ignore

        tqdm.write(text, end=end)
        return
    except Exception:
        # Either tqdm isn't installed or tqdm.write failed; fall back.
        pass

    sys.stdout.write(text + (end or ""))
    try:
        sys.stdout.flush()
    except Exception:
        pass


class _ColumnFormatter(logging.Formatter):
    """
    Custom log formatter with column-aligned prefix and automatic line wrapping.

    Long messages are wrapped to max_width, with continuation lines showing
    the same prefix but with an indented continuation marker.
    """

    # Continuation marker for wrapped lines (indented to show it's a continuation)
    _CONTINUATION_MARKER = "    ↳ "

    def __init__(
        self,
        *,
        columns: _LogColumns,
        use_colors: bool = True,
        max_width: int = _DEFAULT_MAX_WIDTH,
    ) -> None:
        super().__init__()
        self._columns = columns
        self._use_colors = use_colors
        self._max_width = max_width

    def _level_color(self, levelno: int) -> str:
        if not self._use_colors:
            return ""
        return _ANSI_COLORS_BY_LEVEL.get(levelno, "")

    def _build_prefix(
        self,
        filename: str,
        funcname: str,
        levelname: str,
        levelno: int,
    ) -> tuple[str, int]:
        """
        Build the log prefix and return (styled_prefix, visible_length).

        :return: Tuple of (prefix string with ANSI codes, visible character count).
        """
        # Truncate and pad columns
        filename = _truncate_ellipsis(filename, self._columns.filename).ljust(self._columns.filename)
        funcname = _truncate_ellipsis(funcname, self._columns.funcname).ljust(self._columns.funcname)
        levelname_padded = _truncate_ellipsis(levelname, self._columns.level).ljust(self._columns.level)

        color = self._level_color(levelno)
        reset = _ANSI_RESET if (self._use_colors and color) else ""

        # Build prefix
        prefix = f"[{filename}] [{funcname}] {color}[{levelname_padded}]{reset}: "

        # Calculate visible length (without ANSI codes)
        visible_len = len(f"[{filename}] [{funcname}] [{levelname_padded}]: ")

        return prefix, visible_len

    def _wrap_message(self, msg: str, prefix: str, prefix_visible_len: int) -> str:
        """
        Wrap a message to fit within max_width, with continuation prefix on subsequent lines.

        :param msg: The message to wrap.
        :param prefix: The styled prefix for the first line.
        :param prefix_visible_len: Visible length of the prefix (without ANSI).
        :return: Wrapped message string with newlines.
        """
        # Determine effective max width
        if self._max_width <= 0:
            max_width = get_terminal_width(fallback=300)
        else:
            max_width = min(self._max_width, get_terminal_width(fallback=300))

        # Calculate available width for message text
        first_line_width = max_width - prefix_visible_len
        continuation_width = max_width - len(self._CONTINUATION_MARKER) - prefix_visible_len

        if first_line_width <= 10 or continuation_width <= 10:
            # Terminal too narrow, just return as-is
            return prefix + msg

        # Strip ANSI from message for length calculations, but preserve original for output
        msg_plain = strip_ansi(msg)

        # If message fits on one line, no wrapping needed
        if len(msg_plain) <= first_line_width:
            return prefix + msg

        # Wrap the plain text message
        lines: list[str] = []
        remaining = msg_plain

        # First line
        if len(remaining) > first_line_width:
            # Find a good break point (word boundary)
            break_at = remaining.rfind(" ", 0, first_line_width)
            if break_at <= first_line_width // 2:
                # No good break point, hard break
                break_at = first_line_width
            lines.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at:].lstrip()
        else:
            lines.append(remaining)
            remaining = ""

        # Continuation lines
        while remaining:
            if len(remaining) > continuation_width:
                break_at = remaining.rfind(" ", 0, continuation_width)
                if break_at <= continuation_width // 2:
                    break_at = continuation_width
                lines.append(remaining[:break_at].rstrip())
                remaining = remaining[break_at:].lstrip()
            else:
                lines.append(remaining)
                remaining = ""

        # Build output with prefix on first line and continuation marker on subsequent
        if len(lines) == 1:
            return prefix + lines[0]

        # For multi-line: first line gets normal prefix, rest get continuation
        result_lines = [prefix + lines[0]]
        for line in lines[1:]:
            # Continuation line: same prefix structure but with continuation marker
            cont_prefix = " " * prefix_visible_len + self._CONTINUATION_MARKER
            result_lines.append(cont_prefix + line)

        return "\n".join(result_lines)

    def format(self, record: logging.LogRecord) -> str:
        # Make message single-line first (normalize newlines)
        msg = record.getMessage()
        msg = msg.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

        filename = getattr(record, "filename", "")
        funcname = getattr(record, "funcName", "")
        levelname = getattr(record, "levelname", "")

        # Build prefix
        prefix, prefix_visible_len = self._build_prefix(filename, funcname, levelname, record.levelno)

        # Handle exceptions
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            exc_text = exc_text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            if exc_text:
                msg = f"{msg} | {exc_text}"

        # Wrap message if needed
        return self._wrap_message(msg, prefix, prefix_visible_len)


class _CliLogger(logging.Logger, LoggerUIMixin):
    """Project logger with `write()` for pure CLI output without prefix."""

    def write(self, message: str, *, end: str = "\n", tqdm_safe: bool = True) -> None:
        """
        Writes raw text to STDOUT without any logger prefix.

        If `tqdm` is installed and `tqdm_safe=True`, uses `tqdm.write(...)` to avoid
        breaking active progress bars.
        """

        if message is None:
            message = ""
        text = str(message)

        # Keep the original behavior w.r.t. `end`-handling.
        if end and text.endswith(end):
            text_to_write = text[: -len(end)]
            end_to_write = end
        else:
            text_to_write = text
            end_to_write = end or ""

        if tqdm_safe:
            _tqdm_write(text_to_write, end=end_to_write)
            return

        sys.stdout.write(text_to_write + (end_to_write or ""))
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def _render_prefixed_prompt(self, prompt: str) -> str:
        """Render a prompt line with the logger prefix, but without emitting a log record.

        We can't use `logger.info(prompt)` here because logging always terminates the line,
        forcing user input onto the next line.

        Implementation notes:
        - We purposely render the prompt without ANSI colors to keep it readable in any terminal
          and avoid edge cases with cursor positioning.
        - We also avoid automatic wrapping for prompts (prompts should be short; wrapping would
          break inline input UX).
        """

        app_logger = logging.getLogger(_APP_LOGGER_NAME)
        handler: logging.Handler | None = None
        for h in list(app_logger.handlers):
            if isinstance(h, logging.StreamHandler):
                handler = h
                break

        if handler is None or handler.formatter is None:
            return str(prompt)

        formatter = handler.formatter

        # If we use our _ColumnFormatter, temporarily render without colors and without wrapping.
        if isinstance(formatter, _ColumnFormatter):
            tmp = _ColumnFormatter(
                columns=formatter._columns,  # type: ignore[attr-defined]
                use_colors=False,
                max_width=10_000,  # effectively no wrapping for prompts
            )
            fmt = tmp
        else:
            fmt = formatter

        record = app_logger.makeRecord(
            name=app_logger.name,
            level=logging.INFO,
            fn=__file__,
            lno=0,
            msg="%s",
            args=(prompt,),
            exc_info=None,
            func="read_text",
            extra=None,
        )
        try:
            return fmt.format(record)
        except Exception:
            return str(prompt)

    def _read_text_impl(
        self,
        prompt: str,
        *,
        prompt_with_prefix: bool,
        echo: bool,
        echo_level: int,
        default: str | None,
        strip: bool,
        log_events: bool,
    ) -> str | None:
        """Internal helper implementing the 4 public read_text* variants."""

        try:
            if prompt_with_prefix:
                rendered = self._render_prefixed_prompt(prompt)
                # Avoid forcing an extra space if the prompt already ends with whitespace.
                suffix = "" if (rendered.endswith(" ") or rendered.endswith("\t")) else " "
                sys.stdout.write(rendered + suffix)
                sys.stdout.flush()
            else:
                sys.stdout.write(str(prompt))
                sys.stdout.flush()

            line = sys.stdin.readline()
            if line == "":
                if log_events:
                    self.warning("Terminal input returned EOF; using default.")
                return default

            value = line.replace("\r\n", "\n").replace("\r", "\n")
            value = value.rstrip("\n")
            if strip:
                value = value.strip()

            if echo and value is not None:
                self.log(echo_level, "%s", value)

            return value
        except KeyboardInterrupt:
            if log_events:
                self.info("Terminal input cancelled by user (KeyboardInterrupt); using default.")
            return default
        except Exception as exc:
            raise RuntimeError(f"Failed to read terminal input: {exc}") from exc

    def read_text_raw_echo(
        self,
        prompt: str,
        *,
        echo_level: int = logging.INFO,
        default: str | None = None,
        strip: bool = True,
        log_events: bool = True,
    ) -> str | None:
        """Reads one line from STDIN.

        - Shows the prompt without the logger prefix.
        - Echoes the read string as a prefixed log message.
        """

        return self._read_text_impl(
            prompt,
            prompt_with_prefix=False,
            echo=True,
            echo_level=echo_level,
            default=default,
            strip=strip,
            log_events=log_events,
        )

    def read_text_raw(
        self,
        prompt: str,
        *,
        default: str | None = None,
        strip: bool = True,
        log_events: bool = True,
    ) -> str | None:
        """Reads one line from STDIN.

        - Shows the prompt without the logger prefix.
        - Does not echo the read string.
        """

        return self._read_text_impl(
            prompt,
            prompt_with_prefix=False,
            echo=False,
            echo_level=logging.INFO,
            default=default,
            strip=strip,
            log_events=log_events,
        )

    def read_text_echo(
        self,
        prompt: str,
        *,
        echo_level: int = logging.INFO,
        default: str | None = None,
        strip: bool = True,
        log_events: bool = True,
    ) -> str | None:
        """Reads one line from STDIN.

        - Shows the prompt with the logger prefix.
        - Echoes the read string as a prefixed log message.
        """

        return self._read_text_impl(
            prompt,
            prompt_with_prefix=True,
            echo=True,
            echo_level=echo_level,
            default=default,
            strip=strip,
            log_events=log_events,
        )

    def read_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        strip: bool = True,
        log_events: bool = True,
    ) -> str | None:
        """Reads one line from STDIN.

        - Shows the prompt with the logger prefix.
        - Does not echo the read string.
        """

        return self._read_text_impl(
            prompt,
            prompt_with_prefix=True,
            echo=False,
            echo_level=logging.INFO,
            default=default,
            strip=strip,
            log_events=log_events,
        )


def tqdm_logging_redirect() -> "_TqdmLoggingRedirect":
    """Return a context-manager that makes `print()` + logging tqdm-safe within the block."""

    return _TqdmLoggingRedirect()


class _TqdmLoggingRedirect:
    """
    Redirects stdout/stderr + logging handler stream to a tqdm-safe writer.

    Use when you have active progress bars and want to ensure log output doesn't
    corrupt bar rendering.

    Notes:
    - Uses tqdm.write() if tqdm is installed.
    - Falls back to regular stdout if tqdm isn't available.
    """

    def __init__(self) -> None:
        self._orig_stdout = None
        self._orig_stderr = None
        self._prev_handler_stream = None
        self._handler = None

    def __enter__(self) -> "_TqdmLoggingRedirect":
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        sys.stdout = _TqdmFileProxy(self._orig_stdout)  # type: ignore[assignment]
        sys.stderr = _TqdmFileProxy(self._orig_stderr)  # type: ignore[assignment]

        # Also update our app logger handler stream (if configured).
        app_logger = logging.getLogger(_APP_LOGGER_NAME)
        for h in list(app_logger.handlers):
            if isinstance(h, logging.StreamHandler):
                self._handler = h
                self._prev_handler_stream = h.stream
                h.setStream(sys.stdout)
                break

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._handler is not None and self._prev_handler_stream is not None:
                self._handler.setStream(self._prev_handler_stream)
        finally:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout  # type: ignore[assignment]
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr  # type: ignore[assignment]


class _TqdmFileProxy:
    """A minimal file-like proxy that routes writes through tqdm.write."""

    def __init__(self, underlying) -> None:
        self._underlying = underlying

    def write(self, s: str) -> int:
        if s is None:
            return 0
        text = str(s)
        # tqdm/print often write partial lines; normalize by stripping trailing newlines and
        # letting tqdm.write add its own end.
        if text == "":
            return 0

        # Split to preserve multi-line prints without introducing wrapping.
        parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for i, part in enumerate(parts):
            if part == "" and i == len(parts) - 1:
                continue
            _tqdm_write(part, end="\n")
        return len(text)

    def flush(self) -> None:
        try:
            self._underlying.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._underlying, "isatty")())
        except Exception:
            return False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


class DailyFileHandler(logging.FileHandler):
    """
    A logging handler that writes to a daily log file named YYYY-MM-DD.log
    inside the specified directory. Automatically rolls over at midnight.
    """
    def __init__(self, directory: str | os.PathLike[str], mode: str = "a", encoding: str = "utf-8", delay: bool = False) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        initial_date = datetime.now().strftime("%Y-%m-%d")
        initial_file = self.directory / f"{initial_date}.log"
        super().__init__(filename=str(initial_file), mode=mode, encoding=encoding, delay=delay)
        self.current_date = initial_date

    def emit(self, record: logging.LogRecord) -> None:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.stream.close()
            self.baseFilename = os.path.abspath(self.directory / f"{today}.log")
            self.stream = self._open()
        super().emit(record)


def _configure_file_logging(
    *,
    log_file: str | os.PathLike[str] | None,
    level: int,
    rotate: bool = True,
) -> None:
    """Attach a file handler to the app logger once.

    File logging is plaintext (ANSI stripped) and uses the same column formatter,
    but with colors disabled.

    When rotate=True (default), uses RotatingFileHandler with size-based rotation.
    When rotate=False, uses a plain FileHandler — ideal for single pipeline runs
    where the entire log should be in one file.
    """

    if not log_file:
        return

    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    rotation_type = os.getenv("LOG_ROTATION", "rotating").strip().lower()

    # Avoid duplicates (e.g. multiple get_logger calls)
    for h in list(app_logger.handlers):
        if isinstance(h, DailyFileHandler):
            try:
                if Path(h.directory).resolve() == Path(log_file).expanduser().resolve():
                    return
            except Exception:
                return
        elif isinstance(h, (RotatingFileHandler, logging.FileHandler)):
            try:
                if Path(getattr(h, "baseFilename", "")).resolve() == Path(log_file).expanduser().resolve():
                    return
            except Exception:
                return

    path = Path(log_file).expanduser()

    if not rotate:
        # Plain FileHandler — no rotation, single file for the entire run
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            filename=str(path),
            mode="w",  # Overwrite on each run
            encoding="utf-8",
        )
    elif rotation_type == "daily":
        path.mkdir(parents=True, exist_ok=True)
        file_handler = DailyFileHandler(
            directory=path,
            encoding="utf-8",
        )
    else:
        max_bytes = _env_int(_ENV_LOG_FILE_MAX_BYTES, _DEFAULT_LOG_FILE_MAX_BYTES)
        backup_count = _env_int(_ENV_LOG_FILE_BACKUP_COUNT, _DEFAULT_LOG_FILE_BACKUP_COUNT)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=max(0, int(max_bytes)),
            backupCount=max(0, int(backup_count)),
            encoding="utf-8",
        )

    file_handler.setLevel(level)
    file_handler.setFormatter(
        _ColumnFormatter(
            columns=_LogColumns(),
            use_colors=False,
            max_width=_DEFAULT_MAX_WIDTH,
        )
    )

    app_logger.addHandler(file_handler)
    logging.getLogger().addHandler(file_handler)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).addHandler(file_handler)



def _configure_once(*, level: str | int | None = None, use_colors: bool | None = None) -> None:
    """
    Configure the top-level application logger exactly once.
    """

    if getattr(_configure_once, "_done", False):
        return

    # Handle encoding errors on Windows console gracefully
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(errors="replace")
            sys.stderr.reconfigure(errors="replace")
        except Exception:
            pass

    # Decide coloring (default: True).
    resolved_use_colors = _resolve_use_colors(use_colors)

    resolved_level = _to_level(level)

    # Ensure our custom logger class is used for all child loggers too.
    logging.setLoggerClass(_CliLogger)

    # Stream handler to STDOUT
    handler = logging.StreamHandler(stream=sys.stdout)

    handler.setFormatter(
        _ColumnFormatter(
            columns=_LogColumns(),
            use_colors=resolved_use_colors,
        )
    )

    # Create/get app root logger and attach the handler
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.setLevel(resolved_level)
    app_logger.addHandler(handler)
    app_logger.propagate = False  # prevent bubbling to root and double printing

    # Optional file logging via env vars
    _configure_file_logging(log_file=os.getenv(_ENV_LOG_FILE), level=resolved_level)

    # Forward Python warnings (warnings.warn) into logging as WARNING
    logging.captureWarnings(True)

    _configure_once._done = True


def get_logger(
    name: str | None = None,
    level: str | int | None = None,
    *,
    use_colors: bool | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> logging.Logger:
    """
    Return a child logger for the given module/package.
    Usage in modules: logger = get_logger(__name__)

    Parameters
    ----------
    use_colors:
        Enable ANSI colors in terminal output.
        - None (default): enabled unless LOG_COLORS is falsy.
        - True/False: force enable/disable.

    log_file:
        Optional path to a log file. If provided, a rotating file handler is attached
        to the app logger (plaintext, no ANSI). You can also set env var LOG_FILE.
    """

    _configure_once(level=level, use_colors=use_colors)

    # Allow attaching file logger after initial configure_once
    if log_file:
        app_logger = logging.getLogger(_APP_LOGGER_NAME)
        _configure_file_logging(log_file=log_file, level=app_logger.level)

    if not name:
        return logging.getLogger(_APP_LOGGER_NAME)
    return logging.getLogger(_APP_LOGGER_NAME + "." + name)



def set_log_level(level: str | int) -> None:
    """
    Dynamically change the level of the app logger at runtime.
    """

    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.setLevel(_to_level(level))


def enable_file_logging(
    log_file: str | os.PathLike[str],
    *,
    level: str | int | None = None,
    rotate: bool = True,
) -> None:
    """Enable/attach file logging at runtime.

    This is useful for CLIs where you want to start with console logging and later
    also persist logs.

    Parameters
    ----------
    rotate:
        If True (default), uses RotatingFileHandler with size-based rotation.
        If False, uses a plain FileHandler — ideal for single pipeline runs
        where the entire log should be captured in one file.
    """

    _configure_once(level=level, use_colors=None)
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    effective_level = _to_level(level) if level is not None else app_logger.level
    _configure_file_logging(log_file=log_file, level=effective_level, rotate=rotate)


def disable_file_logging() -> None:
    """Disable and close all attached file logging handlers.

    This is necessary to release file locks on Windows when moving/renaming log folders.
    """
    loggers = [
        logging.getLogger(_APP_LOGGER_NAME),
        logging.getLogger(),
    ]
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        loggers.append(logging.getLogger(logger_name))

    for logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                except Exception:
                    pass
                logger.removeHandler(handler)


