import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import install as pretty_install
from rich.theme import Theme
from rich.traceback import install as traceback_install

log = None


def setup_logging(debug: bool = False) -> logging.Logger:
    global log

    if log is not None:
        return log

    # Optional file sink: set LOG_FILE env var to write to a file.
    # Defaults to console-only (correct for Docker/stdout capture).
    log_file = os.getenv("LOG_FILE")
    if log_file:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(pathname)s | %(message)s",
            filename=log_file,
            filemode="a",
            encoding="utf-8",
            force=True,
        )

    console = Console(
        log_time=True,
        log_time_format="%H:%M:%S-%f",
        theme=Theme({
            "traceback.border": "black",
            "traceback.border.syntax_error": "black",
            "inspect.value.border": "black",
        }),
    )
    pretty_install(console=console)
    traceback_install(
        console=console,
        extra_lines=1,
        width=console.width,
        word_wrap=False,
        indent_guides=False,
        suppress=[],
    )
    level = logging.DEBUG if debug else logging.INFO
    rh = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S-%f",
        level=level,
        console=console,
    )
    log = logging.getLogger("timecast")
    log.setLevel(level)
    log.addHandler(rh)

    return log
