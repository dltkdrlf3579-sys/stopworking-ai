from __future__ import annotations

import builtins
from datetime import datetime


_ORIGINAL_PRINT = builtins.print
_INSTALLED = False


def install_timestamped_print() -> None:
    """Prefix process logs with local timestamps without touching each caller."""
    global _INSTALLED
    if _INSTALLED:
        return

    def timestamped_print(*args, **kwargs) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kwargs.setdefault("flush", True)
        _ORIGINAL_PRINT(f"[{timestamp}]", *args, **kwargs)

    builtins.print = timestamped_print
    _INSTALLED = True
