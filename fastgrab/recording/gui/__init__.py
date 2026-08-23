"""Tkinter-based region selector and config dialog.

Imports tkinter lazily so the rest of ``fastgrab.recording`` stays
usable in headless contexts where Tk isn't available.
"""
from .selector import select_region
from .config import show_config_dialog
from .progress import record_with_progress

__all__ = ["select_region", "show_config_dialog", "record_with_progress"]
