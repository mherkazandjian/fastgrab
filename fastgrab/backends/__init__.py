"""Backend registry and auto-detection for fastgrab.

The high-level :class:`fastgrab.screenshot.Screenshot` class delegates
the actual capture work to a backend. Five backends ship in tree:

* ``x11``     — libX11 ``XGetImage``. Linux. Always available in the
  default install.
* ``wlr``     — Wayland ``wlr-screencopy-v1``. Linux. Opt-in via
  ``pip install fastgrab[wayland]``. Works on wlroots compositors
  (Sway, Hyprland, river, niri, cage).
* ``portal``  — ``xdg-desktop-portal`` + PipeWire. Linux. Opt-in via
  ``pip install fastgrab[wayland-portal]``. Currently stubbed.
* ``windows`` — Win32 ``BitBlt`` + ``CreateDIBSection``. Windows
  10/11. Pure ``ctypes``, no extras.
* ``macos``   — CoreGraphics ``CGDisplayCreateImage``. macOS. Pure
  ``ctypes``, no extras.

Auto-detection (``backend=None``) routes by ``sys.platform`` first,
then on Linux prefers Wayland-native paths when ``$WAYLAND_DISPLAY``
is set, and finally falls back to X11/XWayland.
"""
import os
import sys


_WAYLAND_HINT = "pip install fastgrab[wayland]"
_PORTAL_HINT = "pip install fastgrab[wayland-portal]"


def _resolve_backend(name=None):
    """Return a :class:`BaseBackend` instance for the requested name.

    With ``name=None`` the function auto-detects based on the platform
    and (on Linux) the display server. With an explicit name it imports
    just that backend and raises a :class:`RuntimeError` with an install
    hint if the optional deps are missing.
    """
    if name is None:
        return _autodetect()

    if name == "x11":
        from .x11 import X11Backend
        return X11Backend()

    if name == "wlr":
        try:
            from .wlr import WlrBackend
        except ImportError as exc:
            raise RuntimeError(
                "backend 'wlr' requires the wayland extra: " + _WAYLAND_HINT
            ) from exc
        return WlrBackend()

    if name == "portal":
        try:
            from .portal import PortalBackend
        except ImportError as exc:
            raise RuntimeError(
                "backend 'portal' requires the wayland-portal extra: "
                + _PORTAL_HINT
            ) from exc
        return PortalBackend()

    if name == "windows":
        from .windows import WindowsBackend
        return WindowsBackend()

    if name == "macos":
        from .macos import MacosBackend
        return MacosBackend()

    raise ValueError(
        "unknown backend {!r}; expected one of "
        "'x11', 'wlr', 'portal', 'windows', 'macos'".format(name)
    )


def _autodetect():
    """Pick the best available backend based on platform + environment."""
    if sys.platform == "win32":
        from .windows import WindowsBackend
        return WindowsBackend()

    if sys.platform == "darwin":
        from .macos import MacosBackend
        return MacosBackend()

    # Linux (and other POSIX-likes that fastgrab doesn't officially
    # support). Try Wayland first, then X11.
    if os.environ.get("WAYLAND_DISPLAY"):
        try:
            from .wlr import WlrBackend
            return WlrBackend()
        except Exception:  # pywayland may raise ValueError, OSError, etc.
            pass
        try:
            from .portal import PortalBackend
            return PortalBackend()
        except Exception:
            pass
        # Wayland session but no working Wayland backend — fall through to
        # X11/XWayland, which still works for X11 clients in a Wayland session.

    if os.environ.get("DISPLAY"):
        from .x11 import X11Backend
        return X11Backend()

    raise RuntimeError(
        "no usable display server detected (need DISPLAY or WAYLAND_DISPLAY)"
    )
