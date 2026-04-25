"""Backend registry and auto-detection for fastgrab.

The high-level :class:`fastgrab.screenshot.Screenshot` class delegates
the actual capture work to a backend. Three backends ship in tree:

* ``x11``    — libX11 ``XGetImage`` (always available; default install).
* ``wlr``    — Wayland ``wlr-screencopy-v1``; opt-in via
  ``pip install fastgrab[wayland]``.
* ``portal`` — ``xdg-desktop-portal`` + PipeWire; opt-in via
  ``pip install fastgrab[wayland-portal]``.

Auto-detection (``backend=None``) prefers Wayland-native paths when
``$WAYLAND_DISPLAY`` is set, then falls back to X11/XWayland.
"""
import os


_WAYLAND_HINT = "pip install fastgrab[wayland]"
_PORTAL_HINT = "pip install fastgrab[wayland-portal]"


def _resolve_backend(name=None):
    """Return a :class:`BaseBackend` instance for the requested name.

    With ``name=None`` the function auto-detects based on the display
    server (Wayland prefers wlr → portal; X11 picks x11). With an
    explicit name it imports just that backend and raises a
    :class:`RuntimeError` with an install hint if the optional deps
    are missing.
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

    raise ValueError(
        "unknown backend {!r}; expected 'x11', 'wlr', or 'portal'".format(name)
    )


def _autodetect():
    """Pick the best available backend based on environment + installed deps."""
    if os.environ.get("WAYLAND_DISPLAY"):
        try:
            from .wlr import WlrBackend
            return WlrBackend()
        except Exception:  # pywayland may raise ValueError, OSError, etc.
            pass
        try:
            from .portal import PortalBackend
            return PortalBackend()
        except Exception:  # pywayland may raise ValueError, OSError, etc.
            pass
        # Wayland session but no working Wayland backend — fall through to
        # X11/XWayland, which still works for X11 clients in a Wayland session.

    if os.environ.get("DISPLAY"):
        from .x11 import X11Backend
        return X11Backend()

    raise RuntimeError(
        "no usable display server detected (need DISPLAY or WAYLAND_DISPLAY)"
    )
