"""Backend registry and auto-detection for fastgrab.

The high-level :class:`fastgrab.screenshot.Screenshot` class delegates
the actual capture work to a backend. Five backends ship in tree:

* ``x11``     — libX11 ``XGetImage``. Linux. Always available in the
  default install.
* ``wlr``     — Wayland ``wlr-screencopy-v1``. Linux. Opt-in via
  ``pip install fastgrab[wayland]``. Works on wlroots compositors
  (Sway, Hyprland, river, niri, cage).
* ``portal``  — ``xdg-desktop-portal`` + PipeWire. Linux. Opt-in via
  ``pip install fastgrab[portal]``, which also needs GStreamer system
  packages. For GNOME and KDE, where the wlr backend does not work.
  Asks the user's permission on first capture.
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
_PORTAL_HINT = "pip install fastgrab[portal]"


def _resolve_backend(name=None, **options):
    """Return a :class:`BaseBackend` instance for the requested name.

    With ``name=None`` the function auto-detects based on the platform
    and (on Linux) the display server. With an explicit name it imports
    just that backend and raises a :class:`RuntimeError` with an install
    hint if the optional deps are missing.

    ``options`` are forwarded to the backend's constructor. They need an
    explicit ``name``: auto-detection may pick a backend that does not
    accept them, and silently dropping an option that governs something
    like screen-share consent would be worse than refusing it.
    """
    if name is None:
        if options:
            raise TypeError(
                "backend options ({}) need an explicit backend=, because "
                "auto-detection may pick a backend that does not accept "
                "them. The portal backend's persist mode is also settable "
                "with $FASTGRAB_PORTAL_PERSIST, which works either way."
                .format(", ".join(sorted(options)))
            )
        return _autodetect()

    if name == "x11":
        from .x11 import X11Backend
        return X11Backend(**options)

    if name == "wlr":
        try:
            from .wlr import WlrBackend
        except ImportError as exc:
            raise RuntimeError(
                "backend 'wlr' requires the wayland extra: " + _WAYLAND_HINT
            ) from exc
        return WlrBackend(**options)

    if name == "portal":
        try:
            from .portal import PortalBackend
        except ImportError as exc:
            raise RuntimeError(
                "backend 'portal' requires the portal extra: " + _PORTAL_HINT
            ) from exc
        return PortalBackend(**options)

    if name == "windows":
        from .windows import WindowsBackend
        return WindowsBackend(**options)

    if name == "macos":
        from .macos import MacosBackend
        return MacosBackend(**options)

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
            # Constructing one probes for PyGObject and GStreamer, so a
            # session without the extra raises here and falls through
            # rather than claiming every GNOME and KDE desktop for a
            # backend that cannot work.
            return PortalBackend()
        except Exception as exc:
            # A declined screen-share must not become a silent fallback:
            # XWayland would then capture the very screen the user just
            # refused to share.
            if type(exc).__name__ == "PortalCancelled":
                raise
            # Nor must a typo. $FASTGRAB_PORTAL_PERSIST is validated in
            # the constructor, and "this value is not a persist mode" is
            # a configuration error, not "this backend is unavailable" --
            # swallowing it would answer a misspelt setting by quietly
            # capturing through XWayland instead, losing native Wayland
            # windows, with nothing said.
            if isinstance(exc, ValueError):
                raise
            # Everything else here does mean the backend is unusable,
            # which is what the fallback is for.
        # Wayland session but no working Wayland backend — fall through to
        # X11/XWayland, which still works for X11 clients in a Wayland session.

    if os.environ.get("DISPLAY"):
        from .x11 import X11Backend
        return X11Backend()

    raise RuntimeError(
        "no usable display server detected (need DISPLAY or WAYLAND_DISPLAY)"
    )
