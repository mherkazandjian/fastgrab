"""xdg-desktop-portal + PipeWire fallback backend.

Status: **stub.** The full portal flow is non-trivial to implement and
*cannot be exercised inside docker compose CI* (the consent dialog on
GNOME/KDE is a fundamental Wayland-security feature, not something we
can mock away). Rather than ship 400 lines of untested D-Bus + PipeWire
code, this file declares the dependency surface (so
``pip install fastgrab[wayland-portal]`` resolves cleanly) and raises
:class:`NotImplementedError` from ``__init__`` so the dispatcher's
fallback chain logs a helpful message and continues to the X11 backend
where applicable.

When implemented, the flow is:

1. Connect to the D-Bus session bus via ``dbus-next``.
2. Proxy ``org.freedesktop.portal.Desktop`` at
   ``/org/freedesktop/portal/desktop`` with interface
   ``org.freedesktop.portal.ScreenCast``.
3. ``CreateSession(handle_token, session_handle_token)`` → wait for
   ``Response`` signal, capture ``session_handle``.
4. ``SelectSources(session_handle, types=1, multiple=False, cursor_mode=2)``.
5. ``Start(session_handle, parent_window="", options={})`` —
   **this is when GNOME/KDE pop the consent dialog**. Response yields
   ``streams: [(node_id, props), ...]``. Cache ``node_id``.
6. ``OpenPipeWireRemote(session_handle, options)`` returns a unix fd
   for a pre-authenticated PipeWire connection.
7. Per-capture: connect a ``pipewire-python`` stream to ``node_id``
   over the returned fd, request format ``SPA_VIDEO_FORMAT_BGRA``,
   pull one buffer, copy into the caller's numpy ndarray, deactivate
   the stream until next call.

Probe the portal's ``interface_version`` after binding and degrade
options the implementation does not advertise (Fedora's
xdg-desktop-portal-gnome 4.x and Ubuntu's -gtk 1.18 LTS differ on
``cursor_mode`` / ``persist_mode``).

Open issue link: TBD — file under the fastgrab tracker once this stub
lands.
"""
from .base import BaseBackend


_NOT_IMPLEMENTED_MSG = (
    "PortalBackend is not yet implemented. The wayland-portal extra is "
    "wired (dbus-next + pipewire-python) but the screencast flow is "
    "stubbed. On GNOME/KDE Wayland sessions, log in to an X11 session "
    "for now, or use a wlroots compositor (Sway, Hyprland) so the wlr "
    "backend works without a consent prompt."
)


class PortalBackend(BaseBackend):
    def __init__(self):
        # Verify the extras are installed so the user gets a useful
        # error message rather than ImportError much later.
        try:
            import dbus_next  # noqa: F401
            import pipewire  # noqa: F401  (from pipewire-python)
        except ImportError as exc:
            raise ImportError(
                "wayland-portal extra is not installed: "
                "pip install fastgrab[wayland-portal]"
            ) from exc

        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def resolution(self):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
