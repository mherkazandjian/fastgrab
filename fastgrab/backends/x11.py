"""X11 backend — thin shim over the ``fastgrab._linux_x11`` C extension."""
import functools
import os

from fastgrab import _linux_x11

from .base import BaseBackend


# The C extension maps every NULL from XOpenDisplay onto a single error
# code, so an unset variable, a server that is not running and a server
# that refused this one connection all surface as the same sentence. That
# is fine locally, where the next run reproduces it, and useless in a CI
# log for an intermittent failure — which is how #44 went uninvestigable.
_DISPLAY_ERROR = "cannot open X display"


def _describe_display_state():
    """Say what DISPLAY was, and whether a second attempt fares better.

    The retry goes back through the extension rather than parsing DISPLAY
    and probing a socket directly. Xlib accepts far more than ``:0`` — a
    transport prefix (``unix/:0``, and ``inet6/host:0`` which pins the
    address family), a bare socket pathname, bracketed IPv6 literals —
    and a hand-rolled parser gets enough of those wrong to make the
    diagnostic lie, which is worse than having none. Asking the real code
    path costs one connection on a path that has already failed, and
    answers the question #44 actually poses: is the display gone, or was
    that single connection refused?
    """
    display = os.environ.get("DISPLAY")
    if display is None:
        return "DISPLAY is not set"
    if not display:
        return "DISPLAY is set but empty"
    try:
        _linux_x11.resolution()
    except Exception:
        return "DISPLAY={!r}, still unreachable on retry".format(display)
    return (
        "DISPLAY={!r}, reachable on retry — that connection was refused "
        "rather than the server being gone".format(display)
    )


def _with_display_context(func):
    """Append the DISPLAY state to a failed-to-open-display error.

    Applied to every entry point that opens a display of its own — which
    is all three, ``bytes_per_pixel`` included. That one matters more
    than it looks: once ``Screenshot`` has cached the screen size, it is
    the first server call a subsequent ``capture()`` makes, and so the
    likeliest place to meet a server that went away between captures.

    Only the open-display error is touched; every other RuntimeError from
    the extension already names its own cause and passes through
    untouched. The retry runs on the failure path only, so capture is
    unaffected.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            if _DISPLAY_ERROR not in str(exc):
                raise
            try:
                hint = _describe_display_state()
            except Exception:
                # A diagnostic must never displace the error it exists to
                # describe. Callers catch RuntimeError; handing them
                # something else would break them at the worst possible
                # moment — and the likeliest reason this retry fails is
                # the same descriptor exhaustion that just took
                # XOpenDisplay down.
                raise exc
            raise RuntimeError("{} [{}]".format(exc, hint)) from exc

    return wrapper


class X11Backend(BaseBackend):
    @_with_display_context
    def resolution(self):
        return _linux_x11.resolution()

    @_with_display_context
    def bytes_per_pixel(self):
        return _linux_x11.bytes_per_pixel()

    @_with_display_context
    def screenshot(self, x, y, img):
        _linux_x11.screenshot(x, y, img)
