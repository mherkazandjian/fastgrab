"""X11 backend — thin shim over the ``fastgrab._linux_x11`` C extension."""
import functools

from fastgrab import _linux_x11

from ._display import describe_display
from .base import BaseBackend


# The C extension maps every NULL from XOpenDisplay onto a single error
# code, so an unset variable, a server that is not running and a server
# that refused this one connection all surface as the same sentence. That
# is fine locally, where the next run reproduces it, and useless in a CI
# log for an intermittent failure — which is how #44 went uninvestigable.
# The state of DISPLAY is therefore attached at the moment of failure.
_DISPLAY_ERROR = "cannot open X display"


def _with_display_context(func):
    """Append the DISPLAY state to a failed-to-open-display error.

    Applied to every entry point that opens a display of its own — which
    is all three, ``bytes_per_pixel`` included. That one matters more
    than it looks: once ``Screenshot`` has cached the screen size, it is
    the first server call a subsequent ``capture()`` makes, and so the
    likeliest place to meet a server that went away between captures.

    Only the open-display error is touched; every other RuntimeError from
    the extension already names its own cause and passes through
    untouched. The probe runs on the failure path only, so capture is
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
                hint = describe_display()
            except Exception:
                # A diagnostic must never displace the error it exists to
                # describe. Callers catch RuntimeError; handing them
                # something else instead would break them at the worst
                # possible moment — and the likeliest reason this probe
                # fails is the same descriptor exhaustion that just took
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
