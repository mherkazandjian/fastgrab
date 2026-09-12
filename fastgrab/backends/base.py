"""Backend interface for fastgrab.

A backend's job is to fill a caller-provided ``(H, W, 4)`` uint8 numpy
buffer with a region of the screen, in BGRA byte order on
little-endian Linux x86_64. The high-level
:class:`fastgrab.screenshot.Screenshot` wrapper owns the buffer and
calls into the backend.

**Coordinates are device pixels.** Every size and offset crossing this
interface — what :meth:`BaseBackend.resolution` reports, the ``(x, y)``
given to :meth:`BaseBackend.screenshot`, and the ``bbox`` accepted by
:meth:`fastgrab.screenshot.Screenshot.capture` — counts physical pixels
in the backing store, never logical/scaled units. On a 2x Retina panel
an 1800x1169-point desktop reports and captures 3600x2338.

A backend whose platform speaks logical coordinates (macOS points,
Wayland's logical surface size, the portal's ScreenCast source) must
convert at its own boundary rather than leaking those units upward.
Callers that mix the two — sizing a GUI window in points from a capture
measured in pixels, say — get a silently wrong region rather than an
error, so the conversion belongs in one place.

That is the contract new backends are held to, but not every shipped
backend honours it yet. Known gaps, so this docstring is not read as a
promise the code keeps:

* ``x11`` and ``macos`` honour it.
* ``wlr`` honours it. ``wlr-screencopy`` takes a sub-region in *logical*
  coordinates, so the backend converts the device-pixel bbox through
  the output's ``xdg_output`` logical size, rounds the region outward,
  and crops the returned frame back to the exact bbox. One gap is left:
  wlroots applies the output transform *before* the scale, so on a
  rotated or flipped output the region lands on a transposed backing
  store — the backend refuses a sub-region request there rather than
  returning the wrong one. Full-output capture is unaffected either
  way. See issue #38.
* ``windows`` honours it on Windows 10 1607 and later, where the
  backend temporarily gives its own thread per-monitor-v2 DPI
  awareness around the metrics query, the screen-DC acquisition and
  the blit, so neither the host process's DPI context nor the display
  scale changes what it reports or captures. On older Windows
  ``SetThreadDpiAwarenessContext`` does not exist; there the backend
  falls back to the host's context and, at a display scale other than
  100%, back to logical units. Primary monitor only in either case —
  the desktop DC's origin is the primary monitor's top-left, and
  reaching a second monitor at its own scale is the future
  ``Screenshot(display=N)`` feature, not a DPI question.
* ``portal`` is a placeholder that raises ``NotImplementedError``.
"""
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    _closed = False
    """Set on the instance by :meth:`close`; a class attribute so that a
    backend needs no ``__init__`` of its own to have one."""

    def close(self):
        """Release any OS resources this backend holds.

        The default is a no-op, which is right for a backend that keeps
        nothing between calls: ``x11`` shares one process-wide connection
        owned by the C extension, and ``macos`` re-resolves the display
        every time. A backend that allocates a frame buffer per instance
        — ``wlr``'s SHM buffer, ``windows``' DIBSection — overrides this
        and must also refuse to capture afterwards, because the handles
        it would blit through are gone.

        Closing is optional. Each such backend keeps its handles in a
        small helper object carrying a :mod:`weakref` finalizer, so
        dropping the backend frees them too; ``close`` only makes the
        moment deterministic. It must be safe to call twice.
        """
        self._closed = True

    def _check_open(self):
        """Raise if :meth:`close` has already run."""
        if self._closed:
            raise RuntimeError(
                "this backend has been closed; construct a new "
                "Screenshot() rather than reusing a closed one"
            )

    @abstractmethod
    def resolution(self):
        """Return ``(width, height)`` of the primary screen/output.

        In device pixels — see the module docstring.
        """

    @abstractmethod
    def bytes_per_pixel(self):
        """Return the number of bytes per captured pixel (always 4 today)."""

    def refresh(self):
        """Re-resolve any display handle latched at construction time.

        The default is a no-op, which is correct for backends that
        resolve the display on every call. A backend that stores a
        display id (macOS) overrides this so
        :meth:`fastgrab.screenshot.Screenshot.refresh` can pick up a
        change of main display.
        """

    @abstractmethod
    def screenshot(self, x, y, img):
        """Fill ``img`` with the region starting at ``(x, y)``.

        ``x`` and ``y`` are device pixels — see the module docstring.
        ``img`` is a pre-allocated ``(H, W, 4)`` uint8 numpy ndarray
        whose shape implicitly carries the requested capture size. The
        backend must write the whole buffer in BGRA byte order.

        :class:`fastgrab.screenshot.Screenshot` validates the region
        before calling, but this method is public and reachable
        directly, so a backend should refuse a region it cannot honour
        rather than silently capturing a different one.
        """
