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
* ``wlr`` honours it for full-output capture. Sub-region capture takes
  the region in *logical* coordinates (a ``wlr-screencopy`` protocol
  requirement), which matches device pixels only at scale 1; the
  backend refuses a sub-region request on a scaled output rather than
  returning the wrong one. See issue #38.
* ``windows`` reports and captures in whatever DPI context the host
  process happens to have. Windows virtualizes screen metrics for
  DPI-unaware threads and ``BitBlt`` takes logical units, and neither
  the backend nor CPython's manifest establishes per-monitor awareness,
  so at a display scale other than 100% these are not physical pixels.
  See issue #39.
* ``portal`` is a placeholder that raises ``NotImplementedError``.
"""
from abc import ABC, abstractmethod


class BaseBackend(ABC):
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
