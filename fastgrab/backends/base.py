"""Backend interface for fastgrab.

A backend's job is to fill a caller-provided ``(H, W, 4)`` uint8 numpy
buffer with a region of the screen, in BGRA byte order on
little-endian Linux x86_64. The high-level
:class:`fastgrab.screenshot.Screenshot` wrapper owns the buffer and
calls into the backend.
"""
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    @abstractmethod
    def resolution(self):
        """Return ``(width, height)`` of the primary screen/output."""

    @abstractmethod
    def bytes_per_pixel(self):
        """Return the number of bytes per captured pixel (always 4 today)."""

    @abstractmethod
    def screenshot(self, x, y, img):
        """Fill ``img`` with the region starting at ``(x, y)``.

        ``img`` is a pre-allocated ``(H, W, 4)`` uint8 numpy ndarray
        whose shape implicitly carries the requested capture size. The
        backend must write the whole buffer in BGRA byte order.
        """
