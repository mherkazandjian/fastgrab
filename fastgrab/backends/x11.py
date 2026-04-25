"""X11 backend — thin shim over the ``fastgrab._linux_x11`` C extension."""
from fastgrab import _linux_x11

from .base import BaseBackend


class X11Backend(BaseBackend):
    def resolution(self):
        return _linux_x11.resolution()

    def bytes_per_pixel(self):
        return _linux_x11.bytes_per_pixel()

    def screenshot(self, x, y, img):
        _linux_x11.screenshot(x, y, img)
