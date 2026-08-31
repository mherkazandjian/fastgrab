"""Direct exercises of the libX11 C extension.

These tests bypass the Screenshot dispatcher and call into
``fastgrab._linux_x11`` directly. They lock down the wire-level
contract (shape, BGRA byte order, resolution-equality with the
high-level wrapper) that the X11Backend depends on.

Skipped on non-Linux: the C extension is not built into wheels for
Windows or macOS.
"""
import sys

import numpy
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="fastgrab._linux_x11 is the libX11 C extension; Linux only",
)

from fastgrab import screenshot  # noqa: E402
from fastgrab import _linux_x11  # noqa: E402


def test_low_level_resolution_returns_positive_2_tuple():
    res = _linux_x11.resolution()
    assert isinstance(res, tuple)
    assert len(res) == 2
    width, height = res
    assert isinstance(width, int) and isinstance(height, int)
    assert width > 0 and height > 0


def test_low_level_bytes_per_pixel_is_4():
    # X11 ZPixmap on every supported platform we ship to is 32-bit (BGRA).
    assert _linux_x11.bytes_per_pixel() == 4


def test_low_level_screenshot_fills_buffer():
    width, height = _linux_x11.resolution()
    buf = numpy.zeros((height, width, 4), dtype="uint8")
    _linux_x11.screenshot(0, 0, buf)
    assert buf.shape == (height, width, 4)
    assert buf.dtype == numpy.uint8


def test_screensize_matches_low_level_resolution():
    grab = screenshot.Screenshot(backend="x11")
    assert grab.screensize == _linux_x11.resolution()


def test_unreachable_display_raises_instead_of_segfaulting(monkeypatch):
    # Regression: every entry point used to dereference a NULL Display*
    # when XOpenDisplay failed, crashing the interpreter (exit 139).
    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.resolution()
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.bytes_per_pixel()
    with pytest.raises(RuntimeError, match="cannot open X display"):
        _linux_x11.screenshot(0, 0, numpy.zeros((2, 2, 4), dtype=numpy.uint8))
    # And through the public API — Screenshot() itself doesn't touch the
    # server, the first capture() does.
    from fastgrab import screenshot
    with pytest.raises(RuntimeError, match="cannot open X display"):
        screenshot.Screenshot(backend="x11").capture()


@pytest.mark.parametrize("x, y", [(-1, -1), (-1, 0), (0, -1)])
def test_low_level_screenshot_rejects_negative_origin(x, y):
    """Regression: a negative origin used to kill the interpreter.

    It reaches XGetImage as a BadMatch, and X protocol errors are
    delivered to Xlib's default error handler, which prints a
    diagnostic and calls exit() — no exception, no traceback, and the
    NULL check in the C code never gets a chance to run.
    """
    buf = numpy.zeros((8, 8, 4), dtype="uint8")
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(x, y, buf)


def test_low_level_screenshot_rejects_region_past_the_right_edge():
    width, height = _linux_x11.resolution()
    buf = numpy.zeros((8, 8, 4), dtype="uint8")
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(width - 4, 0, buf)
    with pytest.raises(RuntimeError, match="outside the screen"):
        _linux_x11.screenshot(0, height - 4, buf)


def test_low_level_screenshot_rejects_empty_region():
    with pytest.raises(RuntimeError, match="non-positive"):
        _linux_x11.screenshot(0, 0, numpy.zeros((0, 8, 4), dtype="uint8"))


def test_screenshot_rejects_non_3d_buffer():
    with pytest.raises(ValueError, match="height, width, 4"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8), dtype=numpy.uint8))
