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
    with pytest.raises(ValueError, match="must be positive"):
        _linux_x11.screenshot(0, 0, numpy.zeros((0, 8, 4), dtype="uint8"))


def test_screenshot_rejects_non_3d_buffer():
    with pytest.raises(ValueError, match="height, width, 4"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8), dtype=numpy.uint8))


def test_screenshot_rejects_buffer_without_four_channels():
    """Regression: this used to overrun the destination buffer.

    Only ``ndim == 3`` was checked, so an (8, 8, 3) array was accepted
    and the 32-bpp copy wrote 8*8*4 = 256 bytes into a 192-byte
    allocation — a heap overflow through the advertised low-level API.
    """
    with pytest.raises(ValueError, match="height, width, 4"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8, 3), dtype=numpy.uint8))


def test_screenshot_rejects_wrong_dtype():
    """Rejected, not coerced: a coerced copy would be filled and dropped."""
    with pytest.raises(ValueError, match="dtype uint8"):
        _linux_x11.screenshot(0, 0, numpy.zeros((8, 8, 4), dtype=numpy.float64))


def test_screenshot_rejects_non_contiguous_buffer():
    """A strided view cannot be filled in place, so it must not be taken."""
    view = numpy.zeros((8, 16, 4), dtype=numpy.uint8)[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]  # sanity: the case under test
    with pytest.raises(ValueError, match="C-contiguous"):
        _linux_x11.screenshot(0, 0, view)


def test_screenshot_rejects_read_only_buffer():
    buf = numpy.zeros((8, 8, 4), dtype=numpy.uint8)
    buf.flags.writeable = False
    with pytest.raises(ValueError, match="C-contiguous|writable"):
        _linux_x11.screenshot(0, 0, buf)


def test_screenshot_rejects_non_array_buffer():
    with pytest.raises(TypeError, match="ndarray"):
        _linux_x11.screenshot(0, 0, [[0, 0, 0, 0]])


def test_display_failure_names_the_display_it_tried(monkeypatch):
    # The extension reports every XOpenDisplay failure with one sentence,
    # so an intermittent CI failure used to be uninvestigable: the log
    # could not say whether DISPLAY was unset, pointed somewhere dead, or
    # had simply been refused that once. Regression for issue #44.
    from fastgrab.backends.x11 import X11Backend

    monkeypatch.setenv("DISPLAY", ":77")
    backend = X11Backend()
    with pytest.raises(RuntimeError, match=r"cannot open X display.*DISPLAY=':77'"):
        backend.resolution()
    with pytest.raises(RuntimeError, match="still unreachable on retry"):
        backend.screenshot(0, 0, numpy.zeros((2, 2, 4), dtype=numpy.uint8))

    # The original error stays reachable as the cause rather than being
    # swallowed and reworded.
    try:
        backend.resolution()
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
        assert "cannot open X display" in str(exc.__cause__)


def test_bytes_per_pixel_also_reports_the_display(monkeypatch):
    # It opens a display of its own, and once Screenshot has cached the
    # screen size it is the first server call a later capture() makes —
    # so it is the likeliest entry point to meet a vanished server.
    from fastgrab.backends.x11 import X11Backend

    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match=r"DISPLAY=':77'"):
        X11Backend().bytes_per_pixel()


def test_unset_and_empty_display_are_distinguished(monkeypatch):
    from fastgrab.backends.x11 import _describe_display_state

    monkeypatch.delenv("DISPLAY", raising=False)
    assert _describe_display_state() == "DISPLAY is not set"
    monkeypatch.setenv("DISPLAY", "")
    assert _describe_display_state() == "DISPLAY is set but empty"


def test_a_transient_refusal_is_called_out(monkeypatch):
    # The discriminator #44 needs: if the very next connection succeeds,
    # the server never went away and the failure was transient.
    import fastgrab.backends.x11 as x11_mod

    class _Works:
        @staticmethod
        def resolution():
            return (1920, 1080)

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(x11_mod, "_linux_x11", _Works)
    assert "reachable on retry" in x11_mod._describe_display_state()


def test_a_persistent_outage_is_called_out(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":77")
    from fastgrab.backends.x11 import _describe_display_state

    assert "still unreachable on retry" in _describe_display_state()


def test_non_display_errors_pass_through_unchanged():
    # Only the open-display error is annotated. Driven through the
    # decorator directly because the extension's other RuntimeErrors
    # (a failed XGetImage) cannot be provoked from a healthy server.
    from fastgrab.backends.x11 import _with_display_context

    @_with_display_context
    def boom():
        raise RuntimeError("XGetImage returned NULL")

    with pytest.raises(RuntimeError) as excinfo:
        boom()
    assert "DISPLAY=" not in str(excinfo.value)
    assert str(excinfo.value) == "XGetImage returned NULL"


def test_a_failing_diagnostic_does_not_replace_the_real_error(monkeypatch):
    # Callers catch RuntimeError. If describing DISPLAY blows up, the
    # original error must still be what comes out.
    import fastgrab.backends.x11 as x11_mod

    def boom():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(x11_mod, "_describe_display_state", boom)
    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(RuntimeError, match="cannot open X display") as excinfo:
        x11_mod.X11Backend().resolution()
    assert "DISPLAY=" not in str(excinfo.value)


def test_display_context_preserves_the_wrapped_signature():
    from fastgrab.backends.x11 import X11Backend

    assert X11Backend.resolution.__name__ == "resolution"
    assert X11Backend.screenshot.__name__ == "screenshot"
    assert X11Backend.bytes_per_pixel.__name__ == "bytes_per_pixel"
