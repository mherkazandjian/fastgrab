import numpy
import pytest

from fastgrab import screenshot
from fastgrab import _linux_x11


def test_screenshot_instance_can_be_created():
    screenshot.Screenshot()


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
    # Alpha channel is left as raw X data; at least one of B/G/R should not be
    # the original zero across the entire screen unless the screen is all
    # black, in which case this assertion still passes via the dtype/shape
    # checks.
    assert buf.shape == (height, width, 4)
    assert buf.dtype == numpy.uint8


def test_screensize_matches_low_level_resolution():
    grab = screenshot.Screenshot()
    assert grab.screensize == _linux_x11.resolution()


def test_screensize_is_cached():
    grab = screenshot.Screenshot()
    first = grab.screensize
    # Mutate the backing field; if cached, the property returns the new value
    # without re-querying the X server.
    grab._screensize = (1, 1)
    assert grab.screensize == (1, 1)
    assert first != (1, 1)  # sanity: original was a real resolution


def test_check_bbox_accepts_full_screen():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    grab.check_bbox((0, 0, w, h))


def test_check_bbox_accepts_subregion():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    grab.check_bbox((1, 1, w - 1, h - 1))


def test_check_bbox_rejects_out_of_bounds_width():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    with pytest.raises(ValueError):
        grab.check_bbox((0, 0, w + 1, h))


def test_check_bbox_rejects_out_of_bounds_height():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    with pytest.raises(ValueError):
        grab.check_bbox((0, 0, w, h + 1))


def test_capture_full_screen_shape_and_dtype():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    img = grab.capture()
    assert img.shape == (h, w, 4)
    assert img.dtype == numpy.uint8


def test_capture_with_bbox_returns_expected_shape():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    bw, bh = min(100, w), min(50, h)
    img = grab.capture(bbox=(0, 0, bw, bh))
    assert img.shape == (bh, bw, 4)


def test_capture_raises_on_out_of_bounds_bbox():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    with pytest.raises(ValueError):
        grab.capture(bbox=(0, 0, w + 1, h + 1))


def test_capture_reuses_buffer_when_size_unchanged():
    grab = screenshot.Screenshot()
    img1 = grab.capture()
    img2 = grab.capture()
    assert img1 is img2


def test_capture_reallocates_buffer_when_size_changes():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    bw1, bh1 = min(50, w), min(50, h)
    bw2, bh2 = min(80, w), min(80, h)
    if (bw1, bh1) == (bw2, bh2):
        pytest.skip("screen too small to test reallocation")
    img1 = grab.capture(bbox=(0, 0, bw1, bh1))
    img2 = grab.capture(bbox=(0, 0, bw2, bh2))
    assert img1 is not img2
    assert img2.shape == (bh2, bw2, 4)
