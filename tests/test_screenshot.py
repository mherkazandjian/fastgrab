"""Cross-platform API contract tests.

These tests run on Linux (X11 or Wayland), Windows, and macOS — they
exercise the public :class:`fastgrab.screenshot.Screenshot` API
without touching any platform-specific internals. The four
``_linux_x11``-direct tests live in ``tests/test_x11_lowlevel.py``
and are skipped on non-Linux.
"""
import fractions

import numpy
import pytest

from fastgrab import screenshot


def test_screenshot_instance_can_be_created():
    screenshot.Screenshot()


def test_screensize_is_cached():
    grab = screenshot.Screenshot()
    first = grab.screensize
    # Mutate the backing field; if cached, the property returns the new value
    # without re-querying the underlying backend.
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


class _SpyBackend:
    """Minimal backend that records what ``capture()`` forwards to it."""

    def __init__(self, size=(200, 100)):
        self._size = size
        self.calls = []

    def resolution(self):
        return self._size

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        self.calls.append((x, y))


def _spying_screenshot():
    """A Screenshot whose backend records calls instead of capturing."""
    grab = screenshot.Screenshot()
    grab._backend = _SpyBackend()
    grab._screensize = None
    return grab


@pytest.mark.parametrize('bbox', [
    (0.5, 0, 10, 10),
    (0, 0.5, 10, 10),
    (0, 0, 10.5, 10),
    (0, 0, 10, 10.5),
])
def test_check_bbox_rejects_fractional_components(bbox):
    """Half a pixel cannot be captured, and rounding it would lie.

    Integral floats are fine (see the normalization test); only a real
    fractional part is refused, because floor/round/ceil would each
    silently return a different region than the caller asked for.
    """
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='whole number'):
        grab.check_bbox(bbox)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_check_bbox_rejects_non_finite_components(value):
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='finite'):
        grab.check_bbox((value, 0, 10, 10))


def test_check_bbox_rejects_near_integral_fraction():
    """Exact reals must be compared exactly, not through float().

    ``float()`` rounds this Fraction to exactly 1.0, which would let a
    fractional value through as pixel 1.
    """
    grab = screenshot.Screenshot()
    almost_one = fractions.Fraction(2 ** 54 + 1, 2 ** 54)
    assert float(almost_one) == 1.0  # the trap being guarded against
    with pytest.raises(ValueError, match='whole number'):
        grab.check_bbox((almost_one, 0, 10, 10))


def test_check_bbox_accepts_integral_fraction():
    grab = screenshot.Screenshot()
    normalized = grab.check_bbox((fractions.Fraction(8, 2), 0, 10, 10))
    assert normalized == (4, 0, 10, 10)
    assert type(normalized[0]) is int


def test_check_bbox_reports_oversized_real_as_value_error():
    """A huge exact value must not leak OverflowError from float()."""
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError):
        grab.check_bbox((fractions.Fraction(10 ** 400), 0, 10, 10))


@pytest.mark.parametrize('bbox', [
    ('0', 0, 10, 10),
    (None, 0, 10, 10),
    (0, 0, [10], 10),
])
def test_check_bbox_rejects_non_numeric_components(bbox):
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='must be a number'):
        grab.check_bbox(bbox)


@pytest.mark.parametrize('bbox', [
    (True, 0, 10, 10),
    (0, 0, 10, False),
])
def test_check_bbox_rejects_booleans(bbox):
    """bool is a numbers.Integral, but a boolean pixel count is a mistake."""
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='boolean'):
        grab.check_bbox(bbox)


@pytest.mark.parametrize('bbox', [(-1, 0, 10, 10), (0, -1, 10, 10)])
def test_check_bbox_rejects_negative_origin(bbox):
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='must not be negative'):
        grab.check_bbox(bbox)


@pytest.mark.parametrize('bbox', [
    (0, 0, 0, 10),
    (0, 0, 10, 0),
    (0, 0, -1, 10),
    (0, 0, 10, -1),
])
def test_check_bbox_rejects_non_positive_size(bbox):
    """An empty region has nothing to capture and reaches every backend."""
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='must be positive'):
        grab.check_bbox(bbox)


@pytest.mark.parametrize('bbox', [(0, 0, 10), (0, 0, 10, 10, 10), ()])
def test_check_bbox_rejects_wrong_component_count(bbox):
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError, match='exactly four components'):
        grab.check_bbox(bbox)


def test_check_bbox_normalizes_to_plain_ints():
    """Accepted values come back as plain ints, whatever went in.

    Backends do pointer arithmetic with the origin: a numpy integer
    propagates into ctypes calls that reject it, and numpy arithmetic
    wraps on overflow instead of growing.
    """
    grab = screenshot.Screenshot()
    normalized = grab.check_bbox((numpy.int64(4), 6.0, numpy.int32(10), 8))
    assert normalized == (4, 6, 10, 8)
    assert all(type(value) is int for value in normalized)


def test_check_bbox_rejects_numpy_overflowing_bbox():
    """numpy ints wrap on overflow; the bounds check must still catch it."""
    grab = screenshot.Screenshot()
    huge = numpy.int64(numpy.iinfo(numpy.int64).max)
    with pytest.raises(ValueError):
        grab.check_bbox((huge, 0, huge, 10))


def test_capture_accepts_integral_float_bbox():
    """``w / 2`` is a float in Python 3 and must keep working."""
    grab = screenshot.Screenshot()
    img = grab.capture(bbox=(0.0, 0.0, 10.0, 8.0))
    assert img.shape == (8, 10, 4)


def test_capture_rejects_fractional_bbox_origin():
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError):
        grab.capture(bbox=(0.5, 0, 10, 10))


def test_capture_rejects_negative_bbox_origin():
    """Regression: a negative origin must not reach the backend.

    The bounds check only ever tested the far edge, so a negative origin
    reached the platform capture call — on X11 that is a BadMatch which
    the default Xlib error handler turns into an outright process exit.
    """
    grab = screenshot.Screenshot()
    with pytest.raises(ValueError):
        grab.capture(bbox=(-1, -1, 10, 10))


def test_capture_forwards_plain_ints_to_the_backend():
    """Regression: the backend must never receive a numpy int or a float.

    Validating without normalizing left this path broken — the macOS
    backend derives a memory address from the origin, and ctypes
    rejects a numpy.int64 one exactly as it rejected a float.
    """
    grab = _spying_screenshot()
    grab.capture(bbox=(numpy.int64(10), 4.0, 20, 8))
    (x, y), = grab._backend.calls
    assert (x, y) == (10, 4)
    assert type(x) is int and type(y) is int


@pytest.mark.parametrize('bbox', [
    (0, 0, 10, 0),
    (-1, 0, 10, 10),
    (0.5, 0, 10, 10),
    (True, 0, 10, 10),
])
def test_invalid_bbox_never_reaches_the_backend(bbox):
    grab = _spying_screenshot()
    with pytest.raises(ValueError):
        grab.capture(bbox=bbox)
    assert grab._backend.calls == []


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
