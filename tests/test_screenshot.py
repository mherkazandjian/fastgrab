"""Cross-platform API contract tests.

These tests run on Linux (X11 or Wayland), Windows, and macOS — they
exercise the public :class:`fastgrab.screenshot.Screenshot` API
without touching any platform-specific internals. The four
``_linux_x11``-direct tests live in ``tests/test_x11_lowlevel.py``
and are skipped on non-Linux.
"""
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


# --------------------------------------------------------------------
# Blur / redaction (fastgrab.effects wired into capture)
# --------------------------------------------------------------------

# A colour no real desktop is uniformly painted with, so "the frame is
# not entirely this" is a safe assertion against live screen content.
_MARKER = (7, 11, 13)  # B, G, R


def _fill_style(color=_MARKER):
    from fastgrab.effects import BlurStyle
    return BlurStyle(method="fill", color=color)


class _CheckerBackend:
    """Deterministic stand-in backend: a fixed non-uniform pattern.

    Lets the blur tests assert on exact pixel values without depending on
    whatever the real desktop happens to be showing.
    """

    def __init__(self, width=64, height=48):
        self._size = (width, height)

    def resolution(self):
        return self._size

    def bytes_per_pixel(self):
        return 4

    def screenshot(self, x, y, img):
        h, w = img.shape[:2]
        rows = numpy.arange(h, dtype=numpy.int32).reshape(h, 1)
        cols = numpy.arange(w, dtype=numpy.int32).reshape(1, w)
        img[..., 0] = (rows * 7 + cols * 3 + x + y) % 256
        img[..., 1] = (rows * 3 + cols * 11) % 256
        img[..., 2] = (rows + cols * 5) % 256
        img[..., 3] = 255


def _stub_grab(**kwargs):
    grab = screenshot.Screenshot(**kwargs)
    grab._backend = _CheckerBackend()
    grab._screensize = None
    grab._img = None
    return grab


def test_capture_with_blur_keeps_shape_and_dtype():
    grab = screenshot.Screenshot()
    w, h = grab.screensize
    img = grab.capture(blur=[(0, 0, min(16, w), min(16, h))])
    assert img.shape == (h, w, 4)
    assert img.dtype == numpy.uint8


def test_capture_blur_redacts_only_the_requested_region():
    grab = _stub_grab(blur_style=_fill_style())
    plain = grab.capture(blur=False).copy()
    img = grab.capture(blur=[(4, 6, 10, 8)])
    assert (img[6:14, 4:14, 0] == _MARKER[0]).all()
    assert (img[6:14, 4:14, 1] == _MARKER[1]).all()
    assert (img[6:14, 4:14, 2] == _MARKER[2]).all()
    # Everything outside the rectangle is byte-identical to a plain capture.
    assert (img[0:6] == plain[0:6]).all()
    assert (img[14:] == plain[14:]).all()
    assert (img[:, 0:4] == plain[:, 0:4]).all()
    assert (img[:, 14:] == plain[:, 14:]).all()


def test_blur_regions_are_screen_absolute_for_a_subregion_capture():
    """A region is given in screen coordinates, not frame coordinates."""
    grab = _stub_grab(blur_style=_fill_style())
    img = grab.capture(bbox=(20, 10, 24, 20), blur=[(24, 14, 8, 6)])
    # Screen (24, 14) is frame (4, 4) once the bbox origin is subtracted.
    assert (img[4:10, 4:12, 0] == _MARKER[0]).all()
    assert img[0, 0, 0] != _MARKER[0]


def test_constructor_blur_applies_to_every_capture():
    grab = _stub_grab(blur=True, blur_style=_fill_style())
    for _ in range(2):
        img = grab.capture()
        assert (img[..., 0] == _MARKER[0]).all()
        assert (img[..., 2] == _MARKER[2]).all()


def test_capture_blur_false_overrides_the_constructor():
    grab = _stub_grab(blur=True, blur_style=_fill_style())
    assert (grab.capture()[..., 0] == _MARKER[0]).all()
    # The backend rewrites the whole buffer, so switching the blur off
    # gives back a real capture rather than the previous redacted frame.
    plain = grab.capture(blur=False)
    assert not (plain[..., 0:3] == _MARKER).all()


def test_blur_does_not_accumulate_across_captures():
    """Capturing twice must blur a fresh frame, not the blurred one."""
    from fastgrab.effects import BlurStyle, blur_regions

    grab = _stub_grab(blur=True, blur_style=BlurStyle(radius=4))
    first = grab.capture().copy()
    second = grab.capture().copy()
    assert numpy.array_equal(first, second)

    # And it matches blurring a clean capture exactly once.
    plain = grab.capture(blur=False).copy()
    expected = blur_regions(plain, None, BlurStyle(radius=4))
    assert numpy.array_equal(second, expected)


def test_capture_blur_reuses_the_same_buffer():
    grab = _stub_grab(blur=True, blur_style=_fill_style())
    first = grab.capture()
    assert grab.capture() is first


def test_effects_is_not_imported_until_a_blur_is_requested():
    """The default install path must not pay for the blur module."""
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import fastgrab.screenshot;"
        "assert 'fastgrab.effects' not in sys.modules, 'imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_blur_regions_given_as_a_generator_survive_every_capture():
    """Regression: a one-shot iterable redacted frame 1 and nothing after.

    Storing the generator meant the second capture iterated an exhausted
    object and silently returned an unredacted frame — a redaction
    feature failing open.
    """
    grab = _stub_grab(
        blur=(r for r in [(4, 4, 8, 8)]), blur_style=_fill_style()
    )
    for attempt in range(3):
        img = grab.capture()
        assert (img[4:12, 4:12, 0] == _MARKER[0]).all(), (
            "capture {} came back unredacted".format(attempt)
        )


def test_per_call_blur_generator_is_materialised_too():
    grab = _stub_grab(blur_style=_fill_style())
    img = grab.capture(blur=iter([(0, 0, 6, 6)]))
    assert (img[0:6, 0:6, 0] == _MARKER[0]).all()


def test_blur_regions_are_stored_as_plain_tuples():
    grab = _stub_grab(blur=[[1, 2, 3, 4]], blur_style=_fill_style())
    assert grab.blur == ((1, 2, 3, 4),)


def test_malformed_blur_regions_raise_instead_of_being_skipped():
    for bad in ([(1, 2, 3)], [(1, 2, 3, 4, 5)]):
        with pytest.raises(ValueError, match="x, y, width, height"):
            _stub_grab(blur=bad)
    with pytest.raises(ValueError):
        _stub_grab().capture(blur=[("a", 2, 3, 4)])
