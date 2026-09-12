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
        self.size = size
        self.calls = []
        self.refreshed = 0

    def resolution(self):
        return self.size

    def bytes_per_pixel(self):
        return 4

    def refresh(self):
        self.refreshed += 1

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


def test_refresh_picks_up_a_new_resolution():
    """The cache is deliberate; refresh() is how a mode switch lands.

    Without it a full-screen capture keeps covering the old region and
    boxes in the newly available area are rejected as out of bounds.
    """
    grab = _spying_screenshot()
    assert grab.screensize == (200, 100)

    grab._backend.size = (300, 150)
    assert grab.screensize == (200, 100)  # still cached, by design
    with pytest.raises(ValueError):
        grab.check_bbox((0, 0, 300, 150))

    grab.refresh()
    assert grab.screensize == (300, 150)
    assert grab.check_bbox((0, 0, 300, 150)) == (0, 0, 300, 150)


def test_refresh_captures_the_new_full_screen_size():
    grab = _spying_screenshot()
    assert grab.capture().shape == (100, 200, 4)
    grab._backend.size = (300, 150)
    grab.refresh()
    assert grab.capture().shape == (150, 300, 4)


def test_refresh_keeps_the_cache_when_the_backend_fails():
    """A failed refresh must not leave the object worse than before.

    The backend is asked first precisely so the cached size and buffer
    survive when it raises, rather than being discarded in favour of
    nothing.
    """
    grab = _spying_screenshot()
    assert grab.screensize == (200, 100)
    first = grab.capture()

    def boom():
        raise RuntimeError('display went away')

    grab._backend.refresh = boom
    with pytest.raises(RuntimeError, match='display went away'):
        grab.refresh()

    assert grab._screensize == (200, 100)
    assert grab._img is first


def test_refresh_asks_the_backend_to_re_resolve_its_display():
    """macOS latches CGMainDisplayID at construction; it must re-read."""
    grab = _spying_screenshot()
    grab.refresh()
    assert grab._backend.refreshed == 1


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


# -------- close() and the context manager --------
#
# The resources at stake are per-backend (a wlr SHM buffer, a Windows
# DIBSection and its memory DC), so what they actually free is asserted
# in tests/test_integration_wlr.py and tests/test_windows_backend.py.
# These cover the contract every platform shares.

def test_close_is_idempotent():
    grab = screenshot.Screenshot()
    grab.capture((0, 0, 2, 2))
    grab.close()
    grab.close()


def test_close_releases_the_image_buffer():
    """The numpy buffer is the one resource Screenshot owns directly."""
    grab = screenshot.Screenshot()
    grab.capture((0, 0, 2, 2))
    assert grab._img is not None
    grab.close()
    assert grab._img is None


def test_capture_after_close_raises():
    """Refusing beats the alternative on Windows.

    A closed backend has deleted the memory DC its BitBlt targets, and
    blitting through a freed GDI handle is undefined rather than an
    error. Every backend refuses uniformly so the behaviour does not
    depend on which one was autodetected.
    """
    grab = screenshot.Screenshot()
    grab.capture((0, 0, 2, 2))
    grab.close()
    with pytest.raises(RuntimeError, match="closed"):
        grab.capture((0, 0, 2, 2))


def test_context_manager_yields_the_instance_and_closes_it():
    with screenshot.Screenshot() as grab:
        img = grab.capture((0, 0, 2, 2))
        assert img.shape == (2, 2, 4)
    assert grab._closed is True


def test_context_manager_closes_on_an_exception():
    """A capture that raises must still release the frame buffer."""
    boom = RuntimeError("caller blew up")
    try:
        with screenshot.Screenshot() as grab:
            raise boom
    except RuntimeError as exc:
        assert exc is boom
    assert grab._closed is True


def test_a_closed_screenshot_does_not_close_a_new_one():
    """Backends share process-wide state; closing must stay per instance.

    The wlr backend's display connection is a process-wide singleton, so
    an over-broad close would take out every future instance too. This is
    the cheap check that it does not.
    """
    first = screenshot.Screenshot()
    first.capture((0, 0, 2, 2))
    first.close()

    second = screenshot.Screenshot()
    assert second.capture((0, 0, 2, 2)).shape == (2, 2, 4)
    second.close()
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

    def refresh(self):
        """Part of the backend contract since the refresh() work landed."""

    def close(self):
        """Ditto — a stub must satisfy what Screenshot may call on it."""

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


def test_assigning_blur_after_construction_is_normalised_too():
    """Regression: the public attribute bypassed validation entirely."""
    grab = _stub_grab(blur_style=_fill_style())
    grab.blur = (r for r in [(4, 4, 8, 8)])
    for attempt in range(3):
        img = grab.capture()
        assert (img[4:12, 4:12, 0] == _MARKER[0]).all(), (
            "capture {} came back unredacted".format(attempt)
        )
    assert grab.blur == ((4, 4, 8, 8),)


def test_assigning_a_malformed_blur_after_construction_raises():
    grab = _stub_grab()
    with pytest.raises(ValueError):
        grab.blur = [(1, 2, 3)]


@pytest.mark.parametrize("region", [
    (10, 10, 0, 20),      # zero width
    (10, 10, 20, 0),      # zero height
    (10, 10, -5, 20),     # negative width
])
def test_empty_blur_regions_raise_rather_than_being_skipped(region):
    """An empty rectangle would be stored, then silently clipped away."""
    with pytest.raises(ValueError, match="positive"):
        _stub_grab(blur=[region])


@pytest.mark.parametrize("region", [
    (10.5, 10, 20, 20),
    (10, 10, 20.7, 20),
    (10, 10, 0.4, 20),    # would truncate to an empty rectangle
])
def test_fractional_blur_coordinates_raise(region):
    """Truncating a coordinate could shift the box off part of the secret."""
    with pytest.raises(ValueError, match="whole pixels"):
        _stub_grab(blur=[region])


def test_integral_floats_are_accepted():
    grab = _stub_grab(blur=[(10.0, 10.0, 20.0, 20.0)], blur_style=_fill_style())
    assert grab.blur == ((10, 10, 20, 20),)


def test_a_rejected_blur_override_does_not_clear_the_previous_frame():
    """Regression: the backend overwrote the shared buffer before validating.

    capture() hands back its internal buffer, so validating the override
    after the backend wrote into it turned a caller's previously redacted
    frame into a clear capture — via the very call that raised.
    """
    grab = _stub_grab(blur=True, blur_style=_fill_style())
    frame = grab.capture()
    assert (frame[..., 0] == _MARKER[0]).all()

    with pytest.raises(ValueError):
        grab.capture(blur=[(0, 0, 0, 10)])

    # Same buffer object; it must still hold the redacted frame.
    assert (frame[..., 0] == _MARKER[0]).all()


def test_blur_style_is_validated_on_assignment():
    """Regression: a bad style was stored and only raised mid-capture.

    By then the backend had overwritten the shared buffer, so a caller's
    previously redacted frame had already gone clear.
    """
    grab = _stub_grab(blur=True, blur_style=_fill_style())
    frame = grab.capture()
    assert (frame[..., 0] == _MARKER[0]).all()

    for bad in ("fill", {}, object()):
        with pytest.raises(TypeError, match="BlurStyle"):
            grab.blur_style = bad
    with pytest.raises(TypeError):
        _stub_grab(blur=True, blur_style="fill")

    # Rejected before anything captured, so the held frame is untouched.
    assert (frame[..., 0] == _MARKER[0]).all()
    assert grab.blur_style is not None


def test_empty_blur_list_does_not_import_the_effects_module():
    """blur=[] is documented as "capture unmodified", so it must stay lazy."""
    import subprocess
    import sys

    probe = (
        "import sys;"
        "from fastgrab import screenshot;"
        "g = screenshot.Screenshot(blur=[]);"
        "g.capture(blur=[]);"
        "assert 'fastgrab.effects' not in sys.modules, 'imported for a no-op';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_exhausted_generator_raises_on_the_per_call_path():
    """A per-call override stores nothing, so reuse must not fail open."""
    grab = _stub_grab(blur_style=_fill_style())
    gen = (r for r in [(4, 4, 8, 8)])
    assert (grab.capture(blur=gen)[4:12, 4:12, 0] == _MARKER[0]).all()
    with pytest.raises(ValueError, match="empty or already consumed"):
        grab.capture(blur=gen)


def test_exhausted_generator_raises_on_assignment():
    grab = _stub_grab(blur_style=_fill_style())
    gen = (r for r in [(4, 4, 8, 8)])
    grab.blur = gen
    with pytest.raises(ValueError, match="empty or already consumed"):
        grab.blur = gen
    # The rejected assignment left the previous value in place.
    assert grab.blur == ((4, 4, 8, 8),)


def test_an_empty_generator_raises_without_importing_effects():
    import subprocess
    import sys

    probe = (
        "import sys;"
        "from fastgrab import screenshot;"
        "\ntry:\n"
        "    screenshot._normalise_blur((r for r in []))\n"
        "except ValueError:\n"
        "    assert 'fastgrab.effects' not in sys.modules, 'imported anyway';"
        "    print('ok')\n"
        "else:\n"
        "    raise AssertionError('an empty generator was accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
