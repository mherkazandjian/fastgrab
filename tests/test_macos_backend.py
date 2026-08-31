"""Tests for the macOS backend: the points-vs-pixels split on Retina.

The rect arithmetic is a pure function, tested directly. The copy path
memmoves out of a CFData pointer, so it gets a fake ctypes layer — the
only place ``off_x``/``off_y`` ever run, since CI Macs are 1x.

The fakes deliberately model CoreGraphics where it is *unforgiving*:
``_FakeCoreGraphics._image`` refuses an out-of-bounds rect rather than
letting numpy clip it into a zero-padded image, the display mode can
under-report its pixel size so an oversized capture can be exercised,
and CoreGraphics and CoreFoundation are separate objects so a symbol
called on the wrong framework handle fails here the way it would on a
real machine.
"""
import ctypes
from types import SimpleNamespace

import numpy
import pytest

from fastgrab.backends.macos import (
    MacosBackend,
    _snap_rect_to_points,
    _K_CG_BITMAP_BYTE_ORDER_32_LITTLE,
)

# a 40x30-pixel backing store; RETINA drives it from a 20x15-point desktop
RETINA = dict(pixel_w=40, pixel_h=30, point_w=20, point_h=15)
FLAT = dict(pixel_w=40, pixel_h=30, point_w=40, point_h=30)


@pytest.mark.parametrize("rect, expected", [
    # (x, y, w, h) -> (pt_x, pt_y, pt_w, pt_h, off_x, off_y)
    ((0, 0, 8, 6), (0, 0, 4, 3, 0, 0)),        # aligned to the scale
    ((2, 4, 10, 8), (1, 2, 5, 4, 0, 0)),       # even offset and size
    ((7, 5, 11, 9), (3, 2, 6, 5, 1, 1)),       # odd -> snaps out, offset 1,1
    ((1, 1, 1, 1), (0, 0, 1, 1, 1, 1)),        # single pixel inside a point
    ((39, 29, 1, 1), (19, 14, 1, 1, 1, 1)),    # bottom-right corner
])
def test_rect_snaps_outward_to_whole_points(rect, expected):
    assert _snap_rect_to_points(*rect, **RETINA) == expected


@pytest.mark.parametrize("rect", [
    (0, 0, 8, 6),
    (7, 5, 11, 9),
    (1, 1, 1, 1),        # degenerate single pixel, the 1x twin of the above
    (39, 29, 1, 1),
])
def test_rect_is_unchanged_without_scaling(rect):
    x, y, w, h = rect
    assert _snap_rect_to_points(*rect, **FLAT) == (x, y, w, h, 0, 0)


class _FakeCoreGraphics:
    """The slice of CoreGraphics the backend calls, over a numpy screen."""

    def __init__(self, screen, point_w, point_h, row_pad=0,
                 mode_pixel_w=None, mode_pixel_h=None):
        self.screen = screen
        self.pixel_h, self.pixel_w = screen.shape[:2]
        self.point_w, self.point_h = point_w, point_h
        self.row_pad = row_pad
        # What CGDisplayModeGetPixel{Width,Height} claim. Defaults to the
        # real backing store; a test can make the mode under-report to
        # model a display whose mode dict omits the HiDPI pixel fields.
        self.mode_pixel_w = self.pixel_w if mode_pixel_w is None else mode_pixel_w
        self.mode_pixel_h = self.pixel_h if mode_pixel_h is None else mode_pixel_h
        self.bits_per_component = 8
        self.main_display = 1
        self._meta = {}
        self._buffers = {}

    def CGMainDisplayID(self):
        return self.main_display

    def CGDisplayCopyDisplayMode(self, display):
        return 1

    def CGDisplayModeRelease(self, mode):
        pass

    def CGDisplayModeGetWidth(self, mode):
        return self.point_w

    def CGDisplayModeGetHeight(self, mode):
        return self.point_h

    def CGDisplayModeGetPixelWidth(self, mode):
        return self.mode_pixel_w

    def CGDisplayModeGetPixelHeight(self, mode):
        return self.mode_pixel_h

    def _image(self, x, y, w, h):
        if x < 0 or y < 0 or x + w > self.pixel_w or y + h > self.pixel_h:
            # numpy would clip this to a short slice and leave the rest of
            # the buffer zeroed while the metadata still claimed the full
            # size — a silently "valid" image no real API would return.
            raise AssertionError(
                "rect %r falls outside the %dx%d screen; real CoreGraphics "
                "clips to the display bounds and returns a smaller image"
                % ((x, y, w, h), self.pixel_w, self.pixel_h))
        stride = w * 4 + self.row_pad
        buf = (ctypes.c_ubyte * (stride * h))()
        for index, row in enumerate(self.screen[y:y + h, x:x + w]):
            ctypes.memmove(ctypes.byref(buf, index * stride), row.tobytes(),
                           w * 4)
        token = len(self._meta) + 1
        self._meta[token] = (w, h, stride)
        self._buffers[token] = buf
        return token

    def CGDisplayCreateImage(self, display):
        # Always the real backing store, which is what makes an
        # under-reporting mode produce an oversized image.
        return self._image(0, 0, self.pixel_w, self.pixel_h)

    def CGDisplayCreateImageForRect(self, display, rect):
        asked = (rect.origin.x, rect.origin.y,
                 rect.size.width, rect.size.height)
        # the real API rounds a fractional rect outward, which is the
        # trap the backend exists to avoid -- so refuse to model it
        assert all(v == int(v) for v in asked), \
            "rect must be whole points, got %r" % (asked,)
        scale_x = self.pixel_w / self.point_w
        scale_y = self.pixel_h / self.point_h
        return self._image(int(asked[0] * scale_x), int(asked[1] * scale_y),
                           int(asked[2] * scale_x), int(asked[3] * scale_y))

    def CGImageGetWidth(self, image):
        return self._meta[image][0]

    def CGImageGetHeight(self, image):
        return self._meta[image][1]

    def CGImageGetBytesPerRow(self, image):
        return self._meta[image][2]

    def CGImageGetBitsPerPixel(self, image):
        return 32

    def CGImageGetBitsPerComponent(self, image):
        return self.bits_per_component

    def CGImageGetBitmapInfo(self, image):
        return _K_CG_BITMAP_BYTE_ORDER_32_LITTLE

    def CGImageGetDataProvider(self, image):
        return image

    def CGDataProviderCopyData(self, provider):
        return provider

    def CGImageRelease(self, image):
        pass


class _FakeCoreFoundation:
    """The CoreFoundation symbols, deliberately a separate object.

    Collapsing this into the CoreGraphics fake would hide a symbol
    looked up on the wrong framework handle — which on a real machine
    either raises or resolves with no argtypes set and truncates a
    pointer to 32 bits.
    """

    def __init__(self, cg):
        self._cg = cg

    def CFDataGetBytePtr(self, data):
        return ctypes.addressof(self._cg._buffers[data])

    def CFDataGetLength(self, data):
        return len(self._cg._buffers[data])

    def CFRelease(self, data):
        pass


def _backend(point_w=20, point_h=15, row_pad=0,
             mode_pixel_w=None, mode_pixel_h=None):
    """A MacosBackend wired to fakes, bypassing framework loading."""
    screen = numpy.random.default_rng(0).integers(
        0, 256, (30, 40, 4), dtype=numpy.uint8)
    cg = _FakeCoreGraphics(screen, point_w, point_h, row_pad,
                           mode_pixel_w, mode_pixel_h)
    cf = _FakeCoreFoundation(cg)
    backend = object.__new__(MacosBackend)
    backend._cg = cg
    backend._cf = cf
    backend._display = 1
    # the backend builds these with kwargs; the fake only reads attributes
    backend._CGPoint = backend._CGSize = backend._CGRect = SimpleNamespace
    return backend, cg, cf


def _capture(backend, x, y, w, h):
    img = numpy.zeros((h, w, 4), numpy.uint8)
    backend.screenshot(x, y, img)
    return img


def test_resolution_reports_pixels_not_points():
    """The original bug: resolution() returned points, capture got pixels."""
    backend, _, _ = _backend()
    assert backend.resolution() == (40, 30)


@pytest.mark.parametrize("row_pad", [0, 12])
@pytest.mark.parametrize("x, y, w, h", [
    (0, 0, 40, 30),      # full screen, via CGDisplayCreateImage
    (0, 0, 8, 6),
    (7, 5, 11, 9),       # odd offset and size -> sub-rectangle copy
    (8, 5, 8, 9),        # off_x == 0 but off_y == 1: the off_y-only path
    (39, 29, 1, 1),
])
def test_capture_matches_the_backing_store(x, y, w, h, row_pad):
    backend, cg, _ = _backend(row_pad=row_pad)
    img = _capture(backend, x, y, w, h)
    assert img.shape == (h, w, 4)
    assert numpy.array_equal(img, cg.screen[y:y + h, x:x + w])


def test_capture_matches_without_scaling():
    backend, cg, _ = _backend(point_w=40, point_h=30)
    assert numpy.array_equal(_capture(backend, 7, 5, 11, 9),
                             cg.screen[5:14, 7:18])


def test_undersized_image_is_rejected_rather_than_over_read():
    backend, cg, _ = _backend()
    real = cg.CGImageGetWidth
    cg.CGImageGetWidth = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="too small"):
        _capture(backend, 7, 5, 11, 9)


def test_undersized_image_height_is_rejected_rather_than_over_read():
    """The height half of the guard; only the width half was covered."""
    backend, cg, _ = _backend()
    real = cg.CGImageGetHeight
    cg.CGImageGetHeight = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="too small"):
        _capture(backend, 7, 5, 11, 9)


def test_oversized_full_screen_image_is_rejected():
    """A mode that under-reports must not yield a silent top-left crop.

    Models a virtual/Sidecar-style display whose mode dict omits the
    HiDPI pixel fields, so the mode claims 20x16 while the capture comes
    back as the real 40x30 backing store. A lower-bound size check
    passes that happily and returns the top-left corner.
    """
    backend, _, _ = _backend(mode_pixel_w=20, mode_pixel_h=16)
    assert backend.resolution() == (20, 16)
    with pytest.raises(RuntimeError, match="does not match"):
        _capture(backend, 0, 0, 20, 16)


def test_oversized_cfdata_is_rejected():
    """The parent-bitmap layout makes the provider *larger*, not shorter.

    A short-buffer check alone never fires for it: a cropped CGImage
    sharing its parent's storage reports the parent's stride and hands
    back the parent's bytes, so every row is read from the wrong origin
    while all the size checks pass.
    """
    backend, _, cf = _backend()
    real = cf.CFDataGetLength
    cf.CFDataGetLength = lambda data: real(data) + 4096
    with pytest.raises(RuntimeError, match="unsupported provider extent"):
        _capture(backend, 7, 5, 11, 9)


def test_stride_narrower_than_a_row_is_rejected():
    backend, cg, _ = _backend()
    real = cg.CGImageGetBytesPerRow
    cg.CGImageGetBytesPerRow = lambda image: real(image) - 4
    with pytest.raises(RuntimeError, match="unsupported provider extent"):
        _capture(backend, 7, 5, 11, 9)


def test_short_cfdata_is_rejected_rather_than_over_read():
    """The provider bytes must cover what the stride arithmetic reads.

    A CGImage that CoreGraphics backs with a parent bitmap reports the
    parent's stride while its data starts at the parent's origin; the
    size checks all pass and every row is copied from the wrong place.
    """
    backend, _, cf = _backend()
    real = cf.CFDataGetLength
    cf.CFDataGetLength = lambda data: real(data) - 1
    with pytest.raises(RuntimeError, match="CFData"):
        _capture(backend, 7, 5, 11, 9)


def test_non_8_bit_components_are_rejected():
    """A 32-bpp 10-10-10-2 surface passes every other check."""
    backend, cg, _ = _backend()
    cg.bits_per_component = 10
    with pytest.raises(RuntimeError, match="bits per component"):
        _capture(backend, 0, 0, 8, 6)


@pytest.mark.parametrize("dimension", [
    "mode_pixel_w", "mode_pixel_h", "point_w", "point_h",
])
def test_degenerate_display_mode_raises_a_clear_error(dimension):
    """A zero dimension must not surface as a bare ZeroDivisionError."""
    backend, cg, _ = _backend()
    setattr(cg, dimension, 0)
    with pytest.raises(RuntimeError, match="display mode"):
        backend.screenshot(1, 1, numpy.zeros((2, 2, 4), numpy.uint8))


@pytest.mark.parametrize("x, y, w, h", [
    (-1, 0, 8, 6),
    (0, -1, 8, 6),
    (36, 0, 8, 6),       # runs past the right edge
    (0, 26, 8, 6),       # runs past the bottom edge
    (0, 0, 0, 6),        # empty region
])
def test_out_of_range_region_is_refused_at_the_backend(x, y, w, h):
    """The backend is reachable directly, not only via check_bbox.

    CoreGraphics clips such a rect rather than refusing it, so without
    this the caller gets a confusing "too small" error or, worse,
    silently shifted pixels.
    """
    backend, _, _ = _backend()
    with pytest.raises(ValueError, match="outside the"):
        backend.screenshot(x, y, numpy.zeros((h, w, 4), numpy.uint8))


def test_refresh_re_reads_the_main_display():
    """The display id is latched in __init__; refresh() must re-resolve."""
    backend, cg, _ = _backend()
    assert backend._display == 1
    cg.main_display = 7
    backend.refresh()
    assert backend._display == 7


def test_refresh_rejects_a_vanished_main_display():
    backend, cg, _ = _backend()
    cg.main_display = 0
    with pytest.raises(RuntimeError, match="no main display"):
        backend.refresh()


def test_out_of_bounds_rect_is_not_silently_zero_padded():
    """Guards the fake itself: numpy slicing used to clip these away."""
    backend, cg, _ = _backend()
    with pytest.raises(AssertionError, match="outside the"):
        cg._image(35, 28, 10, 10)
